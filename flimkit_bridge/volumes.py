import re
from pathlib import Path

import numpy as np

SLICE_RE = re.compile(r'^z(\d+)$')

CHANNELS = (
    ('intensity', 'photons'),
    ('tau_mean_int', 'ns'),
    ('tau_mean_amp', 'ns'),
    ('tau_mean', 'ns'),
    ('alpha_1', ''),
    ('alpha_2', ''),
    ('alpha_3', ''),
    ('bound_fraction', ''),
    ('chi2_r', ''),
    ('calibrated_chi2_r', ''),
)


class NothingToStack(Exception):
    pass


def slice_dirs(group_dir):
    found = []
    for entry in sorted(Path(group_dir).iterdir()):
        if not entry.is_dir():
            continue
        matched = SLICE_RE.match(entry.name)
        if matched:
            found.append((int(matched.group(1)), entry))
    return [entry for _z, entry in sorted(found)]


def z_values(group_dir):
    found = []
    for entry in Path(group_dir).iterdir():
        matched = SLICE_RE.match(entry.name) if entry.is_dir() else None
        if matched:
            found.append(int(matched.group(1)))
    return sorted(found)


def _stacked(group_dir, prefix, name, dirs):
    whole = Path(group_dir) / f'{prefix}_{name}_stack.npy'
    if whole.exists():
        return np.load(str(whole))
    frames = []
    for entry in dirs:
        plane = entry / f'{name}.npy'
        if not plane.exists():
            return None
        frames.append(np.load(str(plane)))
    if not frames:
        return None
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise NothingToStack(
            f'the {name} maps under {group_dir} are not all one shape: '
            f'{sorted(shapes)}')
    return np.stack(frames)


def read_group(group_dir, prefix):
    dirs = slice_dirs(group_dir)
    if not dirs:
        raise NothingToStack(f'{group_dir} holds no z0000-style slice folders')
    names = []
    units = []
    planes = []
    for name, unit in CHANNELS:
        found = _stacked(group_dir, prefix, name, dirs)
        if found is None:
            continue
        found = np.asarray(found, dtype=np.float32)
        if found.ndim != 3:
            continue
        names.append(name)
        units.append(unit)
        planes.append(found)
    if not planes:
        raise NothingToStack(f'{group_dir} holds no per-slice maps to stack')
    shapes = {plane.shape for plane in planes}
    if len(shapes) != 1:
        raise NothingToStack(
            f'the slice maps in {group_dir} are not all one shape: {sorted(shapes)}')
    return np.stack(planes), names, units


def _omero(names, units):
    channels = []
    for name, unit in zip(names, units):
        channels.append({
            'label': f'{name} ({unit})' if unit else name,
            'active': True,
            'color': 'FFFFFF',
            'window': {'start': 0.0, 'end': 1.0, 'min': 0.0, 'max': 1.0},
        })
    return {'channels': channels, 'rdefs': {'model': 'greyscale'}}


ZARR_FORMAT = 2

NGFF_VERSION = '0.4'


def _compressor():
    """A codec the readers can actually decompress.

    zarr-python 3 defaults to zstd, and QuPath's jzarr refuses it outright:
    "Compressor id:'zstd' not supported". zlib is understood by every Zarr v2
    reader and, on lifetime maps that are largely NaN, compresses better than
    blosc anyway (4.3 MB against 6.7 MB on a 16.8 MB volume).
    """
    import numcodecs

    return numcodecs.Zlib(level=5)


def _escape(text):
    from xml.sax.saxutils import escape

    return escape(str(text), {'"': '&quot;'})


def _ome_xml(name, shape, names, units, voxel_size_um):
    """The OME-XML a bioformats2raw store carries beside its arrays.

    The NGFF metadata alone gets the pixels across but nothing else: QuPath
    reads channel names and physical pixel sizes out of this file, and
    without it every channel arrives called "Channel 1" with no calibration.
    """
    t, c, z, y, x = shape
    z_um, y_um, x_um = voxel_size_um
    channels = ''.join(
        '<Channel ID="Channel:0:{i}" Name="{name}" SamplesPerPixel="1">'
        '<LightPath/></Channel>'.format(i=i, name=_escape(label))
        for i, label in enumerate(_labels(names, units)))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 '
        'http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">'
        f'<Image ID="Image:0" Name="{_escape(name)}">'
        f'<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="float" '
        f'SizeX="{x}" SizeY="{y}" SizeZ="{z}" SizeC="{c}" SizeT="{t}" '
        f'PhysicalSizeX="{x_um}" PhysicalSizeXUnit="\u00b5m" '
        f'PhysicalSizeY="{y_um}" PhysicalSizeYUnit="\u00b5m" '
        f'PhysicalSizeZ="{z_um}" PhysicalSizeZUnit="\u00b5m">'
        f'{channels}<MetadataOnly/>'
        '</Pixels></Image></OME>')


def _labels(names, units):
    labelled = []
    for i, name in enumerate(names):
        unit = units[i] if i < len(units) else ''
        labelled.append(f'{name} ({unit})' if unit else name)
    return labelled


def _multiscales(name, voxel_size_um):
    z_um, y_um, x_um = voxel_size_um
    return [{
        'version': NGFF_VERSION,
        'name': name,
        'axes': [
            {'name': 't', 'type': 'time'},
            {'name': 'c', 'type': 'channel'},
            {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
            {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
            {'name': 'x', 'type': 'space', 'unit': 'micrometer'},
        ],
        # One resolution level, deliberately. A pyramid is built by
        # interpolating between neighbours, and a lifetime map is mostly NaN
        # where nothing was fitted, so the coarse levels a viewer shows when
        # zoomed out would smear that NaN over the pixels that did fit.
        # Chunking is what keeps a large volume readable, not a pyramid.
        'datasets': [{
            'path': '0',
            'coordinateTransformations': [
                {'type': 'scale', 'scale': [1.0, 1.0, z_um, y_um, x_um]}],
        }],
    }]


def _write_json(path, payload):
    import json

    path.write_text(json.dumps(payload, indent=2))


def write_ome_zarr(path, volume, names, units, voxel_size_um):
    """Write one fitted stack as an OME-Zarr in the bioformats2raw layout.

    The layout is not a preference: QuPath reads OME-Zarr through
    Bio-Formats, which wants a root marked `bioformats2raw.layout`, the image
    under a numbered series group, and an OME/METADATA.ome.xml beside it. A
    plain NGFF store at the root opens with no channel names and no pixel
    calibration, and older ome-zarr defaults would not open at all.
    """
    import shutil

    import zarr

    path = Path(path)
    # A second run over the same output folder would otherwise read back a
    # mixture of the two writes.
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()

    z_um, y_um, x_um = [float(v) for v in voxel_size_um]
    volume = np.ascontiguousarray(volume, dtype=np.float32)
    # Bio-Formats expects the five axes it knows; a fit has no time axis, so
    # it gets a single timepoint rather than a different number of axes.
    five = volume[np.newaxis]
    _, channels, depth, height, width = five.shape

    path.mkdir(parents=True)
    _write_json(path / '.zgroup', {'zarr_format': ZARR_FORMAT})
    _write_json(path / '.zattrs', {'bioformats2raw.layout': 3})

    series = path / '0'
    series.mkdir()
    _write_json(series / '.zgroup', {'zarr_format': ZARR_FORMAT})
    _write_json(series / '.zattrs', {
        'multiscales': _multiscales(path.name, (z_um, y_um, x_um)),
        'omero': _omero(names, units),
        'flimkit': {'channels': list(names), 'units': list(units),
                    'voxel_size_um': [z_um, y_um, x_um]},
    })
    array = _open_array(zarr, series / '0', five.shape,
                        (1, 1, 1, min(256, height), min(256, width)))
    array[:] = five

    ome = path / 'OME'
    ome.mkdir()
    _write_json(ome / '.zgroup', {'zarr_format': ZARR_FORMAT})
    _write_json(ome / '.zattrs', {'series': ['0']})
    (ome / 'METADATA.ome.xml').write_text(_ome_xml(
        path.name, five.shape, names, units, (z_um, y_um, x_um)),
        encoding='utf-8')
    return str(path)


def _open_array(zarr, where, shape, chunks):
    options = dict(mode='w', shape=shape, chunks=chunks, dtype='float32',
                   compressor=_compressor())
    try:
        return zarr.open_array(str(where), zarr_format=ZARR_FORMAT, **options)
    except TypeError:
        # zarr 2 has no zarr_format argument and writes v2 regardless.
        return zarr.open_array(str(where), **options)


def write_ome_tiff(path, volume, names, units, voxel_size_um):
    import tifffile

    path = Path(path)
    z_um, y_um, x_um = [float(v) for v in voxel_size_um]
    labelled = [f'{name} ({unit})' if unit else name
                for name, unit in zip(names, units)]
    tifffile.imwrite(
        str(path),
        np.ascontiguousarray(np.moveaxis(volume, 0, 1), dtype=np.float32),
        ome=True, photometric='minisblack',
        resolution=(1.0 / x_um, 1.0 / y_um) if x_um and y_um else None,
        metadata={'axes': 'ZCYX', 'Channel': {'Name': labelled},
                  'PhysicalSizeX': x_um, 'PhysicalSizeXUnit': 'µm',
                  'PhysicalSizeY': y_um, 'PhysicalSizeYUnit': 'µm',
                  'PhysicalSizeZ': z_um, 'PhysicalSizeZUnit': 'µm'})
    return str(path)


def read_metadata(path):
    """Read back what write_ome_zarr recorded, from the series group where
    the bioformats2raw layout keeps it."""
    import json

    series = Path(path) / '0' / '.zattrs'
    if not series.exists():
        series = Path(path) / '.zattrs'
    attrs = json.loads(series.read_text()) if series.exists() else {}
    return {
        'flimkit': attrs.get('flimkit') or {},
        'omero': attrs.get('omero') or {},
        'multiscales': attrs.get('multiscales') or [],
    }


FORMATS = ('ome-zarr', 'ome-tiff')


def save_volume(output_dir, name, volume, names, units, voxel_size_um,
                prefer='ome-zarr'):
    if prefer not in FORMATS:
        raise ValueError(f'format must be one of {list(FORMATS)}, got {prefer!r}')
    output_dir = Path(output_dir)
    if prefer == 'ome-zarr':
        target = output_dir / f'{name}.ome.zarr'
        return write_ome_zarr(target, volume, names, units, voxel_size_um), 'ome-zarr'
    target = output_dir / f'{name}.ome.tif'
    return write_ome_tiff(target, volume, names, units, voxel_size_um), 'ome-tiff'
