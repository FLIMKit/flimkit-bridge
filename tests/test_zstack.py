import os
from pathlib import Path

import numpy as np
import pytest

from flimkit_bridge import volumes, zstack

STACK_DIR = os.environ.get('FLIMKIT_TEST_ZSTACK', '')

needs_zstack = pytest.mark.skipif(
    not STACK_DIR or not Path(STACK_DIR).is_dir(),
    reason='set FLIMKIT_TEST_ZSTACK to a folder of region_zN.ptu files')


def test_defaults_describe_every_parameter():
    described = zstack.defaults()
    assert set(described['values']) == {entry['key'] for entry in zstack.SCHEMA}
    assert all(entry['applies_to'] == ['zstack'] for entry in described['schema'])


def test_unknown_parameters_are_refused():
    with pytest.raises(ValueError) as raised:
        zstack.merge_params({'nexp': 2})
    assert 'unknown z-stack parameter' in str(raised.value)


def test_tau_bounds_are_checked():
    with pytest.raises(ValueError):
        zstack.merge_params({'tau_min_ns': 6.0, 'tau_max_ns': 1.0})


def test_a_half_supplied_reference_is_refused():
    with pytest.raises(ValueError) as raised:
        zstack.merge_params({'n_exp': 2, 'ref_tau1_ns': 0.4})
    assert 'all 2 components' in str(raised.value)


def test_a_whole_reference_is_kept():
    merged = zstack.merge_params({'n_exp': 2, 'ref_tau1_ns': 0.4,
                                  'ref_tau2_ns': 2.9})
    assert merged['ref_tau1_ns'] == 0.4
    assert merged['ref_tau3_ns'] is None


def test_a_missing_folder_is_refused():
    with pytest.raises(FileNotFoundError):
        zstack.scan('/nowhere/slices')


def test_a_folder_without_slices_is_refused(tmp_path):
    (tmp_path / 'notes.txt').write_text('nothing here')
    with pytest.raises(ValueError) as raised:
        zstack.resolve(tmp_path)
    assert 'region_zN.ptu' in str(raised.value)


def test_slices_group_by_their_names(tmp_path):
    for z in (1, 2, 3):
        (tmp_path / f'RegionA_z{z}.ptu').write_bytes(b'')
    (tmp_path / 'RegionB_t1_s2_z1.ptu').write_bytes(b'')
    groups = {group['label']: group for group in zstack.scan(tmp_path)}
    assert groups['RegionA']['n_slices'] == 3
    assert groups['RegionA']['z_first'] == 1 and groups['RegionA']['z_last'] == 3
    assert 'RegionB_t0001_s2' in groups


def test_the_output_folder_sits_beside_the_slices(tmp_path):
    (tmp_path / 'RegionA_z1.ptu').write_bytes(b'')
    _ptu_dir, output_dir, groups = zstack.resolve(tmp_path)
    assert output_dir.parent == tmp_path
    assert output_dir.name.endswith('_flimkit_zstack')
    assert len(groups) == 1


def _fake_group(root, label, n_z=3, shape=(4, 5)):
    group_dir = root / label
    for z in range(n_z):
        slice_dir = group_dir / f'z{z:04d}'
        slice_dir.mkdir(parents=True)
        for name in ('intensity', 'tau_mean_int', 'tau_mean_amp', 'alpha_1'):
            np.save(str(slice_dir / f'{name}.npy'),
                    np.full(shape, float(z + 1), dtype=np.float32))
    return group_dir


def test_slice_maps_stack_into_one_volume(tmp_path):
    group_dir = _fake_group(tmp_path, 'RegionA')
    volume, names, units = volumes.read_group(group_dir, 'RegionA')
    assert volume.shape == (4, 3, 4, 5)
    assert names == ['intensity', 'tau_mean_int', 'tau_mean_amp', 'alpha_1']
    assert units == ['photons', 'ns', 'ns', '']
    assert volume[0, 2].max() == 3.0


def test_a_saved_stack_is_preferred_over_the_slices(tmp_path):
    group_dir = _fake_group(tmp_path, 'RegionA')
    np.save(str(group_dir / 'RegionA_intensity_stack.npy'),
            np.zeros((3, 4, 5), dtype=np.float32))
    volume, names, _units = volumes.read_group(group_dir, 'RegionA')
    assert volume[names.index('intensity')].max() == 0.0


def test_a_group_without_slices_is_refused(tmp_path):
    (tmp_path / 'RegionA').mkdir()
    with pytest.raises(volumes.NothingToStack):
        volumes.read_group(tmp_path / 'RegionA', 'RegionA')


def test_ragged_slices_are_refused(tmp_path):
    group_dir = _fake_group(tmp_path, 'RegionA')
    np.save(str(group_dir / 'z0002' / 'intensity.npy'),
            np.zeros((6, 7), dtype=np.float32))
    with pytest.raises(volumes.NothingToStack):
        volumes.read_group(group_dir, 'RegionA')


def test_a_volume_round_trips_through_ome_tiff(tmp_path):
    import tifffile

    group_dir = _fake_group(tmp_path, 'RegionA')
    volume, names, units = volumes.read_group(group_dir, 'RegionA')
    written, kind = volumes.save_volume(
        tmp_path, 'RegionA', volume, names, units, (2.0, 0.5, 0.5),
        prefer='ome-tiff')
    assert kind == 'ome-tiff'
    read_back = tifffile.imread(written)
    assert read_back.shape == (3, 4, 4, 5)
    assert np.allclose(np.moveaxis(read_back, 1, 0), volume)


def test_a_volume_round_trips_through_ome_zarr(tmp_path):
    group_dir = _fake_group(tmp_path, 'RegionA')
    volume, names, units = volumes.read_group(group_dir, 'RegionA')
    written, kind = volumes.save_volume(
        tmp_path, 'RegionA', volume, names, units, (2.0, 0.5, 0.5))
    assert kind == 'ome-zarr'
    assert written.endswith('.ome.zarr')

    import zarr

    # (T, C, Z, Y, X): Bio-Formats expects the five axes it knows, and a fit
    # has no time axis, so it gets a single timepoint.
    stored = np.asarray(zarr.open_array(str(Path(written) / '0' / '0'), mode='r'))
    assert stored.shape == (1,) + volume.shape
    assert np.allclose(stored[0], volume)
    described = volumes.read_metadata(written)
    assert described['flimkit']['channels'] == names
    assert described['flimkit']['voxel_size_um'] == [2.0, 0.5, 0.5]
    labels = [c['label'] for c in described['omero']['channels']]
    assert labels[0] == 'intensity (photons)'
    assert len(described['multiscales'][0]['datasets']) == 1, \
        'the fitted values should be the only resolution level'
    scale = described['multiscales'][0]['datasets'][0][
        'coordinateTransformations'][0]['scale']
    assert scale == [1.0, 1.0, 2.0, 0.5, 0.5]


def test_writing_twice_replaces_the_store(tmp_path):
    group_dir = _fake_group(tmp_path, 'RegionA')
    volume, names, units = volumes.read_group(group_dir, 'RegionA')
    volumes.save_volume(tmp_path, 'RegionA', volume, names, units, (1.0, 1.0, 1.0))
    thinner = volume[:, :2]
    written, _kind = volumes.save_volume(
        tmp_path, 'RegionA', thinner, names, units, (1.0, 1.0, 1.0))

    import zarr

    stored = zarr.open_array(str(Path(written) / '0' / '0'), mode='r')
    assert stored.shape == (1,) + thinner.shape


def test_a_run_is_summarised_into_products(tmp_path):
    group_dir = _fake_group(tmp_path, 'RegionA')
    (group_dir / 'RegionA_zseries.csv').write_text('z,tau_mean_mean\n0,1.9\n')
    params = zstack.merge_params({'z_step_um': 2.0, 'volume_format': 'ome-tiff'})
    summary = zstack.summarise(
        {'RegionA': {'group_dir': str(group_dir), 'n_slices': 3,
                     'taus_ns': [0.4, 2.9],
                     'z_series': {0: {'tau_mean_mean': 1.9, 'path': '/tmp/a.ptu'}}}},
        tmp_path, [{'label': 'RegionA', 'files': []}], params)
    product = summary['products'][0]
    assert product['image_id'] == 'RegionA'
    assert product['format'] == 'ome-tiff'
    assert product['n_z'] == 3
    assert product['axes'] == 'CZYX'
    assert product['voxel_size_um'][0] == 2.0
    assert Path(product['file']).exists()
    stack = summary['stacks'][0]
    assert stack['taus_ns'] == [0.4, 2.9]
    assert stack['z_series'] == [{'z': 0, 'tau_mean_mean': 1.9, 'path': '/tmp/a.ptu'}]
    assert stack['z_series_csv'].endswith('RegionA_zseries.csv')


def test_an_unknown_volume_format_is_refused(tmp_path):
    group_dir = _fake_group(tmp_path, 'RegionA')
    volume, names, units = volumes.read_group(group_dir, 'RegionA')
    with pytest.raises(ValueError):
        volumes.save_volume(tmp_path, 'RegionA', volume, names, units,
                            (1.0, 1.0, 1.0), prefer='nifti')


def test_a_group_that_cannot_be_stacked_is_reported_not_raised(tmp_path):
    (tmp_path / 'RegionA').mkdir()
    params = zstack.merge_params({})
    summary = zstack.summarise(
        {'RegionA': {'group_dir': str(tmp_path / 'RegionA')}},
        tmp_path, [{'label': 'RegionA', 'files': []}], params)
    assert summary['products'] == []
    assert 'could not stack' in summary['stacks'][0]['error']


@needs_zstack
def test_the_fit_gets_the_arguments_it_needs():
    ptu_dir, output_dir, groups = zstack.resolve(STACK_DIR)
    params = zstack.merge_params({'n_exp': 1, 'z_step_um': 1.5})
    args = zstack.build_args(ptu_dir, output_dir, params)
    for name in ('ptu_dir', 'output_dir', 'nexp', 'tau_min', 'tau_max',
                 'estimate_irf', 'machine_irf', 'min_photons', 'save_stack',
                 'bound_fraction', 'optimizer', 'workers'):
        assert hasattr(args, name), f'fit_zstack reads args.{name}'
    assert args.nexp == 1
    assert groups and groups[0]['n_slices'] >= 1


@needs_zstack
def test_a_real_folder_reports_its_stacks():
    groups = zstack.scan(STACK_DIR)
    assert groups
    assert all(group['n_slices'] >= 1 for group in groups)


def test_the_pooled_fit_travels_with_the_result(tmp_path):
    import json

    group_dir = _fake_group(tmp_path, 'RegionA')
    (group_dir / 'RegionA_reference_fit.json').write_text(json.dumps({
        'taus_ns': [3.443, 0.568], 'nexp': 2, 'total_pooled_photons': 1731281.0,
        'estimate_irf': 'machine_irf', 'user_supplied_tau': False,
        'calibrated_chi2_pearson': 151.4, 'n_slices': 8,
        'z_slices': [1, 2, 3, 4, 5, 6, 7, 8]}))
    params = zstack.merge_params({'volume_format': 'ome-tiff'})

    summary = zstack.summarise(
        {'RegionA': {'group_dir': str(group_dir)}},
        tmp_path, [{'label': 'RegionA', 'files': []}], params)

    pooled = summary['stacks'][0]['pooled']
    assert pooled['taus_ns'] == [3.443, 0.568]
    assert pooled['total_pooled_photons'] == 1731281.0
    assert pooled['estimate_irf'] == 'machine_irf'
    assert 'z_slices' not in pooled, 'one column per slice is not a summary'


def test_a_stack_without_a_pooled_fit_says_none(tmp_path):
    group_dir = _fake_group(tmp_path, 'RegionA')
    params = zstack.merge_params({'volume_format': 'ome-tiff'})

    summary = zstack.summarise(
        {'RegionA': {'group_dir': str(group_dir)}},
        tmp_path, [{'label': 'RegionA', 'files': []}], params)

    assert summary['stacks'][0]['pooled'] is None


def test_the_pooled_fit_rides_on_the_product_too(tmp_path):
    import json

    group_dir = _fake_group(tmp_path, 'RegionA')
    (group_dir / 'RegionA_reference_fit.json').write_text(
        json.dumps({'taus_ns': [3.443, 0.568], 'nexp': 2,
                    'calibrated_chi2_pearson': 151.4}))
    params = zstack.merge_params({'volume_format': 'ome-tiff'})

    summary = zstack.summarise(
        {'RegionA': {'group_dir': str(group_dir)}},
        tmp_path, [{'label': 'RegionA', 'files': []}], params)

    pooled = summary['products'][0]['pooled']
    assert pooled['calibrated_chi2_pearson'] == 151.4
    assert pooled['nexp'] == 2


def test_the_store_is_written_in_the_layout_readers_expect(tmp_path):
    import json

    group_dir = _fake_group(tmp_path, 'RegionA')
    volume, names, units = volumes.read_group(group_dir, 'RegionA')
    written, _kind = volumes.save_volume(
        tmp_path, 'RegionA', volume, names, units, (2.0, 0.65, 0.65))

    store = Path(written)
    # bioformats2raw on Zarr v2. QuPath reads OME-Zarr through Bio-Formats,
    # which needs the layout marker, the image under a numbered series, and
    # the OME-XML beside it; a plain NGFF store at the root opens with no
    # channel names and no calibration, and Zarr v3 does not open at all.
    assert json.loads((store / '.zattrs').read_text())['bioformats2raw.layout'] == 3
    assert not (store / 'zarr.json').exists(), 'zarr.json means Zarr v3'
    assert (store / 'OME' / 'METADATA.ome.xml').exists()
    assert json.loads((store / 'OME' / '.zattrs').read_text())['series'] == ['0']
    series = json.loads((store / '0' / '.zattrs').read_text())
    assert series['multiscales'][0]['version'] == '0.4'
    assert [a['name'] for a in series['multiscales'][0]['axes']] == \
        ['t', 'c', 'z', 'y', 'x']
    assert json.loads((store / '0' / '0' / '.zarray').read_text())['zarr_format'] == 2


def test_the_store_uses_a_codec_the_readers_have(tmp_path):
    import json

    group_dir = _fake_group(tmp_path, 'RegionA')
    volume, names, units = volumes.read_group(group_dir, 'RegionA')
    written, _kind = volumes.save_volume(
        tmp_path, 'RegionA', volume, names, units, (1.0, 1.0, 1.0))

    codec = json.loads(
        (Path(written) / '0' / '0' / '.zarray').read_text())['compressor']
    # zarr-python 3 defaults to zstd and QuPath's jzarr refuses it outright:
    # "Compressor id:'zstd' not supported".
    assert codec['id'] == 'zlib', codec


def test_the_ome_xml_carries_the_names_and_the_calibration(tmp_path):
    group_dir = _fake_group(tmp_path, 'RegionA')
    volume, names, units = volumes.read_group(group_dir, 'RegionA')
    written, _kind = volumes.save_volume(
        tmp_path, 'RegionA', volume, names, units, (2.0, 0.65, 0.65))

    xml = (Path(written) / 'OME' / 'METADATA.ome.xml').read_text(encoding='utf-8')
    # Without this file QuPath names every channel "Channel 1" and reports no
    # pixel size at all; the NGFF metadata alone does not reach it.
    assert 'Name="intensity (photons)"' in xml
    assert 'Name="tau_mean_int (ns)"' in xml
    assert 'Name="alpha_1"' in xml, 'a unitless map gains no empty brackets'
    assert 'PhysicalSizeZ="2.0"' in xml
    assert 'PhysicalSizeX="0.65"' in xml
    assert f'SizeC="{len(names)}"' in xml and 'SizeZ="3"' in xml and 'SizeT="1"' in xml


def test_a_name_with_xml_in_it_cannot_break_the_metadata(tmp_path):
    group_dir = _fake_group(tmp_path, 'RegionA')
    volume, _names, _units = volumes.read_group(group_dir, 'RegionA')
    written, _kind = volumes.save_volume(
        tmp_path, 'RegionA', volume, ['a<b&c"d'] + ['x'] * (volume.shape[0] - 1),
        [''] * volume.shape[0], (1.0, 1.0, 1.0))

    from xml.etree import ElementTree

    xml = (Path(written) / 'OME' / 'METADATA.ome.xml').read_text(encoding='utf-8')
    ElementTree.fromstring(xml)
    assert 'a&lt;b&amp;c&quot;d' in xml
