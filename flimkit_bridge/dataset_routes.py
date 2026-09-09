import re
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import numpy as np
import tifffile

from flimkit_bridge.datasets import StackTooLarge

PLANE_RE = re.compile(r'^/v1/datasets/([^/]+)/planes/([^/]+)\.tif$')
PLANES_RE = re.compile(r'^/v1/datasets/([^/]+)/planes$')
DATASET_RE = re.compile(r'^/v1/datasets/([^/]+)$')

UNITS = {'intensity': 'photons'}


class RouteError(Exception):

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _registry(state):
    registry = getattr(state, 'datasets', None)
    if registry is None:
        raise RouteError(503, 'this FLIMKit has no dataset registry')
    return registry


def open_dataset(state, payload):
    from flimkit_bridge import formats
    registry = _registry(state)
    path = (payload or {}).get('path')
    formats.identify(path)
    channel = (payload or {}).get('channel')
    try:
        ident = registry.open(path, channel=channel)
    except Exception as exc:
        raise RouteError(400, str(exc))
    return registry.metadata(ident)


def list_datasets(state):
    return {'datasets': _registry(state).list()}


def dataset(state, ident):
    registry = _registry(state)
    try:
        return registry.metadata(ident)
    except KeyError:
        raise RouteError(404, f'no such dataset: {ident}')


def close_dataset(state, ident):
    registry = _registry(state)
    try:
        registry.metadata(ident)
    except KeyError:
        raise RouteError(404, f'no such dataset: {ident}')
    return {'closed': registry.close(ident)}


def planes(state, ident):
    registry = _registry(state)
    try:
        meta = registry.metadata(ident)
    except KeyError:
        raise RouteError(404, f'no such dataset: {ident}')
    held = {p['id']: p for p in registry.planes(ident)}
    listed = []
    for name in registry.plane_names(ident):
        entry = held.get(name)
        if entry is None:
            listed.append({
                'id': name,
                'unit': UNITS.get(name, ''),
                'dtype': 'uint16',
                'shape': [meta['height'], meta['width']],
            })
        else:
            listed.append(entry)
    return {'planes': listed}


def plane_tiff(state, ident, name, query):
    from flimkit_bridge.server import encode_image
    registry = _registry(state)
    try:
        registry.metadata(ident)
    except KeyError:
        raise RouteError(404, f'no such dataset: {ident}')
    options = parse_qs(query)
    binning = int(options.get('binning', ['1'])[0])
    if binning < 1:
        raise RouteError(400, 'binning must be 1 or more')
    if name == 'stack':
        return _plane_stack(registry, ident, options, binning)
    if name == 'intensity':
        try:
            array = registry.intensity(ident, binning=binning)
        except StackTooLarge as exc:
            raise RouteError(413, str(exc))
    else:
        array = registry.plane(ident, name)
        if array is None:
            raise RouteError(404, f'no such plane: {name}')
    array = np.asarray(array)
    unit = UNITS.get(name) or registry.plane_unit(ident, name)
    array = _shrink(_crop(array, options), options)
    buffer = BytesIO()
    tifffile.imwrite(buffer, encode_image(name, array))
    return buffer.getvalue(), unit, binning, array.shape


PLANE_STATS_RE = re.compile(r'^/v1/datasets/([^/]+)/planes/stats$')


def plane_stats(state, ident, payload):
    from flimkit_bridge import fitting

    registry = _registry(state)
    try:
        meta = registry.metadata(ident)
    except KeyError:
        raise RouteError(404, f'no such dataset: {ident}')
    payload = payload or {}
    collection = payload.get('rois')
    if not isinstance(collection, dict) or collection.get('type') != 'FeatureCollection':
        raise RouteError(400, 'rois must be a GeoJSON FeatureCollection')
    binning = int(payload.get('binning', 1))
    asked = payload.get('planes')
    if not asked:
        asked = [n for n in registry.plane_names(ident) if n.startswith('tau')]
    if not asked:
        raise RouteError(409, 'this dataset has no fitted planes yet; fit per-pixel first')
    shape = (meta['height'], meta['width'])
    masks = fitting.masks_from_geojson(collection, shape, binning=binning)
    found = []
    for name, mask in masks:
        region = {'name': name, 'planes': {}}
        covered = np.asarray(mask, dtype=bool)
        region['n_pixels'] = int(covered.sum())
        for plane in asked:
            array = registry.plane(ident, plane)
            if array is None:
                continue
            array = np.asarray(array, dtype=float)
            if array.shape != covered.shape:
                raise RouteError(
                    409,
                    f'{plane} is {array.shape} and the mask is {covered.shape}; '
                    f'fit per-pixel at the binning you are asking stats for')
            values = array[covered]
            values = values[np.isfinite(values)]
            if values.size == 0:
                region['planes'][plane] = None
                continue
            region['planes'][plane] = {
                'mean': float(values.mean()),
                'median': float(np.median(values)),
                'std': float(values.std()),
                'n': int(values.size),
                'unit': UNITS.get(plane) or registry.plane_unit(ident, plane),
            }
        found.append(region)
    return {'dataset': ident, 'binning': binning, 'regions': found}


def _plane_stack(registry, ident, options, binning):
    from flimkit_bridge.server import encode_image

    asked = [n for n in (options.get('planes', [''])[0]).split(',') if n]
    if not asked:
        raise RouteError(400, 'planes must name at least one plane, comma separated')
    layers = []
    names = []
    for name in asked:
        if name == 'intensity':
            try:
                array = registry.intensity(ident, binning=binning)
            except StackTooLarge as exc:
                raise RouteError(413, str(exc))
        else:
            array = registry.plane(ident, name)
            if array is None:
                raise RouteError(404, f'no such plane: {name}')
        array = _shrink(_crop(np.asarray(array), options), options)
        layers.append(np.asarray(array, dtype=np.float32))
        unit = UNITS.get(name) or registry.plane_unit(ident, name)
        names.append(f'{name} ({unit})' if unit else name)
    shapes = {layer.shape for layer in layers}
    if len(shapes) != 1:
        raise RouteError(409, f'these planes are not the same shape: {sorted(shapes)}')
    stacked = np.stack(layers)
    buffer = BytesIO()
    tifffile.imwrite(buffer, stacked, ome=True, photometric='minisblack',
                     metadata={'axes': 'CYX', 'Channel': {'Name': names}})
    return buffer.getvalue(), ','.join(names), binning, stacked.shape


def _crop(array, options):
    wanted = [k for k in ('x', 'y', 'w', 'h') if k in options]
    if not wanted:
        return array
    if len(wanted) != 4:
        raise RouteError(400, 'x, y, w and h must all be given together')
    x = int(options['x'][0])
    y = int(options['y'][0])
    w = int(options['w'][0])
    h = int(options['h'][0])
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise RouteError(400, 'x and y must be positive and w and h non-zero')
    height, width = array.shape[:2]
    if x + w > width or y + h > height:
        raise RouteError(
            400,
            f'requested region {w}x{h} at ({x},{y}) falls outside {width}x{height}')
    return array[y:y + h, x:x + w]


def _shrink(array, options):
    try:
        step = int(round(float(options.get('downsample', ['1'])[0])))
    except ValueError:
        raise RouteError(400, 'downsample must be a number')
    if step < 1:
        raise RouteError(400, 'downsample must be 1 or more')
    if step > 1:
        array = array[::step, ::step]
    wanted = [k for k in ('ow', 'oh') if k in options]
    if not wanted:
        return array
    if len(wanted) != 2:
        raise RouteError(400, 'ow and oh must both be given together')
    try:
        out_w = int(options['ow'][0])
        out_h = int(options['oh'][0])
    except ValueError:
        raise RouteError(400, 'ow and oh must be whole numbers')
    if out_w <= 0 or out_h <= 0:
        raise RouteError(400, 'ow and oh must be non-zero')
    return _fit_exactly(array, out_h, out_w)


def _fit_exactly(array, out_h, out_w):
    height, width = array.shape[:2]
    if (height, width) == (out_h, out_w):
        return array
    trimmed = array[:min(height, out_h), :min(width, out_w)]
    if trimmed.shape[:2] == (out_h, out_w):
        return trimmed
    padded = np.zeros((out_h, out_w) + array.shape[2:], dtype=array.dtype)
    padded[:trimmed.shape[0], :trimmed.shape[1]] = trimmed
    return padded


FIT_ROI_RE = re.compile(r'^/v1/datasets/([^/]+)/fit/roi$')


def fit_defaults(state):
    from flimkit_bridge import fitting
    return fitting.defaults()


def fit_rois(state, ident, payload):
    from flimkit_bridge import fitting
    registry = _registry(state)
    try:
        meta = registry.metadata(ident)
    except KeyError:
        raise RouteError(404, f'no such dataset: {ident}')
    payload = payload or {}
    collection = payload.get('rois')
    if not isinstance(collection, dict) or collection.get('type') != 'FeatureCollection':
        raise RouteError(400, 'rois must be a GeoJSON FeatureCollection')
    try:
        params = fitting.merge_params(payload.get('params'))
    except ValueError as exc:
        raise RouteError(400, str(exc))
    binning = int(params['binning'])
    try:
        stack = registry.stack(ident, binning=binning)
    except StackTooLarge as exc:
        raise RouteError(413, str(exc))
    except ValueError as exc:
        raise RouteError(400, str(exc))
    try:
        masks = fitting.masks_from_geojson(
            collection, (meta['height'], meta['width']), binning=binning)
    except Exception as exc:
        raise RouteError(400, f'could not read the ROIs: {exc}')
    if not masks:
        raise RouteError(400, 'no usable regions in the payload')
    irf = _session_irf(state, meta['path'])
    results = []
    for name, mask in masks:
        try:
            found = fitting.fit_masked_decay(
                stack, mask, meta['tcspc_res'], meta['n_bins'], params,
                irf_prompt=irf, n_sync=meta.get('n_sync'))
        except ValueError as exc:
            results.append({'name': name, 'error': str(exc)})
            continue
        found['name'] = name
        results.append(found)
    return {'dataset': ident, 'binning': binning, 'results': results,
            'params_used': params}


def _session_irf(state, path):
    app = getattr(state, 'app', None)
    if app is None:
        return None
    preview = getattr(app, '_fov_preview', None)
    if preview is None:
        return None
    import os
    open_path = getattr(preview, '_ptu_path', None)
    if not open_path or os.path.realpath(open_path) != os.path.realpath(path):
        return None
    return getattr(preview, '_irf_prompt', None)


PHASOR_RE = re.compile(r'^/v1/datasets/([^/]+)/phasor$')
PHASOR_POINTS_RE = re.compile(r'^/v1/datasets/([^/]+)/phasor/points$')
PHASOR_MASK_RE = re.compile(r'^/v1/datasets/([^/]+)/phasor/mask$')


PHASOR_SETTINGS_PATH = '/v1/phasor/settings'


def phasor_settings(state):
    from flimkit_bridge import phasor as phasor_module
    return phasor_module.settings()


def _phasor_options(source):
    from flimkit_bridge import phasor as phasor_module
    if isinstance(source, str):
        source = {key: values[-1] for key, values in parse_qs(source).items()}
    try:
        return phasor_module.normalise(source or {})
    except (TypeError, ValueError) as exc:
        raise RouteError(400, f'bad phasor settings: {exc}')


def _phasor_state(state, ident, options=None):
    from flimkit_bridge import phasor as phasor_module
    registry = _registry(state)
    try:
        meta = registry.metadata(ident)
    except KeyError:
        raise RouteError(404, f'no such dataset: {ident}')
    if meta['modality'] != 'time':
        raise RouteError(
            409,
            f"phasor from a decay needs a time-domain file, not {meta['format']}")
    options = _phasor_options(options)
    held = getattr(state, '_phasor_cache', None)
    if held is None:
        held = {}
        state._phasor_cache = held
    key = phasor_module.cache_key(ident, options)
    found = held.get(key)
    if found is None:
        try:
            found = phasor_module.compute(
                meta['path'], channel=meta['channel'], options=options)
        except ValueError as exc:
            raise RouteError(400, f'phasor failed: {exc}')
        except Exception as exc:
            raise RouteError(500, f'phasor failed: {exc}')
        found['binning'] = DatasetRegistryBinning(meta, found['real'].shape)
        held[key] = found
    return meta, found


def DatasetRegistryBinning(meta, phasor_shape):
    from flimkit_bridge.datasets import DatasetRegistry
    try:
        return DatasetRegistry.infer_binning(
            (meta['height'], meta['width']), phasor_shape)
    except ValueError:
        return None


def phasor_summary(state, ident, query=''):
    meta, found = _phasor_state(state, ident, query)
    return {
        'dataset': ident,
        'width': int(found['real'].shape[1]),
        'height': int(found['real'].shape[0]),
        'binning': found['binning'],
        'frequency_mhz': found['frequency'],
        'channel': found['channel'],
        'calibrated': found['calibrated'],
        'options': found['options'],
    }


def phasor_points(state, ident, query):
    from flimkit_bridge import phasor as phasor_module
    options = parse_qs(query)
    bins = int(options.get('bins', ['256'])[0])
    if bins < 8 or bins > 1024:
        raise RouteError(400, 'bins must be between 8 and 1024')
    min_photons = float(options.get('min_photons', ['0.01'])[0])
    _, found = _phasor_state(state, ident, query)
    payload = phasor_module.density_payload(
        found['real'], found['imag'], found['mean'],
        bins=bins, min_photons=min_photons)
    payload['dataset'] = ident
    payload['options'] = found['options']
    return payload


def phasor_mask(state, ident, payload):
    from flimkit_bridge import phasor as phasor_module
    payload = payload or {}
    _, found = _phasor_state(state, ident, payload.get('options'))
    cursors = payload.get('cursors')
    if not isinstance(cursors, list) or not cursors:
        raise RouteError(400, 'cursors must be a non-empty list')
    min_photons = float(payload.get('min_photons', 0.01))
    try:
        masks = phasor_module.cursor_masks(
            found['real'], found['imag'], found['mean'], cursors, min_photons)
    except ValueError as exc:
        raise RouteError(400, str(exc))
    counts = phasor_module.cursor_stats(
        found['real'], found['imag'], found['mean'], masks, found['frequency'])
    if payload.get('output') == 'labels':
        labels = phasor_module.label_image(
            found['real'], found['imag'], found['mean'], cursors, min_photons)
        return {'dataset': ident, 'binning': found['binning'],
                'cursors': counts, 'labels': _encode_labels(labels),
                'width': int(labels.shape[1]), 'height': int(labels.shape[0]),
                'options': found['options']}
    return {'dataset': ident, 'binning': found['binning'], 'cursors': counts,
            'options': found['options']}


def _encode_labels(labels):
    import base64
    return base64.b64encode(np.ascontiguousarray(labels).tobytes()).decode('ascii')


FIT_PIXELS_RE = re.compile(r'^/v1/datasets/([^/]+)/fit/pixels$')
JOB_RE = re.compile(r'^/v1/jobs/([^/]+)$')


def irfs(state):
    from flimkit_bridge import irf as irf_module
    found = irf_module.available()
    return {'default': irf_module.default_path(),
            'strategies': list(irf_module.STRATEGIES),
            'has_machine_irf': bool(found),
            'machine_irfs': found,
            'note': ('A machine IRF describes one microscope and is not shipped '
                     'with FLIMKit. Build one under Tools > Machine IRF Builder.')
                    if not found else ''}


def _jobs(state):
    registry = getattr(state, 'jobs', None)
    if registry is None:
        raise RouteError(503, 'this FLIMKit has no job registry')
    return registry


def pipeline_defaults(state):
    from flimkit_bridge import pipeline
    return pipeline.defaults()


def run_pipeline(state, payload):
    from flimkit_bridge import pipeline
    jobs = _jobs(state)
    payload = payload or {}
    container = payload.get('container') or payload.get('path')
    if not container:
        raise RouteError(400, 'a container is required, a .lif or .xlif')
    try:
        params = pipeline.merge_params(payload.get('params'))
    except ValueError as exc:
        raise RouteError(400, str(exc))
    try:
        container, tile_dir, output_dir, basename, n_tiles = pipeline.resolve(
            container, payload.get('tile_dir'), payload.get('output_dir'),
            payload.get('basename'))
    except ValueError as exc:
        raise RouteError(400, str(exc))
    except FileNotFoundError as exc:
        raise RouteError(404, str(exc))
    args = pipeline.build_args(container, tile_dir, output_dir, basename, params)
    chosen = params['pipeline']

    def work(progress, cancel):
        progress(0, n_tiles, f'{chosen} over {n_tiles} tiles')
        result = pipeline.run(args, chosen, progress, cancel)
        if cancel.is_set():
            return None
        return pipeline.summarise(result, output_dir)

    job_id = jobs.submit(chosen, work)
    return {'job': job_id, 'container': str(container),
            'tile_dir': str(tile_dir), 'output_dir': str(output_dir),
            'basename': basename, 'n_tiles': n_tiles, 'params_used': params}


def zstack_defaults(state):
    from flimkit_bridge import zstack
    return zstack.defaults()


def scan_zstack(state, payload):
    from flimkit_bridge import zstack
    directory = (payload or {}).get('ptu_dir') or (payload or {}).get('path')
    if not directory:
        raise RouteError(400, 'a ptu_dir is required, the folder holding the slices')
    try:
        groups = zstack.scan(directory)
    except FileNotFoundError as exc:
        raise RouteError(404, str(exc))
    except Exception as exc:
        raise RouteError(400, str(exc))
    from flimkit_bridge import volumes
    return {'ptu_dir': str(directory), 'stacks': groups,
            'n_stacks': len(groups),
            'n_slices': sum(group['n_slices'] for group in groups),
            'volume_formats': list(volumes.FORMATS)}


def run_zstack(state, payload):
    from flimkit_bridge import zstack
    jobs = _jobs(state)
    payload = payload or {}
    directory = payload.get('ptu_dir') or payload.get('path')
    if not directory:
        raise RouteError(400, 'a ptu_dir is required, the folder holding the slices')
    try:
        params = zstack.merge_params(payload.get('params'))
    except ValueError as exc:
        raise RouteError(400, str(exc))
    try:
        ptu_dir, output_dir, groups = zstack.resolve(
            directory, payload.get('output_dir'))
    except FileNotFoundError as exc:
        raise RouteError(404, str(exc))
    except ValueError as exc:
        raise RouteError(400, str(exc))
    args = zstack.build_args(ptu_dir, output_dir, params)
    n_slices = sum(group['n_slices'] for group in groups)

    def work(progress, cancel):
        progress(0, n_slices,
                 f'{len(groups)} z-stack(s) over {n_slices} slices')
        result = zstack.run(args, progress, cancel)
        if cancel.is_set():
            return None
        return zstack.summarise(result, output_dir, groups, params)

    job_id = jobs.submit('zstack_fit', work)
    return {'job': job_id, 'ptu_dir': str(ptu_dir),
            'output_dir': str(output_dir),
            'stacks': [{k: v for k, v in group.items() if k != 'files'}
                       for group in groups],
            'n_stacks': len(groups), 'n_slices': n_slices,
            'params_used': params}


ZSTACK_VOLUME_PATH = '/v1/zstack/volume.ome.tif'


def zstack_volume(state, query):
    from flimkit_bridge import volumes, zstack
    options = {key: values[-1] for key, values in parse_qs(query).items()}
    group_dir = options.get('group_dir')
    if not group_dir:
        raise RouteError(400, 'group_dir is required, one stack of fitted slices')
    try:
        z_step = float(options.get('z_step_um', 1.0))
        side = float(options['pixel_size_um']) if 'pixel_size_um' in options else None
    except ValueError:
        raise RouteError(400, 'z_step_um and pixel_size_um must be numbers')
    try:
        return zstack.stream_volume(group_dir, label=options.get('label'),
                                    z_step_um=z_step, pixel_size_um=side)
    except FileNotFoundError as exc:
        raise RouteError(404, str(exc))
    except volumes.NothingToStack as exc:
        raise RouteError(409, str(exc))
    except Exception as exc:
        raise RouteError(500, str(exc))


def export_zstack(state, payload):
    from flimkit_bridge import zstack
    jobs = _jobs(state)
    payload = payload or {}
    group_dir = payload.get('group_dir')
    if not group_dir:
        raise RouteError(400, 'a group_dir is required, one stack of fitted slices')
    from flimkit_bridge import volumes
    wanted = payload.get('format', 'ome-zarr')
    if wanted not in volumes.FORMATS:
        raise RouteError(400, f'format must be one of {list(volumes.FORMATS)}')
    from pathlib import Path
    if not Path(group_dir).expanduser().is_dir():
        raise RouteError(404, f'no such folder: {group_dir}')

    def work(progress, cancel):
        progress(0, 1, f'writing {wanted}')
        found = zstack.export(
            group_dir, label=payload.get('label'),
            output_dir=payload.get('output_dir'), volume_format=wanted,
            z_step_um=payload.get('z_step_um', 1.0),
            pixel_size_um=payload.get('pixel_size_um'))
        progress(1, 1, 'written')
        return {'products': [found]}

    job_id = jobs.submit('zstack_export', work)
    return {'job': job_id, 'group_dir': str(group_dir), 'format': wanted}


def fit_pixels(state, ident, payload):
    from flimkit_bridge import fitting
    registry = _registry(state)
    jobs = _jobs(state)
    try:
        meta = registry.metadata(ident)
    except KeyError:
        raise RouteError(404, f'no such dataset: {ident}')
    try:
        params = fitting.merge_params((payload or {}).get('params'))
    except ValueError as exc:
        raise RouteError(400, str(exc))
    binning = int(params['binning'])
    estimated = registry.estimated_stack_bytes(ident, binning)
    irf_prompt = _session_irf(state, meta['path'])

    def work(progress, cancel):
        stack = registry.stack(ident, binning=binning)
        found = fitting.fit_pixels(
            stack, meta['tcspc_res'], meta['n_bins'], params,
            irf_prompt=irf_prompt, bands=8, progress=progress, cancel=cancel,
            n_sync=meta.get('n_sync'))
        if found is None:
            return None
        for name, array in found['maps'].items():
            registry.put_plane(ident, name, array,
                               unit='ns' if 'tau' in name else '')
        wrote_back = _write_back(state, meta['path'], found['maps'])
        return {
            'dataset': ident,
            'binning': binning,
            'planes': sorted(found['maps']),
            'global': found['global'],
            'wrote_back': wrote_back,
        }

    job_id = jobs.submit('fit_pixels', work)
    return {'job': job_id, 'dataset': ident, 'params_used': params,
            'estimated_stack_bytes': estimated}


def _write_back(state, path, maps):
    import os
    app = getattr(state, 'app', None)
    if app is None:
        return False
    preview = getattr(app, '_fov_preview', None)
    if preview is None:
        return False
    open_path = getattr(preview, '_ptu_path', None)
    if not open_path or os.path.realpath(open_path) != os.path.realpath(path):
        return False
    try:
        preview._pixel_maps = dict(maps)
        for candidate in ('tau_mean_int', 'tau_mean_amp', 'tau_1'):
            if candidate in maps:
                preview._lifetime_map = maps[candidate]
                break
    except Exception:
        return False
    return True


def job_status(state, job_id):
    try:
        return _jobs(state).status(job_id)
    except KeyError:
        raise RouteError(404, f'no such job: {job_id}')


def job_result(state, job_id):
    jobs = _jobs(state)
    try:
        return {'result': jobs.result(job_id)}
    except KeyError:
        raise RouteError(404, f'no such job: {job_id}')
    except ValueError as exc:
        raise RouteError(409, str(exc))


def job_cancel(state, job_id):
    return {'cancelled': _jobs(state).cancel(job_id)}
