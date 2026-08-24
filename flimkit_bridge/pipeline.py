import argparse
from pathlib import Path

SCHEMA = (
    {'key': 'n_exp', 'label': 'Exponentials', 'type': 'int', 'min': 1, 'max': 3,
     'default': 2},
    {'key': 'tau_min_ns', 'label': 'Minimum tau (ns)', 'type': 'float',
     'min': 0.01, 'max': 100.0, 'default': 0.2},
    {'key': 'tau_max_ns', 'label': 'Maximum tau (ns)', 'type': 'float',
     'min': 0.05, 'max': 200.0, 'default': 6.0},
    {'key': 'binning', 'label': 'Binning', 'type': 'choice',
     'choices': (1, 2, 4, 8), 'default': 1},
    {'key': 'min_photons', 'label': 'Minimum photons per pixel', 'type': 'int',
     'min': 0, 'max': 100000, 'default': 5},
    {'key': 'irf_strategy', 'label': 'Instrument response', 'type': 'choice',
     'choices': ('machine_irf', 'machine_irf_sigma_full', 'machine_irf_sigma_half'),
     'default': 'machine_irf'},
    {'key': 'irf_path', 'label': 'IRF file', 'type': 'path', 'default': ''},
    {'key': 'rotate_tiles', 'label': 'Rotate tiles', 'type': 'bool',
     'default': True},
    {'key': 'register_tiles', 'label': 'Register tiles', 'type': 'bool',
     'default': True},
    {'key': 'reg_max_shift_px', 'label': 'Maximum registration shift (px)',
     'type': 'int', 'min': 0, 'max': 2000, 'default': 120},
    {'key': 'correct_pileup', 'label': 'Correct pile-up', 'type': 'bool',
     'default': False},
    {'key': 'channel', 'label': 'Channel', 'type': 'int', 'min': 0, 'max': 16,
     'default': None, 'advanced': True},
    {'key': 'pipeline', 'label': 'Pipeline', 'type': 'choice',
     'choices': ('tile_fit', 'stitch_fit'), 'default': 'tile_fit'},
    {'key': 'mode', 'label': 'Fit', 'type': 'choice',
     'choices': ('summed', 'both'), 'default': 'both'},
)


def defaults():
    values = {entry['key']: entry['default'] for entry in SCHEMA}
    schema = []
    for entry in SCHEMA:
        described = {'key': entry['key'], 'label': entry['label'],
                     'type': entry['type'],
                     'advanced': bool(entry.get('advanced', False)),
                     'applies_to': ['pipeline']}
        for optional in ('min', 'max'):
            if optional in entry:
                described[optional] = entry[optional]
        if entry['type'] == 'choice':
            described['choices'] = list(entry['choices'])
        schema.append(described)
    return {'values': values, 'schema': schema}


def _match_choice(key, value, choices):
    if value in choices:
        return value
    for choice in choices:
        if str(choice) == str(value):
            return choice
    raise ValueError(f'{key} must be one of {list(choices)}, got {value!r}')


def merge_params(supplied):
    known = {entry['key']: entry for entry in SCHEMA}
    merged = defaults()['values']
    for key, value in (supplied or {}).items():
        if key not in known:
            raise ValueError(f'unknown tile fit parameter: {key}')
        merged[key] = value
    for key, entry in known.items():
        value = merged[key]
        if value is None and key == 'channel':
            continue
        if entry['type'] == 'int':
            value = int(value)
        elif entry['type'] == 'float':
            value = float(value)
        elif entry['type'] == 'bool':
            value = bool(value)
        elif entry['type'] == 'path':
            value = str(value or '')
        elif entry['type'] == 'choice':
            value = _match_choice(key, value, entry['choices'])
        if 'min' in entry and value < entry['min']:
            raise ValueError(f'{key} must be at least {entry["min"]}, got {value}')
        if 'max' in entry and value > entry['max']:
            raise ValueError(f'{key} must be at most {entry["max"]}, got {value}')
        merged[key] = value
    if merged['tau_min_ns'] >= merged['tau_max_ns']:
        raise ValueError('tau_min_ns must be below tau_max_ns')
    return merged


SEARCH_ROOTS_TO_SKIP = ('/', '/Volumes', '/Users', '/home', '/mnt', '/media')

TILES_TO_PROBE = 5


def _holds_a_tile(directory, positions):
    for position in positions[:TILES_TO_PROBE]:
        if (directory / position['file']).exists():
            return True
    return False


def _subdirectories(directory):
    try:
        return sorted(entry for entry in directory.iterdir() if entry.is_dir())
    except OSError:
        return []


def locate_tiles(container, positions, tile_dir=None):
    searched = []
    direct = [Path(tile_dir)] if tile_dir is not None else []
    direct.append(container.parent)
    for candidate in direct:
        searched.append(candidate)
        if _holds_a_tile(candidate, positions):
            return candidate
    for root in (container.parent, container.parent.parent):
        if str(root) in SEARCH_ROOTS_TO_SKIP or root in searched[1:]:
            continue
        searched.append(root)
        for candidate in _subdirectories(root):
            if _holds_a_tile(candidate, positions):
                return candidate
    listed = ', '.join(str(entry) for entry in searched)
    raise FileNotFoundError(
        f'none of the {len(positions)} tiles named in {container.name} are in '
        f'{listed}, or in any folder directly inside those; say where the tiles '
        'are and try again')


def resolve(container, tile_dir=None, output_dir=None, basename=None):
    from flimkit.utils.xml_utils import parse_tile_positions

    container = Path(container).expanduser()
    if not container.exists():
        raise ValueError(f'no such tile container: {container}')
    basename = basename or container.stem
    positions = parse_tile_positions(container, basename)
    if not positions:
        raise ValueError(
            f'{container.name} describes no tiles for {basename!r}')
    found = locate_tiles(container, positions, tile_dir)
    if output_dir:
        output = Path(output_dir).expanduser()
    else:
        output = found / f'{basename.replace(" ", "_")}_flimkit'
    return container, found, output, basename, len(positions)


def build_args(container, tile_dir, output_dir, basename, params):
    from flimkit.configs import (
        MACHINE_IRF_DEFAULT_PATH, Optimizer, lm_restarts, de_population,
        de_maxiter, n_workers, IRF_BINS, IRF_FIT_WIDTH, IRF_FWHM)

    args = argparse.Namespace()
    args.xlif = str(container)
    args.ptu_dir = str(tile_dir)
    args.output_dir = str(output_dir)
    args.ptu_basename = basename
    args.rotate_tiles = bool(params['rotate_tiles'])
    args.nexp = int(params['n_exp'])
    args.tau_min = float(params['tau_min_ns'])
    args.tau_max = float(params['tau_max_ns'])
    args.binning = int(params['binning'])
    args.min_photons = int(params['min_photons'])
    args.register_tiles = bool(params['register_tiles'])
    args.reg_max_shift_px = int(params['reg_max_shift_px'])
    args.estimate_irf = params['irf_strategy']
    args.machine_irf = params['irf_path'] or str(MACHINE_IRF_DEFAULT_PATH)
    args.correct_pileup = bool(params['correct_pileup'])
    args.channel = params['channel']
    args.dist_type = 'discrete'
    args.mode = params['mode']
    args.irf = None
    args.irf_xlsx = None
    args.no_plots = True
    args.no_polish = False
    args.optimizer = Optimizer
    args.restarts = lm_restarts
    args.de_population = de_population
    args.de_maxiter = de_maxiter
    args.workers = n_workers
    args.irf_bins = IRF_BINS
    args.irf_fit_width = IRF_FIT_WIDTH
    args.irf_fwhm = IRF_FWHM
    return args


def run(args, pipeline, progress, cancel):
    from flimkit.FLIM.assemble import Cancelled
    from flimkit.interactive import _run_stitch_and_fit, _run_tile_fit
    runner = _run_tile_fit if pipeline == 'tile_fit' else _run_stitch_and_fit
    try:
        return runner(args, progress_callback=progress, cancel_event=cancel)
    except Cancelled:
        cancel.set()
        return None


def _canvas_from_metadata(output_dir):
    import json
    found = sorted(Path(output_dir).glob('*_metadata.json'))
    if not found:
        return None
    try:
        metadata = json.loads(found[0].read_text())
    except (OSError, ValueError):
        return None
    shape = metadata.get('canvas_shape')
    return [int(shape[0]), int(shape[1])] if shape else None


def _real_unit_products(output_dir):
    import numpy as np
    import tifffile

    output_dir = Path(output_dir)
    found = sorted(output_dir.glob('*_intensity.npy'))
    if not found:
        return []
    roi_name = found[0].name[:-len('_intensity.npy')]
    products = []
    target = output_dir / f'{roi_name}_intensity_photons.tif'
    if not target.exists():
        counts = np.load(found[0])
        whole = (np.isfinite(counts).all() and float(counts.max()) <= 65535
                 and bool(np.array_equal(counts, np.round(counts))))
        tifffile.imwrite(str(target),
                         counts.astype(np.uint16) if whole
                         else counts.astype(np.float32))
    products.append({'file': str(target),
                     'image_id': f'{roi_name}_intensity',
                     'unit': 'photons'})
    for name, unit in ((f'{roi_name}_tau_intensity_weighted_fullrange.tif', 'ns'),):
        candidate = output_dir / name
        if candidate.exists():
            products.append({'file': str(candidate),
                             'image_id': f'{roi_name}_lifetime',
                             'unit': unit})
    return products


def summarise(result, output_dir):
    output_dir = Path(output_dir)
    result = result or {}
    canvas = result.get('canvas') or {}
    shape = None
    for array in canvas.values():
        if hasattr(array, 'shape') and len(array.shape) == 2:
            shape = [int(array.shape[0]), int(array.shape[1])]
            break
    if shape is None:
        shape = _canvas_from_metadata(output_dir)
    summary = {}
    for key, value in (result.get('global_summary') or {}).items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            summary[key] = value
    written = (sorted(entry.name for entry in output_dir.iterdir())
               if output_dir.is_dir() else [])
    try:
        products = _real_unit_products(output_dir)
    except Exception as problem:
        products = []
        print(f'[QuPath bridge] could not prepare images for QuPath: {problem}')
    return {
        'output_dir': str(output_dir),
        'canvas_shape': shape,
        'maps': sorted(canvas),
        'files': written,
        'global_summary': summary,
        'products': products,
    }
