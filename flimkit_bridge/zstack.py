import argparse
from pathlib import Path

SCHEMA = (
    {'key': 'n_exp', 'label': 'Exponentials', 'type': 'int', 'min': 1, 'max': 3,
     'default': 2},
    {'key': 'tau_min_ns', 'label': 'Minimum tau (ns)', 'type': 'float',
     'min': 0.01, 'max': 100.0, 'default': 0.2},
    {'key': 'tau_max_ns', 'label': 'Maximum tau (ns)', 'type': 'float',
     'min': 0.05, 'max': 200.0, 'default': 6.0},
    {'key': 'min_photons', 'label': 'Minimum photons per pixel', 'type': 'int',
     'min': 0, 'max': 100000, 'default': 5},
    {'key': 'irf_strategy', 'label': 'Instrument response', 'type': 'choice',
     'choices': ('machine_irf', 'machine_irf_sigma_full', 'machine_irf_sigma_half',
                 'gaussian', 'parametric', 'raw'),
     'default': 'machine_irf'},
    {'key': 'irf_path', 'label': 'IRF file', 'type': 'path', 'default': ''},
    {'key': 'z_step_um', 'label': 'Z step (um)', 'type': 'float',
     'min': 0.001, 'max': 1000.0, 'default': 1.0},
    {'key': 'bound_fraction', 'label': 'Bound fraction', 'type': 'bool',
     'default': False},
    {'key': 'correct_pileup', 'label': 'Correct pile-up', 'type': 'bool',
     'default': False},
    {'key': 'volume_format', 'label': 'Volume format', 'type': 'choice',
     'choices': ('ome-zarr', 'ome-tiff'), 'default': 'ome-zarr'},
    {'key': 'channel', 'label': 'Channel', 'type': 'int', 'min': 0, 'max': 16,
     'default': None, 'advanced': True},
    {'key': 'ref_tau1_ns', 'label': 'Reference tau 1 (ns)', 'type': 'float',
     'min': 0.0, 'max': 200.0, 'default': None, 'advanced': True},
    {'key': 'ref_tau2_ns', 'label': 'Reference tau 2 (ns)', 'type': 'float',
     'min': 0.0, 'max': 200.0, 'default': None, 'advanced': True},
    {'key': 'ref_tau3_ns', 'label': 'Reference tau 3 (ns)', 'type': 'float',
     'min': 0.0, 'max': 200.0, 'default': None, 'advanced': True},
    {'key': 'save_plots', 'label': 'Save per-slice plots', 'type': 'bool',
     'default': False, 'advanced': True},
)

OPTIONAL = ('channel', 'ref_tau1_ns', 'ref_tau2_ns', 'ref_tau3_ns')


def defaults():
    values = {entry['key']: entry['default'] for entry in SCHEMA}
    schema = []
    for entry in SCHEMA:
        described = {'key': entry['key'], 'label': entry['label'],
                     'type': entry['type'],
                     'advanced': bool(entry.get('advanced', False)),
                     'applies_to': ['zstack']}
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
            raise ValueError(f'unknown z-stack parameter: {key}')
        merged[key] = value
    for key, entry in known.items():
        value = merged[key]
        if value is None and key in OPTIONAL:
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
    supplied_taus = [merged[f'ref_tau{i}_ns'] for i in range(1, merged['n_exp'] + 1)]
    if any(t is not None for t in supplied_taus) and not all(
            t is not None for t in supplied_taus):
        raise ValueError(
            f'give a reference tau for all {merged["n_exp"]} components or none')
    return merged


def scan(ptu_dir):
    from flimkit.utils.batch_fit import group_zstack_files, zstack_group_label

    directory = Path(ptu_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f'no such folder: {directory}')
    groups = group_zstack_files(directory)
    found = []
    for (region, t, s), zslices in sorted(groups.items()):
        planes = sorted(zslices)
        found.append({
            'label': zstack_group_label(region, t, s),
            'region': region,
            't': t,
            's': s,
            'n_slices': len(planes),
            'z_first': planes[0],
            'z_last': planes[-1],
            'files': [str(zslices[z]) for z in planes],
        })
    return found


def resolve(ptu_dir, output_dir=None):
    directory = Path(ptu_dir).expanduser()
    groups = scan(directory)
    if not groups:
        raise ValueError(
            f'no z-stack PTU files in {directory}; FLIMKit expects '
            'region_zN.ptu, optionally region_tN_sN_zN.ptu')
    if output_dir:
        output = Path(output_dir).expanduser()
    else:
        output = directory / f'{directory.name.replace(" ", "_")}_flimkit_zstack'
    return directory, output, groups


def build_args(ptu_dir, output_dir, params):
    from flimkit.configs import (
        MACHINE_IRF_DEFAULT_PATH, Optimizer, lm_restarts, de_population,
        de_maxiter, n_workers, IRF_BINS, IRF_FIT_WIDTH, IRF_FWHM)

    args = argparse.Namespace()
    args.ptu_dir = str(ptu_dir)
    args.output_dir = str(output_dir)
    args.nexp = int(params['n_exp'])
    args.tau_min = float(params['tau_min_ns'])
    args.tau_max = float(params['tau_max_ns'])
    args.min_photons = int(params['min_photons'])
    args.estimate_irf = params['irf_strategy']
    args.machine_irf = params['irf_path'] or str(MACHINE_IRF_DEFAULT_PATH)
    args.correct_pileup = bool(params['correct_pileup'])
    args.channel = params['channel']
    args.bound_fraction = bool(params['bound_fraction'])
    args.ref_tau1 = params['ref_tau1_ns']
    args.ref_tau2 = params['ref_tau2_ns']
    args.ref_tau3 = params['ref_tau3_ns']
    args.pileup_in_model = False
    args.bg_in_model = False
    args.fit_start_ns = None
    args.fit_end_ns = None
    args.exclude_ns = None
    args.cost_function = 'poisson'
    args.optimizer = Optimizer
    args.restarts = lm_restarts
    args.de_population = de_population
    args.de_maxiter = de_maxiter
    args.workers = n_workers
    args.no_polish = False
    args.irf_bins = IRF_BINS
    args.irf_fit_width = IRF_FIT_WIDTH
    args.irf_fwhm = IRF_FWHM
    args.save_stack = True
    args.save_npy = True
    args.no_plots = not bool(params['save_plots'])
    args.save_lifetime = bool(params['save_plots'])
    args.save_rgb = bool(params['save_plots'])
    args.save_intensity = bool(params['save_plots'])
    args.save_ind = False
    return args


def run(args, progress, cancel):
    from flimkit.FLIM.assemble import Cancelled
    from flimkit.interactive import _run_zstack_fit
    try:
        return _run_zstack_fit(args, progress_callback=progress, cancel_event=cancel)
    except Cancelled:
        cancel.set()
        return None


def pixel_size_um(files):
    from flimkit.formats import FLIMFile

    for path in files:
        try:
            opened = FLIMFile(str(path), verbose=False)
            tags = getattr(opened, 'tags', {}) or {}
            found = tags.get('ImgHdr_PixResol') or tags.get('ImgHdr_PixRes')
            if found:
                return float(found)
        except Exception:
            continue
    return None


def stream_volume(group_dir, label=None, z_step_um=1.0, pixel_size_um=None):
    """Write one stack to a temporary OME-TIFF and describe it.

    This is the route out for a client that cannot see the bridge's disk, an
    SSH-forwarded port being the usual reason. The caller streams the file and
    is responsible for deleting it.
    """
    import tempfile

    from flimkit_bridge import volumes

    group_dir = Path(group_dir).expanduser()
    if not group_dir.is_dir():
        raise FileNotFoundError(f'no such folder: {group_dir}')
    label = label or group_dir.name
    volume, names, units = volumes.read_group(group_dir, label)
    side = float(pixel_size_um or 1.0)
    holding = Path(tempfile.mkdtemp(prefix='flimkit-zstack-'))
    written = volumes.write_ome_tiff(
        holding / f'{label}.ome.tif', volume, names, units,
        (float(z_step_um), side, side))
    return {
        'file': written,
        'holding': str(holding),
        'label': label,
        'channels': names,
        'units': units,
        'shape': [int(n) for n in volume.shape],
        'n_z': int(volume.shape[1]),
        'voxel_size_um': [float(z_step_um), side, side],
    }


def export(group_dir, label=None, output_dir=None, volume_format='ome-zarr',
           z_step_um=1.0, pixel_size_um=None):
    from flimkit_bridge import volumes

    group_dir = Path(group_dir).expanduser()
    if not group_dir.is_dir():
        raise FileNotFoundError(f'no such folder: {group_dir}')
    label = label or group_dir.name
    output = Path(output_dir).expanduser() if output_dir else group_dir.parent
    output.mkdir(parents=True, exist_ok=True)
    volume, names, units = volumes.read_group(group_dir, label)
    side = float(pixel_size_um or 1.0)
    written, kind = volumes.save_volume(
        output, label, volume, names, units,
        (float(z_step_um), side, side), prefer=volume_format)
    return {
        'file': written,
        'group_dir': str(group_dir),
        'image_id': label,
        'unit': 'ns',
        'format': kind,
        'axes': 'CZYX',
        'channels': names,
        'units': units,
        'n_z': int(volume.shape[1]),
        'voxel_size_um': [float(z_step_um), side, side],
        'z_series_csv': _existing(group_dir / f'{label}_zseries.csv'),
        'taus_ns': [],
    }


def summarise(result, output_dir, groups, params):
    from flimkit_bridge import volumes

    output_dir = Path(output_dir)
    result = result or {}
    by_label = {group['label']: group for group in groups}
    lateral = None
    stacks = []
    products = []
    for label in sorted(result):
        found = result[label] or {}
        group_dir = Path(found.get('group_dir') or (output_dir / label))
        described = {
            'label': label,
            'group_dir': str(group_dir),
            'n_slices': found.get('n_slices'),
            'taus_ns': [float(t) for t in (found.get('taus_ns') or [])],
            'z_series_csv': _existing(group_dir / f'{label}_zseries.csv'),
            'z_series_json': _existing(group_dir / f'{label}_zseries.json'),
            'reference_fit': _existing(group_dir / f'{label}_reference_fit.json'),
            'pooled': _pooled(group_dir, label),
            'z_series': _rows(found.get('z_series')),
        }
        if lateral is None:
            lateral = pixel_size_um(by_label.get(label, {}).get('files') or [])
        try:
            volume, names, units = volumes.read_group(group_dir, label)
        except Exception as problem:
            described['error'] = f'could not stack the slice maps: {problem}'
            stacks.append(described)
            continue
        side = float(lateral or 1.0)
        try:
            written, kind = volumes.save_volume(
                output_dir, label, volume, names, units,
                (float(params['z_step_um']), side, side),
                prefer=params['volume_format'])
        except Exception as problem:
            described['error'] = f'could not write the volume: {problem}'
            stacks.append(described)
            continue
        described.update({
            'volume': written,
            'format': kind,
            'channels': names,
            'units': units,
            'shape': [int(n) for n in volume.shape],
            'voxel_size_um': [float(params['z_step_um']), side, side],
        })
        products.append({
            'file': written,
            'group_dir': str(group_dir),
            'image_id': label,
            'unit': 'ns',
            'format': kind,
            'axes': 'CZYX',
            'channels': names,
            'units': units,
            'n_z': int(volume.shape[1]),
            'voxel_size_um': [float(params['z_step_um']), side, side],
            'z_series_csv': described['z_series_csv'],
            'taus_ns': described['taus_ns'],
            'pooled': described['pooled'],
        })
        stacks.append(described)
    return {
        'output_dir': str(output_dir),
        'pixel_size_um': lateral,
        'stacks': stacks,
        'products': products,
    }


def _existing(path):
    return str(path) if Path(path).exists() else None


POOLED_SKIP = ('z_slices',)


def _pooled(group_dir, label):
    """The fit of the decay pooled over the whole stack, flattened for a
    client that would rather not go and read the file."""
    import json

    found = Path(group_dir) / f'{label}_reference_fit.json'
    if not found.exists():
        return None
    try:
        described = json.loads(found.read_text())
    except (OSError, ValueError):
        return None
    flat = {}
    for key, value in described.items():
        if key in POOLED_SKIP:
            continue
        if isinstance(value, list):
            flat[key] = [v for v in value
                         if isinstance(v, (int, float)) and not isinstance(v, bool)]
        elif isinstance(value, (int, float, str, bool)) or value is None:
            flat[key] = value
    return flat


def _rows(series):
    if not series:
        return []
    listed = []
    for z in sorted(series):
        row = {'z': z}
        for key, value in (series[z] or {}).items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                row[key] = value
        listed.append(row)
    return listed
