import numpy as np

SCHEMA = (
    {'key': 'fit_model', 'label': 'Model', 'type': 'choice',
     'choices': ('reconv', 'tail'), 'applies_to': ('roi', 'per_pixel'),
     'advanced': False, 'default': 'reconv'},
    {'key': 'n_exp', 'label': 'Exponentials', 'type': 'int', 'min': 1, 'max': 3,
     'applies_to': ('roi', 'per_pixel'), 'advanced': False, 'default': 2},
    {'key': 'tau_min_ns', 'label': 'Minimum tau (ns)', 'type': 'float',
     'min': 0.01, 'max': 100.0, 'applies_to': ('roi', 'per_pixel'),
     'advanced': False, 'default': 0.2},
    {'key': 'tau_max_ns', 'label': 'Maximum tau (ns)', 'type': 'float',
     'min': 0.05, 'max': 200.0, 'applies_to': ('roi', 'per_pixel'),
     'advanced': False, 'default': 6.0},
    {'key': 'binning', 'label': 'Binning', 'type': 'choice',
     'choices': (1, 2, 4, 8), 'applies_to': ('roi', 'per_pixel'),
     'advanced': False, 'default': 1},
    {'key': 'min_photons', 'label': 'Minimum photons per pixel', 'type': 'int',
     'min': 0, 'max': 100000, 'applies_to': ('per_pixel',),
     'advanced': False, 'default': 5},
    {'key': 'irf_strategy', 'label': 'Instrument response', 'type': 'choice',
     'choices': ('machine_irf', 'machine_irf_sigma_full', 'machine_irf_sigma_half',
                 'session', 'gaussian', 'scatter_file', 'pck'),
     'applies_to': ('roi', 'per_pixel'), 'advanced': False,
     'default': 'machine_irf'},
    {'key': 'irf_path', 'label': 'IRF file', 'type': 'path',
     'applies_to': ('roi', 'per_pixel'), 'advanced': False, 'default': ''},
    {'key': 'irf_fwhm_ns', 'label': 'Gaussian IRF FWHM (ns)', 'type': 'float',
     'min': 0.01, 'max': 5.0, 'applies_to': ('roi', 'per_pixel'),
     'advanced': True, 'default': 0.2},
    {'key': 'channel', 'label': 'Channel', 'type': 'int', 'min': 0, 'max': 16,
     'applies_to': ('roi', 'per_pixel'), 'advanced': False, 'default': 0},
    {'key': 'cost_function', 'label': 'Cost function', 'type': 'choice',
     'choices': ('poisson', 'chi2'), 'applies_to': ('roi', 'per_pixel'),
     'advanced': True, 'default': 'poisson'},
    {'key': 'optimizer', 'label': 'Optimizer', 'type': 'choice',
     'choices': ('de', 'lm_multistart'), 'applies_to': ('roi', 'per_pixel'),
     'advanced': True, 'default': 'de'},
    {'key': 'use_gpu', 'label': 'Use the GPU for per-pixel', 'type': 'bool',
     'applies_to': ('per_pixel',), 'advanced': False, 'default': True},
    {'key': 'fit_bg', 'label': 'Fit background', 'type': 'bool',
     'applies_to': ('roi', 'per_pixel'), 'advanced': True, 'default': True},
    {'key': 'fit_sigma', 'label': 'Fit IRF broadening', 'type': 'bool',
     'applies_to': ('roi', 'per_pixel'), 'advanced': True, 'default': False},
    {'key': 'free_tau', 'label': 'Free tau per pixel', 'type': 'bool',
     'applies_to': ('per_pixel',), 'advanced': True, 'default': False},
    {'key': 'lm_restarts', 'label': 'LM restarts', 'type': 'int', 'min': 1,
     'max': 64, 'applies_to': ('roi', 'per_pixel'), 'advanced': True,
     'default': 8},
    {'key': 'de_population', 'label': 'DE population', 'type': 'int', 'min': 5,
     'max': 200, 'applies_to': ('roi', 'per_pixel'), 'advanced': True,
     'default': 30},
    {'key': 'de_maxiter', 'label': 'DE iterations', 'type': 'int', 'min': 100,
     'max': 20000, 'applies_to': ('roi', 'per_pixel'), 'advanced': True,
     'default': 5000},
    {'key': 'fit_start_ns', 'label': 'Fit window start (ns, 0 = auto)',
     'type': 'float', 'min': 0.0, 'max': 200.0,
     'applies_to': ('roi', 'per_pixel'), 'advanced': True, 'default': 0.0},
    {'key': 'fit_end_ns', 'label': 'Fit window end (ns, 0 = auto)',
     'type': 'float', 'min': 0.0, 'max': 200.0,
     'applies_to': ('roi', 'per_pixel'), 'advanced': True, 'default': 0.0},
    {'key': 'correct_pileup', 'label': 'Correct pile-up (Coates)', 'type': 'bool',
     'applies_to': ('per_pixel',), 'advanced': True, 'default': False},
    {'key': 'pileup_in_model', 'label': 'Pile-up in the model (needs free tau)',
     'type': 'bool', 'applies_to': ('roi', 'per_pixel'), 'advanced': True,
     'default': False},
)


def defaults():
    values = {entry['key']: entry['default'] for entry in SCHEMA}
    schema = []
    for entry in SCHEMA:
        described = {
            'key': entry['key'],
            'label': entry['label'],
            'type': entry['type'],
            'applies_to': list(entry['applies_to']),
            'advanced': entry['advanced'],
        }
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
            raise ValueError(f'unknown fit parameter: {key}')
        merged[key] = value
    _validate(merged, known)
    return merged


def _validate(merged, known):
    for key, entry in known.items():
        value = merged[key]
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
    if merged.get('correct_pileup') and merged.get('pileup_in_model'):
        raise ValueError('pick one pile-up route: correct_pileup rescales the '
                         'measured decay, pileup_in_model folds pile-up into the '
                         'fitted model')
    start, end = merged.get('fit_start_ns', 0.0), merged.get('fit_end_ns', 0.0)
    if start and end and start >= end:
        raise ValueError('fit_start_ns must be below fit_end_ns')


def masks_from_geojson(collection, image_shape, binning=1):
    try:
        from flimkit.utils.roi import RoiManager
    except ImportError:
        from flimkit.UI.roi_tools import RoiManager
    manager = RoiManager()
    scaled = _scale_collection(collection, binning)
    ids = manager.add_geojson(scaled, mode='replace')
    height, width = image_shape[0] // binning, image_shape[1] // binning
    found = []
    for region_id in ids:
        region = manager.get_region(region_id)
        mask = manager.compute_region_mask(region_id, (height, width))
        if mask is None:
            continue
        found.append((region.get('name', f'Region {region_id}'), mask))
    return found


def _scale_collection(collection, binning):
    if binning == 1:
        return collection
    scaled = {'type': 'FeatureCollection', 'features': []}
    for feature in collection.get('features', []):
        geometry = feature.get('geometry') or {}
        rings = geometry.get('coordinates') or []
        moved = [[[x / binning, y / binning] for x, y in ring] for ring in rings]
        scaled['features'].append({
            **feature,
            'geometry': {**geometry, 'coordinates': moved},
        })
    return scaled


def build_irf(n_bins, tcspc_res, decay, cached=None, params=None):
    from flimkit_bridge import irf as irf_module
    params = params or {}
    strategy = params.get('irf_strategy', 'machine_irf')
    if strategy == 'session' and cached is None:
        strategy = 'machine_irf'
    return irf_module.build(
        strategy, n_bins, tcspc_res, decay,
        path=(params.get('irf_path') or None),
        session_irf=cached,
        fwhm_ns=params.get('irf_fwhm_ns', 0.2))


def roi_sync(n_sync, n_pixels, n_total):
    if not n_sync or not n_total:
        return None
    return int(float(n_sync) * float(n_pixels) / float(n_total))


def fit_masked_decay(stack, mask, tcspc_res, n_bins, params, irf_prompt=None,
                     n_sync=None):
    stack = np.asarray(stack)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != stack.shape[:2]:
        raise ValueError(
            f'mask shape {mask.shape} does not match stack shape {stack.shape[:2]}')
    if not mask.any():
        raise ValueError('the region selects no pixels')
    values = np.asarray(stack[mask], dtype=np.float64)
    decay = values.sum(axis=0)
    return fit_decay(decay, int(values.shape[0]), tcspc_res, n_bins, params,
                     irf_prompt=irf_prompt,
                     n_sync=roi_sync(n_sync, values.shape[0], mask.size))


def _window(params):
    start = float(params.get('fit_start_ns') or 0.0) or None
    end = float(params.get('fit_end_ns') or 0.0) or None
    return start, end


def accepts(func, name):
    from inspect import signature
    try:
        return name in signature(func).parameters
    except (TypeError, ValueError):
        return True


def _pileup_kwargs(params, func, n_sync):
    wanted = bool(params.get('pileup_in_model'))
    if wanted and not accepts(func, 'pileup_in_model'):
        from flimkit_bridge.version import flimkit_version
        raise ValueError(
            f'pile-up in the model needs a FLIMKit that offers it; this one is '
            f'{flimkit_version()}, which has only the Coates route. Upgrade '
            f'FLIMKit or untick the setting.')
    found = {'correct_pileup': bool(params.get('correct_pileup', False)),
             'n_sync': n_sync}
    if accepts(func, 'pileup_in_model'):
        found['pileup_in_model'] = wanted
    return found


def _tuning(params):
    return dict(optimizer=params['optimizer'],
                n_restarts=int(params.get('lm_restarts', 8)),
                de_popsize=int(params.get('de_population', 30)),
                de_maxiter=int(params.get('de_maxiter', 5000)),
                workers=1)


def fit_decay(decay, n_pixels, tcspc_res, n_bins, params, irf_prompt=None,
              n_sync=None):
    from flimkit.FLIM.fitters import fit_summed, fit_summed_tail
    decay = np.asarray(decay, dtype=float)
    start_ns, end_ns = _window(params)
    sync = n_sync if params.get('pileup_in_model') else None
    if params.get('fit_model') == 'tail':
        irf_source = 'none, the tail fit ignores the instrument response'
        popt, summary = fit_summed_tail(
            decay, tcspc_res, n_bins,
            bool(params['fit_bg']),
            int(params['n_exp']),
            float(params['tau_min_ns']),
            float(params['tau_max_ns']),
            cost_function=params['cost_function'],
            fit_start_ns=start_ns, fit_end_ns=end_ns, n_sync=sync,
            **_tuning(params),
        )
    else:
        irf, irf_source = build_irf(n_bins, tcspc_res, decay, cached=irf_prompt,
                                    params=params)
        popt, summary = fit_summed(
            decay, tcspc_res, n_bins, irf,
            False, bool(params['fit_bg']), bool(params['fit_sigma']),
            int(params['n_exp']),
            float(params['tau_min_ns']),
            float(params['tau_max_ns']),
            cost_function=params['cost_function'],
            fit_start_ns=start_ns, fit_end_ns=end_ns, n_sync=sync,
            **_tuning(params),
        )
    taus = [float(t) for t in summary['taus_ns']]
    amps = [float(a) for a in summary['amps']]
    total = float(np.sum(amps))
    tau_mean = float(np.dot(amps, taus) / total) if total else None
    return {
        'taus_ns': taus,
        'amps': amps,
        'fractions': [float(f) for f in summary['fractions']],
        'tau_mean_ns': tau_mean,
        'tau_mean_amp_ns': float(summary['tau_mean_amp_ns']),
        'tau_mean_int_ns': float(summary['tau_mean_int_ns']),
        'chi2_r': float(summary['reduced_chi2']),
        'chi2_r_tail': float(summary['reduced_chi2_tail']),
        'photon_count': int(decay.sum()),
        'n_pixels': int(n_pixels),
        'irf_source': irf_source,
        'n_exp': int(params['n_exp']),
        'fit_model': params.get('fit_model', 'reconv'),
    }


def fit_pixels(stack, tcspc_res, n_bins, params, irf_prompt=None, bands=8,
               n_sync=None,
               progress=None, cancel=None):
    """Runs fit_per_pixel over row bands.

    The GPU backends accept a progress_callback and never call it, so a single
    call reports nothing and cannot be interrupted. Banding gives a progress
    tick and a cancellation point per band on every backend, and caps peak
    memory. global_popt is fitted once over the whole field so the bands stay
    independent of each other. Each band still goes to the GPU when there is
    one, unless use_gpu is off.
    """
    from flimkit.FLIM.fitters import fit_per_pixel, fit_summed, fit_summed_tail
    stack = np.asarray(stack)
    height = stack.shape[0]
    bands = max(1, min(int(bands), height))
    summed = stack.reshape(-1, stack.shape[2]).sum(axis=0).astype(float)
    fit_model = params.get('fit_model', 'reconv')
    start_ns, end_ns = _window(params)
    sync = n_sync if params.get('pileup_in_model') else None
    pileup = _pileup_kwargs(params, fit_per_pixel, n_sync)
    if fit_model == 'tail':
        irf = None
        irf_source = 'none, the tail fit ignores the instrument response'
        global_popt, summary = fit_summed_tail(
            summed, tcspc_res, n_bins,
            bool(params['fit_bg']),
            int(params['n_exp']),
            float(params['tau_min_ns']),
            float(params['tau_max_ns']),
            cost_function=params['cost_function'],
            fit_start_ns=start_ns, fit_end_ns=end_ns, n_sync=sync,
            **_tuning(params),
        )
    else:
        irf, irf_source = build_irf(n_bins, tcspc_res, summed, cached=irf_prompt,
                                    params=params)
        global_popt, summary = fit_summed(
            summed, tcspc_res, n_bins, irf,
            False, bool(params['fit_bg']), bool(params['fit_sigma']),
            int(params['n_exp']),
            float(params['tau_min_ns']),
            float(params['tau_max_ns']),
            cost_function=params['cost_function'],
            fit_start_ns=start_ns, fit_end_ns=end_ns, n_sync=sync,
            **_tuning(params),
        )
    edges = np.linspace(0, height, bands + 1).astype(int)
    collected = []
    where = 'the GPU' if params.get('use_gpu', True) else 'the CPU'
    print(f'  Per-pixel fit over {height} rows in {bands} bands on {where}...')
    for index in range(bands):
        if cancel is not None and cancel.is_set():
            return None
        start, stop = edges[index], edges[index + 1]
        if start == stop:
            continue
        maps = fit_per_pixel(
            stack[start:stop], tcspc_res, n_bins, irf,
            False, bool(params['fit_bg']), bool(params['fit_sigma']),
            global_popt, int(params['n_exp']),
            min_photons=int(params['min_photons']),
            tau_min_ns=float(params['tau_min_ns']),
            tau_max_ns=float(params['tau_max_ns']),
            fit_idx=summary.get('fit_idx'),
            use_gpu='auto' if params.get('use_gpu', True) else False,
            fit_model=fit_model,
            free_tau=bool(params.get('free_tau', False)),
            **pileup,
        )
        collected.append(maps)
        print(f'    band {index + 1} of {bands}')
        if progress is not None:
            progress(index + 1, bands, f'band {index + 1} of {bands}')
    if not collected:
        return None
    merged = {}
    for name in collected[0]:
        pieces = [part[name] for part in collected]
        merged[name] = np.concatenate(pieces, axis=0)
    return {
        'maps': merged,
        'global': {
            'taus_ns': [float(t) for t in summary['taus_ns']],
            'amps': [float(a) for a in summary['amps']],
            'fractions': [float(f) for f in summary['fractions']],
            'chi2_r': float(summary['reduced_chi2']),
            'irf_source': irf_source,
            'n_exp': int(params['n_exp']),
        },
        'bands': bands,
    }
