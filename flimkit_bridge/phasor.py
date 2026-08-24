import base64

import numpy as np

DEFAULT_MIN_PHOTONS = 0.01


PHASOR_SCHEMA = (
    {'key': 'phasor_filter', 'label': 'Phasor filter', 'type': 'choice',
     'applies_to': ('phasor',), 'advanced': False, 'default': 'none'},
    {'key': 'filter_sigma', 'label': 'Gaussian sigma (px)', 'type': 'float',
     'min': 0.1, 'max': 10.0, 'applies_to': ('phasor',), 'advanced': True,
     'default': 1.0},
    {'key': 'filter_size', 'label': 'Median window (px)', 'type': 'int',
     'min': 3, 'max': 15, 'applies_to': ('phasor',), 'advanced': True,
     'default': 3},
    {'key': 'irf', 'label': 'IRF calibration', 'type': 'choice',
     'applies_to': ('phasor',), 'advanced': False, 'default': 'none'},
)

_NOTHING = ('', 'none', 'None', 'null')


def _choices(key):
    if key == 'phasor_filter':
        from flimkit.phasor.filters import phasor_filter_methods
        return ['none'] + list(phasor_filter_methods())
    from flimkit_bridge import irf as irf_module
    return ['none'] + [entry['id'] for entry in irf_module.available()]


def settings():
    values = {}
    schema = []
    for entry in PHASOR_SCHEMA:
        described = {'key': entry['key'], 'label': entry['label'],
                     'type': entry['type'],
                     'applies_to': list(entry['applies_to']),
                     'advanced': entry['advanced']}
        for optional in ('min', 'max'):
            if optional in entry:
                described[optional] = entry[optional]
        if entry['type'] == 'choice':
            described['choices'] = _choices(entry['key'])
        values[entry['key']] = entry['default']
        schema.append(described)
    return {'values': values, 'schema': schema}


def normalise(options):
    options = options or {}
    found = {entry['key']: entry['default'] for entry in PHASOR_SCHEMA}
    for entry in PHASOR_SCHEMA:
        key = entry['key']
        value = options.get(key)
        if value is None:
            continue
        if entry['type'] == 'float':
            found[key] = float(value)
        elif entry['type'] == 'int':
            found[key] = int(value)
        else:
            found[key] = str(value)
    for key in ('phasor_filter', 'irf'):
        if found[key] in _NOTHING:
            found[key] = 'none'
    return found


def cache_key(ident, options):
    found = normalise(options)
    return (ident, found['phasor_filter'], found['filter_sigma'],
            found['filter_size'], found['irf'])


def apply_filter(real, imag, mean, options):
    found = normalise(options)
    if found['phasor_filter'] == 'none':
        return real, imag
    from flimkit.phasor.filters import phasor_filter
    return phasor_filter(
        np.asarray(real, dtype=float),
        np.asarray(imag, dtype=float),
        found['phasor_filter'],
        mean=np.asarray(mean, dtype=float),
        sigma=found['filter_sigma'],
        size=found['filter_size'])


def resolve_irf(choice):
    import os
    choice = str(choice or '').strip()
    if choice in _NOTHING:
        return None
    if os.path.exists(choice):
        return choice
    from flimkit_bridge import irf as irf_module
    for entry in irf_module.available():
        if entry['id'] == choice:
            return entry['path']
    raise ValueError(
        f'no such IRF: {choice}. Pass an installed machine IRF id from '
        f'GET /v1/irfs, or the path to an IRF workbook.')


def _signal_array(stack, handle, frequency_mhz):
    import xarray as xr
    time_ns = getattr(handle, 'time_ns', None)
    if time_ns is None or len(time_ns) != stack.shape[2]:
        raise ValueError(
            'the reader gives no per-bin time axis matching this decay, so an '
            'IRF cannot be interpolated onto it; read the phasor without '
            'calibration')
    signal = xr.DataArray(np.asarray(stack, dtype=float), dims=('Y', 'X', 'H'),
                          coords={'H': np.asarray(time_ns, dtype=float)})
    signal.attrs['frequency'] = frequency_mhz
    return signal


def valid_pixels(real, mean, min_photons=DEFAULT_MIN_PHOTONS):
    real = np.asarray(real, dtype=float)
    mean = np.asarray(mean, dtype=float)
    return (mean >= min_photons) & ~np.isnan(real)


def _number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def cursor_stats(real, imag, mean, masks, frequency_mhz):
    from phasorpy.lifetime import phasor_to_apparent_lifetime
    real = np.asarray(real, dtype=float)
    imag = np.asarray(imag, dtype=float)
    mean = np.asarray(mean, dtype=float)
    found = []
    for identifier, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        entry = {'id': identifier, 'n_pixels': int(mask.sum())}
        if entry['n_pixels'] and frequency_mhz:
            g, s = real[mask], imag[mask]
            with np.errstate(invalid='ignore', divide='ignore'):
                tau_phi, tau_mod = phasor_to_apparent_lifetime(
                    g, s, float(frequency_mhz))
            entry.update({
                'mean_g': _number(np.nanmean(g)),
                'mean_s': _number(np.nanmean(s)),
                'tau_phi_ns': _number(np.nanmedian(tau_phi)),
                'tau_mod_ns': _number(np.nanmedian(tau_mod)),
                'tau_phi_min_ns': _number(np.nanmin(tau_phi)),
                'tau_phi_max_ns': _number(np.nanmax(tau_phi)),
                'photons': _number(np.nansum(mean[mask])),
            })
        found.append(entry)
    return found


def cursor_masks(real, imag, mean, cursors, min_photons=DEFAULT_MIN_PHOTONS):
    from phasorpy.cursor import mask_from_elliptic_cursor
    real = np.asarray(real, dtype=float)
    imag = np.asarray(imag, dtype=float)
    valid = valid_pixels(real, mean, min_photons)
    found = {}
    for index, cursor in enumerate(cursors or []):
        kind = cursor.get('type', 'ellipse')
        identifier = cursor.get('id', f'c{index + 1}')
        if kind == 'ellipse':
            radius = float(cursor.get('radius', 0.05))
            minor = cursor.get('radius_minor')
            mask = mask_from_elliptic_cursor(
                real, imag,
                np.array([float(cursor['center_g'])]),
                np.array([float(cursor['center_s'])]),
                radius=radius,
                radius_minor=radius if minor is None else float(minor),
                angle=cursor.get('angle_mode', 'semicircle'),
            )
            if mask.ndim > real.ndim:
                mask = mask[0]
        elif kind == 'polygon':
            mask = _polygon_mask(real, imag, cursor.get('vertices') or [])
        else:
            raise ValueError(f'unknown cursor type: {kind}')
        found[identifier] = np.asarray(mask, dtype=bool) & valid
    return found


def _polygon_mask(real, imag, vertices):
    from matplotlib.path import Path as MplPath
    if len(vertices) < 3:
        raise ValueError('a polygon cursor needs at least three vertices')
    points = np.column_stack([real.ravel(), imag.ravel()])
    inside = MplPath([(float(g), float(s)) for g, s in vertices]).contains_points(points)
    return inside.reshape(real.shape)


def label_image(real, imag, mean, cursors, min_photons=DEFAULT_MIN_PHOTONS):
    masks = cursor_masks(real, imag, mean, cursors, min_photons)
    labels = np.zeros(np.asarray(real).shape, dtype=np.uint8)
    for index, cursor in enumerate(cursors or []):
        identifier = cursor.get('id', f'c{index + 1}')
        labels[masks[identifier] & (labels == 0)] = index + 1
    return labels


def density(real, imag, mean, bins=256, min_photons=DEFAULT_MIN_PHOTONS,
            g_range=None, s_range=None):
    real = np.asarray(real, dtype=float)
    imag = np.asarray(imag, dtype=float)
    valid = valid_pixels(real, mean, min_photons)
    g = real[valid]
    s = imag[valid]
    if g_range is None:
        g_range = (-0.05, 1.05)
    if s_range is None:
        s_range = (-0.05, 0.65)
    counts, _, _ = np.histogram2d(
        s, g, bins=bins, range=[list(s_range), list(g_range)])
    counts = counts.astype(np.uint32)
    return {
        'bins': int(bins),
        'counts': counts,
        'g_min': float(g_range[0]),
        'g_max': float(g_range[1]),
        's_min': float(s_range[0]),
        's_max': float(s_range[1]),
        'max_count': int(counts.max()) if counts.size else 0,
        'n_valid': int(valid.sum()),
        'n_in_range': int(counts.sum()),
    }


def density_payload(real, imag, mean, bins=256, min_photons=DEFAULT_MIN_PHOTONS):
    grid = density(real, imag, mean, bins=bins, min_photons=min_photons)
    payload = dict(grid)
    payload['counts'] = base64.b64encode(
        np.ascontiguousarray(grid['counts']).tobytes()).decode('ascii')
    return payload


def _first_harmonic(array):
    """phasorpy returns a leading harmonic axis, so real_cal on a single
    harmonic arrives as (1, Y, X) rather than (Y, X)."""
    array = np.asarray(array, dtype=float)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f'expected a 2-D phasor map, got shape {array.shape}')
    return array


def compute(path, channel=None, binning=4, options=None):
    """Phasor coordinates for any time-domain reader FLIMKit can open.

    FLIMKit's own phasor entry point goes through signal_from_PTUFile and is
    therefore PTU-only. Reading the cube through FLIMFile instead gives the
    same numbers, verified bit-identical on a real PTU, and works for every
    format FLIMFile supports.

    options is a settings dict in the shape normalise() returns. Calibration
    runs before filtering because flimkit/phasor_launcher.py does it in that
    order, and the two halves have to agree on the same file.
    """
    from phasorpy.phasor import phasor_from_signal
    from flimkit.formats import FLIMFile
    found_options = normalise(options)
    handle = FLIMFile(path, verbose=False)
    stack = handle.raw_pixel_stack(channel=channel, binning=binning)
    mean, real, imag = phasor_from_signal(stack, axis=2)
    frequency = getattr(handle, 'sync_rate', None)
    if not frequency:
        raise ValueError(
            f'{path} does not report a laser repetition rate, so phasor '
            'coordinates cannot be placed on the universal semicircle')
    found = {
        'real': _first_harmonic(real),
        'imag': _first_harmonic(imag),
        'mean': _first_harmonic(mean),
        'frequency': float(frequency) / 1e6,
        'channel': channel,
        'calibrated': False,
        'options': found_options,
    }
    irf_path = resolve_irf(found_options['irf'])
    if irf_path:
        found.update(_calibrate(found, handle, irf_path, stack))
    found['real'], found['imag'] = apply_filter(
        found['real'], found['imag'], found['mean'], found_options)
    return found


def _calibrate(found, handle, irf_path, stack):
    from flimkit.phasor.signal import (calibrate_signal_with_irf,
                                       calibrate_signal_with_machine_irf)
    signal = _signal_array(stack, handle, found['frequency'])
    if str(irf_path).endswith('.npy'):
        real_cal, imag_cal = calibrate_signal_with_machine_irf(
            signal, found['real'], found['imag'], irf_path, found['frequency'])
    else:
        from flimkit.phasor.signal import get_phasor_irf
        irf_time_ns, irf_counts = get_phasor_irf(irf_path)
        real_cal, imag_cal = calibrate_signal_with_irf(
            signal, found['real'], found['imag'], irf_time_ns, irf_counts,
            found['frequency'])
    return {'real': _first_harmonic(real_cal),
            'imag': _first_harmonic(imag_cal),
            'calibrated': True}
