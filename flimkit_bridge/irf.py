import os

import numpy as np

STRATEGIES = ('machine_irf', 'machine_irf_sigma_full', 'machine_irf_sigma_half',
              'session', 'gaussian', 'scatter_file', 'pck')


def _machine_irf_dir():
    import flimkit
    return os.path.join(os.path.dirname(flimkit.__file__), 'machine_irf')


def available():
    """Machine IRFs present on this install.

    A machine IRF is specific to one microscope, so none are distributed and a
    fresh install has none until the user measures their own. Generating one is
    FLIMKit's job, under Tools > Machine IRF Builder. This only lists what
    already exists so a client can offer a choice.
    """
    found = []
    directory = _machine_irf_dir()
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            if not name.endswith('.npy'):
                continue
            path = os.path.join(directory, name)
            entry = {'id': name[:-4], 'path': path, 'source': 'bundled',
                     'default': False}
            meta = path[:-4] + '_meta.json'
            if os.path.exists(meta):
                import json
                try:
                    with open(meta, encoding='utf-8') as handle:
                        entry['meta'] = json.load(handle)
                except (OSError, ValueError):
                    pass
            found.append(entry)
    configured = _configured_default()
    for entry in found:
        entry['default'] = os.path.realpath(entry['path']) == configured
    return found


def _configured_default():
    try:
        from flimkit.configs import MACHINE_IRF_DEFAULT_PATH
        return os.path.realpath(str(MACHINE_IRF_DEFAULT_PATH))
    except Exception:
        return ''


def has_machine_irf():
    return bool(available())


def default_path():
    """FLIMKit already configures which machine IRF is the default, including
    a user override, so ask it rather than picking one alphabetically."""
    try:
        from flimkit.configs import MACHINE_IRF_DEFAULT_PATH
        configured = str(MACHINE_IRF_DEFAULT_PATH)
        if configured and os.path.exists(configured):
            return configured
    except Exception:
        pass
    found = available()
    return found[0]['path'] if found else None


def build(strategy, n_bins, tcspc_res, decay, path=None, session_irf=None,
          fwhm_ns=0.2, align_bin=None):
    """Returns (irf_prompt, description).

    Every branch calls FLIMKit rather than reimplementing the maths, so a fit
    run from QuPath uses the same instrument response as one run in FLIMKit.
    """
    from flimkit.FLIM.irf_tools import (gaussian_irf_from_fwhm, irf_from_pck,
                                        machine_irf_prompt)
    if strategy not in STRATEGIES:
        raise ValueError(
            f'unknown IRF strategy {strategy!r}, expected one of {list(STRATEGIES)}')
    if align_bin is None:
        align_bin = int(np.argmax(decay))

    if strategy == 'session':
        if session_irf is None or len(session_irf) != n_bins:
            raise ValueError(
                'FLIMKit has no instrument response loaded for this file, so '
                'the session strategy is unavailable; pick a machine IRF')
        return np.asarray(session_irf, dtype=float), 'flimkit session'

    if strategy.startswith('machine_irf'):
        chosen = path or default_path()
        if not chosen:
            raise ValueError(
                'no machine IRF is installed. A machine IRF describes your own '
                'instrument, so it cannot be shipped with FLIMKit and has to be '
                'measured on the microscope the data came from. Build one in '
                'FLIMKit under Tools > Machine IRF Builder, or pick a different '
                'strategy. Do not substitute a gaussian estimate for real work: '
                'it will return lifetimes that look reasonable and are wrong.')
        if not os.path.exists(chosen):
            raise ValueError(f'no such machine IRF: {chosen}')
        prompt, description, *_ = machine_irf_prompt(
            chosen, n_bins, align_bin, variant=strategy)
        return np.asarray(prompt, dtype=float), description

    if strategy == 'gaussian':
        prompt = gaussian_irf_from_fwhm(n_bins, tcspc_res, float(fwhm_ns), align_bin)
        return np.asarray(prompt, dtype=float), f'gaussian {fwhm_ns} ns FWHM'

    if strategy == 'pck':
        if not path or not os.path.exists(path):
            raise ValueError('the pck strategy needs an existing .pck file')
        return np.asarray(irf_from_pck(path, n_bins), dtype=float), f'pck {path}'

    if not path or not os.path.exists(path):
        raise ValueError('the scatter_file strategy needs an existing file')
    from flimkit.FLIM.irf_tools import irf_from_measured_file
    from flimkit.formats import FLIMFile
    reference = FLIMFile(path, verbose=False)
    prompt = irf_from_measured_file(path, reference)
    return np.asarray(prompt, dtype=float), f'measured {os.path.basename(path)}'
