import numpy as np
import pytest

from flimkit_bridge import fitting
from test_fitting import _synthetic_stack


def _stack(tau_ns=2.0, shape=(8, 8)):
    return _synthetic_stack(tau_ns=tau_ns, shape=shape, photons=800000)


def test_banded_and_unbanded_agree():
    stack, irf = _stack(shape=(8, 8))
    params = fitting.merge_params({'n_exp': 1, 'tau_min_ns': 0.05,
                                   'tau_max_ns': 10.0, 'min_photons': 1,
                                   'irf_strategy': 'session'})

    single = fitting.fit_pixels(stack, 5e-11, 256, params, irf_prompt=irf, bands=1)
    banded = fitting.fit_pixels(stack, 5e-11, 256, params, irf_prompt=irf, bands=4)

    assert set(single['maps']) == set(banded['maps'])
    for name in single['maps']:
        np.testing.assert_allclose(
            banded['maps'][name], single['maps'][name],
            rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg=f'banding changed the {name} map')


def test_maps_have_the_stack_shape():
    stack, irf = _stack(shape=(8, 8))
    params = fitting.merge_params({'n_exp': 1, 'min_photons': 1,
                                   'irf_strategy': 'session'})

    found = fitting.fit_pixels(stack, 5e-11, 256, params, irf_prompt=irf, bands=2)

    for name, array in found['maps'].items():
        assert array.shape == (8, 8), f'{name} has shape {array.shape}'


def test_recovered_lifetime_is_right_and_well_fitted():
    stack, irf = _stack(tau_ns=2.0, shape=(8, 8))
    params = fitting.merge_params({'n_exp': 1, 'tau_min_ns': 0.05,
                                   'tau_max_ns': 10.0, 'min_photons': 1,
                                   'irf_strategy': 'session'})

    found = fitting.fit_pixels(stack, 5e-11, 256, params, irf_prompt=irf, bands=2)

    tau = found['maps']['tau_mean_int']
    valid = np.isfinite(tau)
    assert valid.any(), 'no pixel was fitted'
    assert np.median(tau[valid]) == pytest.approx(2.0, rel=0.05)

    chi2 = found['maps']['chi2_r']
    finite = np.isfinite(chi2)
    assert np.median(chi2[finite]) < 5.0


def test_progress_reaches_the_end():
    stack, irf = _stack(shape=(8, 8))
    params = fitting.merge_params({'n_exp': 1, 'min_photons': 1,
                                   'irf_strategy': 'session'})
    seen = []

    fitting.fit_pixels(stack, 5e-11, 256, params, irf_prompt=irf, bands=4,
                       progress=lambda c, t, m='': seen.append((c, t)))

    assert seen, 'no progress was reported'
    assert seen[-1][0] == seen[-1][1]
    assert [c for c, _ in seen] == sorted(c for c, _ in seen)


def test_cancelling_stops_between_bands():
    import threading
    stack, irf = _stack(shape=(8, 8))
    params = fitting.merge_params({'n_exp': 1, 'min_photons': 1,
                                   'irf_strategy': 'session'})
    cancel = threading.Event()
    cancel.set()

    found = fitting.fit_pixels(stack, 5e-11, 256, params, irf_prompt=irf,
                               bands=4, cancel=cancel)

    assert found is None


def test_band_count_is_clamped_to_the_rows():
    stack, irf = _stack(shape=(4, 4))
    params = fitting.merge_params({'n_exp': 1, 'min_photons': 1,
                                   'irf_strategy': 'session'})

    found = fitting.fit_pixels(stack, 5e-11, 256, params, irf_prompt=irf, bands=99)

    assert found['maps']['tau_mean_int'].shape == (4, 4)
