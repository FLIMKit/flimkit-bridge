import json
from urllib.request import Request, urlopen

import numpy as np
import pytest

from flimkit_bridge import phasor


@pytest.fixture
def two_populations():
    rng = np.random.default_rng(0)
    real = np.empty((16, 16), dtype=float)
    imag = np.empty((16, 16), dtype=float)
    real[:8] = rng.normal(0.30, 0.01, (8, 16))
    imag[:8] = rng.normal(0.40, 0.01, (8, 16))
    real[8:] = rng.normal(0.70, 0.01, (8, 16))
    imag[8:] = rng.normal(0.35, 0.01, (8, 16))
    mean = np.full((16, 16), 100.0)
    return real, imag, mean


def test_cursor_mask_selects_one_population(two_populations):
    real, imag, mean = two_populations

    masks = phasor.cursor_masks(
        real, imag, mean,
        cursors=[{'id': 'c1', 'center_g': 0.30, 'center_s': 0.40, 'radius': 0.05}],
        min_photons=1.0)

    assert list(masks) == ['c1']
    mask = masks['c1']
    assert mask.shape == (16, 16)
    assert mask[:8].all()
    assert not mask[8:].any()


def test_two_cursors_are_separable(two_populations):
    real, imag, mean = two_populations

    masks = phasor.cursor_masks(
        real, imag, mean,
        cursors=[
            {'id': 'a', 'center_g': 0.30, 'center_s': 0.40, 'radius': 0.05},
            {'id': 'b', 'center_g': 0.70, 'center_s': 0.35, 'radius': 0.05},
        ],
        min_photons=1.0)

    assert masks['a'].sum() == 128
    assert masks['b'].sum() == 128
    assert not (masks['a'] & masks['b']).any()


def test_low_photon_pixels_are_excluded(two_populations):
    real, imag, mean = two_populations
    mean = mean.copy()
    mean[0, :] = 0.0

    masks = phasor.cursor_masks(
        real, imag, mean,
        cursors=[{'id': 'c1', 'center_g': 0.30, 'center_s': 0.40, 'radius': 0.05}],
        min_photons=10.0)

    assert not masks['c1'][0].any()
    assert masks['c1'][1:8].all()


def test_nan_phasor_pixels_are_excluded(two_populations):
    real, imag, mean = two_populations
    real = real.copy()
    real[0, 0] = np.nan

    masks = phasor.cursor_masks(
        real, imag, mean,
        cursors=[{'id': 'c1', 'center_g': 0.30, 'center_s': 0.40, 'radius': 0.05}],
        min_photons=1.0)

    assert not masks['c1'][0, 0]


def test_a_polygon_cursor_selects_its_vertices(two_populations):
    real, imag, mean = two_populations

    masks = phasor.cursor_masks(
        real, imag, mean,
        cursors=[{'id': 'poly', 'type': 'polygon',
                  'vertices': [[0.2, 0.3], [0.45, 0.3], [0.45, 0.5], [0.2, 0.5]]}],
        min_photons=1.0)

    assert masks['poly'][:8].all()
    assert not masks['poly'][8:].any()


def test_label_image_numbers_cursors_from_one(two_populations):
    real, imag, mean = two_populations

    labels = phasor.label_image(
        real, imag, mean,
        cursors=[
            {'id': 'a', 'center_g': 0.30, 'center_s': 0.40, 'radius': 0.05},
            {'id': 'b', 'center_g': 0.70, 'center_s': 0.35, 'radius': 0.05},
        ],
        min_photons=1.0)

    assert labels.dtype == np.uint8
    assert set(np.unique(labels)) == {1, 2}
    assert (labels[:8] == 1).all()
    assert (labels[8:] == 2).all()


def test_density_histogram_counts_every_valid_pixel(two_populations):
    real, imag, mean = two_populations

    grid = phasor.density(real, imag, mean, bins=64, min_photons=1.0)

    assert grid['counts'].shape == (64, 64)
    assert grid['counts'].dtype == np.uint32
    assert int(grid['counts'].sum()) == 256
    assert grid['n_in_range'] == 256
    assert grid['g_min'] < grid['g_max']
    assert grid['max_count'] >= 1


def test_density_skips_invalid_pixels(two_populations):
    real, imag, mean = two_populations
    mean = mean.copy()
    mean[0, :] = 0.0

    grid = phasor.density(real, imag, mean, bins=32, min_photons=10.0)

    assert int(grid['counts'].sum()) == 256 - 16


def test_an_unknown_cursor_type_is_refused(two_populations):
    real, imag, mean = two_populations

    with pytest.raises(ValueError, match='unknown cursor type'):
        phasor.cursor_masks(real, imag, mean,
                            cursors=[{'id': 'x', 'type': 'banana'}],
                            min_photons=1.0)



def _single_exponential(tau_ns, frequency_mhz, side=4):
    omega = 2.0 * np.pi * frequency_mhz * 1e6
    product = omega * tau_ns * 1e-9
    g = 1.0 / (1.0 + product ** 2)
    s = product / (1.0 + product ** 2)
    return (np.full((side, side), g), np.full((side, side), s),
            np.full((side, side), 500.0))


@pytest.mark.parametrize('tau_ns', [0.5, 2.4, 4.1])
def test_cursor_stats_recover_a_known_lifetime(tau_ns):
    real, imag, mean = _single_exponential(tau_ns, 40.0)
    masks = {'c1': np.ones(real.shape, dtype=bool)}
    found = phasor.cursor_stats(real, imag, mean, masks, 40.0)[0]
    assert found['tau_phi_ns'] == pytest.approx(tau_ns, abs=1e-6)
    assert found['tau_mod_ns'] == pytest.approx(tau_ns, abs=1e-6)
    assert found['n_pixels'] == real.size
    assert found['photons'] == pytest.approx(500.0 * real.size)


def test_cursor_stats_report_the_phasor_coordinates():
    real, imag, mean = _single_exponential(2.4, 40.0)
    masks = {'c1': np.ones(real.shape, dtype=bool)}
    found = phasor.cursor_stats(real, imag, mean, masks, 40.0)[0]
    assert found['mean_g'] == pytest.approx(float(real[0, 0]))
    assert found['mean_s'] == pytest.approx(float(imag[0, 0]))


def test_an_empty_cursor_reports_pixels_and_no_lifetime():
    real, imag, mean = _single_exponential(2.4, 40.0)
    masks = {'c1': np.zeros(real.shape, dtype=bool)}
    found = phasor.cursor_stats(real, imag, mean, masks, 40.0)[0]
    assert found['n_pixels'] == 0
    assert 'tau_phi_ns' not in found


def test_cursor_stats_without_a_frequency_report_no_lifetime():
    real, imag, mean = _single_exponential(2.4, 40.0)
    masks = {'c1': np.ones(real.shape, dtype=bool)}
    found = phasor.cursor_stats(real, imag, mean, masks, None)[0]
    assert found['n_pixels'] == real.size
    assert 'tau_phi_ns' not in found

import os
from pathlib import Path

PTU_PATH = os.environ.get('FLIMKIT_TEST_PTU', '')
SDT_PATH = os.environ.get('FLIMKIT_TEST_SDT', '')


@pytest.mark.skipif(not PTU_PATH or not Path(PTU_PATH).exists(),
                    reason='set FLIMKIT_TEST_PTU')
def test_generic_path_matches_flimkits_ptu_phasor():
    """FLIMKit's own entry point is PTU-only. Routing through FLIMFile must
    give the same numbers, or the two front ends would disagree."""
    import io
    import contextlib
    from flimkit.phasor_launcher import _process_ptu

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        reference = _process_ptu(PTU_PATH, channel=0)
        ours = phasor.compute(PTU_PATH, channel=0)

    np.testing.assert_allclose(
        ours['real'], phasor._first_harmonic(reference['real_cal']),
        equal_nan=True, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        ours['imag'], phasor._first_harmonic(reference['imag_cal']),
        equal_nan=True, rtol=1e-12, atol=1e-12)
    assert ours['frequency'] == pytest.approx(float(reference['frequency']))


@pytest.mark.skipif(not SDT_PATH or not Path(SDT_PATH).exists(),
                    reason='set FLIMKIT_TEST_SDT')
def test_becker_hickl_sdt_gets_a_phasor():
    found = phasor.compute(SDT_PATH, channel=0)

    assert found['real'].ndim == 2
    assert found['real'].shape == found['imag'].shape == found['mean'].shape
    assert found['frequency'] > 0
    valid = phasor.valid_pixels(found['real'], found['mean'], min_photons=1.0)
    assert valid.any(), 'no pixel had enough photons for a phasor'


def test_settings_lists_the_registered_filters():
    from flimkit.phasor.filters import phasor_filter_methods

    found = phasor.settings()

    entry = next(e for e in found['schema'] if e['key'] == 'phasor_filter')
    assert entry['choices'][0] == 'none'
    assert set(phasor_filter_methods()) <= set(entry['choices'])
    assert found['values']['phasor_filter'] == 'none'


def test_settings_lists_every_installed_machine_irf():
    from flimkit_bridge import irf as irf_module

    found = phasor.settings()

    entry = next(e for e in found['schema'] if e['key'] == 'irf')
    for installed in irf_module.available():
        assert installed['id'] in entry['installed']


def test_normalise_fills_the_defaults():
    found = phasor.normalise(None)

    assert found == {'phasor_filter': 'none', 'filter_sigma': 1.0,
                     'filter_size': 3, 'irf': 'none'}


def test_normalise_coerces_the_wire_types():
    found = phasor.normalise({'filter_sigma': '2.5', 'filter_size': '5'})

    assert found['filter_sigma'] == 2.5
    assert found['filter_size'] == 5


def test_normalise_treats_an_empty_filter_as_none():
    assert phasor.normalise({'phasor_filter': ''})['phasor_filter'] == 'none'
    assert phasor.normalise({'phasor_filter': 'None'})['phasor_filter'] == 'none'


def test_cache_key_separates_two_filters():
    plain = phasor.cache_key('d1', {})
    smoothed = phasor.cache_key('d1', {'phasor_filter': 'median'})

    assert plain != smoothed


def test_cache_key_separates_calibrated_from_uncalibrated():
    plain = phasor.cache_key('d1', {})
    calibrated = phasor.cache_key('d1', {'irf': 'machine_2026'})

    assert plain != calibrated


def test_cache_key_is_stable_across_equivalent_requests():
    assert phasor.cache_key('d1', {'filter_size': '3'}) == phasor.cache_key('d1', {})


def test_no_filter_returns_the_input_untouched(two_populations):
    real, imag, mean = two_populations

    filtered_real, filtered_imag = phasor.apply_filter(
        real, imag, mean, {'phasor_filter': 'none'})

    assert filtered_real is real
    assert filtered_imag is imag


def test_the_median_filter_tightens_a_noisy_population(two_populations):
    real, imag, mean = two_populations

    filtered_real, _ = phasor.apply_filter(
        real, imag, mean, {'phasor_filter': 'median', 'filter_size': 3})

    assert filtered_real.shape == real.shape
    assert filtered_real[:8].std() < real[:8].std()


def test_the_gaussian_filter_honours_sigma():
    rng = np.random.default_rng(1)
    real = rng.normal(0.30, 0.02, (32, 32))
    imag = rng.normal(0.40, 0.02, (32, 32))
    mean = np.full((32, 32), 100.0)

    gentle, _ = phasor.apply_filter(
        real, imag, mean, {'phasor_filter': 'gaussian', 'filter_sigma': 0.5})
    heavy, _ = phasor.apply_filter(
        real, imag, mean, {'phasor_filter': 'gaussian', 'filter_sigma': 3.0})

    assert gentle.std() < real.std()
    assert heavy.std() < gentle.std()


def test_an_unknown_filter_is_rejected(two_populations):
    real, imag, mean = two_populations

    with pytest.raises(ValueError):
        phasor.apply_filter(real, imag, mean, {'phasor_filter': 'nonsense'})


def test_resolve_irf_returns_nothing_for_none():
    assert phasor.resolve_irf('none') is None
    assert phasor.resolve_irf('') is None


def test_resolve_irf_accepts_a_path_on_disk(tmp_path):
    xlsx = tmp_path / 'irf.xlsx'
    xlsx.write_bytes(b'not really an xlsx')

    assert phasor.resolve_irf(str(xlsx)) == str(xlsx)


def test_resolve_irf_rejects_an_unknown_id():
    with pytest.raises(ValueError, match='no such IRF'):
        phasor.resolve_irf('not-installed')


def test_calibration_gets_a_time_axis():
    import xarray as xr

    class Handle:
        time_ns = np.arange(32) * 0.1

    stack = np.zeros((4, 4, 32), dtype=float)
    signal = phasor._signal_array(stack, Handle(), 80.0)

    assert isinstance(signal, xr.DataArray)
    assert signal.dims == ('Y', 'X', 'H')
    assert signal.coords['H'].values[1] == pytest.approx(0.1)
    assert signal.attrs['frequency'] == 80.0


def test_a_reader_without_a_time_axis_is_refused():
    class Handle:
        time_ns = None

    with pytest.raises(ValueError, match='no per-bin time axis'):
        phasor._signal_array(np.zeros((4, 4, 32)), Handle(), 80.0)


def test_phasor_settings_are_served(serve_state):
    from flimkit_bridge.datasets import DatasetRegistry
    from flimkit_bridge.server import BridgeState
    state = BridgeState(images={})
    state.datasets = DatasetRegistry()
    url = serve_state(state)

    request = Request(f'{url}/v1/phasor/settings',
                      headers={'Authorization': 'Bearer test-token'})
    with urlopen(request) as response:
        payload = json.load(response)

    assert payload['values']['phasor_filter'] == 'none'
    keys = [entry['key'] for entry in payload['schema']]
    assert 'phasor_filter' in keys and 'irf' in keys


def test_a_polygon_selects_one_population(two_populations):
    real, imag, mean = two_populations

    masks = phasor.cursor_masks(
        real, imag, mean,
        cursors=[{'id': 'p1', 'type': 'polygon',
                  'vertices': [[0.25, 0.35], [0.35, 0.35],
                               [0.35, 0.45], [0.25, 0.45]]}],
        min_photons=1.0)

    mask = masks['p1']
    assert mask[:8].all()
    assert not mask[8:].any()


def test_a_polygon_and_an_ellipse_share_one_request(two_populations):
    real, imag, mean = two_populations

    masks = phasor.cursor_masks(
        real, imag, mean,
        cursors=[
            {'id': 'p1', 'type': 'polygon',
             'vertices': [[0.25, 0.35], [0.35, 0.35],
                          [0.35, 0.45], [0.25, 0.45]]},
            {'id': 'e1', 'center_g': 0.70, 'center_s': 0.35, 'radius': 0.05},
        ],
        min_photons=1.0)

    assert masks['p1'].sum() == 128
    assert masks['e1'].sum() == 128
    assert not (masks['p1'] & masks['e1']).any()


def test_a_polygon_drops_low_photon_pixels(two_populations):
    real, imag, mean = two_populations
    mean = mean.copy()
    mean[0, :] = 0.0

    masks = phasor.cursor_masks(
        real, imag, mean,
        cursors=[{'id': 'p1', 'type': 'polygon',
                  'vertices': [[0.25, 0.35], [0.35, 0.35],
                               [0.35, 0.45], [0.25, 0.45]]}],
        min_photons=10.0)

    assert not masks['p1'][0].any()
    assert masks['p1'][1:8].all()


def test_a_polygon_needs_three_vertices(two_populations):
    real, imag, mean = two_populations

    with pytest.raises(ValueError, match='three vertices'):
        phasor.cursor_masks(
            real, imag, mean,
            cursors=[{'id': 'p1', 'type': 'polygon',
                      'vertices': [[0.25, 0.35], [0.35, 0.45]]}],
            min_photons=1.0)


def test_a_polygon_with_no_vertices_is_rejected(two_populations):
    real, imag, mean = two_populations

    with pytest.raises(ValueError, match='three vertices'):
        phasor.cursor_masks(
            real, imag, mean,
            cursors=[{'id': 'p1', 'type': 'polygon'}],
            min_photons=1.0)


def test_an_unknown_cursor_type_is_rejected(two_populations):
    real, imag, mean = two_populations

    with pytest.raises(ValueError, match='unknown cursor type'):
        phasor.cursor_masks(
            real, imag, mean,
            cursors=[{'id': 'x1', 'type': 'freehand'}],
            min_photons=1.0)


def test_a_polygon_gets_its_own_label(two_populations):
    real, imag, mean = two_populations

    labels = phasor.label_image(
        real, imag, mean,
        cursors=[
            {'id': 'e1', 'center_g': 0.70, 'center_s': 0.35, 'radius': 0.05},
            {'id': 'p1', 'type': 'polygon',
             'vertices': [[0.25, 0.35], [0.35, 0.35],
                          [0.35, 0.45], [0.25, 0.45]]},
        ],
        min_photons=1.0)

    assert set(np.unique(labels)) == {1, 2}
    assert (labels[8:] == 1).all()
    assert (labels[:8] == 2).all()


def test_a_polygon_reports_lifetimes_like_an_ellipse(two_populations):
    real, imag, mean = two_populations
    cursors = [{'id': 'p1', 'type': 'polygon',
                'vertices': [[0.25, 0.35], [0.35, 0.35],
                             [0.35, 0.45], [0.25, 0.45]]}]

    masks = phasor.cursor_masks(real, imag, mean, cursors, min_photons=1.0)
    found = phasor.cursor_stats(real, imag, mean, masks, 80.0)

    assert len(found) == 1
    assert found[0]['n_pixels'] == 128
    assert found[0]['tau_phi_ns'] > 0
    assert found[0]['mean_g'] == pytest.approx(0.30, abs=0.01)


def test_the_irf_setting_is_a_path_field():
    found = phasor.settings()

    entry = next(e for e in found['schema'] if e['key'] == 'irf')
    assert entry['type'] == 'path', (
        'a dropdown cannot point at a reference measurement on disk')


def test_resolve_irf_still_accepts_an_installed_id():
    from flimkit_bridge import irf as irf_module

    installed = irf_module.available()
    if not installed:
        pytest.skip('no machine IRF installed')
    assert phasor.resolve_irf(installed[0]['id']) == installed[0]['path']


def test_a_reference_measurement_is_read_as_a_decay(tmp_path):
    class Reference:
        time_ns = np.arange(64) * 0.05

        def summed_decay(self, channel=None):
            counts = np.zeros(64)
            counts[8] = 100.0
            counts[9] = 40.0
            return counts

    irf_time_ns, irf_counts = phasor._irf_from_reference(Reference())

    assert irf_time_ns[8] == pytest.approx(0.4)
    assert irf_counts[8] == 100.0
    assert irf_counts.sum() == 140.0


def test_a_reference_with_no_photons_is_refused():
    class Empty:
        time_ns = np.arange(64) * 0.05

        def summed_decay(self, channel=None):
            return np.zeros(64)

    with pytest.raises(ValueError, match='no photons'):
        phasor._irf_from_reference(Empty())


def test_an_unreadable_reference_says_what_is_supported(tmp_path):
    bogus = tmp_path / 'reference.pqres'
    bogus.write_bytes(b'not a flim file')

    with pytest.raises(ValueError, match='pqres'):
        phasor._reference_irf(str(bogus))
