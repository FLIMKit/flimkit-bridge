import numpy as np
import pytest

from flimkit_bridge import fitting


def _synthetic_stack(tau_ns=2.0, n_bins=256, tcspc_res=5e-11, shape=(8, 8),
                     photons=400000, background=2.0, noise=True, seed=0):
    """A decay built the way the reconvolution model expects: an exponential
    convolved with the IRF the fit will be given. Building a bare exponential
    instead makes the model systematically broader than the data, which biases
    tau low by a few percent and pushes reduced chi-squared into the hundreds."""
    from flimkit.FLIM.irf_tools import gaussian_irf
    t_ns = np.arange(n_bins) * tcspc_res * 1e9
    irf = gaussian_irf(n_bins, 30, 4.0)
    model = np.real(np.fft.ifft(np.fft.fft(np.exp(-t_ns / tau_ns))
                                * np.fft.fft(irf)))
    model = np.clip(model, 0, None)
    model = model / model.sum() * photons + background
    per_pixel = np.tile(model, shape + (1,)) / (shape[0] * shape[1])
    if noise:
        stack = np.random.default_rng(seed).poisson(per_pixel)
    else:
        stack = np.rint(per_pixel)
    return stack.astype(np.uint32), irf


def test_defaults_have_values_and_a_schema():
    payload = fitting.defaults()

    assert set(payload) == {'values', 'schema'}
    keys = {entry['key'] for entry in payload['schema']}
    assert {'n_exp', 'tau_min_ns', 'tau_max_ns', 'cost_function'} <= keys
    for entry in payload['schema']:
        assert entry['key'] in payload['values']
        assert entry['type'] in ('int', 'float', 'choice', 'bool', 'path')
        assert entry['applies_to']


def test_schema_choices_are_listed_for_choice_fields():
    for entry in fitting.defaults()['schema']:
        if entry['type'] == 'choice':
            assert entry['choices'], f"{entry['key']} has no choices"


def test_merge_params_keeps_defaults_for_absent_keys():
    merged = fitting.merge_params({'n_exp': 3})

    assert merged['n_exp'] == 3
    assert merged['tau_min_ns'] == fitting.defaults()['values']['tau_min_ns']


def test_merge_params_rejects_an_unknown_key():
    with pytest.raises(ValueError, match='unknown fit parameter'):
        fitting.merge_params({'nonsense': 1})


def test_merge_params_rejects_a_bad_component_count():
    with pytest.raises(ValueError, match='n_exp'):
        fitting.merge_params({'n_exp': 9})


@pytest.mark.parametrize('truth', [0.5, 1.0, 2.0, 4.0])
def test_a_known_single_exponential_is_recovered(truth):
    stack, irf = _synthetic_stack(tau_ns=truth)
    mask = np.ones(stack.shape[:2], dtype=bool)

    result = fitting.fit_masked_decay(
        stack, mask, tcspc_res=5e-11, n_bins=256,
        params=fitting.merge_params({'n_exp': 1, 'tau_min_ns': 0.05,
                                     'tau_max_ns': 10.0,
                                     'irf_strategy': 'session'}),
        irf_prompt=irf)

    assert result['taus_ns'][0] == pytest.approx(truth, rel=0.02)
    assert result['tau_mean_ns'] == pytest.approx(truth, rel=0.02)
    assert result['photon_count'] > 0


@pytest.mark.parametrize('truth', [0.5, 1.0, 2.0, 4.0])
def test_a_correct_fit_has_reduced_chi_squared_near_one(truth):
    """The assertion that actually catches a model/data mismatch. A bare
    exponential fitted with the reconvolution model gives chi2r in the
    hundreds while still landing within a few percent of the right tau."""
    stack, irf = _synthetic_stack(tau_ns=truth)

    result = fitting.fit_masked_decay(
        stack, np.ones(stack.shape[:2], dtype=bool), tcspc_res=5e-11,
        n_bins=256,
        params=fitting.merge_params({'n_exp': 1, 'tau_min_ns': 0.05,
                                     'tau_max_ns': 10.0,
                                     'irf_strategy': 'session'}),
        irf_prompt=irf)

    assert 0.5 < result['chi2_r'] < 2.0, (
        f"reduced chi-squared {result['chi2_r']:.3f} suggests the model and "
        'the data disagree')


def test_mask_selects_only_its_pixels():
    stack, irf = _synthetic_stack(shape=(4, 4))
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True

    result = fitting.fit_masked_decay(
        stack, mask, tcspc_res=5e-11, n_bins=256,
        params=fitting.merge_params({'n_exp': 1, 'irf_strategy': 'session'}),
        irf_prompt=irf)

    assert result['n_pixels'] == 1
    assert result['photon_count'] == int(stack[0, 0].sum())


def test_an_empty_mask_is_refused():
    stack, _ = _synthetic_stack(shape=(4, 4))

    with pytest.raises(ValueError, match='no pixels'):
        fitting.fit_masked_decay(
            stack, np.zeros((4, 4), dtype=bool), tcspc_res=5e-11,
            n_bins=256, params=fitting.merge_params({'irf_strategy': 'session'}))


def test_mask_shape_must_match_the_stack():
    stack, _ = _synthetic_stack(shape=(4, 4))

    with pytest.raises(ValueError, match='does not match'):
        fitting.fit_masked_decay(
            stack, np.ones((8, 8), dtype=bool), tcspc_res=5e-11,
            n_bins=256, params=fitting.merge_params({'irf_strategy': 'session'}))


def test_geojson_becomes_masks_at_the_stack_resolution():
    collection = {
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'properties': {'name': 'Cell 1'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0.0, 0.0], [8.0, 0.0], [8.0, 8.0],
                                 [0.0, 8.0], [0.0, 0.0]]],
            },
        }],
    }

    masks = fitting.masks_from_geojson(collection, image_shape=(16, 16),
                                       binning=2)

    assert len(masks) == 1
    name, mask = masks[0]
    assert name == 'Cell 1'
    assert mask.shape == (8, 8)
    assert mask.any()


def test_geojson_masks_scale_with_binning():
    collection = {
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'properties': {'name': 'Half'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0.0, 0.0], [16.0, 0.0], [16.0, 8.0],
                                 [0.0, 8.0], [0.0, 0.0]]],
            },
        }],
    }

    unbinned = fitting.masks_from_geojson(collection, (16, 16), binning=1)[0][1]
    binned = fitting.masks_from_geojson(collection, (16, 16), binning=2)[0][1]

    assert unbinned.shape == (16, 16)
    assert binned.shape == (8, 8)
    assert binned.sum() == pytest.approx(unbinned.sum() / 4, rel=0.35)


def test_the_default_irf_is_a_machine_irf():
    """The user asked for a machine IRF by default rather than a guess."""
    assert fitting.defaults()['values']['irf_strategy'] == 'machine_irf'


def test_irf_strategy_is_offered_as_a_choice():
    entry = next(e for e in fitting.defaults()['schema']
                 if e['key'] == 'irf_strategy')

    assert entry['advanced'] is False
    assert 'machine_irf' in entry['choices']
    assert 'session' in entry['choices']
    assert 'gaussian' in entry['choices']


def test_min_photons_is_a_per_pixel_setting_only():
    entry = next(e for e in fitting.defaults()['schema']
                 if e['key'] == 'min_photons')

    assert entry['applies_to'] == ['per_pixel']
    assert fitting.defaults()['values']['min_photons'] == 5


def test_dim_pixels_still_count_towards_a_region_fit():
    stack, irf = _synthetic_stack(shape=(4, 4))
    stack = stack.copy()
    stack[0, 0] = 0

    result = fitting.fit_masked_decay(
        stack, np.ones((4, 4), dtype=bool), tcspc_res=5e-11, n_bins=256,
        params=fitting.merge_params({'n_exp': 1, 'min_photons': 5,
                                     'irf_strategy': 'session'}),
        irf_prompt=irf)

    assert result['n_pixels'] == 16


def test_a_sparse_region_keeps_every_photon():
    stack, irf = _synthetic_stack(shape=(8, 8))
    sparse = (stack > 0).astype(stack.dtype)

    result = fitting.fit_masked_decay(
        sparse, np.ones((8, 8), dtype=bool), tcspc_res=5e-11, n_bins=256,
        params=fitting.merge_params({'n_exp': 1, 'min_photons': 5,
                                     'irf_strategy': 'session'}),
        irf_prompt=irf)

    assert result['photon_count'] == int(sparse.sum())
    assert result['n_pixels'] == 64


def test_roi_sync_scales_with_the_fraction_of_the_field_selected():
    assert fitting.roi_sync(1000000, 256, 1024) == 250000
    assert fitting.roi_sync(1000000, 1024, 1024) == 1000000
    assert fitting.roi_sync(None, 256, 1024) is None
    assert fitting.roi_sync(1000000, 256, 0) is None


def test_a_region_fit_reaches_the_model_with_a_scaled_sync_count(monkeypatch):
    stack, irf = _synthetic_stack(shape=(8, 8))
    mask = np.zeros((8, 8), dtype=bool)
    mask[:4] = True
    seen = {}

    def spy(decay, n_pixels, tcspc_res, n_bins, params, irf_prompt=None,
            n_sync=None):
        seen['n_sync'] = n_sync
        seen['n_pixels'] = n_pixels
        return {}

    monkeypatch.setattr(fitting, 'fit_decay', spy)
    fitting.fit_masked_decay(
        stack, mask, tcspc_res=5e-11, n_bins=256,
        params=fitting.merge_params({'n_exp': 1, 'pileup_in_model': True,
                                     'irf_strategy': 'session'}),
        irf_prompt=irf, n_sync=6400000)

    assert seen['n_pixels'] == 32
    assert seen['n_sync'] == 3200000


def test_tail_is_offered_and_defaults_to_reconv():
    values = fitting.defaults()['values']
    assert values['fit_model'] == 'reconv'
    assert fitting.merge_params({'fit_model': 'tail'})['fit_model'] == 'tail'
    with pytest.raises(ValueError):
        fitting.merge_params({'fit_model': 'deconv'})


def test_the_stack_limit_is_configurable(monkeypatch):
    from flimkit_bridge import datasets

    monkeypatch.setenv('FLIMKIT_BRIDGE_MAX_STACK_BYTES', '9000000000')
    assert datasets.default_max_stack() == 9000000000
    assert datasets.DatasetRegistry()._max_stack == 9000000000
    monkeypatch.setenv('FLIMKIT_BRIDGE_MAX_STACK_BYTES', 'not a number')
    assert datasets.default_max_stack() >= datasets.DEFAULT_MAX_STACK


def test_the_cap_scales_with_the_machine(monkeypatch):
    import psutil

    from flimkit_bridge import datasets

    monkeypatch.delenv('FLIMKIT_BRIDGE_MAX_STACK_BYTES', raising=False)
    ram = psutil.virtual_memory().total
    assert datasets.default_max_stack() == max(datasets.DEFAULT_MAX_STACK,
                                               int(ram * 0.25))


def test_the_summed_fit_does_not_spawn_a_process_pool(monkeypatch):
    from flimkit.FLIM import fitters

    seen = {}
    real = fitters.fit_summed

    def watched(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(fitters, 'fit_summed', watched)
    stack, irf = _synthetic_stack(shape=(4, 4))
    decay = np.asarray(stack).reshape(-1, 256).sum(axis=0).astype(float)
    fitting.fit_decay(
        decay, 16, 5e-11, 256,
        fitting.merge_params({'n_exp': 1, 'irf_strategy': 'session'}),
        irf_prompt=irf)
    assert seen['workers'] == 1
    assert seen['optimizer'] == 'de'


def test_per_pixel_uses_the_gpu_unless_it_is_turned_off():
    assert fitting.defaults()['values']['use_gpu'] is True
    assert fitting.merge_params({'use_gpu': False})['use_gpu'] is False


def test_accepts_reports_a_missing_keyword():
    from flimkit_bridge.fitting import accepts

    def older(stack, correct_pileup=False, n_sync=None):
        return None

    def newer(stack, correct_pileup=False, n_sync=None, pileup_in_model=False):
        return None

    assert not accepts(older, 'pileup_in_model')
    assert accepts(newer, 'pileup_in_model')


def test_pileup_in_the_model_is_refused_on_a_flimkit_without_it():
    from flimkit_bridge.fitting import _pileup_kwargs

    def older(stack, correct_pileup=False, n_sync=None):
        return None

    with pytest.raises(ValueError, match='needs a FLIMKit that offers it'):
        _pileup_kwargs({'pileup_in_model': True}, older, 1000)


def test_coates_still_reaches_a_flimkit_without_the_model_route():
    from flimkit_bridge.fitting import _pileup_kwargs

    def older(stack, correct_pileup=False, n_sync=None):
        return None

    found = _pileup_kwargs({'correct_pileup': True}, older, 1000)
    assert found == {'correct_pileup': True, 'n_sync': 1000}


def test_the_model_route_passes_through_when_flimkit_has_it():
    from flimkit_bridge.fitting import _pileup_kwargs

    def newer(stack, correct_pileup=False, n_sync=None, pileup_in_model=False):
        return None

    found = _pileup_kwargs({'pileup_in_model': True}, newer, 1000)
    assert found['pileup_in_model'] is True
    assert found['n_sync'] == 1000
