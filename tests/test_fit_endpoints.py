import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from flimkit_bridge.datasets import DatasetRegistry
from flimkit_bridge.server import BridgeState

PTU_PATH = os.environ.get('FLIMKIT_TEST_PTU', '')

pytestmark = pytest.mark.skipif(
    not PTU_PATH or not Path(PTU_PATH).exists(),
    reason='set FLIMKIT_TEST_PTU to run the fit endpoint tests',
)


@pytest.fixture
def served(serve_state):
    state = BridgeState(images={})
    state.datasets = DatasetRegistry()
    return serve_state(state)


def _call(url, path, method='GET', body=None):
    headers = {'Authorization': 'Bearer test-token'}
    data = None
    if body is not None:
        data = json.dumps(body).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    request = Request(f'{url}{path}', data=data, method=method, headers=headers)
    with urlopen(request) as response:
        return json.load(response)


def _square(name, size):
    return {
        'type': 'Feature',
        'properties': {'name': name},
        'geometry': {'type': 'Polygon', 'coordinates': [[
            [0.0, 0.0], [float(size), 0.0], [float(size), float(size)],
            [0.0, float(size)], [0.0, 0.0]]]},
    }


def test_defaults_are_served(served):
    payload = _call(served, '/v1/fit/defaults')

    assert 'values' in payload and 'schema' in payload
    assert payload['values']['n_exp'] >= 1


def test_fitting_a_real_roi_returns_measurements(served):
    opened = _call(served, '/v1/datasets', 'POST', {'path': PTU_PATH})
    half = opened['width'] // 2

    payload = _call(
        served, f"/v1/datasets/{opened['id']}/fit/roi", 'POST',
        {'rois': {'type': 'FeatureCollection',
                  'features': [_square('Whole quadrant', half)]},
         'params': {'n_exp': 1, 'binning': 8, 'tau_min_ns': 0.2,
                    'tau_max_ns': 8.0}})

    assert payload['binning'] == 8
    assert payload['params_used']['n_exp'] == 1
    result = payload['results'][0]
    assert result['name'] == 'Whole quadrant'
    assert result['n_pixels'] > 0
    assert result['photon_count'] > 0
    assert len(result['taus_ns']) == 1
    assert 0.2 <= result['taus_ns'][0] <= 8.0
    assert result['chi2_r'] > 0
    from flimkit_bridge import irf as irf_module
    if irf_module.available():
        assert 'machine_irf_default' in result['irf_source'], (
            "the default should be FLIMKit's configured machine IRF, not a "
            'site-specific one')
    assert payload['params_used']['irf_strategy'] == 'machine_irf'
    assert payload['params_used']['min_photons'] == 5


def test_two_rois_give_two_results(served):
    opened = _call(served, '/v1/datasets', 'POST', {'path': PTU_PATH})
    binning = 1
    while binning < 8 and min(opened['width'], opened['height']) // (binning * 2) >= 32:
        binning *= 2
    quarter = opened['width'] // (4 * binning)
    assert quarter >= 2, (
        f"{opened['width']}x{opened['height']} is too small to hold two nested "
        'regions at any binning')

    payload = _call(
        served, f"/v1/datasets/{opened['id']}/fit/roi", 'POST',
        {'rois': {'type': 'FeatureCollection',
                  'features': [_square('A', quarter), _square('B', quarter * 2)]},
         'params': {'n_exp': 1, 'binning': binning}})

    assert [r['name'] for r in payload['results']] == ['A', 'B']
    for result in payload['results']:
        assert 'error' not in result, result['error']
    assert payload['results'][1]['n_pixels'] > payload['results'][0]['n_pixels']


def test_bad_parameters_are_rejected(served):
    opened = _call(served, '/v1/datasets', 'POST', {'path': PTU_PATH})

    with pytest.raises(HTTPError) as caught:
        _call(served, f"/v1/datasets/{opened['id']}/fit/roi", 'POST',
              {'rois': {'type': 'FeatureCollection', 'features': []},
               'params': {'n_exp': 99}})
    assert caught.value.code == 400


def test_a_missing_collection_is_rejected(served):
    opened = _call(served, '/v1/datasets', 'POST', {'path': PTU_PATH})

    with pytest.raises(HTTPError) as caught:
        _call(served, f"/v1/datasets/{opened['id']}/fit/roi", 'POST', {'params': {}})
    assert caught.value.code == 400


def test_fitting_an_unknown_dataset_is_404(served):
    with pytest.raises(HTTPError) as caught:
        _call(served, '/v1/datasets/ds_999/fit/roi', 'POST',
              {'rois': {'type': 'FeatureCollection', 'features': [_square('X', 4)]}})
    assert caught.value.code == 404
