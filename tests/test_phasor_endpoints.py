import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from flimkit_bridge.datasets import DatasetRegistry
from flimkit_bridge.server import BridgeState

PTU_PATH = os.environ.get('FLIMKIT_TEST_PTU', '')
SDT_PATH = os.environ.get('FLIMKIT_TEST_SDT', '')

pytestmark = pytest.mark.skipif(
    not PTU_PATH or not Path(PTU_PATH).exists(),
    reason='set FLIMKIT_TEST_PTU to run the phasor endpoint tests',
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


@pytest.fixture
def dataset(served):
    opened = _call(served, '/v1/datasets', 'POST', {'path': PTU_PATH})
    return served, opened['id']


def test_phasor_summary(dataset):
    url, ident = dataset

    payload = _call(url, f'/v1/datasets/{ident}/phasor')

    assert payload['width'] > 0 and payload['height'] > 0
    assert payload['frequency_mhz'] > 0
    assert payload['binning'] in (1, 2, 4, 8, 16)
    assert payload['calibrated'] is False


def test_phasor_points_decode_to_a_grid(dataset):
    url, ident = dataset

    payload = _call(url, f'/v1/datasets/{ident}/phasor/points?bins=64')

    counts = np.frombuffer(base64.b64decode(payload['counts']), dtype=np.uint32)
    assert counts.size == 64 * 64
    assert counts.sum() == payload['n_in_range']
    assert 0 < payload['n_in_range'] <= payload['n_valid']
    assert payload['max_count'] >= 1


def test_phasor_points_reject_a_silly_bin_count(dataset):
    url, ident = dataset

    with pytest.raises(HTTPError) as caught:
        _call(url, f'/v1/datasets/{ident}/phasor/points?bins=4')
    assert caught.value.code == 400


def test_cursor_mask_counts_pixels(dataset):
    url, ident = dataset
    summary = _call(url, f'/v1/datasets/{ident}/phasor')

    payload = _call(url, f'/v1/datasets/{ident}/phasor/mask', 'POST',
                    {'cursors': [{'id': 'c1', 'center_g': 0.5, 'center_s': 0.4,
                                  'radius': 0.3}],
                     'min_photons': 1.0})

    assert payload['binning'] == summary['binning']
    assert payload['cursors'][0]['id'] == 'c1'
    assert payload['cursors'][0]['n_pixels'] >= 0


def test_label_output_decodes(dataset):
    url, ident = dataset

    payload = _call(url, f'/v1/datasets/{ident}/phasor/mask', 'POST',
                    {'cursors': [{'id': 'c1', 'center_g': 0.5, 'center_s': 0.4,
                                  'radius': 0.3}],
                     'min_photons': 1.0, 'output': 'labels'})

    labels = np.frombuffer(base64.b64decode(payload['labels']), dtype=np.uint8)
    assert labels.size == payload['width'] * payload['height']
    assert set(np.unique(labels)) <= {0, 1}


def test_empty_cursor_list_is_refused(dataset):
    url, ident = dataset

    with pytest.raises(HTTPError) as caught:
        _call(url, f'/v1/datasets/{ident}/phasor/mask', 'POST', {'cursors': []})
    assert caught.value.code == 400


@pytest.mark.skipif(not SDT_PATH or not Path(SDT_PATH).exists(),
                    reason='set FLIMKIT_TEST_SDT')
def test_phasor_works_for_becker_hickl_sdt(served):
    """Phasor is routed through FLIMFile rather than FLIMKit's PTU-only entry
    point, so every time-domain format it can read gets a phasor."""
    opened = _call(served, '/v1/datasets', 'POST', {'path': SDT_PATH})

    payload = _call(served, f"/v1/datasets/{opened['id']}/phasor")

    assert payload['width'] > 0 and payload['height'] > 0
    assert payload['frequency_mhz'] > 0
    assert payload['binning'] in (1, 2, 4, 8, 16)


def test_phasor_summary_reports_the_options_it_used(dataset):
    url, ident = dataset

    payload = _call(url, f'/v1/datasets/{ident}/phasor?phasor_filter=median')

    assert payload['options']['phasor_filter'] == 'median'
    assert payload['frequency_mhz'] > 0


def test_a_filtered_phasor_is_not_served_from_the_plain_cache(dataset):
    url, ident = dataset

    plain = _call(url, f'/v1/datasets/{ident}/phasor/points?bins=64')
    smoothed = _call(
        url, f'/v1/datasets/{ident}/phasor/points?bins=64&phasor_filter=median')

    assert plain['counts'] != smoothed['counts']


def test_an_unknown_irf_is_a_client_error(dataset):
    url, ident = dataset

    with pytest.raises(HTTPError) as raised:
        _call(url, f'/v1/datasets/{ident}/phasor?irf=not-installed')

    assert raised.value.code == 400


def test_mask_options_ride_on_the_body(dataset):
    url, ident = dataset

    payload = _call(url, f'/v1/datasets/{ident}/phasor/mask', 'POST', {
        'cursors': [{'id': 'c1', 'center_g': 0.5, 'center_s': 0.3, 'radius': 0.2}],
        'options': {'phasor_filter': 'median'},
    })

    assert payload['options']['phasor_filter'] == 'median'


FULL_PLANE = [[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]
LEFT_HALF = [[-5.0, -5.0], [0.5, -5.0], [0.5, 5.0], [-5.0, 5.0]]
RIGHT_HALF = [[0.5, -5.0], [5.0, -5.0], [5.0, 5.0], [0.5, 5.0]]


def _mask(url, ident, cursors, min_photons=1.0, output=None):
    body = {'cursors': cursors, 'min_photons': min_photons}
    if output:
        body['output'] = output
    return _call(url, f'/v1/datasets/{ident}/phasor/mask', 'POST', body)


def test_a_polygon_covering_everything_selects_every_valid_pixel(dataset):
    url, ident = dataset
    points = _call(url, f'/v1/datasets/{ident}/phasor/points?bins=64&min_photons=1.0')

    payload = _mask(url, ident, [
        {'id': 'p1', 'type': 'polygon', 'vertices': FULL_PLANE}])

    assert payload['cursors'][0]['n_pixels'] == points['n_valid']


def test_two_polygons_partition_the_valid_pixels(dataset):
    url, ident = dataset
    points = _call(url, f'/v1/datasets/{ident}/phasor/points?bins=64&min_photons=1.0')

    payload = _mask(url, ident, [
        {'id': 'left', 'type': 'polygon', 'vertices': LEFT_HALF},
        {'id': 'right', 'type': 'polygon', 'vertices': RIGHT_HALF}])

    counts = {entry['id']: entry['n_pixels'] for entry in payload['cursors']}
    assert counts['left'] + counts['right'] == points['n_valid']


def test_a_polygon_reports_a_lifetime(dataset):
    url, ident = dataset

    payload = _mask(url, ident, [
        {'id': 'p1', 'type': 'polygon', 'vertices': FULL_PLANE}])

    entry = payload['cursors'][0]
    assert entry['n_pixels'] > 0
    assert entry['tau_phi_ns'] is not None
    assert entry['photons'] > 0


def test_a_polygon_becomes_a_label_image(dataset):
    url, ident = dataset

    payload = _mask(url, ident, [
        {'id': 'p1', 'type': 'polygon', 'vertices': FULL_PLANE}],
        output='labels')

    labels = np.frombuffer(base64.b64decode(payload['labels']), dtype=np.uint8)
    assert labels.size == payload['width'] * payload['height']
    assert int((labels == 1).sum()) == payload['cursors'][0]['n_pixels']


def test_a_polygon_and_an_ellipse_are_labelled_apart(dataset):
    url, ident = dataset

    payload = _mask(url, ident, [
        {'id': 'e1', 'center_g': 0.5, 'center_s': 0.3, 'radius': 0.1},
        {'id': 'p1', 'type': 'polygon', 'vertices': FULL_PLANE}],
        output='labels')

    labels = np.frombuffer(base64.b64decode(payload['labels']), dtype=np.uint8)
    assert set(np.unique(labels)) <= {0, 1, 2}
    assert int((labels == 1).sum()) == payload['cursors'][0]['n_pixels']


def test_a_polygon_with_two_vertices_is_a_client_error(dataset):
    url, ident = dataset

    with pytest.raises(HTTPError) as raised:
        _mask(url, ident, [
            {'id': 'p1', 'type': 'polygon', 'vertices': [[0.2, 0.3], [0.4, 0.5]]}])

    assert raised.value.code == 400
