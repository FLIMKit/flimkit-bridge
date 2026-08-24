import json
import threading
from io import BytesIO
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import tifffile

from flimkit_bridge.server import BridgeState, create_server


@pytest.fixture
def running_server():
    state = BridgeState(
        images={
            'intensity': np.arange(35, dtype=np.float32).reshape(5, 7),
            'lifetime': np.arange(35, dtype=np.float32).reshape(5, 7) / 10.0,
        },
        units={'intensity': 'photons', 'lifetime': 'ns'},
        exported_rois={
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'properties': {'objectType': 'annotation', 'name': 'FLIMKit rect'},
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[
                            [0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0], [0.0, 0.0],
                        ]],
                    },
                },
                {
                    'type': 'Feature',
                    'properties': {'objectType': 'annotation', 'name': 'FLIMKit triangle'},
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[
                            [1.0, 1.0], [5.0, 1.0], [3.0, 4.0], [1.0, 1.0],
                        ]],
                    },
                },
            ],
        },
    )
    server = create_server('127.0.0.1', 0, 'test-token', state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f'http://{host}:{port}', state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_status_needs_no_token(running_server):
    base_url, _ = running_server
    with urlopen(f'{base_url}/v1/status') as response:
        payload = json.load(response)
    assert payload['protocol'] == 'flimkit-qupath'
    assert payload['protocol_version'] == 1
    assert payload['bridge_version']
    assert payload['flimkit_version']


def test_image_requires_token(running_server):
    base_url, _ = running_server
    with pytest.raises(HTTPError) as caught:
        urlopen(f'{base_url}/v1/images/intensity.tif')
    assert caught.value.code == 401


@pytest.mark.parametrize('image_id,unit,dtype', [
    ('intensity', 'photons', np.uint16),
    ('lifetime', 'ns', np.float32),
])
def test_image_round_trips_with_unit(running_server, image_id, unit, dtype):
    base_url, state = running_server
    request = Request(
        f'{base_url}/v1/images/{image_id}.tif',
        headers={'Authorization': 'Bearer test-token'},
    )
    with urlopen(request) as response:
        received = tifffile.imread(BytesIO(response.read()))
        header_unit = response.headers['X-FLIMKit-Value-Unit']
    assert received.dtype == dtype
    assert header_unit == unit
    np.testing.assert_array_equal(received, state.images[image_id].astype(dtype))


def test_unknown_image_is_404(running_server):
    base_url, _ = running_server
    request = Request(
        f'{base_url}/v1/images/chi2.tif',
        headers={'Authorization': 'Bearer test-token'},
    )
    with pytest.raises(HTTPError) as caught:
        urlopen(request)
    assert caught.value.code == 404


def test_exported_rois_require_token(running_server):
    base_url, _ = running_server
    with pytest.raises(HTTPError) as caught:
        urlopen(f'{base_url}/v1/rois')
    assert caught.value.code == 401


def test_exported_rois_are_served(running_server):
    base_url, state = running_server
    request = Request(
        f'{base_url}/v1/rois',
        headers={'Authorization': 'Bearer test-token'},
    )
    with urlopen(request) as response:
        payload = json.load(response)
    assert payload == state.exported_rois


def test_roi_collection_is_stored(running_server):
    base_url, state = running_server
    payload = {
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'properties': {'name': 'Cell 1'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[1.25, 2.5], [4.5, 2.5], [3.0, 4.0], [1.25, 2.5]]],
            },
        }],
    }
    request = Request(
        f'{base_url}/v1/rois',
        data=json.dumps(payload).encode('utf-8'),
        method='POST',
        headers={
            'Authorization': 'Bearer test-token',
            'Content-Type': 'application/geo+json',
        },
    )
    with urlopen(request) as response:
        reply = json.load(response)
    assert reply == {'received_features': 1}
    assert state.received_rois == [payload]


def test_non_collection_is_rejected(running_server):
    base_url, _ = running_server
    request = Request(
        f'{base_url}/v1/rois',
        data=json.dumps({'type': 'Feature'}).encode('utf-8'),
        method='POST',
        headers={'Authorization': 'Bearer test-token'},
    )
    with pytest.raises(HTTPError) as caught:
        urlopen(request)
    assert caught.value.code == 400
