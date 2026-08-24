import json
from io import BytesIO
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import tifffile

from flimkit_bridge.server import BridgeState, create_server


class FakeApp:
    def __init__(self):
        self.imported = []


@pytest.fixture
def live_server(serve_state, monkeypatch):
    import flimkit_bridge.server as server_module

    app = FakeApp()
    images = {
        'intensity': np.arange(35, dtype=np.float32).reshape(5, 7),
        'lifetime': np.arange(35, dtype=np.float32).reshape(5, 7) / 10.0,
    }
    rois = {
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'properties': {'name': 'Live ROI'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0.0, 0.0], [2.0, 0.0], [1.0, 2.0], [0.0, 0.0]]],
            },
        }],
    }
    monkeypatch.setattr(server_module, '_live_images',
                        lambda state: (images, {'intensity': 'photons', 'lifetime': 'ns'}))
    monkeypatch.setattr(server_module, '_live_rois', lambda state: rois)
    monkeypatch.setattr(server_module, '_live_import',
                        lambda state, payload: state.app.imported.append(payload) or [1])

    state = BridgeState(images={}, units={}, app=app)
    server = create_server('127.0.0.1', 0, 'test-token', state, live=True)
    import threading
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f'http://{host}:{port}', state, app, images, rois
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(base_url, path, token='test-token'):
    return urlopen(Request(f'{base_url}{path}',
                           headers={'Authorization': f'Bearer {token}'}))


def test_connected_is_false_until_a_client_authorises(live_server):
    base_url, state, _, _, _ = live_server
    assert state.connected is False
    with urlopen(f'{base_url}/v1/status'):
        pass
    assert state.connected is False
    with _get(base_url, '/v1/images/intensity.tif'):
        pass
    assert state.connected is True


def test_bad_token_does_not_mark_connected(live_server):
    base_url, state, _, _, _ = live_server
    with pytest.raises(HTTPError):
        _get(base_url, '/v1/images/intensity.tif', token='wrong')
    assert state.connected is False


def test_live_images_come_from_the_app(live_server):
    base_url, _, _, images, _ = live_server
    with _get(base_url, '/v1/images/lifetime.tif') as response:
        received = tifffile.imread(BytesIO(response.read()))
        unit = response.headers['X-FLIMKit-Value-Unit']
    np.testing.assert_array_equal(received, images['lifetime'])
    assert unit == 'ns'


def test_live_rois_come_from_the_app(live_server):
    base_url, _, _, _, rois = live_server
    with _get(base_url, '/v1/rois') as response:
        assert json.load(response) == rois


def test_posted_rois_reach_the_app(live_server):
    base_url, _, app, _, _ = live_server
    payload = {'type': 'FeatureCollection', 'features': []}
    request = Request(
        f'{base_url}/v1/rois',
        data=json.dumps(payload).encode('utf-8'),
        method='POST',
        headers={'Authorization': 'Bearer test-token'},
    )
    with urlopen(request) as response:
        assert json.load(response) == {'received_features': 0}
    assert app.imported == [payload]
