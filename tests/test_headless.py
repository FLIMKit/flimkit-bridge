import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from flimkit_bridge import discovery, headless
from flimkit_bridge.state import build_state

SYNTH = os.environ.get('FLIMKIT_TEST_PTU', '')


def call(url, token='test-token', payload=None, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header('Authorization', 'Bearer ' + token)
    if data:
        request.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


@pytest.fixture
def headless_server(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, 'discovery_path',
                        lambda: tmp_path / 'qupath-bridge.json')
    server, url, token = headless.serve(port=0, token='test-token')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield url, token
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_build_state_attaches_both_registries():
    state = build_state()
    assert state.datasets is not None
    assert state.jobs is not None
    assert state.app is None


def test_plugin_state_attaches_both_registries():
    import flimkit_bridge

    state = flimkit_bridge._live_state(object())
    assert state.datasets is not None
    assert state.jobs is not None


def test_headless_server_answers_status(headless_server):
    url, token = headless_server
    assert call(url + '/v1/status', token)['protocol'] == 'flimkit-qupath'


def test_headless_server_serves_the_dataset_routes(headless_server):
    url, token = headless_server
    assert call(url + '/v1/datasets', token) == {'datasets': []}


def test_headless_server_rejects_a_bad_token(headless_server):
    url, _ = headless_server
    with pytest.raises(urllib.error.HTTPError) as raised:
        call(url + '/v1/datasets', 'wrong-token')
    assert raised.value.code == 401


def test_serve_writes_the_discovery_file(tmp_path, monkeypatch):
    target = tmp_path / 'qupath-bridge.json'
    monkeypatch.setattr(discovery, 'discovery_path', lambda: target)
    server, url, token = headless.serve(port=0, token='test-token')
    try:
        payload = discovery.read()
        assert payload['url'] == url
        assert payload['token'] == token
        assert payload['pid'] == os.getpid()
    finally:
        server.server_close()


def test_serve_can_skip_the_discovery_file(tmp_path, monkeypatch):
    target = tmp_path / 'qupath-bridge.json'
    monkeypatch.setattr(discovery, 'discovery_path', lambda: target)
    server, _, _ = headless.serve(port=0, announce=False)
    try:
        assert not target.exists()
    finally:
        server.server_close()


def test_main_refuses_to_hijack_a_running_bridge(monkeypatch, capsys):
    monkeypatch.setattr(headless, 'running_bridge',
                        lambda: {'url': 'http://127.0.0.1:8765', 'pid': 999999})
    assert headless.main([]) == 1
    assert 'already serving' in capsys.readouterr().out


@pytest.mark.skipif(not SYNTH or not Path(SYNTH).exists(),
                    reason='set FLIMKIT_TEST_PTU to a FLIM file to run this')
def test_headless_fits_a_roi_end_to_end(headless_server):
    url, token = headless_server
    meta = call(url + '/v1/datasets', token, {'path': SYNTH})
    ident = meta['id']
    rois = {
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'properties': {'name': 'annotation-1'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[2, 2], [13, 2], [13, 13], [2, 13], [2, 2]]],
            },
        }],
    }
    params = {'n_exp': 2, 'irf_strategy': 'gaussian', 'irf_fwhm_ns': 0.15}
    out = call(url + f'/v1/datasets/{ident}/fit/roi', token,
               {'rois': rois, 'params': params})
    result = out['results'][0]
    assert result['name'] == 'annotation-1'
    assert len(result['taus_ns']) == 2
    assert result['photon_count'] > 0
