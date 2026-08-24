import json
import os
import numpy as np
import pytest

from flimkit_bridge import discovery
from flimkit_bridge.server import BridgeState, create_server


def test_write_then_read_round_trips(tmp_path):
    target = tmp_path / 'qupath-bridge.json'
    discovery.write('http://127.0.0.1:9001', 'secret-token', path=target)

    payload = discovery.read(target)

    assert payload['url'] == 'http://127.0.0.1:9001'
    assert payload['token'] == 'secret-token'
    assert payload['pid'] == os.getpid()
    assert payload['protocol'] == 'flimkit-qupath'


@pytest.mark.skipif(os.name == 'nt', reason='POSIX permissions only')
def test_file_is_owner_readable_only(tmp_path):
    target = tmp_path / 'qupath-bridge.json'
    discovery.write('http://127.0.0.1:9001', 'secret-token', path=target)

    assert oct(target.stat().st_mode)[-3:] == '600'


def test_read_rejects_a_foreign_file(tmp_path):
    target = tmp_path / 'qupath-bridge.json'
    target.write_text(json.dumps({'protocol': 'something-else', 'url': 'x', 'token': 'y'}))

    assert discovery.read(target) is None


def test_read_of_a_missing_file_is_none(tmp_path):
    assert discovery.read(tmp_path / 'absent.json') is None


def test_read_live_rejects_a_dead_pid(tmp_path):
    target = tmp_path / 'qupath-bridge.json'
    discovery.write('http://127.0.0.1:9001', 'tok', pid=999999, path=target)

    assert discovery.read(target) is not None
    assert discovery.read_live(target) is None


def test_read_live_accepts_this_process(tmp_path):
    target = tmp_path / 'qupath-bridge.json'
    discovery.write('http://127.0.0.1:9001', 'tok', path=target)

    assert discovery.read_live(target) is not None


def test_remove_is_forgiving(tmp_path):
    target = tmp_path / 'qupath-bridge.json'
    discovery.write('http://127.0.0.1:9001', 'tok', path=target)
    discovery.remove(target)
    discovery.remove(target)

    assert not target.exists()


@pytest.fixture
def running(serve_state):
    state = BridgeState(
        images={'intensity': np.zeros((2, 2), dtype=np.float32)},
        units={'intensity': 'photons'},
    )
    return serve_state(state), state


def _status_with_host(base_url, host_header):
    import http.client
    address = base_url.removeprefix('http://')
    connection = http.client.HTTPConnection(address, timeout=10)
    try:
        connection.request('GET', '/v1/status', headers={'Host': host_header})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_foreign_host_header_is_refused(running):
    base_url, _ = running

    status, _ = _status_with_host(base_url, 'evil.example.com')

    assert status == 403


def test_localhost_host_header_is_allowed(running):
    base_url, _ = running
    port = base_url.rsplit(':', 1)[-1]

    status, body = _status_with_host(base_url, f'localhost:{port}')

    assert status == 200
    assert json.loads(body)['protocol'] == 'flimkit-qupath'


def test_port_fallback_when_busy():
    first_state = BridgeState(images={})
    first = create_server('127.0.0.1', 0, 'tok', first_state)
    try:
        busy_port = first.server_address[1]
        second_state = BridgeState(images={})
        with pytest.raises(OSError):
            create_server('127.0.0.1', busy_port, 'tok', second_state)
    finally:
        first.server_close()


def test_status_reports_both_versions():
    from flimkit_bridge.version import report

    found = report()
    assert found['protocol'] == 'flimkit-qupath'
    assert found['protocol_version'] == 1
    assert found['bridge_version'] not in (None, '')
    assert found['flimkit_version'] not in (None, '')
