import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from flimkit_bridge.jobs import JobRegistry
from flimkit_bridge.server import BridgeState


@pytest.fixture
def served(serve_state):
    state = BridgeState(images={})
    state.jobs = JobRegistry()
    yield serve_state(state)
    state.jobs.shutdown()


def _call(url, path, method='GET', body=None):
    headers = {'Authorization': 'Bearer test-token'}
    data = None
    if body is not None:
        data = json.dumps(body).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    request = Request(f'{url}{path}', data=data, method=method, headers=headers)
    with urlopen(request) as response:
        return json.load(response)


def _slices(directory, base='RegionA', count=3):
    for z in range(1, count + 1):
        (directory / f'{base}_z{z}.ptu').write_bytes(b'')
    return directory


def test_the_defaults_are_served(served):
    payload = _call(served, '/v1/zstack/defaults')

    assert payload['values']['n_exp'] >= 1
    assert payload['values']['z_step_um'] > 0
    assert all(entry['applies_to'] == ['zstack'] for entry in payload['schema'])


def test_the_defaults_need_the_token(served):
    with pytest.raises(HTTPError) as raised:
        urlopen(Request(f'{served}/v1/zstack/defaults'))
    assert raised.value.code == 401


def test_a_folder_is_scanned_before_anything_runs(served, tmp_path):
    _slices(tmp_path)
    (tmp_path / 'RegionB_t1_s1_z1.ptu').write_bytes(b'')

    payload = _call(served, '/v1/zstack/scan', 'POST', {'ptu_dir': str(tmp_path)})

    assert payload['n_stacks'] == 2
    assert payload['n_slices'] == 4
    labels = {stack['label'] for stack in payload['stacks']}
    assert 'RegionA' in labels
    assert payload['volume_formats'] == ['ome-zarr', 'ome-tiff']


def test_scanning_an_empty_folder_finds_nothing(served, tmp_path):
    payload = _call(served, '/v1/zstack/scan', 'POST', {'ptu_dir': str(tmp_path)})

    assert payload['n_stacks'] == 0
    assert payload['stacks'] == []


def test_scanning_needs_a_folder(served):
    with pytest.raises(HTTPError) as raised:
        _call(served, '/v1/zstack/scan', 'POST', {})
    assert raised.value.code == 400


def test_scanning_a_missing_folder_is_a_404(served, tmp_path):
    with pytest.raises(HTTPError) as raised:
        _call(served, '/v1/zstack/scan', 'POST',
              {'ptu_dir': str(tmp_path / 'nowhere')})
    assert raised.value.code == 404


def test_a_folder_without_slices_will_not_start(served, tmp_path):
    with pytest.raises(HTTPError) as raised:
        _call(served, '/v1/zstack', 'POST', {'ptu_dir': str(tmp_path)})
    assert raised.value.code == 400


def test_bad_parameters_are_refused_before_the_job(served, tmp_path):
    _slices(tmp_path)
    with pytest.raises(HTTPError) as raised:
        _call(served, '/v1/zstack', 'POST',
              {'ptu_dir': str(tmp_path), 'params': {'tau_min_ns': 9.0}})
    assert raised.value.code == 400


def test_starting_a_run_describes_what_it_found(served, tmp_path, monkeypatch):
    from flimkit_bridge import zstack

    _slices(tmp_path, count=4)
    monkeypatch.setattr(zstack, 'build_args', lambda *a, **k: object())
    monkeypatch.setattr(zstack, 'run', lambda *a, **k: {})
    monkeypatch.setattr(zstack, 'summarise',
                        lambda *a, **k: {'products': [], 'stacks': []})

    started = _call(served, '/v1/zstack', 'POST',
                    {'ptu_dir': str(tmp_path), 'params': {'n_exp': 1}})

    assert started['n_stacks'] == 1
    assert started['n_slices'] == 4
    assert started['params_used']['n_exp'] == 1
    assert started['output_dir'].endswith('_flimkit_zstack')
    assert 'files' not in started['stacks'][0]
    assert started['job']


def test_a_chosen_output_folder_is_kept(served, tmp_path, monkeypatch):
    from flimkit_bridge import zstack

    _slices(tmp_path)
    monkeypatch.setattr(zstack, 'build_args', lambda *a, **k: object())
    monkeypatch.setattr(zstack, 'run', lambda *a, **k: {})
    monkeypatch.setattr(zstack, 'summarise', lambda *a, **k: {})

    started = _call(served, '/v1/zstack', 'POST',
                    {'ptu_dir': str(tmp_path),
                     'output_dir': str(tmp_path / 'elsewhere')})

    assert started['output_dir'] == str(tmp_path / 'elsewhere')


def test_a_finished_run_hands_back_its_products(served, tmp_path, monkeypatch):
    import time

    from flimkit_bridge import zstack

    _slices(tmp_path)
    monkeypatch.setattr(zstack, 'build_args', lambda *a, **k: object())
    monkeypatch.setattr(zstack, 'run', lambda *a, **k: {'RegionA': {}})
    monkeypatch.setattr(zstack, 'summarise', lambda *a, **k: {
        'products': [{'file': '/tmp/RegionA.ome.zarr', 'image_id': 'RegionA',
                      'format': 'ome-zarr', 'n_z': 3}],
        'stacks': [{'label': 'RegionA'}]})

    started = _call(served, '/v1/zstack', 'POST', {'ptu_dir': str(tmp_path)})
    deadline = time.time() + 10
    while time.time() < deadline:
        status = _call(served, f"/v1/jobs/{started['job']}")
        if status['state'] in ('done', 'error', 'cancelled'):
            break
        time.sleep(0.05)

    assert status['state'] == 'done', status
    result = _call(served, f"/v1/jobs/{started['job']}?result")['result']
    assert result['products'][0]['format'] == 'ome-zarr'
    assert result['products'][0]['n_z'] == 3


def test_a_bridge_without_a_job_registry_says_so(serve_state, tmp_path):
    _slices(tmp_path)
    url = serve_state(BridgeState(images={}))
    with pytest.raises(HTTPError) as raised:
        _call(url, '/v1/zstack', 'POST', {'ptu_dir': str(tmp_path)})
    assert raised.value.code == 503


def test_a_finished_stack_can_be_re_exported(served, tmp_path):
    import time

    import numpy as np

    group_dir = tmp_path / 'RegionA'
    for z in range(2):
        slice_dir = group_dir / f'z{z:04d}'
        slice_dir.mkdir(parents=True)
        for name in ('intensity', 'tau_mean_int'):
            np.save(str(slice_dir / f'{name}.npy'),
                    np.zeros((4, 4), dtype=np.float32))

    started = _call(served, '/v1/zstack/export', 'POST',
                    {'group_dir': str(group_dir), 'format': 'ome-tiff',
                     'z_step_um': 3.0})
    assert started['format'] == 'ome-tiff'
    deadline = time.time() + 10
    while time.time() < deadline:
        status = _call(served, f"/v1/jobs/{started['job']}")
        if status['state'] in ('done', 'error', 'cancelled'):
            break
        time.sleep(0.05)
    assert status['state'] == 'done', status
    product = _call(served, f"/v1/jobs/{started['job']}?result")['result']['products'][0]
    assert product['file'].endswith('RegionA.ome.tif')
    assert product['n_z'] == 2
    assert product['voxel_size_um'][0] == 3.0


def test_re_exporting_a_missing_folder_is_a_404(served, tmp_path):
    with pytest.raises(HTTPError) as raised:
        _call(served, '/v1/zstack/export', 'POST',
              {'group_dir': str(tmp_path / 'nowhere')})
    assert raised.value.code == 404


def test_an_unknown_export_format_is_refused(served, tmp_path):
    (tmp_path / 'RegionA').mkdir()
    with pytest.raises(HTTPError) as raised:
        _call(served, '/v1/zstack/export', 'POST',
              {'group_dir': str(tmp_path / 'RegionA'), 'format': 'nifti'})
    assert raised.value.code == 400


def _volume_slices(root, label='RegionA', n_z=2, shape=(4, 5)):
    import numpy as np

    group_dir = root / label
    for z in range(n_z):
        slice_dir = group_dir / f'z{z:04d}'
        slice_dir.mkdir(parents=True)
        for name in ('intensity', 'tau_mean_int'):
            np.save(str(slice_dir / f'{name}.npy'),
                    np.full(shape, float(z + 1), dtype=np.float32))
    return group_dir


def _fetch(url, path):
    request = Request(f'{url}{path}',
                      headers={'Authorization': 'Bearer test-token'})
    with urlopen(request) as response:
        return response.read(), dict(response.headers)


def test_a_volume_streams_to_a_client_that_cannot_see_the_disk(served, tmp_path):
    import io

    import numpy as np
    import tifffile

    group_dir = _volume_slices(tmp_path)

    body, headers = _fetch(served, '/v1/zstack/volume.ome.tif?group_dir='
                           + str(group_dir) + '&z_step_um=2.5')

    assert headers['Content-Type'] == 'image/tiff'
    assert int(headers['Content-Length']) == len(body)
    assert headers['X-FLIMKit-Volume-Label'] == 'RegionA'
    assert headers['X-FLIMKit-Volume-Axes'] == 'ZCYX'
    assert headers['X-FLIMKit-Volume-Channels'] == 'intensity,tau_mean_int'
    assert headers['X-FLIMKit-Volume-Units'] == 'photons,ns'
    assert headers['X-FLIMKit-Volume-Shape'] == '2,2,4,5'
    assert headers['X-FLIMKit-Voxel-Size-Um'].startswith('2.5,')
    read_back = tifffile.imread(io.BytesIO(body))
    assert read_back.shape == (2, 2, 4, 5)
    assert np.allclose(read_back[1, 0], 2.0)


def test_streaming_leaves_no_temporary_file_behind(served, tmp_path):
    import tempfile
    import time
    from pathlib import Path as _Path

    group_dir = _volume_slices(tmp_path)
    holding = _Path(tempfile.gettempdir())
    before = set(holding.glob('flimkit-zstack-*'))

    _fetch(served, '/v1/zstack/volume.ome.tif?group_dir=' + str(group_dir))

    # The server drops the file once the body is out, which can land just
    # after the client has read the last of it, so this waits rather than
    # looking exactly once.
    deadline = time.time() + 5
    while time.time() < deadline:
        left = set(holding.glob('flimkit-zstack-*')) - before
        if not left:
            break
        time.sleep(0.05)
    assert set(holding.glob('flimkit-zstack-*')) == before


def test_streaming_needs_a_folder(served):
    with pytest.raises(HTTPError) as raised:
        _fetch(served, '/v1/zstack/volume.ome.tif')
    assert raised.value.code == 400


def test_streaming_a_missing_folder_is_a_404(served, tmp_path):
    with pytest.raises(HTTPError) as raised:
        _fetch(served, '/v1/zstack/volume.ome.tif?group_dir='
               + str(tmp_path / 'nowhere'))
    assert raised.value.code == 404


def test_streaming_a_folder_with_no_maps_is_a_409(served, tmp_path):
    (tmp_path / 'RegionA').mkdir()
    with pytest.raises(HTTPError) as raised:
        _fetch(served, '/v1/zstack/volume.ome.tif?group_dir='
               + str(tmp_path / 'RegionA'))
    assert raised.value.code == 409


def test_streaming_needs_the_token(served, tmp_path):
    group_dir = _volume_slices(tmp_path)
    with pytest.raises(HTTPError) as raised:
        urlopen(Request(f'{served}/v1/zstack/volume.ome.tif?group_dir='
                        + str(group_dir)))
    assert raised.value.code == 401
