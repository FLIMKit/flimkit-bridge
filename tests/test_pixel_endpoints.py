import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from flimkit_bridge.datasets import DatasetRegistry
from flimkit_bridge.jobs import JobRegistry
from flimkit_bridge.server import BridgeState

PTU_PATH = os.environ.get('FLIMKIT_TEST_PTU', '')

pytestmark = pytest.mark.skipif(
    not PTU_PATH or not Path(PTU_PATH).exists(),
    reason='set FLIMKIT_TEST_PTU to run the per-pixel endpoint tests',
)


@pytest.fixture
def served(serve_state):
    state = BridgeState(images={})
    state.datasets = DatasetRegistry()
    state.jobs = JobRegistry()
    url = serve_state(state)
    yield url, state
    state.jobs.shutdown()


def _call(url, path, method='GET', body=None):
    headers = {'Authorization': 'Bearer test-token'}
    data = None
    if body is not None:
        data = json.dumps(body).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    request = Request(f'{url}{path}', data=data, method=method, headers=headers)
    with urlopen(request) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def _await(url, job_id, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = _call(url, f'/v1/jobs/{job_id}')
        if status['state'] in ('done', 'error', 'cancelled'):
            return status
        time.sleep(0.5)
    raise AssertionError('the job never finished')


def test_irfs_are_listed(served):
    url, _ = served

    payload = _call(url, '/v1/irfs')

    assert 'machine_irf' in payload['strategies']
    assert 'session' in payload['strategies']
    if payload['machine_irfs']:
        assert payload['default'].endswith('.npy')


def test_a_whole_fov_fit_runs_and_produces_planes(served):
    url, _ = served
    opened = _call(url, '/v1/datasets', 'POST', {'path': PTU_PATH})

    from flimkit_bridge import irf as irf_module
    strategy = 'machine_irf' if irf_module.available() else 'gaussian'
    started = _call(url, f"/v1/datasets/{opened['id']}/fit/pixels", 'POST',
                    {'params': {'n_exp': 1, 'binning': 8, 'min_photons': 5,
                                'tau_min_ns': 0.2, 'tau_max_ns': 8.0,
                                'irf_strategy': strategy}})

    assert started['job'].startswith('job_')
    assert started['params_used']['irf_strategy'] == strategy
    assert started['estimated_stack_bytes'] > 0

    status = _await(url, started['job'])
    assert status['state'] == 'done', status.get('error')
    assert status['progress']['fraction'] == 1.0

    result = _call(url, f"/v1/jobs/{started['job']}?result")['result']
    assert 'tau_mean_int' in result['planes']
    assert result['binning'] == 8
    assert result['global']['n_exp'] == 1
    assert result['wrote_back'] is False

    planes = _call(url, f"/v1/datasets/{opened['id']}/planes")
    assert 'tau_mean_int' in [p['id'] for p in planes['planes']]


def test_a_running_job_can_be_cancelled(served):
    url, _ = served
    opened = _call(url, '/v1/datasets', 'POST', {'path': PTU_PATH})

    from flimkit_bridge import irf as irf_module
    started = _call(url, f"/v1/datasets/{opened['id']}/fit/pixels", 'POST',
                    {'params': {'n_exp': 2, 'binning': 2, 'min_photons': 5,
                                'irf_strategy': 'machine_irf'
                                if irf_module.available() else 'gaussian'}})
    _call(url, f"/v1/jobs/{started['job']}", 'DELETE')

    status = _await(url, started['job'])
    assert status['state'] in ('cancelled', 'done')


def test_an_unknown_job_is_404(served):
    url, _ = served
    with pytest.raises(HTTPError) as caught:
        _call(url, '/v1/jobs/job_999')
    assert caught.value.code == 404


def test_bad_pixel_parameters_are_rejected(served):
    url, _ = served
    opened = _call(url, '/v1/datasets', 'POST', {'path': PTU_PATH})

    with pytest.raises(HTTPError) as caught:
        _call(url, f"/v1/datasets/{opened['id']}/fit/pixels", 'POST',
              {'params': {'n_exp': 42}})
    assert caught.value.code == 400
