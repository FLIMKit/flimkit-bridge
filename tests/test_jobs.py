import threading
import time

import pytest

from flimkit_bridge.jobs import JobRegistry


@pytest.fixture
def jobs():
    registry = JobRegistry(history=5)
    yield registry
    registry.shutdown()


def _wait(registry, job_id, state, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = registry.status(job_id)
        if found['state'] == state:
            return found
        time.sleep(0.02)
    raise AssertionError(
        f"job {job_id} stayed in {registry.status(job_id)['state']}, wanted {state}")


def test_a_job_runs_and_reports_its_result(jobs):
    job_id = jobs.submit('demo', lambda progress, cancel: {'answer': 42})

    status = _wait(jobs, job_id, 'done')

    assert status['progress']['fraction'] == 1.0
    assert jobs.result(job_id) == {'answer': 42}


def test_progress_is_reported_and_monotonic(jobs):
    seen = []

    def work(progress, cancel):
        for i in range(5):
            progress(i + 1, 5, f'step {i + 1}')
        return 'ok'

    job_id = jobs.submit('demo', work)
    _wait(jobs, job_id, 'done')

    assert jobs.status(job_id)['progress']['fraction'] == 1.0
    fractions = [f for f in jobs.status(job_id).get('history', [])]
    assert fractions == sorted(fractions)


def test_a_raising_job_reports_the_error(jobs):
    def work(progress, cancel):
        raise ValueError('the stack is too large')

    job_id = jobs.submit('demo', work)
    status = _wait(jobs, job_id, 'error')

    assert status['error']['type'] == 'ValueError'
    assert 'too large' in status['error']['message']
    with pytest.raises(ValueError):
        jobs.result(job_id)


def test_a_job_can_be_cancelled(jobs):
    started = threading.Event()

    def work(progress, cancel):
        started.set()
        for i in range(1000):
            if cancel.is_set():
                return None
            progress(i, 1000)
            time.sleep(0.005)
        return 'finished'

    job_id = jobs.submit('demo', work)
    assert started.wait(5)
    jobs.cancel(job_id)

    status = _wait(jobs, job_id, 'cancelled')
    assert status['state'] == 'cancelled'


def test_cancelling_an_unknown_job_is_false(jobs):
    assert jobs.cancel('nope') is False


def test_unknown_job_status_raises(jobs):
    with pytest.raises(KeyError):
        jobs.status('nope')


def test_history_is_bounded(jobs):
    for _ in range(12):
        job_id = jobs.submit('demo', lambda progress, cancel: 1)
        _wait(jobs, job_id, 'done')

    assert len(jobs.list()) <= 5


def test_jobs_run_one_at_a_time(jobs):
    running = []
    peak = []

    def work(progress, cancel):
        running.append(1)
        peak.append(len(running))
        time.sleep(0.05)
        running.pop()
        return None

    ids = [jobs.submit('demo', work) for _ in range(4)]
    for job_id in ids:
        _wait(jobs, job_id, 'done')

    assert max(peak) == 1


def test_cancellable_is_reported(jobs):
    job_id = jobs.submit('demo', lambda progress, cancel: 1, cancellable=False)
    _wait(jobs, job_id, 'done')

    assert jobs.status(job_id)['cancellable'] is False
