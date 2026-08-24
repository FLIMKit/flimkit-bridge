import queue
import threading
import time
import traceback
from collections import OrderedDict


class Job:

    def __init__(self, job_id, kind, cancellable):
        self.id = job_id
        self.kind = kind
        self.cancellable = cancellable
        self.state = 'queued'
        self.current = 0
        self.total = 0
        self.message = ''
        self.history = []
        self.started = time.time()
        self.finished = None
        self.result = None
        self.error = None
        self.cancel = threading.Event()

    def fraction(self):
        if self.state == 'done':
            return 1.0
        if not self.total:
            return 0.0
        return min(1.0, self.current / self.total)

    def snapshot(self):
        found = {
            'id': self.id,
            'kind': self.kind,
            'state': self.state,
            'cancellable': self.cancellable,
            'progress': {
                'current': self.current,
                'total': self.total,
                'fraction': self.fraction(),
            },
            'message': self.message,
            'elapsed_s': round((self.finished or time.time()) - self.started, 3),
            'history': list(self.history),
        }
        if self.error is not None:
            found['error'] = self.error
        return found


class JobRegistry:

    def __init__(self, history=50):
        self._lock = threading.RLock()
        self._jobs = OrderedDict()
        self._history = history
        self._queue = queue.Queue()
        self._stopping = threading.Event()
        self._worker = threading.Thread(
            target=self._serve, name='flimkit-bridge-job', daemon=True)
        self._worker.start()
        self._next = 0

    def submit(self, kind, work, cancellable=True):
        with self._lock:
            self._next += 1
            job_id = f'job_{self._next}'
            job = Job(job_id, kind, cancellable)
            self._jobs[job_id] = job
            self._trim()
        self._queue.put((job, work))
        return job_id

    def _serve(self):
        while not self._stopping.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            job, work = item
            try:
                self._run(job, work)
            finally:
                self._queue.task_done()

    def _trim(self):
        while len(self._jobs) > self._history:
            for job_id, job in list(self._jobs.items()):
                if job.state in ('done', 'error', 'cancelled'):
                    del self._jobs[job_id]
                    break
            else:
                break

    def _run(self, job, work):
        with self._lock:
            if job.cancel.is_set():
                job.state = 'cancelled'
                job.finished = time.time()
                return
            job.state = 'running'

        def progress(current, total, message=''):
            with self._lock:
                job.current = int(current)
                job.total = int(total)
                if message:
                    job.message = message
                job.history.append(job.fraction())

        try:
            result = work(progress, job.cancel)
        except Exception as exc:
            with self._lock:
                job.state = 'error'
                job.error = {
                    'type': type(exc).__name__,
                    'message': str(exc),
                    'traceback': traceback.format_exc(),
                }
                job.finished = time.time()
            return
        with self._lock:
            job.finished = time.time()
            if job.cancel.is_set():
                job.state = 'cancelled'
            else:
                job.state = 'done'
                job.result = result

    def _job(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f'no such job: {job_id}')
        return job

    def status(self, job_id):
        return self._job(job_id).snapshot()

    def result(self, job_id):
        job = self._job(job_id)
        if job.state == 'error':
            raise ValueError(job.error['message'])
        if job.state != 'done':
            raise ValueError(f'job {job_id} is {job.state}')
        return job.result

    def cancel(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not job.cancellable:
                return False
            job.cancel.set()
            if job.state == 'queued':
                job.state = 'cancelled'
                job.finished = time.time()
            return True

    def list(self):
        with self._lock:
            return [job.snapshot() for job in self._jobs.values()]

    def shutdown(self):
        self._stopping.set()
        with self._lock:
            for job in self._jobs.values():
                if job.state in ('queued', 'running'):
                    job.cancel.set()
