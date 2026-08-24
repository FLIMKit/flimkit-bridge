import os
import subprocess
import threading
from pathlib import Path

import pytest

from flimkit_bridge.server import create_server


@pytest.fixture
def serve_state():
    servers = []

    def start(state):
        server = create_server('127.0.0.1', 0, 'test-token', state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        host, port = server.server_address
        return f'http://{host}:{port}'

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _run_script(qupath_path, script_name, marker, extra_args):
    script = Path(__file__).parents[1] / 'qupath' / script_name
    assert script.exists(), f'missing QuPath script: {script}'
    command = [qupath_path, 'script', str(script)]
    for value in extra_args:
        command += ['--args', value]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    rows = [row for row in output.splitlines() if marker in row]
    assert rows, output
    fields = rows[0].split(marker + ' ')[1].split(' ')
    return dict(field.split('=', 1) for field in fields)


@pytest.fixture
def parse_rois():
    def run(qupath_path, base_url):
        return _run_script(
            qupath_path, 'ParseRois.groovy', 'QUPATH_PARSE_OK',
            [base_url, 'test-token'])

    return run


@pytest.fixture
def verify_image():
    def run(qupath_path, base_url, image_id):
        script = Path(__file__).parents[1] / 'qupath' / 'VerifyImage.groovy'
        assert script.exists(), f'missing QuPath script: {script}'
        completed = subprocess.run(
            [
                qupath_path,
                'script',
                str(script),
                '--args', base_url,
                '--args', 'test-token',
                '--args', image_id,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = completed.stdout + completed.stderr
        assert completed.returncode == 0, output
        rows = [row for row in output.splitlines() if 'QUPATH_REAL_OK' in row]
        assert rows, output
        fields = rows[0].split('QUPATH_REAL_OK ')[1].split(' ')
        return dict(field.split('=', 1) for field in fields)

    return run
