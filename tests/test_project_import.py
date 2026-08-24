import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from flimkit_bridge.server import BridgeState

QUPATH_PATH = os.environ.get('QUPATH_PATH', '')
EXTENSION_JARS = sorted(
    (Path.home() / 'QuPath' / 'v0.7' / 'extensions').glob(
        'qupath-extension-flimkit-bridge-*.jar'))

pytestmark = pytest.mark.skipif(
    not QUPATH_PATH or not EXTENSION_JARS,
    reason='needs QUPATH_PATH and the extension installed in QuPath',
)


def test_flimkit_images_become_project_entries(serve_state, tmp_path):
    state = BridgeState(
        images={
            'intensity': np.arange(35, dtype=np.float32).reshape(5, 7),
            'lifetime': np.arange(35, dtype=np.float32).reshape(5, 7) / 10.0,
        },
        units={'intensity': 'photons', 'lifetime': 'ns'},
    )
    base_url = serve_state(state)
    script = Path(__file__).parents[1] / 'qupath' / 'AddToProject.groovy'
    assert script.exists(), f'missing QuPath script: {script}'
    project_dir = tmp_path / 'project'

    completed = subprocess.run(
        [
            QUPATH_PATH,
            'script',
            str(script),
            '--args', base_url,
            '--args', 'test-token',
            '--args', str(project_dir),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert 'PROJECT_OK entries=2 opened=2' in output, output
    assert 'FLIMKit intensity (photons)' in output, output
    assert 'FLIMKit lifetime (ns)' in output, output
    assert 'width=7 height=5 type=FLOAT32' in output, output
    assert (project_dir / 'flimkit' / 'intensity.tif').exists()
    assert (project_dir / 'flimkit' / 'lifetime.tif').exists()
