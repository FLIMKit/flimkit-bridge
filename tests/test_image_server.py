import json
import os
import subprocess
from pathlib import Path

import pytest

from flimkit_bridge import discovery
from flimkit_bridge.datasets import DatasetRegistry
from flimkit_bridge.server import BridgeState

QUPATH_PATH = os.environ.get('QUPATH_PATH', '')
PTU_PATH = os.environ.get('FLIMKIT_TEST_PTU', '')
SDT_PATH = os.environ.get('FLIMKIT_TEST_SDT', '')
EXTENSION_JARS = sorted(
    (Path.home() / 'QuPath' / 'v0.7' / 'extensions').glob(
        'qupath-extension-flimkit-bridge-*.jar'))

pytestmark = pytest.mark.skipif(
    not QUPATH_PATH or not EXTENSION_JARS or not PTU_PATH or not SDT_PATH
    or not Path(PTU_PATH).exists() or not Path(SDT_PATH).exists(),
    reason='needs QUPATH_PATH, the extension installed, and both test files',
)


@pytest.fixture
def live_bridge(serve_state):
    """Publishes at the real discovery path, because the QuPath side resolves it
    from user.home in the JVM and cannot be redirected from Python."""
    state = BridgeState(images={})
    state.datasets = DatasetRegistry()
    url = serve_state(state)
    target = discovery.discovery_path()
    saved = target.read_bytes() if target.exists() else None
    discovery.write(url, 'test-token')
    try:
        yield url, target
    finally:
        if saved is None:
            discovery.remove()
        else:
            target.write_bytes(saved)


def _open_in_qupath(paths):
    script = Path(__file__).parents[1] / 'qupath' / 'OpenViaBuilder.groovy'
    assert script.exists(), f'missing script: {script}'
    completed = subprocess.run(
        [QUPATH_PATH, 'script', str(script), '--args', ';'.join(paths)],
        capture_output=True, text=True, timeout=900,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert 'OPEN_DONE' in output, output
    parsed = {'support': {}, 'opened': {}, 'tile': {}}
    for line in output.splitlines():
        for marker, key in (('SUPPORT ', 'support'), ('OPENED ', 'opened'),
                            ('TILE ', 'tile')):
            if line.startswith(marker):
                fields = dict(f.split('=', 1)
                              for f in line.split(marker, 1)[1].split(' ') if '=' in f)
                parsed[key][fields['name']] = fields
    return parsed, output


def test_qupath_opens_ptu_and_sdt_through_the_bridge(live_bridge, tmp_path):
    import numpy as np
    import tifffile

    decoy = tmp_path / 'ordinary.tiff'
    tifffile.imwrite(decoy, np.zeros((8, 8), dtype=np.uint16))

    parsed, output = _open_in_qupath([PTU_PATH, SDT_PATH, str(decoy)])

    ptu = Path(PTU_PATH).name
    sdt = Path(SDT_PATH).name

    assert parsed['support'][ptu]['level'] == '4.0'
    assert parsed['support'][sdt]['level'] == '4.0'
    assert parsed['support']['ordinary.tiff']['level'] == 'none'

    assert parsed['opened'][ptu]['type'] == 'UINT16'
    assert parsed['opened'][ptu]['channels'] == '1'
    assert int(parsed['opened'][ptu]['width']) > 0
    assert parsed['opened'][sdt]['type'] == 'UINT16'

    assert int(parsed['tile'][ptu]['max']) > 0, 'PTU intensity came back empty'
    assert int(parsed['tile'][sdt]['max']) > 0, 'SDT intensity came back empty'
