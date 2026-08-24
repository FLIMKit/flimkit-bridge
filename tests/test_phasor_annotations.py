import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

QUPATH_PATH = os.environ.get('QUPATH_PATH', '')
EXTENSION_JARS = sorted(
    (Path.home() / 'QuPath' / 'v0.7' / 'extensions').glob(
        'qupath-extension-flimkit-bridge-*.jar'))

pytestmark = pytest.mark.skipif(
    not QUPATH_PATH or not EXTENSION_JARS,
    reason='needs QUPATH_PATH and the extension installed',
)


def _trace(labels, binning, tmp_path):
    target = tmp_path / 'labels.bin'
    target.write_bytes(labels.tobytes())
    script = Path(__file__).parents[1] / 'qupath' / 'TracePhasorMask.groovy'
    completed = subprocess.run(
        [QUPATH_PATH, 'script', str(script), '--args', str(target),
         '--args', str(labels.shape[1]), '--args', str(labels.shape[0]),
         '--args', str(binning)],
        capture_output=True, text=True, timeout=600)
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert 'TRACE_DONE' in output, output
    rois = []
    count = None
    for line in output.splitlines():
        if line.startswith('ROI '):
            rois.append(dict(f.split('=', 1)
                             for f in line.split('ROI ', 1)[1].split('|') if '=' in f))
        elif line.startswith('TRACED '):
            count = int(line.split('count=')[1])
    return count, rois


def test_a_single_block_becomes_one_annotation(tmp_path):
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[8:24, 8:24] = 1

    count, rois = _trace(labels, binning=1, tmp_path=tmp_path)

    assert count == 1
    roi = rois[0]
    assert roi['name'] == 'Population A'
    assert roi['class'] == 'Phasor'
    assert float(roi['area']) == pytest.approx(16 * 16, rel=0.05)
    assert float(roi['x']) == pytest.approx(8, abs=1)
    assert float(roi['w']) == pytest.approx(16, abs=1)


def test_binning_scales_the_annotation_into_image_coordinates(tmp_path):
    """The label image is at phasor resolution; tracing with downsample=binning
    is what maps it back onto the full-resolution image."""
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[8:24, 8:24] = 1

    _, unbinned = _trace(labels, binning=1, tmp_path=tmp_path)
    _, binned = _trace(labels, binning=4, tmp_path=tmp_path)

    assert float(binned[0]['w']) == pytest.approx(
        float(unbinned[0]['w']) * 4, rel=0.05)
    assert float(binned[0]['x']) == pytest.approx(
        float(unbinned[0]['x']) * 4, abs=2)
    assert float(binned[0]['area']) == pytest.approx(
        float(unbinned[0]['area']) * 16, rel=0.05)


def test_two_labels_become_two_named_annotations(tmp_path):
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[0:8, 0:8] = 1
    labels[16:28, 16:28] = 2

    count, rois = _trace(labels, binning=1, tmp_path=tmp_path)

    assert count == 2
    assert [r['name'] for r in rois] == ['Population A', 'Population B']
    assert float(rois[1]['area']) > float(rois[0]['area'])


def test_an_empty_mask_yields_nothing(tmp_path):
    count, rois = _trace(np.zeros((16, 16), dtype=np.uint8), 1, tmp_path)

    assert count == 0
    assert rois == []
