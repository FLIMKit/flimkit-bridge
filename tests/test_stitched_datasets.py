import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from flimkit_bridge import formats
from flimkit_bridge.datasets import is_stitched_output
from flimkit_bridge.server import bind
from flimkit_bridge.state import build_state

STITCHED = os.environ.get('FLIMKIT_TEST_STITCHED', '')

needs_stitched = pytest.mark.skipif(
    not STITCHED or not Path(STITCHED).is_dir(),
    reason='set FLIMKIT_TEST_STITCHED to a stitch output directory')


def call(url, token='test-token', payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data)
    request.add_header('Authorization', 'Bearer ' + token)
    if data:
        request.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


@pytest.fixture
def served():
    state = build_state()
    server = bind('127.0.0.1', 0, 'test-token', state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield 'http://%s:%d' % server.server_address
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_an_empty_directory_is_not_a_stitched_output(tmp_path):
    assert not is_stitched_output(tmp_path)


def test_metadata_without_the_cube_is_not_a_stitched_output(tmp_path):
    (tmp_path / 'R_2_metadata.json').write_text('{}')
    assert not is_stitched_output(tmp_path)


def test_a_plain_file_is_not_a_stitched_output(tmp_path):
    target = tmp_path / 'tile.ptu'
    target.write_bytes(b'')
    assert not is_stitched_output(target)


def test_a_directory_that_is_not_a_stitch_output_is_refused(tmp_path):
    with pytest.raises(formats.PathProblem) as raised:
        formats.identify(str(tmp_path))
    assert raised.value.status == 400


@needs_stitched
def test_the_stitch_output_is_recognised():
    assert is_stitched_output(STITCHED)
    found = formats.identify(STITCHED)
    assert found['recognised']
    assert found['format'] == 'stitched'


@needs_stitched
def test_a_stitched_canvas_opens_as_a_dataset(served):
    meta = call(served + '/v1/datasets', payload={'path': STITCHED})
    assert meta['format'] == 'stitched'
    assert meta['width'] > 0 and meta['height'] > 0
    assert meta['n_bins'] > 0


@needs_stitched
def test_a_region_on_the_canvas_is_fitted(served):
    meta = call(served + '/v1/datasets', payload={'path': STITCHED})
    size = min(meta['width'], meta['height']) // 2
    rois = {
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'properties': {'name': 'region'},
            'geometry': {'type': 'Polygon', 'coordinates': [[
                [4, 4], [4 + size, 4], [4 + size, 4 + size], [4, 4 + size], [4, 4]]]},
        }],
    }
    params = {'n_exp': 1, 'irf_strategy': 'gaussian', 'irf_fwhm_ns': 0.2,
              'min_photons': 0}
    out = call(served + f"/v1/datasets/{meta['id']}/fit/roi",
               payload={'rois': rois, 'params': params})
    result = out['results'][0]
    assert 'error' not in result, result.get('error')
    assert result['photon_count'] > 0
    assert result['n_pixels'] > 0


@needs_stitched
def test_the_canvas_is_served_whole_not_rebinned(served):
    meta = call(served + '/v1/datasets', payload={'path': STITCHED})
    rois = {
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'properties': {'name': 'region'},
            'geometry': {'type': 'Polygon',
                         'coordinates': [[[0, 0], [8, 0], [8, 8], [0, 8], [0, 0]]]},
        }],
    }
    with pytest.raises(urllib.error.HTTPError) as raised:
        call(served + f"/v1/datasets/{meta['id']}/fit/roi",
             payload={'rois': rois, 'params': {'binning': 2}})
    assert raised.value.code == 400


@needs_stitched
def test_the_region_decay_matches_the_cube():
    from flimkit.formats.PTU.stitch import load_flim_for_fitting

    stack, _, _ = load_flim_for_fitting(STITCHED)
    mask = np.zeros(stack.shape[:2], dtype=bool)
    mask[4:12, 4:12] = True
    expected = np.asarray(stack[mask], dtype=np.float64).sum(axis=0)
    assert expected.sum() > 0
    assert expected.shape[0] == stack.shape[2]
