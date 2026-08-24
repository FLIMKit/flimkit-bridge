import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from flimkit_bridge.server import BridgeState

PTU_PATH = os.environ.get('FLIMKIT_TEST_PTU', '')

pytestmark = pytest.mark.skipif(
    not PTU_PATH or not Path(PTU_PATH).exists(),
    reason='set FLIMKIT_TEST_PTU to a .ptu file to run the format tests',
)


@pytest.fixture
def base_url(serve_state):
    return serve_state(BridgeState(images={}))


def _get(url, path=''):
    request = Request(f'{url}{path}', headers={'Authorization': 'Bearer test-token'})
    with urlopen(request) as response:
        return json.load(response)


def _identify(url, path):
    request = Request(
        f'{url}/v1/identify',
        data=json.dumps({'path': path}).encode('utf-8'),
        method='POST',
        headers={
            'Authorization': 'Bearer test-token',
            'Content-Type': 'application/json',
        },
    )
    with urlopen(request) as response:
        return json.load(response)


def test_formats_lists_extensions_and_labels(base_url):
    payload = _get(base_url, '/v1/formats')

    assert '.ptu' in payload['extensions']
    assert '.sdt' in payload['extensions']
    ids = {entry['id'] for entry in payload['formats']}
    assert 'ptu' in ids
    entry = next(e for e in payload['formats'] if e['id'] == 'ptu')
    assert entry['exts'] == ['.ptu']
    assert entry['modality'] == 'time'
    assert entry['label']


def test_formats_requires_a_token(base_url):
    with pytest.raises(HTTPError) as caught:
        urlopen(f'{base_url}/v1/formats')
    assert caught.value.code == 401


def test_identify_recognises_a_real_ptu(base_url):
    payload = _identify(base_url, PTU_PATH)

    assert payload['recognised'] is True
    assert payload['format'] == 'ptu'
    assert payload['modality'] == 'time'
    assert payload['ambiguous'] is False


def test_identify_refuses_a_plain_tiff(base_url, tmp_path):
    import numpy as np
    import tifffile

    target = tmp_path / 'ordinary.tiff'
    tifffile.imwrite(target, np.zeros((4, 4), dtype=np.uint16))

    payload = _identify(base_url, str(target))

    assert payload['recognised'] is False
    assert payload['format'] == 'unknown'
    assert payload['ambiguous'] is True


def test_identify_missing_file_is_404(base_url, tmp_path):
    with pytest.raises(HTTPError) as caught:
        _identify(base_url, str(tmp_path / 'absent.ptu'))
    assert caught.value.code == 404


def test_identify_a_directory_is_400(base_url, tmp_path):
    with pytest.raises(HTTPError) as caught:
        _identify(base_url, str(tmp_path))
    assert caught.value.code == 400


def test_identify_requires_a_path(base_url):
    request = Request(
        f'{base_url}/v1/identify',
        data=json.dumps({}).encode('utf-8'),
        method='POST',
        headers={'Authorization': 'Bearer test-token'},
    )
    with pytest.raises(HTTPError) as caught:
        urlopen(request)
    assert caught.value.code == 400


def test_identify_requires_a_token(base_url):
    request = Request(
        f'{base_url}/v1/identify',
        data=json.dumps({'path': PTU_PATH}).encode('utf-8'),
        method='POST',
    )
    with pytest.raises(HTTPError) as caught:
        urlopen(request)
    assert caught.value.code == 401
