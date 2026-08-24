import json
from io import BytesIO
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import tifffile

from flimkit_bridge.datasets import DatasetRegistry
from flimkit_bridge.server import BridgeState

from test_datasets import FakeReader


@pytest.fixture
def served(serve_state, tmp_path):
    readers = {}
    made = tmp_path / 'a.ptu'
    made.write_bytes(b'not really a ptu')

    def opener(path, channel):
        return readers.setdefault(path, FakeReader(path, n_y=8, n_x=8, n_bins=16))

    registry = DatasetRegistry(opener=opener)
    state = BridgeState(images={})
    state.datasets = registry
    return serve_state(state), registry, str(made)


def _call(url, path, method='GET', body=None, token='test-token'):
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    data = None
    if body is not None:
        data = json.dumps(body).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    request = Request(f'{url}{path}', data=data, method=method, headers=headers)
    with urlopen(request) as response:
        raw = response.read()
        return response.status, json.loads(raw) if raw else None, response.headers


def test_open_returns_metadata(served):
    url, _, path = served

    status, payload, _ = _call(url, '/v1/datasets', 'POST', {'path': path})

    assert status == 200
    assert payload['id'].startswith('ds_')
    assert payload['width'] == 8 and payload['height'] == 8
    assert payload['n_bins'] == 16
    assert payload['estimated_stack_bytes']['1'] == 8 * 8 * 16 * 4


def test_open_is_idempotent(served):
    url, registry, path = served

    _, first, _ = _call(url, '/v1/datasets', 'POST', {'path': path})
    _, second, _ = _call(url, '/v1/datasets', 'POST', {'path': path})

    assert first['id'] == second['id']
    assert registry.refcount(first['id']) == 2


def test_list_and_fetch_and_close(served):
    url, _, path = served
    _, opened, _ = _call(url, '/v1/datasets', 'POST', {'path': path})
    ident = opened['id']

    _, listed, _ = _call(url, '/v1/datasets')
    assert [d['id'] for d in listed['datasets']] == [ident]

    _, fetched, _ = _call(url, f'/v1/datasets/{ident}')
    assert fetched['id'] == ident

    _, closed, _ = _call(url, f'/v1/datasets/{ident}', 'DELETE')
    assert closed == {'closed': True}

    _, listed, _ = _call(url, '/v1/datasets')
    assert listed['datasets'] == []


def test_open_missing_file_is_404(served):
    url, _, _ = served
    with pytest.raises(HTTPError) as caught:
        _call(url, '/v1/datasets', 'POST', {'path': '/no/such/file.ptu'})
    assert caught.value.code == 404


def test_unknown_dataset_is_404(served):
    url, _, _ = served
    with pytest.raises(HTTPError) as caught:
        _call(url, '/v1/datasets/ds_999')
    assert caught.value.code == 404


def test_datasets_require_a_token(served):
    url, _, path = served
    with pytest.raises(HTTPError) as caught:
        _call(url, '/v1/datasets', 'POST', {'path': path}, token=None)
    assert caught.value.code == 401


def test_planes_lists_intensity(served):
    url, _, path = served
    _, opened, _ = _call(url, '/v1/datasets', 'POST', {'path': path})

    _, payload, _ = _call(url, f"/v1/datasets/{opened['id']}/planes")

    ids = [p['id'] for p in payload['planes']]
    assert 'intensity' in ids
    entry = next(p for p in payload['planes'] if p['id'] == 'intensity')
    assert entry['unit'] == 'photons'
    assert entry['shape'] == [8, 8]


def _plane(url, ident, name, query=''):
    request = Request(
        f'{url}/v1/datasets/{ident}/planes/{name}.tif{query}',
        headers={'Authorization': 'Bearer test-token'},
    )
    with urlopen(request) as response:
        return tifffile.imread(BytesIO(response.read())), response.headers


def test_intensity_plane_is_served_as_uint16(served):
    url, _, path = served
    _, opened, _ = _call(url, '/v1/datasets', 'POST', {'path': path})

    array, headers = _plane(url, opened['id'], 'intensity')

    assert array.shape == (8, 8)
    assert array.dtype == np.uint16
    assert headers['X-FLIMKit-Value-Unit'] == 'photons'
    assert headers['X-FLIMKit-Plane-Binning'] == '1'
    np.testing.assert_array_equal(array, np.full((8, 8), 16, dtype=np.uint16))


def test_plane_sub_rectangle(served):
    url, _, path = served
    _, opened, _ = _call(url, '/v1/datasets', 'POST', {'path': path})

    array, _ = _plane(url, opened['id'], 'intensity', '?x=2&y=3&w=4&h=2')

    assert array.shape == (2, 4)


def test_plane_downsample_halves_the_tile(served):
    url, _, path = served
    _, opened, _ = _call(url, '/v1/datasets', 'POST', {'path': path})

    array, _ = _plane(url, opened['id'], 'intensity', '?x=0&y=0&w=8&h=8&downsample=2')

    assert array.shape == (4, 4)


def test_plane_returns_exactly_the_tile_size_asked_for(served):
    url, _, path = served
    _, opened, _ = _call(url, '/v1/datasets', 'POST', {'path': path})

    array, _ = _plane(url, opened['id'], 'intensity',
                      '?x=0&y=0&w=8&h=8&downsample=2&ow=4&oh=4')

    assert array.shape == (4, 4)


def test_a_ragged_edge_tile_is_padded_to_the_tile_size(served):
    url, _, path = served
    _, opened, _ = _call(url, '/v1/datasets', 'POST', {'path': path})

    array, _ = _plane(url, opened['id'], 'intensity',
                      '?x=0&y=0&w=7&h=7&downsample=2&ow=4&oh=4')

    assert array.shape == (4, 4), 'QuPath needs every tile at the size it asked for'


def test_a_downsampled_tile_samples_the_full_resolution_one(served):
    url, _, path = served
    _, opened, _ = _call(url, '/v1/datasets', 'POST', {'path': path})

    full, _ = _plane(url, opened['id'], 'intensity', '?x=0&y=0&w=8&h=8')
    half, _ = _plane(url, opened['id'], 'intensity', '?x=0&y=0&w=8&h=8&downsample=2')

    assert (half == full[::2, ::2]).all()


def test_a_bad_downsample_is_refused(served):
    url, _, path = served
    _, opened, _ = _call(url, '/v1/datasets', 'POST', {'path': path})

    for query in ('?x=0&y=0&w=8&h=8&downsample=0', '?x=0&y=0&w=8&h=8&downsample=wide'):
        with pytest.raises(HTTPError) as caught:
            _plane(url, opened['id'], 'intensity', query)
        assert caught.value.code == 400


def test_plane_sub_rectangle_out_of_bounds_is_400(served):
    url, _, path = served
    _, opened, _ = _call(url, '/v1/datasets', 'POST', {'path': path})

    with pytest.raises(HTTPError) as caught:
        _plane(url, opened['id'], 'intensity', '?x=0&y=0&w=99&h=99')
    assert caught.value.code == 400


def test_unknown_plane_is_404(served):
    url, _, path = served
    _, opened, _ = _call(url, '/v1/datasets', 'POST', {'path': path})

    with pytest.raises(HTTPError) as caught:
        _plane(url, opened['id'], 'tau_mean_int')
    assert caught.value.code == 404


def test_binned_intensity_plane(served):
    url, _, path = served
    _, opened, _ = _call(url, '/v1/datasets', 'POST', {'path': path})

    array, headers = _plane(url, opened['id'], 'intensity', '?binning=2')

    assert array.shape == (4, 4)
    assert headers['X-FLIMKit-Plane-Binning'] == '2'


def test_a_plane_stack_comes_back_as_one_named_multichannel_image(tmp_path):
    import io
    import re

    import numpy as np
    import tifffile

    from flimkit_bridge import dataset_routes

    class _Registry:
        def metadata(self, ident):
            return {'height': 4, 'width': 4}

        def intensity(self, ident, binning=1):
            return np.full((4, 4), 7.0, dtype=np.float32)

        def plane(self, ident, name):
            return np.full((4, 4), 2.5, dtype=np.float32) if name == 'tau_1' else None

        def plane_unit(self, ident, name):
            return 'ns'

    state = type('S', (), {'datasets': _Registry()})()
    body, unit, binning, shape = dataset_routes.plane_tiff(
        state, 'ds_1', 'stack', 'planes=intensity,tau_1')
    assert shape == (2, 4, 4)
    with tifffile.TiffFile(io.BytesIO(body)) as handle:
        assert handle.series[0].dtype == np.float32
        names = re.findall(r'Channel[^>]*Name="([^"]+)"', handle.ome_metadata)
        stacked = handle.asarray()
    assert names == ['intensity (photons)', 'tau_1 (ns)']
    assert stacked[0].max() == 7.0 and stacked[1].max() == 2.5
    assert unit == 'intensity (photons),tau_1 (ns)'


def test_a_plane_stack_needs_planes_named():
    import numpy as np

    from flimkit_bridge import dataset_routes

    class _Registry:
        def metadata(self, ident):
            return {'height': 4, 'width': 4}

    state = type('S', (), {'datasets': _Registry()})()
    with pytest.raises(dataset_routes.RouteError) as caught:
        dataset_routes.plane_tiff(state, 'ds_1', 'stack', '')
    assert caught.value.status == 400


def test_plane_stats_refuses_before_a_fit_and_reports_after(tmp_path):
    import numpy as np

    from flimkit_bridge import dataset_routes

    tau = np.full((8, 8), np.nan, dtype=np.float32)
    tau[:4, :4] = 2.5

    class _Registry:
        def __init__(self, fitted):
            self._fitted = fitted

        def metadata(self, ident):
            return {'height': 8, 'width': 8}

        def plane_names(self, ident):
            return ['intensity'] + (['tau_mean_amp'] if self._fitted else [])

        def plane(self, ident, name):
            return tau if (self._fitted and name == 'tau_mean_amp') else None

        def plane_unit(self, ident, name):
            return 'ns'

    rois = {'type': 'FeatureCollection', 'features': [{
        'type': 'Feature', 'properties': {'name': 'Corner'},
        'geometry': {'type': 'Polygon', 'coordinates': [[
            [0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]]}}]}

    unfitted = type('S', (), {'datasets': _Registry(False)})()
    with pytest.raises(dataset_routes.RouteError) as caught:
        dataset_routes.plane_stats(unfitted, 'ds_1', {'rois': rois})
    assert caught.value.status == 409

    fitted = type('S', (), {'datasets': _Registry(True)})()
    got = dataset_routes.plane_stats(fitted, 'ds_1', {'rois': rois})
    region = got['regions'][0]
    assert region['name'] == 'Corner'
    stats = region['planes']['tau_mean_amp']
    assert stats['unit'] == 'ns'
    assert stats['mean'] == pytest.approx(2.5)
    assert stats['n'] == stats['n']


def test_plane_stats_refuses_a_mask_that_does_not_match_the_maps():
    import numpy as np

    from flimkit_bridge import dataset_routes

    class _Registry:
        def metadata(self, ident):
            return {'height': 8, 'width': 8}

        def plane_names(self, ident):
            return ['tau_mean_amp']

        def plane(self, ident, name):
            return np.zeros((4, 4), dtype=np.float32)

        def plane_unit(self, ident, name):
            return 'ns'

    rois = {'type': 'FeatureCollection', 'features': [{
        'type': 'Feature', 'properties': {'name': 'All'},
        'geometry': {'type': 'Polygon', 'coordinates': [[
            [0, 0], [8, 0], [8, 8], [0, 8], [0, 0]]]}}]}
    state = type('S', (), {'datasets': _Registry()})()
    with pytest.raises(dataset_routes.RouteError) as caught:
        dataset_routes.plane_stats(state, 'ds_1', {'rois': rois})
    assert caught.value.status == 409
    assert 'binning' in str(caught.value)
