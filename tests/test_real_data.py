import json
import os
from pathlib import Path

import numpy as np
import pytest

from flimkit_bridge.server import BridgeState

SESSION_PATH = os.environ.get('FLIMKIT_TEST_SESSION', '')
QUPATH_PATH = os.environ.get('QUPATH_PATH', '')

pytestmark = pytest.mark.skipif(
    not SESSION_PATH or not Path(SESSION_PATH).exists() or not QUPATH_PATH,
    reason='set FLIMKIT_TEST_SESSION and QUPATH_PATH to run the real-data tests',
)


@pytest.fixture(scope='module')
def session():
    return np.load(SESSION_PATH, allow_pickle=True)


@pytest.fixture(scope='module')
def intensity_image(session):
    return np.asarray(session['fov_intensity_map'], dtype=np.float32)


@pytest.fixture(scope='module')
def lifetime_map(session):
    return np.asarray(session['fov_lifetime_map'], dtype=np.float32)


@pytest.fixture(scope='module')
def exported_rois(session):
    roi_tools = pytest.importorskip('flimkit.UI.roi_tools')
    manager = roi_tools.RoiManager.from_json(str(session['fov_regions']))
    assert manager.regions, 'session carries no regions'
    return manager.to_geojson()


def test_real_intensity_survives_the_trip(serve_state, verify_image, intensity_image):
    state = BridgeState(
        images={'intensity': intensity_image},
        units={'intensity': 'photons'},
    )
    reported = verify_image(QUPATH_PATH, serve_state(state), 'intensity')

    height, width = intensity_image.shape
    assert int(reported['width']) == width
    assert int(reported['height']) == height
    assert reported['type'] == 'UINT16'
    assert reported['unit'] == 'photons'
    assert int(reported['nan']) == 0
    assert float(reported['max']) == pytest.approx(float(intensity_image.max()))
    assert float(reported['sum']) == pytest.approx(
        float(intensity_image.sum()), rel=1e-6)


def test_real_lifetime_map_survives_the_trip(serve_state, verify_image, lifetime_map):
    state = BridgeState(
        images={'lifetime': lifetime_map},
        units={'lifetime': 'ns'},
    )
    reported = verify_image(QUPATH_PATH, serve_state(state), 'lifetime')

    height, width = lifetime_map.shape
    finite = np.isfinite(lifetime_map)
    assert int(reported['width']) == width
    assert int(reported['height']) == height
    assert reported['type'] == 'FLOAT32'
    assert reported['unit'] == 'ns'
    assert int(reported['finite']) == int(finite.sum())
    assert int(reported['nan']) == int((~finite).sum())
    assert float(reported['max']) == pytest.approx(
        float(lifetime_map[finite].max()), rel=1e-6)
    assert float(reported['sum']) == pytest.approx(
        float(lifetime_map[finite].sum()), rel=1e-5)


def test_real_flimkit_rois_parse_in_qupath(serve_state, parse_rois, exported_rois):
    state = BridgeState(images={}, exported_rois=exported_rois)
    reported = parse_rois(QUPATH_PATH, serve_state(state))

    assert int(reported['objects']) == len(exported_rois['features'])
    assert int(reported['with_roi']) == len(exported_rois['features'])
    assert int(reported['empty']) == 0


def test_real_freehand_rings_are_simple(exported_rois):
    shapely = pytest.importorskip('shapely.geometry')
    invalid = [
        feature['properties']['name']
        for feature in exported_rois['features']
        if not shapely.Polygon(feature['geometry']['coordinates'][0]).is_valid
    ]
    assert not invalid, f'FLIMKit exported invalid rings: {invalid}'


def test_repaired_rings_are_flagged(exported_rois):
    repaired = [
        feature['properties']['name']
        for feature in exported_rois['features']
        if feature['properties'].get('repaired') == 'self-intersecting'
    ]
    assert repaired, 'this session no longer exercises the repair path'


def test_real_rois_are_closed_polygons(exported_rois):
    assert exported_rois['type'] == 'FeatureCollection'
    for feature in exported_rois['features']:
        geometry = feature['geometry']
        assert geometry['type'] == 'Polygon'
        ring = geometry['coordinates'][0]
        assert len(ring) >= 4
        assert ring[0] == ring[-1]
        assert feature['properties']['tool_type'] in (
            'rect', 'ellipse', 'polygon', 'freehand')
