from io import BytesIO
from urllib.request import Request, urlopen

import numpy as np
import pytest
import tifffile

from flimkit_bridge.server import BridgeState, encode_image


def test_integral_intensity_becomes_uint16():
    counts = np.array([[0.0, 166.0], [3.0, 9.0]], dtype=np.float64)

    encoded = encode_image('intensity', counts)

    assert encoded.dtype == np.uint16
    np.testing.assert_array_equal(encoded, counts.astype(np.uint16))


def test_lifetime_stays_float32():
    assert encode_image('lifetime', np.array([[1.5, 2.5]])).dtype == np.float32


@pytest.mark.parametrize('counts', [
    np.array([[1.5, 2.0]]),
    np.array([[70000.0, 1.0]]),
    np.array([[-1.0, 1.0]]),
    np.array([[np.nan, 1.0]]),
])
def test_intensity_falls_back_to_float32(counts):
    assert encode_image('intensity', counts).dtype == np.float32


def test_uint16_intensity_round_trips_over_http(serve_state):
    counts = np.arange(35, dtype=np.float64).reshape(5, 7)
    state = BridgeState(
        images={'intensity': counts},
        units={'intensity': 'photons'},
    )
    base_url = serve_state(state)

    request = Request(
        f'{base_url}/v1/images/intensity.tif',
        headers={'Authorization': 'Bearer test-token'},
    )
    with urlopen(request) as response:
        received = tifffile.imread(BytesIO(response.read()))

    assert received.dtype == np.uint16
    np.testing.assert_array_equal(received, counts.astype(np.uint16))


def test_float_intensity_still_round_trips(serve_state):
    counts = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    state = BridgeState(images={'intensity': counts}, units={'intensity': 'photons'})
    base_url = serve_state(state)

    request = Request(
        f'{base_url}/v1/images/intensity.tif',
        headers={'Authorization': 'Bearer test-token'},
    )
    with urlopen(request) as response:
        received = tifffile.imread(BytesIO(response.read()))

    assert received.dtype == np.float32
    np.testing.assert_allclose(received, counts)
