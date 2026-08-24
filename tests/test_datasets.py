import os
import threading
from pathlib import Path

import numpy as np
import pytest

from flimkit_bridge.datasets import DatasetRegistry, StackTooLarge


class FakeReader:
    def __init__(self, path, n_y=8, n_x=8, n_bins=16, channels=(0,)):
        self.path = path
        self.n_y = n_y
        self.n_x = n_x
        self.n_bins = n_bins
        self.tcspc_res = 8e-11
        self.channels = list(channels)
        self.decodes = 0

    def metadata(self):
        return {
            'format': 'ptu',
            'modality': 'time',
            'n_x': self.n_x,
            'n_y': self.n_y,
            'n_bins': self.n_bins,
            'tcspc_res': self.tcspc_res,
            'channels': self.channels,
            'pixel_size_um': 0.284,
        }

    def raw_stack(self, channel, binning):
        self.decodes += 1
        y, x = self.n_y // binning, self.n_x // binning
        return np.ones((y, x, self.n_bins), dtype=np.uint32)


@pytest.fixture
def registry():
    readers = {}

    def opener(path, channel):
        reader = readers.setdefault(path, FakeReader(path))
        return reader

    return DatasetRegistry(opener=opener, plane_budget_bytes=2500,
                           max_stack_bytes=100_000), readers


def test_open_is_idempotent_per_path_and_channel(registry):
    reg, readers = registry

    first = reg.open('/data/a.ptu', channel=0)
    second = reg.open('/data/a.ptu', channel=0)

    assert first == second
    assert len(reg.list()) == 1
    assert reg.refcount(first) == 2


def test_different_channels_are_different_datasets(registry):
    reg, _ = registry

    assert reg.open('/data/a.ptu', channel=0) != reg.open('/data/a.ptu', channel=1)
    assert len(reg.list()) == 2


def test_close_is_refcounted(registry):
    reg, _ = registry
    ident = reg.open('/data/a.ptu', channel=0)
    reg.open('/data/a.ptu', channel=0)

    assert reg.close(ident) is False
    assert reg.refcount(ident) == 1
    assert reg.close(ident) is True
    assert reg.list() == []


def test_metadata_reports_geometry_and_estimates(registry):
    reg, _ = registry
    ident = reg.open('/data/a.ptu', channel=0)

    meta = reg.metadata(ident)

    assert meta['id'] == ident
    assert (meta['width'], meta['height']) == (8, 8)
    assert meta['n_bins'] == 16
    assert meta['pixel_size_um'] == 0.284
    assert meta['estimated_stack_bytes']['1'] == 8 * 8 * 16 * 4
    assert meta['estimated_stack_bytes']['2'] == 4 * 4 * 16 * 4


def test_stack_is_uint32_not_float(registry):
    reg, _ = registry
    ident = reg.open('/data/a.ptu', channel=0)

    assert reg.stack(ident, binning=1).dtype == np.uint32


def test_stack_over_budget_is_refused_with_a_suggestion(registry):
    reg, _ = registry
    reg_small = DatasetRegistry(
        opener=reg._opener, plane_budget_bytes=2500, max_stack_bytes=2000)
    ident = reg_small.open('/data/a.ptu', channel=0)

    with pytest.raises(StackTooLarge) as caught:
        reg_small.stack(ident, binning=1)

    assert caught.value.estimated_bytes == 8 * 8 * 16 * 4
    assert caught.value.limit_bytes == 2000
    assert caught.value.suggest_binning == 2


def test_stacks_are_not_retained(registry):
    reg, readers = registry
    ident = reg.open('/data/a.ptu', channel=0)

    reg.stack(ident, binning=1)
    reg.stack(ident, binning=1)

    assert len(readers) == 1
    reader = next(iter(readers.values()))
    assert reader.decodes == 2


def test_planes_are_cached_and_evicted_by_budget(registry):
    reg, _ = registry
    ident = reg.open('/data/a.ptu', channel=0)
    plane = np.zeros((16, 16), dtype=np.float32)

    reg.put_plane(ident, 'tau_1', plane, unit='ns')
    reg.put_plane(ident, 'tau_2', plane, unit='ns')
    reg.put_plane(ident, 'tau_3', plane, unit='ns')

    assert reg.plane(ident, 'tau_1') is None
    assert reg.plane(ident, 'tau_3') is not None
    assert [p['id'] for p in reg.planes(ident)] == ['tau_2', 'tau_3']


def test_eviction_drops_the_least_recently_used(registry):
    reg, _ = registry
    ident = reg.open('/data/a.ptu', channel=0)
    plane = np.zeros((16, 16), dtype=np.float32)

    reg.put_plane(ident, 'a', plane, unit='ns')
    reg.put_plane(ident, 'b', plane, unit='ns')
    reg.plane(ident, 'a')
    reg.put_plane(ident, 'c', plane, unit='ns')

    assert reg.plane(ident, 'a') is not None
    assert reg.plane(ident, 'b') is None
    assert reg.plane(ident, 'c') is not None


@pytest.mark.parametrize('stack_shape,map_shape,expected', [
    ((512, 512), (512, 512), 1),
    ((512, 512), (256, 256), 2),
    ((1024, 1024), (256, 256), 4),
])
def test_binning_is_inferred_from_shapes(stack_shape, map_shape, expected):
    assert DatasetRegistry.infer_binning(stack_shape, map_shape) == expected


def test_binning_inference_rejects_a_non_divisor():
    with pytest.raises(ValueError, match='does not divide'):
        DatasetRegistry.infer_binning((512, 512), (300, 300))


def test_concurrent_opens_yield_one_dataset(registry):
    reg, _ = registry
    seen = []

    def worker():
        seen.append(reg.open('/data/race.ptu', channel=0))

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(seen)) == 1
    assert len(reg.list()) == 1
    assert reg.refcount(seen[0]) == 12


def test_unknown_dataset_raises(registry):
    reg, _ = registry
    with pytest.raises(KeyError):
        reg.metadata('nope')


PTU_PATH = os.environ.get('FLIMKIT_TEST_PTU', '')
SDT_PATH = os.environ.get('FLIMKIT_TEST_SDT', '')

real_ptu = pytest.mark.skipif(
    not PTU_PATH or not Path(PTU_PATH).exists(),
    reason='set FLIMKIT_TEST_PTU to a .ptu file',
)

real_sdt = pytest.mark.skipif(
    not SDT_PATH or not Path(SDT_PATH).exists(),
    reason='set FLIMKIT_TEST_SDT to a .sdt file',
)


@real_ptu
def test_real_ptu_metadata_is_sane():
    reg = DatasetRegistry()
    ident = reg.open(PTU_PATH)

    meta = reg.metadata(ident)

    assert meta['format'] == 'ptu'
    assert meta['modality'] == 'time'
    assert meta['width'] > 0 and meta['height'] > 0
    assert meta['n_bins'] > 0
    assert 0 < meta['tcspc_res'] < 1e-6
    assert meta['estimated_stack_bytes']['1'] == (
        meta['width'] * meta['height'] * meta['n_bins'] * 4)
    assert meta['estimated_stack_bytes']['4'] < meta['estimated_stack_bytes']['1']


@real_ptu
def test_real_ptu_stack_is_uint32_and_binned():
    reg = DatasetRegistry()
    ident = reg.open(PTU_PATH)
    meta = reg.metadata(ident)

    stack = reg.stack(ident, binning=4)

    assert stack.dtype == np.uint32
    assert stack.shape == (meta['height'] // 4, meta['width'] // 4, meta['n_bins'])
    assert DatasetRegistry.infer_binning(
        (meta['height'], meta['width']), stack.shape[:2]) == 4


@real_sdt
def test_becker_hickl_sdt_reports_dimensions_without_decoding():
    reg = DatasetRegistry()
    ident = reg.open(SDT_PATH)

    meta = reg.metadata(ident)

    assert meta['format'] == 'bh_sdt'
    assert meta['modality'] == 'time'
    assert meta['width'] > 0 and meta['height'] > 0
    assert meta['n_bins'] > 0
    assert meta['channels']


@real_sdt
def test_sdt_dimensions_match_the_decoded_stack():
    reg = DatasetRegistry()
    ident = reg.open(SDT_PATH)
    meta = reg.metadata(ident)

    stack = reg.stack(ident, binning=8)

    assert stack.dtype == np.uint32
    assert DatasetRegistry.infer_binning(
        (meta['height'], meta['width']), stack.shape[:2]) == 8


@real_sdt
def test_sdt_has_no_pixel_size_rather_than_a_bogus_zero():
    reg = DatasetRegistry()
    meta = reg.metadata(reg.open(SDT_PATH))

    assert meta['pixel_size_um'] is None


def test_metadata_reports_the_acquisition_window():
    from flimkit_bridge import datasets

    class _File:
        n_bins = 529
        tcspc_res = 9.696969697e-11
        sync_rate = 19.505e6
        period_ns = 51.2689
        n_x = 512
        n_y = 512
        tags = {}
        path = 'made-up.ptu'
        n_channels = 1

    reader = datasets._FlimFileReader.__new__(datasets._FlimFileReader)
    reader._file = _File()
    reader._format = 'ptu'
    reader._modality = 'time'
    reader._channel = None
    meta = reader.metadata()
    assert meta['time_range_ns'] == pytest.approx(51.2970, abs=1e-3)
    assert meta['sync_rate_mhz'] == pytest.approx(19.505)
    assert meta['period_ns'] == pytest.approx(51.2689)


def test_the_window_is_absent_rather_than_wrong_without_a_laser_rate():
    from flimkit_bridge import datasets

    class _File:
        n_bins = 256
        tcspc_res = 5e-11
        sync_rate = 0
        period_ns = 0
        n_x = 8
        n_y = 8
        tags = {}
        path = 'made-up.ptu'
        n_channels = 1

    reader = datasets._FlimFileReader.__new__(datasets._FlimFileReader)
    reader._file = _File()
    reader._format = 'ptu'
    reader._modality = 'time'
    reader._channel = None
    meta = reader.metadata()
    assert meta['time_range_ns'] == pytest.approx(12.8)
    assert meta['sync_rate_mhz'] is None
    assert meta['period_ns'] is None
