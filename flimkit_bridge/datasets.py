import os
import threading
from pathlib import Path
from collections import OrderedDict

import numpy as np

DEFAULT_PLANE_BUDGET = 512 * 1024 * 1024
DEFAULT_MAX_STACK = 2 * 1024 * 1024 * 1024


def default_max_stack():
    override = os.environ.get('FLIMKIT_BRIDGE_MAX_STACK_BYTES')
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    try:
        import psutil
        share = int(psutil.virtual_memory().total * 0.25)
        return max(DEFAULT_MAX_STACK, share)
    except Exception:
        return DEFAULT_MAX_STACK


class StackTooLarge(Exception):

    def __init__(self, estimated_bytes, limit_bytes, suggest_binning):
        super().__init__(
            f'decoding this stack needs about {estimated_bytes / 1e9:.2f} GB, '
            f'over the {limit_bytes / 1e9:.2f} GB limit; '
            f'try binning={suggest_binning}, or raise '
            f'FLIMKIT_BRIDGE_MAX_STACK_BYTES above it')
        self.estimated_bytes = estimated_bytes
        self.limit_bytes = limit_bytes
        self.suggest_binning = suggest_binning


class _FlimFileReader:

    def __init__(self, path, channel):
        from flimkit.formats import FLIMFile, detect_format, file_modality
        self._file = FLIMFile(path, verbose=False)
        self._format = detect_format(path)
        self._modality = file_modality(path)
        self._channel = channel

    def metadata(self):
        tags = getattr(self._file, 'tags', {}) or {}
        width, height = self._dimensions(tags)
        pixel_size = tags.get('ImgHdr_PixResol') or tags.get('ImgHdr_PixRes') or 0
        return {
            'format': self._format,
            'modality': self._modality,
            'n_x': width,
            'n_y': height,
            'n_bins': int(self._file.n_bins),
            'tcspc_res': float(self._file.tcspc_res),
            'channels': self._active_channels(),
            'pixel_size_um': float(pixel_size) if pixel_size else None,
            'time_range_ns': float(self._file.n_bins) * float(self._file.tcspc_res) * 1e9,
            'sync_rate_mhz': self._sync_rate_mhz(),
            'period_ns': self._period_ns(),
            'n_sync': self._n_sync(),
        }

    def _n_sync(self):
        count = getattr(self._file, 'n_sync', 0) or 0
        return int(count) if count else None

    def _sync_rate_mhz(self):
        rate = getattr(self._file, 'sync_rate', 0) or 0
        return round(float(rate) / 1e6, 4) if rate else None

    def _period_ns(self):
        period = getattr(self._file, 'period_ns', 0) or 0
        return round(float(period), 4) if period else None

    def _dimensions(self, tags):
        width, height = self._file.n_x, self._file.n_y
        if width and height:
            return int(width), int(height)
        for x_key, y_key in (('BH_ImageX', 'BH_ImageY'),
                             ('ImgHdr_PixX', 'ImgHdr_PixY')):
            if tags.get(x_key) and tags.get(y_key):
                return int(tags[x_key]), int(tags[y_key])
        raise ValueError(
            f'{self._format} reader does not report image dimensions for '
            f'{self._file.path}; it may not be an image acquisition')

    def _active_channels(self):
        try:
            from flimkit.phasor_launcher import get_ptu_active_channels
            found = list(get_ptu_active_channels(self._file.path))
            if found:
                return found
        except Exception:
            pass
        count = getattr(self._file, 'n_channels', None)
        return list(range(int(count))) if count else []

    def raw_stack(self, channel, binning):
        return self._file.raw_pixel_stack(channel=channel, binning=binning)

    def intensity(self, channel, binning):
        return self._file.intensity_image(channel=channel, binning=binning)


class _StitchedReader:

    def __init__(self, path, channel):
        from flimkit.formats.PTU.stitch import load_stitched_flim
        self._stack, self._time, self._intensity, self._meta = load_stitched_flim(path)
        self.lazy_stack = True

    def metadata(self):
        height, width = self._stack.shape[0], self._stack.shape[1]
        return {
            'format': 'stitched',
            'modality': 'time',
            'n_x': int(width),
            'n_y': int(height),
            'n_bins': int(self._meta['n_time_bins']),
            'tcspc_res': float(self._meta['tcspc_resolution_ps']) * 1e-12,
            'channels': [],
            'pixel_size_um': float(self._meta.get('pixel_size_um') or 0) or None,
            'n_tiles': self._meta.get('tiles_processed'),
        }

    def raw_stack(self, channel, binning):
        if binning != 1:
            raise ValueError('a stitched canvas is served at binning 1')
        return self._stack

    def intensity(self, channel, binning):
        return self._intensity


def is_stitched_output(path):
    directory = Path(path)
    if not directory.is_dir():
        return False
    metadata = list(directory.glob('*_metadata.json')) or (
        [directory / 'metadata.json'] if (directory / 'metadata.json').exists() else [])
    if not metadata:
        return False
    counts = (list(directory.glob('*_stitched_flim_counts.npy'))
              or list(directory.glob('stitched_flim_counts.npy')))
    return bool(counts)


def _default_opener(path, channel):
    if is_stitched_output(path):
        return _StitchedReader(path, channel)
    return _FlimFileReader(path, channel)


class _Entry:

    def __init__(self, ident, path, channel, reader):
        self.id = ident
        self.path = path
        self.channel = channel
        self.reader = reader
        self.refcount = 0
        self.meta = reader.metadata()
        self.planes = OrderedDict()
        self.lock = threading.RLock()


class DatasetRegistry:

    def __init__(self, opener=None, plane_budget_bytes=DEFAULT_PLANE_BUDGET,
                 max_stack_bytes=None):
        self._opener = opener or _default_opener
        self._plane_budget = plane_budget_bytes
        self._max_stack = max_stack_bytes if max_stack_bytes else default_max_stack()
        self._lock = threading.RLock()
        self._by_key = {}
        self._by_id = {}
        self._next = 0

    @staticmethod
    def infer_binning(stack_shape, map_shape):
        sh, sw = stack_shape[0], stack_shape[1]
        mh, mw = map_shape[0], map_shape[1]
        if mh == 0 or mw == 0 or sh % mh or sw % mw:
            raise ValueError(
                f'map shape {map_shape} does not divide stack shape {stack_shape}')
        binning = sh // mh
        if sw // mw != binning:
            raise ValueError(
                f'map shape {map_shape} does not divide stack shape {stack_shape} '
                'by the same factor on both axes')
        return binning

    def open(self, path, channel=None):
        resolved = os.path.realpath(os.path.expanduser(str(path)))
        key = (resolved, channel)
        with self._lock:
            ident = self._by_key.get(key)
            if ident is not None:
                self._by_id[ident].refcount += 1
                return ident
            reader = self._opener(resolved, channel)
            self._next += 1
            ident = f'ds_{self._next}'
            entry = _Entry(ident, resolved, channel, reader)
            entry.refcount = 1
            self._by_key[key] = ident
            self._by_id[ident] = entry
            return ident

    def close(self, ident):
        with self._lock:
            entry = self._by_id.get(ident)
            if entry is None:
                return True
            entry.refcount -= 1
            if entry.refcount > 0:
                return False
            del self._by_id[ident]
            del self._by_key[(entry.path, entry.channel)]
            return True

    def refcount(self, ident):
        with self._lock:
            return self._by_id[ident].refcount

    def list(self):
        with self._lock:
            return [self.metadata(i) for i in self._by_id]

    def reader(self, ident):
        return self._entry(ident).reader

    def _entry(self, ident):
        with self._lock:
            entry = self._by_id.get(ident)
        if entry is None:
            raise KeyError(f'no such dataset: {ident}')
        return entry

    def estimated_stack_bytes(self, ident, binning):
        entry = self._entry(ident)
        meta = entry.meta
        y = meta['n_y'] // binning
        x = meta['n_x'] // binning
        return int(y * x * meta['n_bins'] * 4)

    def metadata(self, ident):
        entry = self._entry(ident)
        meta = entry.meta
        estimates = {
            str(b): self.estimated_stack_bytes(ident, b) for b in (1, 2, 4, 8)
        }
        return {
            'id': entry.id,
            'path': entry.path,
            'channel': entry.channel,
            'format': meta['format'],
            'modality': meta['modality'],
            'width': meta['n_x'],
            'height': meta['n_y'],
            'n_bins': meta['n_bins'],
            'tcspc_res': meta['tcspc_res'],
            'channels': meta['channels'],
            'pixel_size_um': meta['pixel_size_um'],
            'time_range_ns': meta.get('time_range_ns'),
            'sync_rate_mhz': meta.get('sync_rate_mhz'),
            'period_ns': meta.get('period_ns'),
            'n_sync': meta.get('n_sync'),
            'estimated_stack_bytes': estimates,
            'planes': self.plane_names(ident),
            'n_tiles': meta.get('n_tiles'),
        }

    def stack(self, ident, binning=1):
        entry = self._entry(ident)
        if getattr(entry.reader, 'lazy_stack', False):
            with entry.lock:
                return entry.reader.raw_stack(entry.channel, binning)
        estimated = self.estimated_stack_bytes(ident, binning)
        if estimated > self._max_stack:
            suggest = binning
            while suggest < 16 and self.estimated_stack_bytes(ident, suggest) > self._max_stack:
                suggest *= 2
            raise StackTooLarge(estimated, self._max_stack, suggest)
        with entry.lock:
            return entry.reader.raw_stack(entry.channel, binning)

    def intensity(self, ident, binning=1):
        name = 'intensity' if binning == 1 else f'intensity@{binning}'
        held = self.plane(ident, name)
        if held is not None:
            return held
        entry = self._entry(ident)
        with entry.lock:
            reader = entry.reader
            if hasattr(reader, 'intensity'):
                array = reader.intensity(entry.channel, binning)
            else:
                array = self.stack(ident, binning).sum(axis=2)
        self.put_plane(ident, name, np.asarray(array), unit='photons')
        return self.plane(ident, name)

    def plane_names(self, ident):
        entry = self._entry(ident)
        with entry.lock:
            held = [n for n in entry.planes if not n.startswith('intensity@')]
        if 'intensity' not in held:
            held.insert(0, 'intensity')
        return held

    def put_plane(self, ident, name, array, unit=''):
        entry = self._entry(ident)
        array = np.asarray(array)
        with entry.lock:
            entry.planes.pop(name, None)
            entry.planes[name] = {'array': array, 'unit': unit}
            self._evict(entry)

    def _evict(self, entry):
        total = sum(p['array'].nbytes for p in entry.planes.values())
        while total > self._plane_budget and len(entry.planes) > 1:
            _, dropped = entry.planes.popitem(last=False)
            total -= dropped['array'].nbytes

    def plane(self, ident, name):
        entry = self._entry(ident)
        with entry.lock:
            found = entry.planes.get(name)
            if found is None:
                return None
            entry.planes.move_to_end(name)
            return found['array']

    def plane_unit(self, ident, name):
        entry = self._entry(ident)
        with entry.lock:
            found = entry.planes.get(name)
            return '' if found is None else found['unit']

    def planes(self, ident):
        entry = self._entry(ident)
        with entry.lock:
            return [
                {
                    'id': name,
                    'unit': held['unit'],
                    'dtype': str(held['array'].dtype),
                    'shape': list(held['array'].shape),
                }
                for name, held in entry.planes.items()
            ]
