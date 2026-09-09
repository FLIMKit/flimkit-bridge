import json
import os
from dataclasses import dataclass, field
import socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import urlparse

import numpy as np
import tifffile


UINT16_MAX = 65535


def encode_image(image_id, image):
    array = np.asarray(image)
    if image_id != 'intensity':
        return np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        return np.asarray(array, dtype=np.float32)
    if array.min() < 0 or array.max() > UINT16_MAX:
        return np.asarray(array, dtype=np.float32)
    rounded = np.rint(array)
    if not np.array_equal(rounded, array):
        return np.asarray(array, dtype=np.float32)
    return rounded.astype(np.uint16)


def _empty_collection():
    return {'type': 'FeatureCollection', 'features': []}


@dataclass
class BridgeState:
    images: dict[str, np.ndarray]
    units: dict[str, str] = field(default_factory=dict)
    exported_rois: dict = field(default_factory=_empty_collection)
    received_rois: list[dict] = field(default_factory=list)
    app: object = None
    connected: bool = False
    datasets: object = None
    jobs: object = None


def _live_images(state):
    from flimkit.plugins import get_current_images
    current = get_current_images(state.app)
    return current.get('images', {}), current.get('units', {})


def _live_rois(state):
    from flimkit.plugins import export_rois_geojson
    return export_rois_geojson(state.app)


def _live_import(state, payload):
    from flimkit.plugins import import_rois_geojson
    return import_rois_geojson(state.app, payload, mode='append')


_ALLOWED_HOSTS = ('127.0.0.1', 'localhost', '::1', '[::1]')


def _host_allowed(header):
    if not header:
        return True
    name = header.rsplit(':', 1)[0] if header.count(':') == 1 else header
    return name.strip().lower() in _ALLOWED_HOSTS


def create_server(host: str, port: int, token: str, state: BridgeState,
                  live: bool = False):
    class Handler(BaseHTTPRequestHandler):
        def _local_only(self):
            if _host_allowed(self.headers.get('Host')):
                return True
            self.close_connection = True
            self.send_error(403, 'the bridge only answers to localhost')
            return False

        def _authorized(self):
            ok = self.headers.get('Authorization') == f'Bearer {token}'
            if ok:
                state.connected = True
            return ok

        def _send_json(self, status, payload):
            body = json.dumps(payload).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._local_only():
                return
            if self.path == '/v1/status':
                from flimkit_bridge.version import report
                self._send_json(200, report())
                return
            if self._dataset_get():
                return
            if self.path == '/v1/pipeline/defaults':
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(lambda r: r.pipeline_defaults(state))
                return
            if self.path == '/v1/zstack/defaults':
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(lambda r: r.zstack_defaults(state))
                return
            if self.path == '/v1/phasor/settings':
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(lambda r: r.phasor_settings(state))
                return
            if self.path == '/v1/irfs':
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(lambda r: r.irfs(state))
                return
            if self.path == '/v1/fit/defaults':
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(lambda r: r.fit_defaults(state))
                return
            if self.path == '/v1/formats':
                if not self._authorized():
                    self.send_error(401)
                    return
                from flimkit_bridge import formats
                try:
                    self._send_json(200, formats.catalogue())
                except Exception as exc:
                    self.send_error(500, str(exc))
                return
            if self.path == '/v1/rois':
                if not self._authorized():
                    self.send_error(401)
                    return
                try:
                    payload = _live_rois(state) if live else state.exported_rois
                except Exception as exc:
                    self.send_error(500, str(exc))
                    return
                self._send_json(200, payload)
                return
            prefix = '/v1/images/'
            if self.path.startswith(prefix) and self.path.endswith('.tif'):
                if not self._authorized():
                    self.send_error(401)
                    return
                image_id = self.path[len(prefix):-len('.tif')]
                images, units = state.images, state.units
                if live:
                    try:
                        images, units = _live_images(state)
                    except Exception as exc:
                        self.send_error(500, str(exc))
                        return
                try:
                    image = images[image_id]
                except KeyError:
                    self.send_error(404)
                    return
                buffer = BytesIO()
                tifffile.imwrite(buffer, encode_image(image_id, image))
                body = buffer.getvalue()
                self.send_response(200)
                self.send_header('Content-Type', 'image/tiff')
                self.send_header('Content-Length', str(len(body)))
                self.send_header(
                    'X-FLIMKit-Value-Unit', units.get(image_id, ''))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self):
            if not self._local_only():
                return
            if self.path == '/v1/identify':
                if not self._authorized():
                    self.send_error(401)
                    return
                self._identify()
                return
            if self.path == '/v1/datasets':
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(lambda routes: routes.open_dataset(state, self._read_json()))
                return
            if self.path == '/v1/pipeline':
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(lambda routes: routes.run_pipeline(state, self._read_json()))
                return
            if self.path == '/v1/zstack/scan':
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(lambda routes: routes.scan_zstack(state, self._read_json()))
                return
            if self.path == '/v1/zstack/export':
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(lambda routes: routes.export_zstack(state, self._read_json()))
                return
            if self.path == '/v1/zstack':
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(lambda routes: routes.run_zstack(state, self._read_json()))
                return
            from flimkit_bridge import dataset_routes as _routes
            stats_match = _routes.PLANE_STATS_RE.match(self.path)
            if stats_match:
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(
                    lambda r: r.plane_stats(state, stats_match.group(1), self._read_json()))
                return
            pixels_match = _routes.FIT_PIXELS_RE.match(self.path)
            if pixels_match:
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(
                    lambda r: r.fit_pixels(state, pixels_match.group(1), self._read_json()))
                return
            mask_match = _routes.PHASOR_MASK_RE.match(self.path)
            if mask_match:
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(
                    lambda r: r.phasor_mask(state, mask_match.group(1), self._read_json()))
                return
            fit_match = _routes.FIT_ROI_RE.match(self.path)
            if fit_match:
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(
                    lambda r: r.fit_rois(state, fit_match.group(1), self._read_json()))
                return
            if self.path != '/v1/rois':
                self.send_error(404)
                return
            if not self._authorized():
                self.send_error(401)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                payload = json.loads(self.rfile.read(length).decode('utf-8'))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                self.send_error(400)
                return
            if payload.get('type') != 'FeatureCollection':
                self.send_error(400)
                return
            state.received_rois.append(payload)
            if live:
                try:
                    _live_import(state, payload)
                except Exception as exc:
                    self.send_error(500, str(exc))
                    return
            self._send_json(200, {
                'received_features': len(payload.get('features', [])),
            })

        def do_DELETE(self):
            if not self._local_only():
                return
            from flimkit_bridge import dataset_routes as routes
            job = routes.JOB_RE.match(self.path)
            if job:
                if not self._authorized():
                    self.send_error(401)
                    return
                self._route(lambda r: r.job_cancel(state, job.group(1)))
                return
            found = routes.DATASET_RE.match(self.path)
            if not found:
                self.send_error(404)
                return
            if not self._authorized():
                self.send_error(401)
                return
            self._route(lambda r: r.close_dataset(state, found.group(1)))

        def _dataset_get(self):
            from flimkit_bridge import dataset_routes as routes
            parsed = urlparse(self.path)
            if parsed.path == routes.ZSTACK_VOLUME_PATH:
                if not self._authorized():
                    self.send_error(401)
                    return True
                self._send_volume(parsed.query)
                return True
            if parsed.path == '/v1/datasets':
                if not self._authorized():
                    self.send_error(401)
                    return True
                self._route(lambda r: r.list_datasets(state))
                return True
            found = routes.PLANE_RE.match(parsed.path)
            if found:
                if not self._authorized():
                    self.send_error(401)
                    return True
                self._send_plane(found.group(1), found.group(2), parsed.query)
                return True
            found = routes.JOB_RE.match(parsed.path)
            if found:
                if not self._authorized():
                    self.send_error(401)
                    return True
                if parsed.query == 'result':
                    self._route(lambda r: r.job_result(state, found.group(1)))
                else:
                    self._route(lambda r: r.job_status(state, found.group(1)))
                return True
            found = routes.PHASOR_POINTS_RE.match(parsed.path)
            if found:
                if not self._authorized():
                    self.send_error(401)
                    return True
                self._route(lambda r: r.phasor_points(state, found.group(1), parsed.query))
                return True
            found = routes.PHASOR_RE.match(parsed.path)
            if found:
                if not self._authorized():
                    self.send_error(401)
                    return True
                self._route(
                    lambda r: r.phasor_summary(state, found.group(1), parsed.query))
                return True
            found = routes.PLANES_RE.match(parsed.path)
            if found:
                if not self._authorized():
                    self.send_error(401)
                    return True
                self._route(lambda r: r.planes(state, found.group(1)))
                return True
            found = routes.DATASET_RE.match(parsed.path)
            if found:
                if not self._authorized():
                    self.send_error(401)
                    return True
                self._route(lambda r: r.dataset(state, found.group(1)))
                return True
            return False

        def _send_volume(self, query):
            import shutil

            from flimkit_bridge import dataset_routes as routes
            try:
                built = routes.zstack_volume(state, query)
            except routes.RouteError as problem:
                self.send_error(problem.status, str(problem))
                return
            except Exception as exc:
                self.send_error(500, str(exc))
                return
            # A fitted stack can be gigabytes, so it goes out of a file in
            # blocks rather than through one bytes object in memory.
            try:
                size = os.path.getsize(built['file'])
                self.send_response(200)
                self.send_header('Content-Type', 'image/tiff')
                self.send_header('Content-Length', str(size))
                self.send_header('X-FLIMKit-Volume-Label', built['label'])
                self.send_header('X-FLIMKit-Volume-Axes', 'ZCYX')
                self.send_header('X-FLIMKit-Volume-Channels',
                                 ','.join(built['channels']))
                self.send_header('X-FLIMKit-Volume-Units',
                                 ','.join(built['units']))
                self.send_header('X-FLIMKit-Volume-Shape',
                                 ','.join(str(n) for n in built['shape']))
                self.send_header('X-FLIMKit-Voxel-Size-Um',
                                 ','.join(str(v) for v in built['voxel_size_um']))
                self.end_headers()
                with open(built['file'], 'rb') as source:
                    shutil.copyfileobj(source, self.wfile, 1024 * 1024)
            finally:
                shutil.rmtree(built['holding'], ignore_errors=True)

        def _send_plane(self, ident, name, query):
            from flimkit_bridge import dataset_routes as routes
            try:
                body, unit, binning, shape = routes.plane_tiff(state, ident, name, query)
            except routes.RouteError as problem:
                self.send_error(problem.status, str(problem))
                return
            except Exception as exc:
                self.send_error(500, str(exc))
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/tiff')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-FLIMKit-Value-Unit', unit)
            self.send_header('X-FLIMKit-Plane', name)
            self.send_header('X-FLIMKit-Plane-Binning', str(binning))
            self.send_header('X-FLIMKit-Plane-Shape', f'{shape[0]},{shape[1]}')
            self.end_headers()
            self.wfile.write(body)

        def _route(self, call):
            from flimkit_bridge import dataset_routes as routes
            from flimkit_bridge import formats
            try:
                self._send_json(200, call(routes))
            except routes.RouteError as problem:
                self.send_error(problem.status, str(problem))
            except formats.PathProblem as problem:
                self.send_error(problem.status, str(problem))
            except Exception as exc:
                self.send_error(500, str(exc))

        def _identify(self):
            from flimkit_bridge import formats
            payload = self._read_json()
            if payload is None:
                return
            try:
                self._send_json(200, formats.identify(payload.get('path')))
            except formats.PathProblem as problem:
                self.send_error(problem.status, str(problem))
            except Exception as exc:
                self.send_error(500, str(exc))

        def _read_json(self):
            try:
                length = int(self.headers.get('Content-Length', '0'))
                return json.loads(self.rfile.read(length).decode('utf-8'))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                self.send_error(400, 'expected a JSON body')
                return None

        def log_message(self, format, *args):
            pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        block_on_close = False
        allow_reuse_address = os.name != 'nt'

        def server_bind(self):
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = self.server_address[:2]

    server = Server((host, port), Handler)
    server.state = state
    return server


def bind(host, port, token, state, live=False):
    try:
        return create_server(host, port, token, state, live=live)
    except OSError:
        if port == 0:
            raise
        return create_server(host, 0, token, state, live=live)
