import atexit
import secrets
import threading

from flimkit.plugins import (
    export_rois_geojson,
    get_current_images,
    import_rois_geojson,
    panel_button,
    plugin_config,
    startup,
    tool,
)

from flimkit_bridge import discovery
from flimkit_bridge.state import build_state
from flimkit_bridge.server import bind

FLIMKIT_PLUGIN_API = 1

PLUGIN_NAME = 'flimkit_bridge'

DEFAULT_PORT = 8765

_state = {
    'server': None,
    'thread': None,
    'token': '',
    'url': '',
    'error': '',
}


def bridge_status():
    return dict(_state)


def _live_state(app):
    state = build_state()
    state.app = app
    return state


def _start_server(app, port=DEFAULT_PORT):
    token = secrets.token_urlsafe(24)
    state = _live_state(app)
    server = bind('127.0.0.1', port, token, state, live=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, bound = server.server_address
    _state.update({
        'server': server,
        'thread': thread,
        'token': token,
        'url': f'http://{host}:{bound}',
        'error': '',
    })
    discovery.write(_state['url'], token)
    atexit.register(discovery.remove)
    return _state


@startup('flimkit_bridge_server', order=200)
def start_bridge(app):
    if _state['server'] is not None:
        return
    cfg = plugin_config(PLUGIN_NAME)
    port = int(cfg.get('port', DEFAULT_PORT) or DEFAULT_PORT)
    try:
        _start_server(app, port)
    except OSError as exc:
        _state['error'] = str(exc)
        print(f'[FLIMKit bridge] could not bind a port: {exc}')
        return
    if _state['url'] != f'http://127.0.0.1:{port}':
        print(f'[FLIMKit bridge] port {port} was busy, using {_state["url"]}')
    print(f'[FLIMKit bridge] listening on {_state["url"]}')
    print(f'[FLIMKit bridge] details written to {discovery.discovery_path()}')


@panel_button('flimkit_bridge_send', 'Send to a viewer', panel='roi', order=200)
def send_to_clients(app):
    from tkinter import messagebox
    server = _state['server']
    if server is None:
        messagebox.showerror(
            'FLIMKit bridge',
            'The bridge is not running.\n\n'
            + (_state['error'] or 'It failed to start with FLIMKit.'),
            parent=getattr(app, 'root', None),
        )
        return
    if not server.state.connected:
        messagebox.showerror(
            'No viewer is connected',
            'Nothing has connected to FLIMKit yet.\n\n'
            'Start QuPath, then choose\n'
            'Extensions > FLIMKit bridge > Connect...\n\n'
            'It finds the address and token itself, from\n'
            f'{discovery.discovery_path()}',
            parent=getattr(app, 'root', None),
        )
        return
    images = get_current_images(app)
    names = sorted(images.get('images', {}))
    if not names:
        messagebox.showerror(
            'FLIMKit bridge',
            'There is nothing to send yet. Fit a dataset first.',
            parent=getattr(app, 'root', None),
        )
        return
    messagebox.showinfo(
        'FLIMKit bridge',
        'FLIMKit is serving ' + ', '.join(names) + '.\n\n'
        'In QuPath choose\n'
        'Extensions > FLIMKit bridge > Add FLIMKit images to project',
        parent=getattr(app, 'root', None),
    )


@tool(id='flimkit_bridge_open', label='FLIMKit Bridge...', menu='Tools', order=510)
def open_bridge(app):
    from tkinter import messagebox
    if _state['server'] is None:
        messagebox.showerror(
            'FLIMKit bridge',
            'The bridge is not running.\n\n'
            + (_state['error'] or 'It failed to start with FLIMKit.'),
            parent=getattr(app, 'root', None),
        )
        return
    connected = 'yes' if _state['server'].state.connected else 'no'
    messagebox.showinfo(
        'FLIMKit bridge',
        f'Address: {_state["url"]}\n'
        f'A viewer is connected: {connected}\n\n'
        f'Pairing details: {discovery.discovery_path()}\n\n'
        'In QuPath choose\n'
        'Extensions > FLIMKit bridge > Connect...',
        parent=getattr(app, 'root', None),
    )
