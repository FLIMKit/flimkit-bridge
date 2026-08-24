import json
import os
import time
from pathlib import Path

FILENAME = 'bridge.json'
LEGACY_FILENAME = 'qupath-bridge.json'
PROTOCOL = 'flimkit-bridge'
LEGACY_PROTOCOL = 'flimkit-qupath'


def discovery_dir():
    return Path(os.path.expanduser('~')) / '.flimkit'


def discovery_path():
    return discovery_dir() / FILENAME


def write(url, token, pid=None, path=None):
    target = Path(path) if path is not None else discovery_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'protocol': PROTOCOL,
        'protocol_version': 1,
        'url': url,
        'token': token,
        'pid': os.getpid() if pid is None else int(pid),
        'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    _write_atomic(target, payload)
    legacy = dict(payload)
    legacy['protocol'] = LEGACY_PROTOCOL
    _write_atomic(target.parent / LEGACY_FILENAME, legacy)
    return target


def _write_atomic(target, payload):
    tmp = target.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, target)


def read(path=None):
    target = Path(path) if path is not None else discovery_path()
    try:
        with open(target, encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if payload.get('protocol') not in (PROTOCOL, LEGACY_PROTOCOL):
        return None
    if not payload.get('url') or not payload.get('token'):
        return None
    return payload


def _windows_process_alive(pid):
    import ctypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) == 0:
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def process_alive(pid):
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name == 'nt':
        try:
            return _windows_process_alive(pid)
        except OSError:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def read_live(path=None):
    payload = read(path)
    if payload is None:
        return None
    if not process_alive(payload.get('pid')):
        return None
    return payload


def remove(path=None):
    target = Path(path) if path is not None else discovery_path()
    for each in (target, target.parent / LEGACY_FILENAME):
        try:
            each.unlink()
        except OSError:
            pass
