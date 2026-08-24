import argparse
import os
import secrets
import sys

from flimkit_bridge import discovery
from flimkit_bridge.server import bind
from flimkit_bridge.state import build_state

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8765


def running_bridge():
    payload = discovery.read_live()
    if payload is None:
        return None
    if payload.get('pid') == os.getpid():
        return None
    return payload


def serve(host=DEFAULT_HOST, port=DEFAULT_PORT, token=None, announce=True):
    token = token or secrets.token_urlsafe(24)
    state = build_state()
    server = bind(host, port, token, state)
    bound_host, bound_port = server.server_address
    url = f'http://{bound_host}:{bound_port}'
    if announce:
        discovery.write(url, token)
    return server, url, token


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='flimkit-bridge',
        description='Serve FLIMKit to QuPath without starting the desktop GUI')
    parser.add_argument('--host', default=DEFAULT_HOST)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--token', default='')
    parser.add_argument('--no-announce', action='store_true',
                        help='do not write the discovery file QuPath reads')
    parser.add_argument('--force', action='store_true',
                        help='take over the discovery file from a running bridge')
    return parser.parse_args(argv)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass
    args = parse_args(argv)
    announce = not args.no_announce
    if announce and not args.force:
        other = running_bridge()
        if other is not None:
            print(f'[FLIMKit bridge] another bridge is already serving on '
                  f'{other["url"]} (pid {other.get("pid")})')
            print('[FLIMKit bridge] stop it, or pass --force to take over '
                  'the discovery file, or --no-announce to run alongside it')
            return 1
    try:
        server, url, token = serve(args.host, args.port, args.token or None, announce)
    except OSError as exc:
        print(f'[FLIMKit bridge] could not bind {args.host}:{args.port}: {exc}')
        return 1
    if args.port and url != f'http://{args.host}:{args.port}':
        print(f'[FLIMKit bridge] port {args.port} was busy, using {url}')
    print(f'[FLIMKit bridge] listening on {url}')
    if announce:
        print(f'[FLIMKit bridge] details written to {discovery.discovery_path()}')
    else:
        print(f'[FLIMKit bridge] token {token}')
    print('[FLIMKit bridge] no GUI, Ctrl-C to stop')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('[FLIMKit bridge] stopping')
    finally:
        server.shutdown()
        server.server_close()
        if announce:
            discovery.remove()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
