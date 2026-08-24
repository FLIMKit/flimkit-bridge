def bridge_version():
    try:
        from importlib.metadata import version
        return version('flimkit-bridge')
    except Exception:
        return 'unknown'


def flimkit_version():
    try:
        from flimkit._version import __version__
        return __version__
    except Exception:
        return 'unknown'


def report():
    return {
        'protocol': 'flimkit-qupath',
        'protocol_version': 1,
        'bridge_version': bridge_version(),
        'flimkit_version': flimkit_version(),
    }
