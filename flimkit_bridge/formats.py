import os

AMBIGUOUS_EXTS = ('.tif', '.tiff', '.ome.tif', '.json', '.bin')


class PathProblem(Exception):

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _flim_file():
    from flimkit.formats import flim_file
    return flim_file


def catalogue():
    module = _flim_file()
    formats = []
    for entry in module.supported_formats():
        formats.append({
            'id': entry['id'],
            'label': entry.get('label', entry['id']),
            'exts': list(entry.get('exts', ())),
            'modality': entry.get('modality', 'unknown'),
        })
    return {
        'extensions': list(module.supported_extensions()),
        'formats': formats,
    }


def is_ambiguous(path):
    lowered = str(path).lower()
    return any(lowered.endswith(ext) for ext in AMBIGUOUS_EXTS)


def identify(path):
    if not path or not str(path).strip():
        raise PathProblem(400, 'a path is required')
    resolved = os.path.expanduser(str(path))
    if not os.path.exists(resolved):
        raise PathProblem(404, f'no such file: {resolved}')
    if os.path.isdir(resolved):
        from flimkit_bridge.datasets import is_stitched_output
        if is_stitched_output(resolved):
            return {
                'recognised': True,
                'format': 'stitched',
                'modality': 'time',
                'ambiguous': True,
            }
        raise PathProblem(400, f'not a stitched output directory: {resolved}')
    if not os.path.isfile(resolved):
        raise PathProblem(400, f'not a regular file: {resolved}')
    module = _flim_file()
    try:
        found = module.detect_format(resolved)
    except Exception:
        found = 'unknown'
    if found and found != 'unknown':
        try:
            modality = module.file_modality(resolved)
        except Exception:
            modality = 'unknown'
    else:
        found = 'unknown'
        modality = 'unknown'
    return {
        'recognised': found != 'unknown',
        'format': found,
        'modality': modality,
        'ambiguous': is_ambiguous(resolved),
    }
