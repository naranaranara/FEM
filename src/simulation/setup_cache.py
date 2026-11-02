import hashlib
import json
import os
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import issparse, csr_matrix


def _digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def source_signature():
    root = Path(__file__).resolve().parents[1]
    value = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.suffix in ('.py', '.cpp'):
            value.update(path.relative_to(root).as_posix().encode())
            value.update(path.read_bytes())
    return value.hexdigest()


def matrix_signature(matrix):
    value = hashlib.sha256(str(matrix.shape).encode())
    for array in (matrix.data, matrix.indices, matrix.indptr):
        value.update(str(array.dtype).encode())
        value.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    return value.hexdigest()


class SetupCache:
    def __init__(self, root, settings, progress=None):
        self.progress = progress
        self.hits = {}
        self.metadata = dict(schema=1, settings=settings, source=source_signature(),
                             numpy=np.__version__, scipy=scipy.__version__,
                             python=list(sys.version_info[:2]))
        encoded = json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        self.key = hashlib.sha256(encoded.encode()).hexdigest()
        self.folder = Path(root).resolve() / self.key

    def report(self, message):
        if self.progress:
            self.progress(message)

    def load(self, name):
        path = self.folder / (name + '.npz')
        self.hits[name] = False
        try:
            if _digest(path) != path.with_suffix('.sha256').read_text().strip():
                raise ValueError('checksum mismatch')
            with np.load(path, allow_pickle=False) as archive:
                descriptor = json.loads(str(archive['__descriptor__']))
                def unpack(item):
                    kind, data = item
                    if kind == 'array': return archive[data]
                    if kind == 'csr':
                        pieces = unpack(data)
                        return csr_matrix((pieces['data'], pieces['indices'], pieces['indptr']),
                                          shape=tuple(pieces['shape']))
                    if kind == 'dict': return {key: unpack(val) for key, val in data.items()}
                    if kind == 'tuple': return tuple(unpack(val) for val in data)
                    if kind == 'list': return [unpack(val) for val in data]
                    if kind == 'scalar': return data
                    raise ValueError('unknown cache entry')
                result = unpack(descriptor)
            self.hits[name] = True
            self.report('Setup cache HIT: ' + name)
            return result
        except FileNotFoundError:
            self.report('Setup cache MISS: ' + name)
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
            self.report('Setup cache invalid (%s): %s; rebuilding' % (name, exc))
        return None

    def store(self, name, value):
        arrays = {}
        def pack(item):
            if issparse(item):
                item = item.tocsr()
                return ['csr', pack(dict(data=item.data, indices=item.indices,
                                         indptr=item.indptr, shape=item.shape))]
            if isinstance(item, np.ndarray):
                if item.dtype.hasobject: raise ValueError('object arrays cannot be cached')
                key = 'a%d' % len(arrays); arrays[key] = item
                return ['array', key]
            if isinstance(item, dict): return ['dict', {key: pack(val) for key, val in item.items()}]
            if isinstance(item, tuple): return ['tuple', [pack(val) for val in item]]
            if isinstance(item, list): return ['list', [pack(val) for val in item]]
            if isinstance(item, np.generic): item = item.item()
            return ['scalar', item]
        arrays['__descriptor__'] = np.array(json.dumps(pack(value), allow_nan=False))
        temporary = None
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.folder, suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                np.savez(stream, **arrays)
            checksum = _digest(temporary)
            target = self.folder / (name + '.npz')
            os.replace(str(temporary), str(target))
            target.with_suffix('.sha256').write_text(checksum, encoding='ascii')
            (self.folder / 'settings.json').write_text(
                json.dumps(self.metadata, indent=2, sort_keys=True), encoding='utf-8')
            self.report('Setup cache SAVED: ' + name)
        except OSError as exc:
            warnings.warn('Setup cache could not be saved: ' + str(exc))
        finally:
            if temporary is not None and temporary.exists(): temporary.unlink()
