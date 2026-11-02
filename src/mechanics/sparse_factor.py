"""##r## Reusable sparse factors for constant linear FEM condensation.

PARDISO uses its tested general real matrix path (mtype=11). This changes
linear algebra only; it does not regularize contact or change its energy.
"""
import weakref
import os
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits


_dll_directories = []


def _prepare_windows_mkl():
    """##r## A venv may reuse MKL from its base Conda installation."""
    if os.name != 'nt' or os.environ.get('PYPARDISO_MKL_RT'):
        return
    for prefix in dict.fromkeys((sys.prefix, sys.base_prefix)):
        directory = Path(prefix)/'Library'/'bin'
        candidates = sorted(directory.glob('mkl_rt*.dll'), key=lambda p: len(p.name))
        if candidates:
            if hasattr(os, 'add_dll_directory'):
                _dll_directories.append(os.add_dll_directory(str(directory)))
            os.environ['PYPARDISO_MKL_RT'] = str(candidates[0])
            return


class PardisoFactor:
    def __init__(self, matrix):
        _prepare_windows_mkl()
        try:
            from pypardiso import PyPardisoSolver
        except (ImportError, OSError) as exc:
            raise ImportError('PARDISO requires pypardiso and a working Intel MKL runtime; '
                              'install the pardiso extra or use --linear-backend superlu') from exc
        self.matrix = csr_matrix(matrix, dtype=np.float64, copy=True)
        self.matrix.sum_duplicates()
        self.matrix.sort_indices()
        n = self.matrix.shape[0]
        if self.matrix.shape != (n, n) or not n:
            raise ValueError('nonempty square matrix required')
        if not np.all(np.isfinite(self.matrix.data)) or np.any(np.diff(self.matrix.indptr) == 0):
            raise ValueError('finite entries and nonempty rows required')
        if max(n, self.matrix.nnz) >= np.iinfo(np.int32).max:
            raise ValueError('PyPardiso wrapper requires 32-bit sparse indexing')
        self.solver = PyPardisoSolver(mtype=11)
        # ##r## No core-count option: keep the established one-thread setting,
        # including MKL loaded lazily after the outer threadpool context.
        try:
            with threadpool_limits(1):
                self.solver.factorize(self.matrix)
        except Exception:
            self.solver.free_memory(everything=True)
            raise
        self._cleanup = weakref.finalize(self, self.solver.free_memory, everything=True)

    def solve(self, rhs):
        if not self._cleanup.alive:
            raise RuntimeError('PARDISO factor has been closed')
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.matrix.shape[0]:
            raise ValueError('invalid right-hand side shape')
        if not np.all(np.isfinite(rhs)):
            raise ValueError('right-hand side must be finite')
        if rhs.ndim == 2 and rhs.shape[1] == 0:
            return rhs.copy()
        with threadpool_limits(1):
            result = self.solver.solve(self.matrix, np.asfortranarray(rhs))
        if not np.all(np.isfinite(result)):
            raise np.linalg.LinAlgError('nonfinite PARDISO solution')
        return result

    def close(self):
        self._cleanup()


def factorize_sparse(matrix, backend='superlu'):
    if backend == 'superlu':
        return splu(matrix.tocsc())
    if backend == 'pardiso':
        return PardisoFactor(matrix)
    raise ValueError('linear backend must be superlu or pardiso')
