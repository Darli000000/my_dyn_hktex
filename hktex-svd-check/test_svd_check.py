"""CPU-only checks of SVD diagnostics and fallback, using real SciPy factorizations."""
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import scipy.linalg

from apply_svd_check import HELPER


def main():
    ns = {'np': np, 'scipy': scipy, '__file__': __file__}
    exec(compile(HELPER, 'svd_helper.py.txt', 'exec'), ns)
    checked = ns['_hktex_checked_alignment_svd']
    real = scipy.linalg.svd
    matrix = np.random.default_rng(4).normal(size=(16, 16)).astype(np.float32)
    original = matrix.copy()
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, HKTEX_SVD_DIAG_DIR=directory):
        expected = real(matrix)
        observed = checked(matrix)
        assert all(np.array_equal(a, b) for a, b in zip(expected, observed))
        assert not list(Path(directory).iterdir())
        print('PASS: normal path unchanged')

        def fail_default(a, *args, **kwargs):
            if kwargs.get('lapack_driver', 'gesdd') == 'gesdd':
                raise np.linalg.LinAlgError('Injected default-driver failure for test')
            assert a.dtype == np.float64
            return real(a, *args, **kwargs)

        with patch.object(scipy.linalg, 'svd', side_effect=fail_default):
            for a in (matrix, np.zeros((16, 16), dtype=np.float32), np.diag(np.arange(16, dtype=np.float64))):
                u, s, vt = checked(a)
                assert u.dtype == a.dtype
                np.testing.assert_allclose((u * s) @ vt, a, atol=2e-6, rtol=2e-6)
        assert np.array_equal(matrix, original)
        print('PASS: real gesvd fallback, singular matrices, dtype and input preservation')

        with patch.object(scipy.linalg, 'svd', side_effect=AssertionError('Must not call SVD')):
            try:
                checked(np.full((2, 2), np.nan))
                raise AssertionError('Should reject non-finite matrix')
            except FloatingPointError:
                pass
        print('PASS: non-finite input rejected and recorded')

        with patch.object(scipy.linalg, 'svd', side_effect=np.linalg.LinAlgError('Both fail')):
            try:
                checked(matrix)
                raise AssertionError('Should stop if both drivers fail')
            except RuntimeError:
                pass
        print('PASS: failed fallback stops execution')

        def bad_fallback(a, *args, **kwargs):
            if not kwargs:
                raise np.linalg.LinAlgError('Injected failure')
            return np.eye(16), np.ones(16), np.eye(16)

        with patch.object(scipy.linalg, 'svd', side_effect=bad_fallback):
            try:
                checked(matrix)
                raise AssertionError('Should reject inaccurate fallback')
            except RuntimeError:
                pass
        print('PASS: inaccurate fallback rejected')
    print('ALL CPU CHECKS PASSED (does not reproduce the server matrix)')


if __name__ == '__main__':
    main()
