"""원통 표면 검증과 축약 접촉에서 결과 및 사용자 콜백 호환성을 검사해요."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.sparse import issparse
from threadpoolctl import threadpool_limits

from src.mechanics import condensation
from src.simulation import test_cylinder_validation as validation
from src.simulation.two_body import build_problem


class CylinderPipelineChecks(unittest.TestCase):
    """실제 표면 영점과 축약 접촉 계산의 동등성 및 호환성을 검사해요."""

    @classmethod
    def setUpClass(cls):
        """작은 수치 검사의 스레드 수를 일정하게 유지해요."""
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.pool = threadpool_limits(1)

    @classmethod
    def tearDownClass(cls):
        """검사 전의 스레드 설정으로 복구해요."""
        cls.pool.restore_original_limits()
        torch.set_num_threads(cls.threads)

    def test_surface_zero_search_keeps_dense_reference_without_dense_storage(self):
        """두 곡면의 영점 탐색이 조밀 보간 기준과 일치하고 희소 행렬만 만드는지 검사해요."""
        sampler = validation.make_surface_sampler
        storage = []

        def record_sampler(reference, faces, **kwargs):
            """실제 표면 샘플러를 실행하고 보간 행렬의 저장 방식만 기록해요."""
            result = sampler(reference, faces, **kwargs)
            matrix = result[1]
            storage.append((kwargs.get('sparse', False), issparse(matrix), matrix.shape,
                            matrix.nnz if issparse(matrix) else np.count_nonzero(matrix)))
            return result

        def dense_sampler(reference, faces, **kwargs):
            """기존 조밀 보간 행렬을 만드는 호출 규약으로 같은 표면을 평가해요."""
            return sampler(reference, faces, sparse=False)

        with patch.object(validation, 'make_surface_sampler', record_sampler):
            actual, arrays = validation.surface_error(32, 7, .65)
        with patch.object(validation, 'make_surface_sampler', dense_sampler):
            expected, expected_arrays = validation.surface_error(32, 7, .65)

        self.assertEqual(actual, expected)
        self.assertEqual(set(arrays), set(expected_arrays))
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], expected_arrays[key])
        self.assertEqual(len(storage), 2)
        for requested, sparse, shape, entries in storage:
            self.assertTrue(requested)
            self.assertTrue(sparse)
            self.assertEqual(entries, 3 * shape[0])
            self.assertEqual(shape[1], 231)
        self.assertTrue(all(body['root_residual'] < 1e-10 for body in actual['bodies']))

    def test_condensation_keeps_callbacks_that_need_inner_autograd(self):
        """축약 전후에 간격 콜백을 교체하거나 별도 문제를 전달해도 내부 자동미분을 허용해요."""
        terms = condensation.multiple_contact_terms
        value_paths = []

        def record_terms(*args, **kwargs):
            """실제 접촉 계산에 전달한 값 전용 콜백의 사용 여부만 기록해요."""
            value_paths.append(kwargs.get('gaps_value_autodiff') is not None)
            return terms(*args, **kwargs)

        with patch.object(condensation, 'multiple_contact_terms', record_terms):
            reference = condensation.condense_contact_problem(build_problem(n=2, sparse=True))
            reference['set_closure'](.025)
            state = reference['initial'].copy()
            expected = reference['objective'](state, 'gn')
            self.assertTrue(value_paths[-1])

            for replacement in ('before', 'after', 'unmarked'):
                with self.subTest(replacement=replacement):
                    problem = build_problem(n=2, sparse=True)
                    original = problem['gaps_ad']

                    def needs_autograd(values, indices=None):
                        r"""내부 자동미분으로 항등 좌표를 복원한 뒤 같은 접촉 간격을 계산해요.

                        $$\nabla_x\left(\frac12 x^Tx\right)=x$$
                        """
                        coordinates = values.clone().requires_grad_(True)
                        # 항등 좌표를 복원할 이차 함수를 평가해요: $$f(x)=\frac12 x^Tx$$
                        scalar = .5 * coordinates.square().sum()
                        # 내부 자동미분으로 원래 좌표를 복원해요: $$\nabla_x f=x$$
                        recovered = torch.autograd.grad(scalar, coordinates, create_graph=True)[0]
                        return original(recovered, indices=indices)

                    if replacement in ('before', 'unmarked'):
                        problem['gaps_ad'] = needs_autograd
                    if replacement == 'unmarked':
                        problem.pop('_no_grad_gaps_ad')
                    actual = condensation.condense_contact_problem(problem)
                    if replacement == 'after':
                        problem['gaps_ad'] = needs_autograd
                    actual['set_closure'](.025)
                    result = actual['objective'](state, 'gn')
                    self.assertFalse(value_paths[-1])
                    self.assertAlmostEqual(result[0], expected[0], places=14)
                    np.testing.assert_allclose(result[1], expected[1], rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(result[2], expected[2], rtol=1e-12, atol=1e-12)
                    self.assertEqual(result[3], expected[3])


if __name__ == '__main__':
    unittest.main()
