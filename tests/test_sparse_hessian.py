"""접촉 헤시안의 희소 조립이 정확한 지지 영역과 미분을 보존하는지 검사해요."""
import unittest
import numpy as np
import torch
from scipy.sparse import issparse
from src.contact.barrier import barrier_arrays
from src.contact.terms import multiple_contact_terms, multiple_contact_terms_sparse


class SparseHessianChecks(unittest.TestCase):
    """조밀 중간 행렬 없이 조립한 헤시안의 물리적 미분을 검사해요."""

    def setUp(self):
        """작은 자동미분 문제에서 스레드 수를 제한해요."""
        torch.set_num_threads(1)

    def test_large_local_support_keeps_only_exact_nonzeros(self):
        """서로 독립적인 접촉점이 전역 대각 헤시안을 만드는지 검사해요."""
        count = 1024
        state = np.linspace(-.005, .005, count)
        weights = np.linspace(.5, 1.5, count)

        def gaps(x, indices=None):
            r"""각 자유도에만 의존하는 비선형 간격을 계산해요.

            $$s_i=0.1+x_i+x_i^2$$
            """
            value = x if indices is None else x[indices]
            # 각 자유도의 간격을 계산해요: $$s_i=0.1+x_i+x_i^2$$
            return .1 + value + value ** 2

        result = multiple_contact_terms_sparse(state, gaps, weights, np.arange(count),
                                               gaps_subset_autodiff=gaps)
        _, b1, b2 = barrier_arrays(.1 + state + state ** 2, .2)
        self.assertTrue(issparse(result['hessian']))
        self.assertEqual(result['hessian'].nnz, count)
        np.testing.assert_allclose(result['gradient'], weights * b1 * (1 + 2 * state), rtol=1e-14)
        np.testing.assert_allclose(result['hessian'].diagonal(), weights * b2 * (1 + 2 * state) ** 2, rtol=1e-14)

    def test_full_and_filtered_unsorted_dofs_match_dense(self):
        """임의 순서의 접촉 자유도와 곡률을 전역 위치에 정확히 배치하는지 검사해요."""
        rng = np.random.default_rng(14)
        state = rng.normal(scale=.001, size=191)
        dofs = rng.choice(len(state), size=137, replace=False)
        ids = torch.tensor(dofs, dtype=torch.long)

        def gaps(x, indices=None):
            r"""선택된 자유도의 비선형 접촉 간격을 계산해요.

            $$s_i=0.015+x_{d_i}+x_{d_i}^2$$
            """
            local = x[ids]
            # 접촉점의 비선형 간격을 계산해요: $$s_i=0.015+x_{d_i}+x_{d_i}^2$$
            values = .015 + local + local ** 2
            return values if indices is None else values[indices]

        weights = np.linspace(.7, 1.3, len(dofs))
        for method in ('gn', 'filtered_gn', 'full'):
            sparse = multiple_contact_terms_sparse(state, gaps, weights, dofs, method=method,
                                                    gaps_subset_autodiff=gaps,
                                                    return_gap_data=method == 'gn')
            dense = multiple_contact_terms(state, gaps, weights, method=method,
                                           gaps_subset_autodiff=gaps)
            np.testing.assert_allclose(sparse['gradient'], dense['gradient'], rtol=1e-14, atol=1e-15)
            np.testing.assert_allclose(sparse['hessian'].toarray(), dense['hessian'], rtol=1e-14, atol=1e-15)
            self.assertEqual(sparse['hessian'].nnz, len(dofs))
            if method == 'gn':
                self.assertIsInstance(sparse['gap_jacobian'], np.ndarray)
                np.testing.assert_array_equal(sparse['gap_dofs'], dofs)
                np.testing.assert_allclose(sparse['gap_jacobian'], np.diag(1 + 2 * state[dofs]))

    def test_jacobian_storage_switch_keeps_later_dense_rows(self):
        """초기 희소 행 뒤에 조밀한 행이 와도 모든 접촉 결합을 보존하는지 검사해요."""
        rng = np.random.default_rng(19)
        matrix = rng.normal(scale=.01, size=(198, 160))
        matrix[:32] = np.eye(160)[:32]
        matrix[0, -1] = 1e-30
        tensor = torch.tensor(matrix, dtype=torch.float64)

        def gaps(x, indices=None):
            r"""조밀한 행과 희소한 행이 섞인 선형 간격을 계산해요.

            $$\boldsymbol{s}=0.1+A\boldsymbol{x}$$
            """
            selected = tensor if indices is None else tensor[indices]
            # 선형 간격을 계산해요: $$\boldsymbol{s}=0.1+A\boldsymbol{x}$$
            return .1 + selected @ x

        result = multiple_contact_terms_sparse(np.zeros(160), gaps, np.ones(198), np.arange(160),
                                               gaps_subset_autodiff=gaps, return_gap_data=True)
        np.testing.assert_array_equal(result['gap_jacobian'], matrix)
        expected = multiple_contact_terms(np.zeros(160), gaps, np.ones(198))
        np.testing.assert_allclose(result['hessian'].toarray(), expected['hessian'], rtol=2e-14, atol=1e-15)

    def test_no_active_contact_and_force_only_preserve_contract(self):
        """비활성 접촉과 기울기 전용 평가의 반환 형식을 검사해요."""
        def gaps(x):
            """접촉 임계값 바깥의 선형 간격을 반환해요."""
            return x[:2] + .3

        for compute in (False, True):
            result = multiple_contact_terms_sparse(np.zeros(7), gaps, np.ones(2), [1, 0],
                                                   compute_hessian=compute)
            np.testing.assert_array_equal(result['gradient'], np.zeros(7))
            self.assertEqual(result['energy'], 0.)
            if compute:
                self.assertEqual(result['hessian'].shape, (7, 7))
                self.assertEqual(result['hessian'].nnz, 0)
            else:
                self.assertIsNone(result['hessian'])

    def test_explicit_value_callback_avoids_only_validation_graph(self):
        """명시적 값 전용 콜백만 그래프를 생략하고 기본 내부 미분은 허용하는지 검사해요."""
        value_modes = []
        derivative_modes = []

        def gaps(x, indices=None):
            r"""내부 미분으로 간격을 계산하고 미분 모드를 기록해요.

            $$s_i=0.1+\frac{\partial}{\partial x_i}\sum_j(x_j^3/3+x_j^2/2)$$
            """
            derivative_modes.append(torch.is_grad_enabled())
            # 간격의 원시함수를 합해요: $$F=\sum_j(x_j^3/3+x_j^2/2)$$
            field = torch.sum(x ** 3 / 3 + x ** 2 / 2)
            derivative = torch.autograd.grad(field, x, create_graph=True)[0]
            # 양의 기준 간격을 더해요: $$s_i=0.1+\partial F/\partial x_i$$
            values = .1 + derivative[:5]
            return values if indices is None else values[indices]

        def values(x):
            r"""같은 간격을 미분 없는 식으로 계산해요.

            $$s_i=0.1+x_i+x_i^2$$
            """
            value_modes.append(torch.is_grad_enabled())
            # 검증할 간격을 직접 계산해요: $$s_i=0.1+x_i+x_i^2$$
            return .1 + x[:5] + x[:5] ** 2

        state = np.linspace(-.003, .003, 7)
        weights = np.linspace(.7, 1.2, 5)
        for method in ('gn', 'filtered_gn'):
            baseline = multiple_contact_terms(state, gaps, weights, method=method,
                                              gaps_subset_autodiff=gaps)
            dense = multiple_contact_terms(state, gaps, weights, method=method,
                                           gaps_subset_autodiff=gaps, gaps_value_autodiff=values)
            sparse = multiple_contact_terms_sparse(state, gaps, weights, [4, 0, 3, 1, 2],
                                                    method=method, gaps_subset_autodiff=gaps,
                                                    gaps_value_autodiff=values)
            for result in (dense, sparse):
                self.assertAlmostEqual(result['energy'], baseline['energy'], places=14)
                np.testing.assert_allclose(result['gradient'], baseline['gradient'], rtol=1e-14, atol=1e-15)
                H = result['hessian'].toarray() if issparse(result['hessian']) else result['hessian']
                np.testing.assert_allclose(H, baseline['hessian'], rtol=1e-14, atol=1e-15)
        self.assertTrue(value_modes)
        self.assertFalse(any(value_modes))
        self.assertTrue(all(derivative_modes))

        def unused(x):
            """기울기나 전체 헤시안 경로에서 호출되면 실패해요."""
            raise AssertionError('value-only callback must be ignored')

        for method, compute in (('full', True), ('gn', False)):
            multiple_contact_terms(state, gaps, weights, method=method,
                                   compute_hessian=compute, gaps_value_autodiff=unused)


if __name__ == '__main__':
    unittest.main()
