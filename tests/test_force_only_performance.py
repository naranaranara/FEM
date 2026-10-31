"""힘만 계산하는 접촉 경로의 활성점 미분과 기존 콜백 호환성을 검사해요."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.contact import terms
from src.simulation.two_body import build_problem


class ForceOnlyChecks(unittest.TestCase):
    """전체 야코비안 없이도 동일한 접촉 에너지와 힘을 구하는지 검사해요."""

    @classmethod
    def setUpClass(cls):
        """작은 수치 검사의 스레드 수를 일정하게 유지해요."""
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.pool = threadpool_limits(1)

    @classmethod
    def tearDownClass(cls):
        """검사 전의 스레드 설정을 복구해요."""
        cls.pool.restore_original_limits()
        torch.set_num_threads(cls.threads)

    def test_chunked_force_matches_independent_energy_with_constants_and_unused_dofs(self):
        """부분 묶음과 상수 간격 및 사용하지 않는 자유도를 포함해 기준 에너지 미분과 비교해요."""
        rng = np.random.default_rng(701)
        state = rng.normal(scale=.003, size=9)
        matrix = torch.tensor(rng.normal(scale=.01, size=(81, 4)), dtype=torch.float64)
        base = torch.tensor(np.where(np.arange(81) % 5 == 0, .3, .06), dtype=torch.float64)
        weights = np.linspace(.8, 1.2, 83)

        def gaps(x):
            r"""일부 자유도에 의존하는 비선형 간격과 두 상수 간격을 만들어요.

            $$s_i=b_i+a_i^Tx+0.01(a_i^Tx)^2$$
            """
            # 자유도의 선형 조합을 구해요: $$v_i=a_i^Tx$$
            linear = matrix @ x[:4]
            # 간격의 이차 성분을 구해요: $$q_i=0.01v_i^2$$
            quadratic = .01 * linear.square()
            # 비선형 접촉 간격을 구해요: $$s_i=b_i+v_i+q_i$$
            values = base + linear + quadratic
            return torch.cat((values, x.new_tensor([.1, .35])))

        x = torch.tensor(state, dtype=torch.float64, requires_grad=True)
        s = gaps(x)
        active = s < .2
        w = torch.tensor(weights, dtype=torch.float64)
        # 독립적인 전체 스칼라 장벽 에너지를 구해요: $$E=-\sum_{s_i<\hat d}w_i(s_i-\hat d)^2\log(s_i/\hat d)$$
        energy = torch.sum(-w[active] * (s[active] - .2).square() * torch.log(s[active] / .2))
        expected = torch.autograd.grad(energy, x)[0].numpy()
        calls = []
        cache = {}

        def selected(value, ids):
            """같은 그래프를 공유하는 선택 콜백에서도 묶음별 역전파를 지원하는지 확인해요."""
            calls.append(ids.detach().numpy().copy())
            if 'values' not in cache:
                cache['values'] = gaps(value)
            return cache['values'][ids]

        with patch.object(terms, '_gap_jacobian', side_effect=AssertionError('전체 야코비안을 만들면 안 돼요.')):
            with patch.object(terms, '_jacobian_rows', side_effect=AssertionError('야코비안 행을 만들면 안 돼요.')):
                actual = terms.multiple_contact_terms(state, gaps, weights, compute_hessian=False,
                    gaps_subset_autodiff=selected, gaps_value_autodiff=gaps)
        self.assertAlmostEqual(actual['energy'], float(energy.detach()), places=14)
        np.testing.assert_allclose(actual['gradient'], expected, rtol=2e-12, atol=2e-14)
        np.testing.assert_array_equal(actual['gradient'][4:], 0.)
        np.testing.assert_array_equal(np.concatenate(calls), np.flatnonzero(s.detach().numpy() < .2))
        self.assertEqual([len(ids) for ids in calls], [32, 32, 1])
        self.assertIsNone(actual['hessian'])

    def test_curved_two_body_dense_sparse_and_method_results_match_full_force_graph(self):
        """움직이는 법선을 포함한 곡면에서 조밀 및 희소 힘 계산이 기존 전체 그래프와 일치해요."""
        problem = build_problem(n=None, nx=5, ny=5, nz=3)
        state = problem['initial'] + np.random.default_rng(706).normal(scale=3e-4, size=len(problem['initial']))
        gaps = problem['gaps_ad']
        weights = problem['weights']
        for method in ('full', 'gn', 'filtered_gn'):
            with self.subTest(method=method):
                expected = terms.multiple_contact_terms(state, gaps, weights,
                    method=method, compute_hessian=False)
                actual = terms.multiple_contact_terms(state, gaps, weights, method=method,
                    compute_hessian=False, gaps_subset_autodiff=gaps, gaps_value_autodiff=gaps)
                sparse = terms.multiple_contact_terms_sparse(state, gaps, weights,
                    problem['contact_dofs'][::-1].copy(), method=method, compute_hessian=False,
                    gaps_subset_autodiff=gaps, gaps_value_autodiff=gaps)
                for result in (actual, sparse):
                    self.assertAlmostEqual(result['energy'], expected['energy'], places=14)
                    np.testing.assert_allclose(result['gradient'], expected['gradient'], rtol=3e-11, atol=3e-13)
                    np.testing.assert_allclose(result['gaps'], expected['gaps'], rtol=1e-13, atol=1e-14)
                    self.assertEqual(result['filter_count'], expected['filter_count'])
                    self.assertIsNone(result['hessian'])

    def test_no_opt_in_and_selected_callback_preserve_inner_autograd(self):
        """명시한 두 콜백이 없으면 기존 경로를 유지하고 선택 콜백 내부의 자동미분도 허용해요."""
        state = np.array([.02, -.03, .04])

        def values(x):
            r"""좌표별 제곱으로 양의 간격을 만들어요.

            $$s_i=0.1+x_i^2$$
            """
            # 접촉 간격을 구해요: $$s_i=0.1+x_i^2$$
            return .1 + x.square()

        def internal_ad(x, ids=None):
            r"""내부 자동미분으로 같은 간격을 계산하는 콜백을 만들어요.

            $$\nabla_x\left(\frac12 x^Tx\right)=x$$
            """
            # 좌표를 복원할 이차 함수를 계산해요: $$f=\frac12 x^Tx$$
            scalar = .5 * x.square().sum()
            # 내부 역전파로 좌표를 복원해요: $$\nabla_x f=x$$
            recovered = torch.autograd.grad(scalar, x, create_graph=True)[0]
            result = values(recovered)
            return result if ids is None else result[ids]

        def unused(*args):
            """명시적으로 함께 선택하지 않은 콜백이 호출되면 검사에 실패해요."""
            raise AssertionError('두 콜백을 함께 지정한 경우에만 선택 경로를 사용해요.')

        expected = terms.multiple_contact_terms(state, values, np.ones(3), compute_hessian=False)
        for callbacks in ({'gaps_subset_autodiff': unused}, {'gaps_value_autodiff': unused},
                          {'gaps_subset_autodiff': internal_ad, 'gaps_value_autodiff': values}):
            actual = terms.multiple_contact_terms(state, internal_ad, np.ones(3),
                compute_hessian=False, **callbacks)
            self.assertAlmostEqual(actual['energy'], expected['energy'], places=14)
            np.testing.assert_allclose(actual['gradient'], expected['gradient'], rtol=1e-12, atol=1e-14)

    def test_invalid_all_gap_values_and_subset_shapes_are_rejected(self):
        """비활성점의 잘못된 간격까지 검사하고 선택 콜백의 출력 크기를 확인해요."""
        state = np.zeros(2)

        def gaps(x):
            """일정한 양의 간격을 두 개 반환해요."""
            return x + .1

        for invalid in (float('nan'), 0., -.1):
            calls = []

            def invalid_values(x):
                """전체 검증에 실패하는 간격을 포함해 반환해요."""
                return torch.stack((x[0] + .1, x.new_tensor(invalid)))

            def selected(x, ids):
                """검증 실패 이후에 선택 평가가 호출되는지 기록해요."""
                calls.append(ids)
                return gaps(x)[ids]

            with self.assertRaises(ValueError):
                terms.multiple_contact_terms(state, gaps, np.ones(2), compute_hessian=False,
                    gaps_value_autodiff=invalid_values, gaps_subset_autodiff=selected)
            self.assertEqual(calls, [])
        for selected in (lambda x, ids: gaps(x)[:1], lambda x, ids: gaps(x)[None, :],
                         lambda x, ids: torch.zeros(len(ids), dtype=x.dtype)):
            with self.assertRaises(ValueError):
                terms.multiple_contact_terms(state, gaps, np.ones(2), compute_hessian=False,
                    gaps_value_autodiff=gaps, gaps_subset_autodiff=selected)

    def test_constant_unused_and_inactive_gradients_are_zero(self):
        """상수 간격이나 상태와 무관한 간격 및 비활성점의 힘은 영으로 유지해요."""
        state = np.zeros(3)
        independent = torch.tensor([.12], dtype=torch.float64, requires_grad=True)
        for constants in ((.1, .15), (.3, .35)):
            def gaps(x):
                """상태와 무관한 두 간격을 반환해요."""
                return x.new_tensor(constants)

            def selected(x, ids):
                """선택된 상수 간격을 반환해요."""
                return gaps(x)[ids]

            actual = terms.multiple_contact_terms(state, gaps, np.ones(2), compute_hessian=False,
                gaps_value_autodiff=gaps, gaps_subset_autodiff=selected)
            np.testing.assert_array_equal(actual['gradient'], 0.)
        actual = terms.multiple_contact_terms(state, lambda x: independent, [1.], compute_hessian=False,
            gaps_value_autodiff=lambda x: independent.detach(),
            gaps_subset_autodiff=lambda x, ids: independent[ids])
        np.testing.assert_array_equal(actual['gradient'], 0.)

    def test_matrix_state_keeps_legacy_force_gradient_shape(self):
        """행렬 모양의 상태에는 기존 힘 계산을 사용해 같은 모양의 기울기를 반환해요."""
        state = np.array([[.01, -.02], [.015, .03]])

        def gaps(x):
            r"""행렬 상태의 각 성분에서 일차원 간격 배열을 계산해요.

            $$s_i=0.1+(\operatorname{vec}X)_i^2$$
            """
            values = x.reshape(-1)
            # 상태 성분별 접촉 간격을 구해요: $$s_i=0.1+(\operatorname{vec}X)_i^2$$
            return .1 + values.square()

        def unused(*args):
            """행렬 상태에서 일차원 전용 최적화가 호출되면 검사에 실패해요."""
            raise AssertionError('행렬 상태는 기존 힘 계산 경로를 유지해요.')

        expected = terms.multiple_contact_terms(state, gaps, np.ones(state.size), compute_hessian=False)
        actual = terms.multiple_contact_terms(state, gaps, np.ones(state.size), compute_hessian=False,
            gaps_value_autodiff=unused, gaps_subset_autodiff=unused)
        self.assertEqual(actual['gradient'].shape, state.shape)
        self.assertEqual(actual['energy'], expected['energy'])
        np.testing.assert_array_equal(actual['gradient'], expected['gradient'])
        np.testing.assert_array_equal(actual['gaps'], expected['gaps'])
        self.assertIsNone(actual['hessian'])


if __name__ == '__main__':
    unittest.main()
