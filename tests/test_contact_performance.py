"""접촉 미분 최적화가 곡률과 예외 처리를 유지하는지 검사해요."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.contact import terms
from src.simulation.two_body import build_problem


class ContactDerivativeChecks(unittest.TestCase):
    """배치 역전파와 검증용 그래프 해제의 동등성을 검사해요."""

    def setUp(self):
        """작은 접촉 문제에서 스레드 생성 비용을 제한해요."""
        torch.set_num_threads(1)

    def test_batched_rows_match_scalar_rows_and_constant_outputs(self):
        """비선형 및 상수 출력의 배치 미분을 스칼라 미분과 비교해요."""
        state = torch.tensor([.2, -.1, .3], dtype=torch.float64, requires_grad=True)
        values = torch.cat((state ** 3, state ** 2, state, state[:1] * 0.))
        expected = torch.autograd.functional.jacobian(
            lambda x: torch.cat((x ** 3, x ** 2, x, x[:1] * 0.)), state)
        actual = terms._jacobian_rows(values, state)
        np.testing.assert_allclose(actual.detach(), expected.detach(), atol=1e-15)
        with patch.object(terms, '_BATCHED_GRAD', False):
            fallback = terms._jacobian_rows(values, state)
        np.testing.assert_array_equal(actual.detach(), fallback.detach())
        np.testing.assert_array_equal(
            terms._jacobian_rows(torch.ones(10), state), np.zeros((10, 3)))
        unused = torch.ones(10, dtype=torch.float64, requires_grad=True)
        np.testing.assert_array_equal(terms._jacobian_rows(unused, state), np.zeros((10, 3)))
        self.assertEqual(tuple(terms._jacobian_rows(state[:0], state).shape), (0, 3))

    def test_full_hessian_matches_unbatched_curved_two_body(self):
        """양쪽 표면이 변형된 상태의 접촉 곡률을 기존 미분 방식과 비교해요."""
        problem = build_problem(n=2, initial_gap=.015, closure=.003)
        state = problem['initial'] + np.random.default_rng(15).normal(
            size=len(problem['initial'])) * 1e-4
        arguments = (state, problem['gaps_ad'], problem['weights'])
        actual = terms.multiple_contact_terms(*arguments, method='full')
        with patch.object(terms, '_BATCHED_GRAD', False):
            expected = terms.multiple_contact_terms(*arguments, method='full')
        np.testing.assert_allclose(actual['gradient'], expected['gradient'], rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(actual['hessian'], expected['hessian'], rtol=1e-12, atol=1e-13)

    def test_gn_internal_differentiation_and_sr1_jacobian(self):
        """간격 함수 내부의 자동미분과 SR1 야코비안 재사용을 유지하는지 검사해요."""
        def gaps(state):
            r"""내부 미분으로 접촉 간격을 계산해요.

            $$s_i = \frac{\partial (x^T x)}{\partial x_i} + 0.1$$
            """
            # 미분할 이차 스칼라장을 계산해요: $$f = x^T x$$
            field = (state * state).sum()
            gradient = torch.autograd.grad(field, state, create_graph=True)[0]
            # 스칼라장의 기울기에 양의 기준 간격을 더해요: $$s_i = (\nabla f)_i + 0.1$$
            return gradient[:2] + .1

        def selected(state, indices):
            """요청한 간격을 선택해요."""
            return gaps(state)[indices]

        for method in ('gn', 'filtered_gn'):
            result = terms.multiple_contact_terms(
                np.zeros(3), gaps, [1., 2.], method=method,
                gaps_subset_autodiff=selected, return_gap_data=method == 'gn')
            expected = terms.multiple_contact_terms(
                np.zeros(3), lambda x: 2. * x[:2] + .1, [1., 2.], method=method)
            np.testing.assert_array_equal(result['gradient'], expected['gradient'])
            np.testing.assert_array_equal(result['hessian'], expected['hessian'])
            self.assertEqual(result['energy'], expected['energy'])
            if method == 'gn':
                np.testing.assert_array_equal(result['gap_jacobian'], [[2., 0., 0.], [0., 2., 0.]])
                np.testing.assert_array_equal(result['gap_dofs'], [0, 1, 2])

    def test_batching_rule_failure_uses_scalar_fallback(self):
        """배치 미분을 지원하지 않는 연산은 기존 역전파로 복구하는지 검사해요."""
        state = torch.tensor([.1, .2], dtype=torch.float64, requires_grad=True)
        original = torch.autograd.grad

        def unsupported_batch(*args, **kwargs):
            """배치 역전파에서만 지원하지 않는 연산 오류를 재현해요."""
            if kwargs.get('is_grads_batched'):
                raise RuntimeError('Batching rule not implemented for fixture')
            return original(*args, **kwargs)

        with patch.object(terms, '_BATCHED_GRAD', True), patch(
                'torch.autograd.grad', side_effect=unsupported_batch):
            actual = terms._jacobian_rows(state ** 2, state)
        np.testing.assert_allclose(actual.detach(), np.diag([.2, .4]), atol=1e-15)


if __name__ == '__main__':
    unittest.main()
