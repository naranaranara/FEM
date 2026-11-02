"""선택 RIMLS 커널의 값과 일차 및 이차 미분이 일반 계산과 같은지 검사해요."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from src.geometry import rimls


class SelectedRIMLSChecks(unittest.TestCase):
    """움직이는 두 표면과 지지 경계 및 자동미분 모드의 동등성을 검사해요."""

    def setUp(self):
        """일관된 배정밀도 입력과 한 스레드 환경을 만들어요."""
        torch.set_num_threads(1)
        rng = np.random.default_rng(29)
        self.queries = torch.tensor(rng.normal(size=(2, 3)) * .1, dtype=torch.float64)
        self.points = torch.tensor(rng.normal(size=(48, 3)) * .2, dtype=torch.float64)
        normals = rng.normal(size=(48, 3)) * .1 + [0., 0., 1.]
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        self.normals = torch.tensor(normals, dtype=torch.float64)

    def test_values_and_all_input_derivatives_match_reference(self):
        """선택 커널의 좌표와 법선에 대한 야코비안 및 헤시안을 비교해요."""
        arguments = self.queries, self.points, self.normals

        def actual(*values):
            """최적화되는 크기의 RIMLS 질의를 평가해요."""
            return rimls.rimls_values_torch(*values, 1.5)

        def reference(*values):
            """일반 계산 그래프로 같은 질의를 평가해요."""
            return rimls._rimls_values_reference(*values, 1.5)

        np.testing.assert_array_equal(actual(*arguments), reference(*arguments))
        for a, b in zip(torch.autograd.functional.jacobian(actual, arguments, vectorize=True),
                        torch.autograd.functional.jacobian(reference, arguments, vectorize=True)):
            np.testing.assert_allclose(a, b, rtol=3e-12, atol=3e-14)
        coefficients = torch.tensor([.7, -.3], dtype=torch.float64)
        ha = torch.autograd.functional.hessian(lambda *x: actual(*x) @ coefficients, arguments, vectorize=True)
        hb = torch.autograd.functional.hessian(lambda *x: reference(*x) @ coefficients, arguments, vectorize=True)
        for row_a, row_b in zip(ha, hb):
            for a, b in zip(row_a, row_b):
                np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-12)

    def test_finite_difference_gradient_and_double_backward(self):
        """독립적인 유한차분으로 일차 및 이차 역전파를 검사해요."""
        arguments = tuple(x.clone().requires_grad_() for x in (self.queries, self.points, self.normals))
        function = lambda *x: rimls.rimls_values_torch(*x, 1.5)
        self.assertTrue(torch.autograd.gradcheck(function, arguments, fast_mode=True))
        self.assertTrue(torch.autograd.gradgradcheck(function, arguments, fast_mode=True))

    def test_kernel_support_boundary_and_zero_support_errors(self):
        """지지 경계의 양쪽에서 값과 미분 및 기존 오류를 유지하는지 검사해요."""
        points = self.points.clone()
        points[0] = torch.tensor([1., 0., 0.], dtype=torch.float64)
        query = torch.zeros((1, 3), dtype=torch.float64)
        arguments = query, points, self.normals
        for offset in (-1e-7, 0., 1e-7):
            points[0, 0] = 1. + offset
            fast = lambda *x: rimls.rimls_values_torch(*x, 1.)
            slow = lambda *x: rimls._rimls_values_reference(*x, 1.)
            for a, b in zip(torch.autograd.functional.jacobian(fast, arguments),
                            torch.autograd.functional.jacobian(slow, arguments)):
                np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-14)
        with self.assertRaisesRegex(ValueError, 'outside kernel support'):
            rimls.rimls_values_torch(query + 100., points, self.normals, 1.)
        with self.assertRaisesRegex(ValueError, 'vanishing robust weights'):
            rimls.rimls_values_torch(query, points, self.normals * 100., 1., sigma_n=1e-20)
        for invalid in (0., -1.):
            with self.assertRaises(ValueError):
                rimls.rimls_values_torch(query, points, self.normals, invalid)
        self.assertEqual(tuple(rimls.rimls_values_torch(query[:0], points, self.normals, 1.).shape), (0,))

    def test_scalar_defaults_and_tensor_scale_derivatives(self):
        """기존 강건 척도 기본값과 텐서 매개변수의 미분을 보존하는지 검사해요."""
        actual = rimls.rimls_values_torch(self.queries, self.points, self.normals, 1.5)
        scalar = torch.stack([rimls.rimls_value_torch(q, self.points, self.normals, 1.5, sigma_n=.5)
                              for q in self.queries])
        np.testing.assert_allclose(actual, scalar, rtol=2e-14, atol=2e-15)
        scales = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (1.5, .5, .5))
        with patch.object(rimls._SelectedRIMLS, 'apply', side_effect=AssertionError('tensor scale fallback required')):
            answer = rimls.rimls_values_torch(self.queries, self.points, self.normals, *scales).sum()
        reference = rimls._rimls_values_reference(self.queries, self.points, self.normals, *scales).sum()
        a = torch.autograd.grad(answer, scales)
        b = torch.autograd.grad(reference, scales)
        for first, second in zip(a, b):
            np.testing.assert_array_equal(first, second)

    def test_dispatch_preserves_small_and_full_graphs(self):
        """작은 표면과 전체 질의는 일반 그래프를 유지하는지 검사해요."""
        with patch.object(rimls._SelectedRIMLS, 'apply', wraps=rimls._SelectedRIMLS.apply) as selected:
            rimls.rimls_values_torch(self.queries, self.points, self.normals, 1.5)
            self.assertEqual(selected.call_count, 1)
            rimls.rimls_values_torch(self.queries, self.points[:20], self.normals[:20], 1.5)
            rimls.rimls_values_torch(self.queries.repeat(20, 1), self.points, self.normals, 1.5)
            self.assertEqual(selected.call_count, 1)

    def test_value_only_chunks_preserve_values_and_bound_pair_arrays(self):
        """값 전용 평가가 지지점을 생략하지 않고 질의 묶음의 크기만 제한하는지 검사해요."""
        queries = self.queries.repeat(200, 1)
        points = self.points.repeat(5, 1)
        normals = self.normals.repeat(5, 1)
        reference = rimls._rimls_values_reference(queries, points, normals, 1.5)
        original = rimls._rimls_values_reference
        sizes = []

        def counted(q, *args):
            """한 번에 평가하는 질의 수를 기록해요."""
            sizes.append(len(q))
            return original(q, *args)

        from src.geometry.local_rimls import local_rimls_mode  # ##r## legacy chunk path only
        with local_rimls_mode(False), patch.object(rimls, 'try_native_values', return_value=None), patch.object(
                rimls, '_rimls_values_reference', side_effect=counted):
            actual = rimls.rimls_values_torch(queries, points, normals, 1.5)
            with torch.no_grad():
                explicit = rimls.rimls_values_torch(queries.requires_grad_(), points, normals, 1.5)
        np.testing.assert_array_equal(actual, reference)
        np.testing.assert_array_equal(explicit, reference)
        self.assertGreater(len(sizes), 2)
        self.assertLessEqual(max(sizes) * len(points), 65536)

    def test_forward_mode_and_function_transforms(self):
        """순방향 미분과 함수 변환이 일반 그래프를 통해 동작하는지 검사해요."""
        arguments = self.queries, self.points, self.normals
        function = lambda *x: rimls.rimls_values_torch(*x, 1.5)
        reference = lambda *x: rimls._rimls_values_reference(*x, 1.5)
        if hasattr(torch.autograd, 'forward_ad'):
            tangent = torch.ones_like(self.queries)
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(self.queries, tangent)
                a = torch.autograd.forward_ad.unpack_dual(function(dual, self.points, self.normals))
                b = torch.autograd.forward_ad.unpack_dual(reference(dual, self.points, self.normals))
                np.testing.assert_array_equal(a.tangent, b.tangent)
        if hasattr(torch, 'func'):
            for a, b in zip(torch.func.jacrev(function, argnums=(0, 1, 2))(*arguments),
                            torch.func.jacrev(reference, argnums=(0, 1, 2))(*arguments)):
                np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
