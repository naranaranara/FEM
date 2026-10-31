"""선택적인 네이티브 RIMLS 미분과 움직이는 표면의 Full Newton 동등성을 검사해요."""
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.geometry import native_rimls, rimls
from src.geometry.surface import make_surface_sampler


def _inputs(count=7):
    """작은 곡면의 좌표와 법선을 재현 가능한 배정밀도 입력으로 만들어요."""
    rng = np.random.default_rng(816)
    queries = rng.normal(scale=.08, size=(2, 3))
    points = rng.normal(scale=.2, size=(count, 3))
    normals = rng.normal(scale=.2, size=(count, 3)) + [0., 0., 1.]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return queries, points, normals


@unittest.skipUnless(native_rimls.available(), '선택적인 C++ 확장이 없어요.')
class NativeKernelChecks(unittest.TestCase):
    """C++ 편미분을 독립적인 일반 텐서 계산 그래프와 비교해요."""

    def setUp(self):
        """작은 수치 검사를 한 스레드에서 실행해요."""
        torch.set_num_threads(1)

    def _compare_vjp_hvp(self, arrays, scales):
        """좌표와 법선 및 역전파 가중치의 혼합 이차 미분을 비교해요."""
        rng = np.random.default_rng(27)
        inputs = tuple(torch.tensor(a, requires_grad=True) for a in arrays)
        cotangent = torch.tensor([.7, -.3], dtype=torch.float64, requires_grad=True)
        vectors = tuple(rng.normal(size=a.shape) for a in arrays)
        values = rimls._rimls_values_reference(*inputs, *scales)
        first = torch.autograd.grad(values @ cotangent, inputs, create_graph=True)
        directional = sum(torch.sum(g * torch.tensor(v)) for g, v in zip(first, vectors))
        expected = torch.autograd.grad(directional, (*inputs, cotangent))
        vjp = tuple(np.empty_like(a) for a in arrays)
        hvp = tuple(np.empty_like(a) for a in (*arrays, cotangent.detach().numpy()))
        native_rimls._rimls_native.vjp(*arrays, cotangent.detach().numpy(), *vjp, *scales)
        native_rimls._rimls_native.hvp(*arrays, cotangent.detach().numpy(), *vectors, *hvp, *scales)
        for actual, reference in zip(vjp, first):
            np.testing.assert_allclose(actual, reference.detach().numpy(), rtol=3e-11, atol=3e-13)
        for actual, reference in zip(hvp, expected):
            np.testing.assert_allclose(actual, reference.detach().numpy(), rtol=4e-10, atol=4e-12)

    def test_exact_vjp_and_hvp_with_moving_normals_and_scalar_scales(self):
        """법선까지 변하는 방향과 서로 다른 강건 척도에서 이차 미분을 검사해요."""
        for count, scales in ((7, (1.5, .5, .5)), (48, (1.1, .35, .7))):
            with self.subTest(count=count, scales=scales):
                self._compare_vjp_hvp(_inputs(count), scales)

    def test_support_boundary_both_sides_preserve_second_derivatives(self):
        """지지 경계의 안쪽과 경계 및 바깥쪽에서 이차 미분을 비교해요."""
        queries, points, normals = _inputs()
        queries[0] = 0.
        for offset in (-1e-7, 0., 1e-7):
            with self.subTest(offset=offset):
                points[0] = [1. + offset, 0., 0.]
                self._compare_vjp_hvp((queries, points, normals), (1., .5, .5))

    def test_native_buffer_validation_and_numerical_errors(self):
        """잘못된 버퍼와 지지가 사라지는 입력이 명확한 오류를 반환하는지 검사해요."""
        inputs = _inputs()
        cotangent = np.ones(2)
        outputs = tuple(np.empty_like(a) for a in inputs)
        kernel = native_rimls._rimls_native.vjp
        with self.assertRaises(TypeError):
            kernel(inputs[0].astype(np.float32), *inputs[1:], cotangent, *outputs, 1.5, .5, .5)
        with self.assertRaises((ValueError, BufferError)):
            kernel(inputs[0][:, ::-1], *inputs[1:], cotangent, *outputs, 1.5, .5, .5)
        readonly = outputs[0].copy()
        readonly.flags.writeable = False
        with self.assertRaises((ValueError, BufferError)):
            kernel(*inputs, cotangent, readonly, *outputs[1:], 1.5, .5, .5)
        with self.assertRaisesRegex(ValueError, 'sizes'):
            kernel(*inputs, cotangent, np.empty(2), *outputs[1:], 1.5, .5, .5)
        with self.assertRaisesRegex(ValueError, 'outside kernel support'):
            kernel(inputs[0] + 100., *inputs[1:], cotangent, *outputs, 1.5, .5, .5)
        with self.assertRaisesRegex(ValueError, 'vanishing robust weights'):
            kernel(*inputs, cotangent, *outputs, 1.5, .5, 1e-20)
        directions = tuple(np.zeros_like(a) for a in inputs)
        with self.assertRaisesRegex(ValueError, 'sizes'):
            native_rimls._rimls_native.hvp(*inputs, cotangent, np.empty(1), *directions[1:],
                                           *outputs, np.empty_like(cotangent), 1.5, .5, .5)


class NativeFallbackChecks(unittest.TestCase):
    """확장이 없거나 지원하지 않는 입력도 기존 미분 동작을 유지하는지 검사해요."""

    def setUp(self):
        """일관된 작은 입력과 스레드 수를 설정해요."""
        torch.set_num_threads(1)
        self.inputs = tuple(torch.tensor(a, requires_grad=True) for a in _inputs())

    def test_missing_extension_preserves_values_gradient_and_hessian(self):
        """컴파일러 없이 설치한 환경에서 일반 그래프의 결과를 유지하는지 검사해요."""
        reference = lambda *x: rimls._rimls_values_reference(*x, 1.5).square().sum()
        actual = lambda *x: native_rimls.native_values(*x, 1.5).square().sum()
        with patch.object(native_rimls, '_rimls_native', None):
            self.assertFalse(native_rimls.available())
            np.testing.assert_array_equal(actual(*self.inputs).detach(), reference(*self.inputs).detach())
            expected = torch.autograd.functional.hessian(reference, self.inputs)
            received = torch.autograd.functional.hessian(actual, self.inputs)
        for row_a, row_b in zip(received, expected):
            for a, b in zip(row_a, row_b):
                np.testing.assert_array_equal(a, b)

    def test_tensor_scales_and_float32_keep_differentiable_reference(self):
        """척도 텐서의 미분과 단정밀도 입력을 일반 경로로 보존해요."""
        scales = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (1.5, .5, .5))
        with patch.object(native_rimls._NativeValue, 'apply', side_effect=AssertionError('reference required')):
            a = native_rimls.native_values(*self.inputs, *scales).sum()
            b = rimls._rimls_values_reference(*self.inputs, *scales).sum()
            for first, second in zip(torch.autograd.grad(a, scales), torch.autograd.grad(b, scales)):
                np.testing.assert_array_equal(first, second)
            single = tuple(value.float() for value in self.inputs)
            np.testing.assert_array_equal(native_rimls.native_values(*single, 1.5).detach(),
                                          rimls._rimls_values_reference(*single, 1.5).detach())

    def test_other_dimensions_and_broadcast_normals_keep_reference(self):
        """네이티브 기준보다 큰 질의에서도 다른 좌표 차원과 법선 방송을 보존해요."""
        rng = np.random.default_rng(64)
        for dimension, normal_count in ((2, 65), (3, 1)):
            with self.subTest(dimension=dimension, normal_count=normal_count):
                arrays = (rng.normal(scale=.08, size=(33, dimension)),
                          rng.normal(scale=.2, size=(65, dimension)),
                          rng.normal(scale=.08, size=(normal_count, dimension)))
                arrays[2][:, -1] += 1.
                inputs = tuple(torch.tensor(a, requires_grad=True) for a in arrays)
                with patch.object(native_rimls._NativeValue, 'apply',
                                  side_effect=AssertionError('reference shape required')):
                    with native_rimls.native_full_mode():
                        actual = rimls.rimls_values_torch(*inputs, 1.5)
                expected = rimls._rimls_values_reference(*inputs, 1.5)
                np.testing.assert_array_equal(actual.detach(), expected.detach())
                actual_gradient = torch.autograd.grad(actual.square().sum(), inputs)
                expected_gradient = torch.autograd.grad(expected.square().sum(), inputs)
                for first, second in zip(actual_gradient, expected_gradient):
                    np.testing.assert_array_equal(first, second)

    def test_forward_ad_and_function_transforms_preserve_reference(self):
        """순방향 미분과 함수 변환에서 지원되는 일반 그래프로 돌아가는지 검사해요."""
        actual = lambda *x: native_rimls.native_values(*x, 1.5)
        reference = lambda *x: rimls._rimls_values_reference(*x, 1.5)
        with patch.object(native_rimls._NativeValue, 'apply', side_effect=AssertionError('reference required')):
            if hasattr(torch.autograd, 'forward_ad'):
                with torch.autograd.forward_ad.dual_level():
                    dual = torch.autograd.forward_ad.make_dual(self.inputs[0], torch.ones_like(self.inputs[0]))
                    first = torch.autograd.forward_ad.unpack_dual(actual(dual, *self.inputs[1:]))
                    second = torch.autograd.forward_ad.unpack_dual(reference(dual, *self.inputs[1:]))
                    np.testing.assert_array_equal(first.tangent.detach(), second.tangent.detach())
            if hasattr(torch, 'func'):
                first = torch.func.jacrev(actual, argnums=(0, 1, 2))(*self.inputs)
                second = torch.func.jacrev(reference, argnums=(0, 1, 2))(*self.inputs)
                for a, b in zip(first, second):
                    np.testing.assert_array_equal(a.detach(), b.detach())

    def test_full_context_restores_nested_and_failed_calls(self):
        """중첩 실행과 예외 이후에도 Full Newton의 선택 범위가 복원되는지 검사해요."""
        self.assertFalse(native_rimls._full_mode.get())
        with native_rimls.native_full_mode():
            self.assertTrue(native_rimls._full_mode.get())
            with native_rimls.native_full_mode(False):
                self.assertFalse(native_rimls._full_mode.get())
            self.assertTrue(native_rimls._full_mode.get())
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with native_rimls.native_full_mode():
                raise RuntimeError('test')
        self.assertFalse(native_rimls._full_mode.get())


@unittest.skipUnless(native_rimls.available(), '선택적인 C++ 확장이 없어요.')
class NativeSurfaceChecks(unittest.TestCase):
    """법선 재계산을 포함한 두 곡면의 장벽 에너지와 완전한 헤시안을 검사해요."""

    def setUp(self):
        """비평면 삼각형 두 개씩으로 이루어진 서로 마주 보는 표면을 만들어요."""
        torch.set_num_threads(1)
        lower = np.array([[0., 0., 0.], [.5, 0., .015], [.5, .5, -.01], [0., .5, .01]])
        upper = lower + [.015, -.01, .11]
        upper[:, 2] += [.005, -.003, .002, -.004]
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        self.lower_sampler = make_surface_sampler(lower, faces)[0]
        self.upper_sampler = make_surface_sampler(upper, faces[:, ::-1].copy())[0]
        self.state = torch.tensor(np.vstack((lower, upper)).ravel(), requires_grad=True)

    def _energy(self, state, function):
        r"""움직이는 두 곡면의 RIMLS 간격으로 접촉 장벽 에너지를 계산해요.

        $$E=-\sum_i(s_i-\hat d)^2\log(s_i/\hat d)$$
        """
        vertices = state.reshape(8, 3)
        pl, nl = self.lower_sampler(vertices[:4])
        pu, nu = self.upper_sampler(vertices[4:])
        gaps = torch.cat((function(pu, pl, nl, 1.5), function(pl, pu, nu, 1.5)))
        # 접촉 장벽 에너지를 합해요: $$E=-\sum_i(s_i-\hat d)^2\log(s_i/\hat d)$$
        return -torch.sum((gaps - .2).square() * torch.log(gaps / .2))

    def test_curved_surface_energy_gradient_hessian_and_finite_difference(self):
        """법선의 일차 및 이차 변화가 포함된 에너지 미분을 독립적으로 비교해요."""
        reference = lambda x: self._energy(x, rimls._rimls_values_reference)
        actual = lambda x: self._energy(x, native_rimls.native_values)
        a, b = actual(self.state), reference(self.state)
        np.testing.assert_array_equal(a.detach(), b.detach())
        ga, gb = torch.autograd.grad(a, self.state)[0], torch.autograd.grad(b, self.state)[0]
        np.testing.assert_allclose(ga, gb, rtol=3e-11, atol=3e-13)
        ha = torch.autograd.functional.hessian(actual, self.state)
        hb = torch.autograd.functional.hessian(reference, self.state)
        np.testing.assert_allclose(ha, hb, rtol=4e-10, atol=4e-12)
        direction = torch.tensor(np.random.default_rng(45).normal(size=len(self.state)))
        direction /= torch.linalg.vector_norm(direction)
        step = 1e-5
        plus = (self.state.detach() + step * direction).requires_grad_()
        minus = (self.state.detach() - step * direction).requires_grad_()
        gp = torch.autograd.grad(reference(plus), plus)[0]
        gm = torch.autograd.grad(reference(minus), minus)[0]
        np.testing.assert_allclose(ha @ direction, (gp - gm) / (2 * step), rtol=3e-7, atol=3e-9)

    def test_higher_derivatives_keep_the_tensor_graph(self):
        """이차 미분 이후에도 필요한 계산 그래프가 유지되는지 검사해요."""
        direction = torch.tensor(np.random.default_rng(81).normal(size=len(self.state)))
        results = []
        for function in (native_rimls.native_values, rimls._rimls_values_reference):
            value = self._energy(self.state, function)
            gradient = torch.autograd.grad(value, self.state, create_graph=True)[0]
            second = torch.autograd.grad(gradient @ direction, self.state, create_graph=True)[0]
            results.append(torch.autograd.grad(second @ direction, self.state)[0])
        np.testing.assert_allclose(*results, rtol=2e-9, atol=2e-10)


@unittest.skipUnless(native_rimls.available(), '선택적인 C++ 확장이 없어요.')
class NativeFullIntegrationChecks(unittest.TestCase):
    """실제 접촉 조립에서 Full Newton만 네이티브 경로를 사용하는지 검사해요."""

    @unittest.skipUnless('is_grads_batched' in inspect.signature(torch.autograd.grad).parameters,
                         '현재 PyTorch가 배치 역전파를 제공하지 않아요.')
    def test_generic_callbacks_can_compute_internal_batched_derivatives(self):
        """일반 간격 함수 내부의 배치 일차 및 이차 미분을 보존해요."""
        from src.contact.terms import multiple_contact_terms

        torch.set_num_threads(1)
        rng = np.random.default_rng(38)
        queries = torch.tensor(rng.normal(scale=.08, size=(33, 3)))
        points, normals = (torch.tensor(a) for a in _inputs(65)[1:])

        def callback(function, order):
            """지정한 RIMLS 함수의 내부 미분을 사용하는 간격 함수를 만들어요."""
            def gaps(state):
                r"""내부 미분 행으로 양의 시험 간격을 만들어요.

                $$s_i=0.1+10^{-3}\|D_i\|^2+10^{-3}x_0$$
                """
                values = function(queries + state, points, normals, 1.5)
                if order == 1:
                    derivative = torch.autograd.grad(values, state,
                        grad_outputs=torch.eye(len(values), dtype=state.dtype),
                        is_grads_batched=True, create_graph=True)[0]
                else:
                    first = torch.autograd.grad(values.sum(), state, create_graph=True)[0]
                    derivative = torch.autograd.grad(first, state,
                        grad_outputs=torch.eye(len(state), dtype=state.dtype),
                        is_grads_batched=True)[0]
                # 미분 행의 크기에 따라 시험 간격을 정해요: $$s_i=0.1+10^{-3}\|D_i\|^2+10^{-3}x_0$$
                return .1 + .001 * derivative.square().sum(dim=1) + .001 * state[0]
            return gaps

        for order in (1, 2):
            with self.subTest(order=order):
                weights = np.ones(len(queries) if order == 1 else 3)
                with patch.object(native_rimls._NativeValue, 'apply',
                                  wraps=native_rimls._NativeValue.apply) as native:
                    actual = multiple_contact_terms(np.zeros(3),
                        callback(rimls.rimls_values_torch, order), weights, method='full')
                    self.assertGreater(native.call_count, 0)
                expected = multiple_contact_terms(np.zeros(3),
                    callback(rimls._rimls_values_reference, order), weights, method='full')
                self.assertAlmostEqual(actual['energy'], expected['energy'], places=13)
                for key in ('gaps', 'gradient', 'hessian'):
                    np.testing.assert_allclose(actual[key], expected[key], rtol=2e-9, atol=2e-11)

    def test_full_contact_matches_reference_and_other_methods_keep_existing_path(self):
        """변형된 메시의 완전한 헤시안과 확장 부재 시의 복귀를 함께 검사해요."""
        from src.simulation import two_body

        torch.set_num_threads(1)
        problem = two_body.build_problem(n=4)
        state = problem['initial'] + np.random.default_rng(72).normal(scale=3e-4, size=len(problem['initial']))
        with patch.object(native_rimls._NativeValue, 'apply', wraps=native_rimls._NativeValue.apply) as native:
            actual = problem['contact'](state, 'full')
            self.assertGreater(native.call_count, 0)
        with patch.object(two_body, 'rimls_values_torch', rimls._rimls_values_reference):
            expected = problem['contact'](state, 'full')
        with patch.object(native_rimls, '_rimls_native', None):
            fallback = problem['contact'](state, 'full')
        for result in (actual, fallback):
            self.assertEqual(result['energy'], expected['energy'])
            np.testing.assert_allclose(result['gradient'], expected['gradient'], rtol=3e-11, atol=3e-13)
            np.testing.assert_allclose(result['hessian'].toarray(), expected['hessian'].toarray(),
                                       rtol=4e-10, atol=4e-12)
        with patch.object(native_rimls._NativeValue, 'apply', side_effect=AssertionError('Full Hessian only')):
            problem['contact'](state, 'gn')
            problem['contact'](state, 'filtered_gn')
            problem['contact'](state, 'full', compute_hessian=False)
            problem['gaps_ad'](torch.tensor(state, requires_grad=True))


if __name__ == '__main__':
    unittest.main()
