"""Full Newton의 미분 배치와 캐시가 접촉 곡률 및 독립 지지 영역을 보존하는지 검사해요."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.contact.terms import multiple_contact_terms, multiple_contact_terms_sparse
from src.geometry import native_rimls, rimls
from src.simulation import two_body


def _reference_terms(state, gaps, weights):
    """공개 접촉 조립과 독립적인 스칼라 에너지에서 기준 미분을 계산해요."""
    values = torch.tensor(state, dtype=torch.float64, requires_grad=True)
    weights_t = torch.tensor(weights, dtype=torch.float64)

    def energy(x):
        r"""활성 접촉점의 기준 장벽 에너지를 합해요.

        $$E=-\sum_{s_i<\hat d}w_i(s_i-\hat d)^2\log(s_i/\hat d)$$
        """
        s = gaps(x)
        selected = s < .2
        # 활성 간격의 장벽 에너지를 합해요: $$E=-\sum_{s_i<\hat d}w_i(s_i-\hat d)^2\log(s_i/\hat d)$$
        return -torch.sum(weights_t[selected] * (s[selected] - .2).square() * torch.log(s[selected] / .2))

    scalar = energy(values)
    gradient = torch.autograd.grad(scalar, values)[0]
    hessian = torch.autograd.functional.hessian(energy, values, vectorize=False)
    return dict(energy=float(scalar.detach()), gradient=gradient.detach().numpy(),
                hessian=hessian.detach().numpy())


class FullHessianAssemblyChecks(unittest.TestCase):
    """분리된 접촉 영역과 비활성 및 상수 간격의 전역 조립을 검사해요."""

    def setUp(self):
        """일관된 작은 검사를 위해 스레드 수를 제한해요."""
        torch.set_num_threads(1)

    def test_disconnected_patches_constants_and_unused_dofs_match_energy(self):
        """부분 배치와 임의 자유도 순서에서도 독립 영역의 곡률을 보존하는지 검사해요."""
        state = np.random.default_rng(918).normal(scale=.01, size=151)
        dofs = np.random.default_rng(921).permutation(137)
        weights = np.array([.7, 1.2, .8, 1.1])

        def gaps(x):
            r"""서로 독립적인 두 접촉 영역과 상수 간격을 만들어요.

            $$\boldsymbol{s}=(0.1+0.03x_2+0.01x_4^2+0.02x_6x_8,\ 0.13+0.02\sin x_{101}+0.01x_{103}^2,\ 0.35,\ 0.14)$$
            """
            # 첫 접촉 영역의 비선형 간격을 구해요: $$s_0=0.1+0.03x_2+0.01x_4^2+0.02x_6x_8$$
            first = .1 + .03 * x[2] + .01 * x[4].square() + .02 * x[6] * x[8]
            # 둘째 접촉 영역의 비선형 간격을 구해요: $$s_1=0.13+0.02\sin x_{101}+0.01x_{103}^2$$
            second = .13 + .02 * torch.sin(x[101]) + .01 * x[103].square()
            return torch.stack((first, second, x.new_tensor(.35), x.new_tensor(.14)))

        expected = _reference_terms(state, gaps, weights)
        dense = multiple_contact_terms(state, gaps, weights, method='full')
        sparse = multiple_contact_terms_sparse(state, gaps, weights, dofs, method='full')
        for actual, hessian in ((dense, dense['hessian']), (sparse, sparse['hessian'].toarray())):
            self.assertAlmostEqual(actual['energy'], expected['energy'], places=14)
            np.testing.assert_allclose(actual['gradient'], expected['gradient'], rtol=2e-12, atol=2e-14)
            np.testing.assert_allclose(hessian, expected['hessian'], rtol=2e-12, atol=2e-14)
            np.testing.assert_array_equal(hessian[np.ix_([2, 4, 6, 8], [101, 103])], 0.)
            unused = np.setdiff1d(np.arange(len(state)), [2, 4, 6, 8, 101, 103])
            np.testing.assert_array_equal(actual['gradient'][unused], 0.)
            np.testing.assert_array_equal(hessian[unused], 0.)


@unittest.skipUnless(native_rimls.available(), '선택적인 C++ 확장이 없어요.')
class NativeHessianAssemblyChecks(unittest.TestCase):
    """움직이는 표면에서 네이티브 미분과 전역 헤시안의 동등성을 검사해요."""

    def setUp(self):
        """수치 비교가 불필요한 스레드 수에 영향을 받지 않도록 해요."""
        torch.set_num_threads(1)

    def test_curved_geometry_partial_batch_dense_sparse_and_missing_extension(self):
        """마지막 미분 배치가 남는 곡면에서 조밀 및 희소 Full Hessian을 비교해요."""
        problem = two_body.build_problem(n=None, nx=5, ny=5, nz=3)
        state = problem['initial'] + np.random.default_rng(504).normal(scale=3e-4, size=len(problem['initial']))
        args = (state, problem['gaps_ad'], problem['weights'])
        dofs = problem['contact_dofs'][::-1].copy()
        with patch.object(two_body, 'rimls_values_torch', rimls._rimls_values_reference):
            expected = _reference_terms(*args)
        dense = multiple_contact_terms(*args, method='full')
        sparse = multiple_contact_terms_sparse(*args, dofs, method='full')
        with patch.object(native_rimls, '_rimls_native', None):
            fallback = multiple_contact_terms_sparse(*args, dofs, method='full')
        for actual, hessian in ((dense, dense['hessian']), (sparse, sparse['hessian'].toarray()),
                                (fallback, fallback['hessian'].toarray())):
            self.assertAlmostEqual(actual['energy'], expected['energy'], places=14)
            np.testing.assert_allclose(actual['gradient'], expected['gradient'], rtol=3e-11, atol=3e-13)
            np.testing.assert_allclose(hessian, expected['hessian'], rtol=4e-10, atol=4e-12)

    def test_old_style_callback_and_missing_batch_api_keep_scalar_result(self):
        """구형 사용자 정의 역전파와 배치 API 부재에서도 기존 접촉 결과를 유지해요."""
        class OldPass(torch.autograd.Function):
            """구형 자동미분 함수의 호출 규약으로 입력을 그대로 전달해요."""

            @staticmethod
            def forward(ctx, value):
                """입력을 복제해 반환해요."""
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                """출력의 미분을 입력에 그대로 전달해요."""
                return gradient

        class OldIdentity(torch.autograd.Function):
            """역전파 내부에서 구형 사용자 정의 함수를 호출해요."""

            @staticmethod
            def forward(ctx, value):
                """입력의 값을 그대로 보존해요."""
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                """이차 미분 중에도 구형 함수가 호출되는 조건을 만들어요."""
                return OldPass.apply(gradient)

        rng = np.random.default_rng(317)
        queries = torch.tensor(np.column_stack((rng.normal(scale=.1, size=(33, 2)), np.full(33, .12))))
        points = torch.tensor(np.column_stack((rng.normal(scale=.2, size=(65, 2)), np.zeros(65))))
        normals = torch.tensor(np.tile([0., 0., 1.], (65, 1)))
        state = np.array([.01, -.02, .005])
        weights = np.linspace(.8, 1.2, len(queries))

        def gaps(x):
            """구형 자동미분 함수를 통과한 변위로 RIMLS 간격을 평가해요."""
            return rimls.rimls_values_torch(queries + OldIdentity.apply(x), points, normals, 1.5)

        actual = multiple_contact_terms(state, gaps, weights, method='full')
        with patch.object(native_rimls, '_BATCH_API', False):
            fallback = multiple_contact_terms(state, gaps, weights, method='full')
        with patch.object(native_rimls, '_rimls_native', None):
            expected = _reference_terms(state, gaps, weights)
        for result in (actual, fallback):
            self.assertAlmostEqual(result['energy'], expected['energy'], places=14)
            np.testing.assert_allclose(result['gradient'], expected['gradient'], rtol=3e-11, atol=3e-13)
            np.testing.assert_allclose(result['hessian'], expected['hessian'], rtol=4e-10, atol=4e-12)

    def test_cache_budget_and_changed_configuration_preserve_curvature(self):
        """메모리 한도와 이전 평가 상태에 관계없이 현재 형상의 곡률을 계산하는지 검사해요."""
        problem = two_body.build_problem(n=4)
        rng = np.random.default_rng(407)
        states = [problem['initial'] + rng.normal(scale=4e-4, size=len(problem['initial'])) for _ in range(2)]
        dofs = problem['contact_dofs']
        for state in (states[0], states[1], states[0]):
            args = (state, problem['gaps_ad'], problem['weights'])
            with patch.object(two_body, 'rimls_values_torch', rimls._rimls_values_reference):
                expected = _reference_terms(*args)
            for budget in (0, 160 * 1024, 64 * 1024 * 1024):
                with self.subTest(budget=budget), patch.object(native_rimls, '_PAIR_CACHE_BYTES', budget):
                    actual = multiple_contact_terms_sparse(*args, dofs, method='full')
                self.assertAlmostEqual(actual['energy'], expected['energy'], places=14)
                np.testing.assert_allclose(actual['gradient'], expected['gradient'], rtol=3e-11, atol=3e-13)
                np.testing.assert_allclose(actual['hessian'].toarray(), expected['hessian'], rtol=4e-10, atol=4e-12)


@unittest.skipUnless(native_rimls.available() and hasattr(native_rimls._rimls_native, 'hvp_batch')
                     and hasattr(native_rimls._rimls_native, 'vjp_batch'),
                     '선택적인 배치 C++ 확장이 없어요.')
class NativeBatchKernelChecks(unittest.TestCase):
    """배치 C++ 미분을 독립적인 텐서 미분과 비교하고 버퍼 경계를 검사해요."""

    def setUp(self):
        """작은 비평면 입력과 서로 다른 미분 방향을 준비해요."""
        torch.set_num_threads(1)
        self.rng = np.random.default_rng(135)
        self.arrays = (self.rng.normal(scale=.08, size=(2, 3)),
                       self.rng.normal(scale=.2, size=(7, 3)),
                       self.rng.normal(scale=.1, size=(7, 3)) + [0., 0., 1.])
        self.scales = 1.4, .35, .7

    def test_fused_and_remainder_batches_match_independent_tensor_derivatives(self):
        """공유 계산과 남은 방향 처리 모두 법선 및 가중치의 혼합 곡률을 보존해요."""
        inputs = tuple(torch.tensor(value, requires_grad=True) for value in self.arrays)
        cotangent = torch.tensor([.7, -.3], dtype=torch.float64, requires_grad=True)
        values = rimls._rimls_values_reference(*inputs, *self.scales)
        first = torch.autograd.grad(values, inputs, cotangent, create_graph=True)
        kernel = native_rimls._rimls_native
        for count in (1, 2, 3, 4, 8, 16, 17):
            with self.subTest(count=count):
                vectors = tuple(self.rng.normal(size=(count, *value.shape)) for value in self.arrays)
                weights = self.rng.normal(size=(count, len(values)))
                hessian_products = tuple(np.empty((count, *value.shape))
                                         for value in (*self.arrays, cotangent.detach().numpy()))
                gradients = tuple(np.empty((count, *value.shape)) for value in self.arrays)
                kernel.hvp_batch(*self.arrays, cotangent.detach().numpy(), *vectors,
                                 *hessian_products, *self.scales)
                kernel.vjp_batch(*self.arrays, weights, *gradients, *self.scales)
                for row in range(count):
                    reference_hessian = torch.autograd.grad(first, (*inputs, cotangent),
                        tuple(torch.tensor(vector[row]) for vector in vectors), retain_graph=True)
                    reference_gradient = torch.autograd.grad(values, inputs,
                        torch.tensor(weights[row]), retain_graph=True)
                    for actual, expected in zip(hessian_products, reference_hessian):
                        np.testing.assert_allclose(actual[row], expected.detach().numpy(),
                                                   rtol=4e-10, atol=4e-12)
                    for actual, expected in zip(gradients, reference_gradient):
                        np.testing.assert_allclose(actual[row], expected.detach().numpy(),
                                                   rtol=3e-11, atol=3e-13)

    def test_batch_buffer_mismatch_and_readonly_outputs_are_rejected(self):
        """배치 크기나 자료형이 잘못된 입력과 읽기 전용 출력 버퍼를 거부해요."""
        count = 3
        cotangent = np.array([.7, -.3])
        vectors = tuple(np.zeros((count, *value.shape)) for value in self.arrays)
        outputs = tuple(np.empty((count, *value.shape))
                        for value in (*self.arrays, cotangent))
        kernel = native_rimls._rimls_native
        with self.assertRaisesRegex(ValueError, 'batch sizes'):
            kernel.hvp_batch(*self.arrays, cotangent, vectors[0], vectors[1][:-1], vectors[2],
                             *outputs, *self.scales)
        with self.assertRaisesRegex(ValueError, 'batch sizes'):
            kernel.hvp_batch(*self.arrays, cotangent, *vectors,
                             *outputs[:3], outputs[3][:-1], *self.scales)
        with self.assertRaisesRegex(ValueError, 'shape'):
            kernel.hvp_batch(*self.arrays, cotangent, vectors[0].ravel(), *vectors[1:],
                             *outputs, *self.scales)
        with self.assertRaises(TypeError):
            kernel.hvp_batch(*self.arrays, cotangent, vectors[0].astype(np.float32),
                             *vectors[1:], *outputs, *self.scales)
        readonly = outputs[0].copy()
        readonly.flags.writeable = False
        with self.assertRaises((ValueError, BufferError)):
            kernel.hvp_batch(*self.arrays, cotangent, *vectors, readonly, *outputs[1:], *self.scales)
        weights = np.ones((count, len(cotangent)))
        with self.assertRaisesRegex(ValueError, 'batch sizes'):
            kernel.vjp_batch(*self.arrays, weights, outputs[0][:-1], *outputs[1:3], *self.scales)
        with self.assertRaisesRegex(ValueError, 'shape'):
            kernel.vjp_batch(*self.arrays, weights.ravel(), *outputs[:3], *self.scales)

    def test_large_pair_workspace_preserves_rigid_translation(self):
        """배치 작업공간 한도를 넘는 표면에서도 강체 병진의 불변성을 보존해요."""
        count, point_count = 8, 41000
        arrays = (np.array([[.01, -.02, .12]]), np.zeros((point_count, 3)),
                  np.tile([0., 0., 1.], (point_count, 1)))
        cotangent = np.array([.7])
        translations = self.rng.normal(size=(count, 1, 3))
        vectors = (translations, np.repeat(translations, point_count, axis=1),
                   np.zeros((count, point_count, 3)))
        outputs = tuple(np.empty((count, *value.shape)) for value in (*arrays, cotangent))
        native_rimls._rimls_native.hvp_batch(*arrays, cotangent, *vectors, *outputs, *self.scales)
        for output in outputs:
            np.testing.assert_array_equal(output, 0.)


if __name__ == '__main__':
    unittest.main()
