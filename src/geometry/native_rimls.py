"""선택적인 C++ 커널로 RIMLS의 값과 정확한 일차·이차 미분을 계산해요."""
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import torch

try:
    from . import _rimls_native
except ImportError:
    _rimls_native = None

_full_mode = ContextVar('native_rimls_full_mode', default=False)
_scalar_retry = ContextVar('native_rimls_scalar_retry', default=False)
_derivative_budget = ContextVar('native_rimls_derivative_budget', default=None)
_PAIR_CACHE_BYTES = 64 * 1024 * 1024
_BATCH_API = hasattr(torch, 'vmap') and hasattr(torch.autograd.Function, 'setup_context')


@contextmanager
def native_full_mode(enabled=True):
    """현재 실행 문맥의 전체 뉴턴 미분에만 네이티브 커널을 허용해요."""
    token = _full_mode.set(enabled)
    plan = {'remaining': _PAIR_CACHE_BYTES if enabled else 0, 'used': False}
    budget_token = _derivative_budget.set(plan if enabled else None)
    try:
        yield plan
    finally:
        _derivative_budget.reset(budget_token)
        _full_mode.reset(token)


@contextmanager
def native_scalar_retry():
    """호출자가 배치 미분 실패를 스칼라 미분으로 재시도할 수 있음을 표시해요."""
    token = _scalar_retry.set(True)
    try:
        yield
    finally:
        _scalar_retry.reset(token)


def _supported(queries, points, normals, scales):
    """네이티브 미분이 지원하는 장치와 자동미분 입력인지 확인해요."""
    inputs = (queries, points, normals)
    if (_rimls_native is None
            or any(value.ndim != 2 or value.shape[1] != 3 for value in inputs)
            or points.shape != normals.shape or not len(queries)
            or any(value.device.type != 'cpu' or value.dtype != torch.float64 for value in inputs)
            or any(torch.is_tensor(value) for value in scales)):
        return False
    if getattr(torch._C, '_are_functorch_transforms_active', lambda: False)():
        return False
    forward_ad = getattr(torch.autograd, 'forward_ad', None)
    return forward_ad is None or not any(
        forward_ad.unpack_dual(value).tangent is not None for value in inputs)


def try_native_rimls(queries, points, normals, R, sigma_r, sigma_n):
    """전체 뉴턴 문맥에서 효과가 있는 입력에만 네이티브 미분을 선택해요."""
    if (not _full_mode.get() or len(queries) * len(points) < 2048
            or not torch.is_grad_enabled()
            or not any(value.requires_grad for value in (queries, points, normals))
            or not _supported(queries, points, normals, (R, sigma_r, sigma_n))):
        return None
    result = _NativeValue.apply(queries, points, normals, R, sigma_r, sigma_n)
    plan = _derivative_budget.get()
    if plan is not None:
        plan['used'] = True
    return result


def try_native_values(queries, points, normals, R, sigma_r, sigma_n):
    """미분이 필요 없는 큰 CPU 간격 평가를 선택적인 네이티브 커널로 처리해요."""
    if (len(queries) * len(points) < 65536 or not hasattr(_rimls_native, 'values')
            or not _supported(queries, points, normals, (R, sigma_r, sigma_n))
            or (torch.is_grad_enabled()
                and any(value.requires_grad for value in (queries, points, normals)))):
        return None
    try:
        scales = np.asarray((R, sigma_r, sigma_n), dtype=float)
    except (OverflowError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(scales)):
        return None
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        squares = np.square(np.asarray((scales[0], scales[0] * scales[1], scales[2])))
    if not np.all(np.isfinite(squares)) or np.any(squares <= 0):
        return None
    try:
        inputs = tuple(_array(value) for value in (queries, points, normals))
    except RuntimeError:
        return None
    if not all(np.all(np.isfinite(value)) for value in inputs):
        return None
    output = np.empty(len(queries))
    try:
        _rimls_native.values(*inputs, output, R, sigma_r, sigma_n)
    except ValueError:
        return None
    if not np.all(np.isfinite(output)):
        return None
    return torch.from_numpy(output)


def available():
    """선택적인 네이티브 확장을 불러왔는지 알려줘요."""
    return _rimls_native is not None


def _array(value):
    """CPU 배정밀도 텐서를 네이티브 커널용 연속 배열로 변환해요."""
    try:
        return value.detach().contiguous().numpy()
    except RuntimeError as exc:
        message = str(exc)
        if any(word in message for word in ('Batching', 'data pointer', 'storage', 'numpy')):
            raise RuntimeError('Batching rule not implemented for native RIMLS derivatives') from exc
        raise


def _torch_vjp(queries, points, normals, cotangent, R, sigma_r, sigma_n):
    r"""고차 미분을 위한 일반 텐서 역전파 식을 평가해요.

    $$\bar q_i=\bar s_i\sum_jA_{ij},\quad\bar p_j=-\sum_i\bar s_iA_{ij},\quad\bar n_j=\sum_i\bar s_iB_{ij}$$
    """
    from .rimls import _rimls_pair_derivatives
    pair, normal = _rimls_pair_derivatives(queries, points, normals, R, sigma_r, sigma_n)
    # 질의 좌표의 기울기를 모아요: $$\bar q_i=\bar s_i\sum_jA_{ij}$$
    query_gradient = cotangent[:, None] * pair.sum(dim=1)
    # 표면점의 기울기를 모아요: $$\bar p_j=-\sum_i\bar s_iA_{ij}$$
    point_gradient = -(cotangent @ pair.reshape(len(queries), -1)).reshape_as(points)
    # 표면 법선의 기울기를 모아요: $$\bar n_j=\sum_i\bar s_iB_{ij}$$
    normal_gradient = (cotangent @ normal.reshape(len(queries), -1)).reshape_as(normals)
    return query_gradient, point_gradient, normal_gradient


def _pair_derivatives(queries, points, normals, scales, budget):
    """한 평가의 메모리 한도 안에서 쌍별 미분 계수를 보관해요."""
    size = len(queries) * (len(points) * 6 + 3) * np.dtype(np.float64).itemsize
    if budget is None or size > budget['remaining'] or not hasattr(_rimls_native, 'pair_jacobian'):
        return None
    inputs = tuple(_array(value) for value in (queries, points, normals))
    budget['remaining'] -= size
    try:
        pair = np.empty((len(queries), len(points), 3))
        normal = np.empty_like(pair)
        _rimls_native.pair_jacobian(*inputs, pair, normal, *scales)
        return pair, normal, pair.sum(axis=1)
    except MemoryError:
        budget['remaining'] += size
        return None
    except BaseException:
        budget['remaining'] += size
        raise


def _cached_vjp(derivatives, cotangent):
    r"""보관한 쌍별 미분에 출력 가중치를 곱해 역전파를 계산해요.

    $$\bar q_i=\bar s_i\sum_jA_{ij},\quad\bar p_j=-\sum_i\bar s_iA_{ij},\quad\bar n_j=\sum_i\bar s_iB_{ij}$$
    """
    pair, normal, query_sum = derivatives
    # 질의 좌표의 기울기를 모아요: $$\bar q_i=\bar s_i\sum_jA_{ij}$$
    query_gradient = cotangent[:, None] * query_sum
    # 표면점의 기울기를 모아요: $$\bar p_j=-\sum_i\bar s_iA_{ij}$$
    point_gradient = -(cotangent @ pair.reshape(len(cotangent), -1)).reshape(pair.shape[1], 3)
    # 표면 법선의 기울기를 모아요: $$\bar n_j=\sum_i\bar s_iB_{ij}$$
    normal_gradient = (cotangent @ normal.reshape(len(cotangent), -1)).reshape(normal.shape[1], 3)
    return query_gradient, point_gradient, normal_gradient


def _native_vjp(queries, points, normals, cotangent, R, sigma_r, sigma_n, derivatives):
    """보관한 편미분이나 네이티브 커널로 일차 역전파를 평가해요."""
    inputs = tuple(_array(value) for value in (queries, points, normals, cotangent))
    if derivatives is None:
        outputs = tuple(np.empty_like(value) for value in inputs[:3])
        _rimls_native.vjp(*inputs, *outputs, R, sigma_r, sigma_n)
    else:
        outputs = _cached_vjp(derivatives, inputs[3])
    return tuple(torch.from_numpy(value) for value in outputs)


class _NativeVJP(torch.autograd.Function):
    """첫 역전파의 계산을 묶고 다음 역전파에 헤시안 방향곱을 제공해요."""

    @staticmethod
    def forward(ctx, queries, points, normals, cotangent, R, sigma_r, sigma_n, derivatives):
        """보관한 편미분이나 네이티브 커널로 좌표와 법선의 역전파를 계산해요."""
        ctx.save_for_backward(queries, points, normals, cotangent)
        ctx.scales = R, sigma_r, sigma_n
        return _native_vjp(queries, points, normals, cotangent, R, sigma_r, sigma_n, derivatives)

    @staticmethod
    def backward(ctx, vq, vp, vn):
        r"""역전파 식의 방향 미분으로 정확한 이차 미분을 계산해요.

        $$\mathrm{d}(J^T\bar s)=H_{\bar s}\mathrm{d}x+J^T\mathrm{d}\bar s$$
        """
        queries, points, normals, cotangent = ctx.saved_tensors
        directions = tuple(torch.zeros_like(value) if direction is None else direction
                           for value, direction in zip((queries, points, normals), (vq, vp, vn)))
        create_graph = torch.is_grad_enabled()

        def tensor_gradients():
            """일반 텐서 식으로 배치 방향과 고차 미분을 지원해요."""
            with torch.enable_grad():
                # 독립적인 입력을 만들고 필요한 고차 계산 그래프를 유지해요.
                inputs = tuple(value.view_as(value) if create_graph and value.requires_grad
                               else value.detach().requires_grad_(True)
                               for value in (queries, points, normals, cotangent))
                outputs = _torch_vjp(*inputs, *ctx.scales)
                gradients = torch.autograd.grad(outputs, inputs, grad_outputs=directions,
                                                create_graph=create_graph, allow_unused=True)
            return gradients

        if create_graph:
            gradients = tensor_gradients()
        else:
            try:
                if _BATCH_API:
                    gradients = _NativeHVP.apply(queries, points, normals, cotangent, *directions, *ctx.scales)
                else:
                    inputs = tuple(_array(value) for value in (queries, points, normals, cotangent))
                    vectors = tuple(_array(value) for value in directions)
                    outputs = tuple(np.empty_like(value) for value in inputs)
                    _rimls_native.hvp(*inputs, *vectors, *outputs, *ctx.scales)
                    gradients = tuple(torch.from_numpy(value) for value in outputs)
            except RuntimeError as exc:
                if (_scalar_retry.get()
                        or 'Batching rule not implemented for native RIMLS derivatives' not in str(exc)):
                    raise
                gradients = tensor_gradients()
        return (*gradients, None, None, None, None)


class _NativeHVP(torch.autograd.Function):
    """이차 미분의 배치 방향을 공개 vmap 규칙으로 네이티브 커널에 전달해요."""

    @staticmethod
    def forward(queries, points, normals, cotangent, vq, vp, vn, R, sigma_r, sigma_n):
        """계산 그래프가 필요 없는 헤시안 방향곱을 평가해요."""
        inputs = tuple(_array(value) for value in (queries, points, normals, cotangent))
        vectors = tuple(_array(value) for value in (vq, vp, vn))
        outputs = tuple(np.empty_like(value) for value in inputs)
        _rimls_native.hvp(*inputs, *vectors, *outputs, R, sigma_r, sigma_n)
        return tuple(torch.from_numpy(value) for value in outputs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        """고차 미분은 호출자가 일반 텐서 식으로 처리하므로 상태를 보관하지 않아요."""
        pass

    @staticmethod
    def backward(ctx, *values):
        """계산 그래프 없는 내부 함수의 직접 재미분을 명확히 차단해요."""
        raise RuntimeError('native HVP helper requires the differentiable tensor fallback for higher derivatives')

    @staticmethod
    def vmap(info, in_dims, *inputs):
        """여러 방향을 한 번에 전달하고 지원하지 않는 배치는 방향별로 처리해요."""
        if hasattr(_rimls_native, 'hvp_batch') and all(dim is None for dim in in_dims[:4]):
            original = tuple(_array(value) for value in inputs[:4])
            vectors = tuple(_array(value.unsqueeze(0).expand(info.batch_size, *value.shape)
                                   if dim is None else value.movedim(dim, 0))
                            for value, dim in zip(inputs[4:7], in_dims[4:7]))
            outputs = tuple(np.empty((info.batch_size, *value.shape)) for value in original)
            _rimls_native.hvp_batch(*original, *vectors, *outputs, *inputs[7:])
            return tuple(torch.from_numpy(value) for value in outputs), (0, 0, 0, 0)
        outputs = []
        for index in range(info.batch_size):
            batch = tuple(tensor if dim is None else tensor.select(dim, index)
                          for tensor, dim in zip(inputs[:7], in_dims[:7]))
            outputs.append(_NativeHVP.apply(*batch, *inputs[7:]))
        return tuple(torch.stack([out[k] for out in outputs]) for k in range(4)), (0, 0, 0, 0)


class _BatchableNativeVJP(_NativeVJP):
    """구형 PyTorch용 역전파를 유지하면서 새 공개 배치 규칙을 제공해요."""

    @staticmethod
    def forward(queries, points, normals, cotangent, R, sigma_r, sigma_n, derivatives):
        """기존 경로와 같은 일차 역전파를 계산해요."""
        return _native_vjp(queries, points, normals, cotangent, R, sigma_r, sigma_n, derivatives)

    @staticmethod
    def setup_context(ctx, inputs, output):
        """부모 역전파가 사용할 좌표와 법선 및 출력 가중치를 보관해요."""
        ctx.save_for_backward(*inputs[:4])
        ctx.scales = inputs[4:7]

    @staticmethod
    def vmap(info, in_dims, *inputs):
        r"""같은 기하에 대한 여러 출력 가중치를 묶어 역전파해요.

        $$\bar q_{bi}=\bar s_{bi}\sum_jA_{ij},\quad\bar p_{bj}=-\sum_i\bar s_{bi}A_{ij},\quad\bar n_{bj}=\sum_i\bar s_{bi}B_{ij}$$
        """
        derivatives = inputs[7]
        if (not torch.is_grad_enabled() and all(dim is None for dim in in_dims[:3])
                and in_dims[3] is not None
                and (derivatives is not None or hasattr(_rimls_native, 'vjp_batch'))):
            cotangent = _array(inputs[3].movedim(in_dims[3], 0))
            if derivatives is not None:
                pair, normal, query_sum = derivatives
                # 배치별 질의 좌표의 기울기를 모아요: $$\bar q_{bi}=\bar s_{bi}\sum_jA_{ij}$$
                query_gradient = cotangent[:, :, None] * query_sum[None, :, :]
                # 배치별 표면점의 기울기를 모아요: $$\bar p_{bj}=-\sum_i\bar s_{bi}A_{ij}$$
                point_gradient = -(cotangent @ pair.reshape(pair.shape[0], -1)).reshape(info.batch_size, pair.shape[1], 3)
                # 배치별 표면 법선의 기울기를 모아요: $$\bar n_{bj}=\sum_i\bar s_{bi}B_{ij}$$
                normal_gradient = (cotangent @ normal.reshape(normal.shape[0], -1)).reshape(info.batch_size, normal.shape[1], 3)
                outputs = (query_gradient, point_gradient, normal_gradient)
            else:
                original = tuple(_array(value) for value in inputs[:3])
                outputs = tuple(np.empty((info.batch_size, *value.shape)) for value in original)
                _rimls_native.vjp_batch(*original, cotangent, *outputs, *inputs[4:7])
            return tuple(torch.from_numpy(out) for out in outputs), (0, 0, 0)
        outputs = []
        for index in range(info.batch_size):
            batch = tuple(tensor if dim is None else tensor.select(dim, index)
                          for tensor, dim in zip(inputs[:4], in_dims[:4]))
            outputs.append(_BatchableNativeVJP.apply(*batch, *inputs[4:]))
        return tuple(torch.stack([out[k] for out in outputs]) for k in range(3)), (0, 0, 0)


class _NativeValue(torch.autograd.Function):
    """기준 RIMLS 값에 네이티브 일차 및 이차 미분을 연결해요."""

    @staticmethod
    def forward(ctx, queries, points, normals, R, sigma_r, sigma_n):
        """작은 질의 묶음으로 값을 구하고 미분에 필요한 입력만 보관해요."""
        from .rimls import _rimls_values_reference
        ctx.save_for_backward(queries, points, normals)
        ctx.scales = R, sigma_r, sigma_n
        ctx.budget = _derivative_budget.get()
        ctx.derivatives = None
        ctx.cache_checked = False
        chunk = max(1, 65536 // max(1, len(points)))
        if len(queries) <= chunk:
            return _rimls_values_reference(queries, points, normals, R, sigma_r, sigma_n)
        return torch.cat([_rimls_values_reference(queries[start:start + chunk], points, normals,
                                                  R, sigma_r, sigma_n)
                          for start in range(0, len(queries), chunk)])

    @staticmethod
    def backward(ctx, cotangent):
        """간격의 역전파 가중치를 네이티브 미분 함수에 전달해요."""
        try:
            if not ctx.cache_checked:
                _array(cotangent)
                ctx.derivatives = _pair_derivatives(*ctx.saved_tensors, ctx.scales, ctx.budget)
                ctx.cache_checked = True
            function = _BatchableNativeVJP if _BATCH_API else _NativeVJP
            gradients = function.apply(*ctx.saved_tensors, cotangent, *ctx.scales, ctx.derivatives)
        except RuntimeError as exc:
            if 'Batching rule not implemented for native RIMLS derivatives' not in str(exc):
                raise
            gradients = _torch_vjp(*ctx.saved_tensors, cotangent, *ctx.scales)
        return (*gradients, None, None, None)


def native_values(queries, points, normals, R, sigma_r=.5, sigma_n=.5):
    """지원하는 CPU 배정밀도 입력에 선택적인 네이티브 미분을 적용해요."""
    from .rimls import _rimls_values_reference
    if not _supported(queries, points, normals, (R, sigma_r, sigma_n)):
        return _rimls_values_reference(queries, points, normals, R, sigma_r, sigma_n)
    result = _NativeValue.apply(queries, points, normals, R, sigma_r, sigma_n)
    plan = _derivative_budget.get()
    if plan is not None:
        plan['used'] = True
    return result
