"""RIMLS 간격과 미분을 계산하고 선택 질의의 역전파를 묶음 처리해요."""
import numpy as np
import torch
from .primitives import sample_plane
from .native_rimls import try_native_rimls, try_native_values

def imls_value_gradient_torch(query, points, normals, R):
    import torch
    if R <= 0:
        raise ValueError('R must be positive')
    diff = query - points
    t = torch.clamp(1.0 - torch.sum(diff * diff, dim=1) / (R * R), min=0.0)
    phi = t ** 4
    grad_phi = (-8.0 / (R * R)) * (t ** 3)[:, None] * diff
    D = phi.sum()
    if D.item() < 1e-14:
        raise ValueError('query outside kernel support')
    local_gap = torch.sum(diff * normals, dim=1)
    f0 = torch.sum(phi * local_gap) / D
    grad0 = torch.sum(phi[:, None] * normals +
                      (local_gap - f0)[:, None] * grad_phi, dim=0) / D
    return f0, grad0, phi, local_gap


def rimls_value_torch(query, points, normals, R, sigma_r=0.5, sigma_n=1.0):
    import torch
    if sigma_r <= 0 or sigma_n <= 0:
        raise ValueError('sigma_r and sigma_n must be positive')
    f0, grad0, phi, local_gap = imls_value_gradient_torch(query, points, normals, R)
    # Algebraically identical squared residuals, without norm/abs singularities.
    residual_squared = (f0 - local_gap) ** 2
    normal_difference_squared = torch.sum((grad0[None, :] - normals) ** 2, dim=1)
    W = phi * torch.exp(-residual_squared / (sigma_r * R) ** 2)
    W = W * torch.exp(-normal_difference_squared / sigma_n ** 2)
    denominator = W.sum()
    if denominator.item() <= 1e-14:
        raise ValueError('vanishing robust kernel weights')
    return torch.sum(W * local_gap) / denominator


def _rimls_values_reference(queries,points,normals,R,sigma_r=.5,sigma_n=.5):
    r"""일반 계산 그래프로 RIMLS 간격을 평가해요.

    $$s_i=\frac{\sum_j w_{ij}\ell_{ij}}{\sum_j w_{ij}}$$
    """
    if R<=0 or sigma_r<=0 or sigma_n<=0:
        raise ValueError('positive R and robust scales required')
    # 질의와 표면점의 상대좌표를 구해요: $$d_{ij}=q_i-p_j$$
    diff=queries[:,None,:]-points[None,:,:]
    # 유한 지지 커널의 밑을 구해요: $$t_{ij}=\max(1-\|d_{ij}\|^2/R^2,0)$$
    t=torch.clamp(1.-torch.sum(diff*diff,dim=2)/(R*R),min=0.)
    # 거리 가중치를 구해요: $$a_{ij}=t_{ij}^{4}$$
    phi=t**4
    # 거리 가중치의 기울기를 구해요: $$G_{ij}=-8t_{ij}^{3}d_{ij}/R^2$$
    grad_phi=(-8./(R*R))*t[:,:,None]**3*diff
    # 가중치 합을 구해요: $$D_i=\sum_j a_{ij}$$
    D=phi.sum(dim=1)
    if torch.any(D<1e-14).item():raise ValueError('query outside kernel support')
    # 국부 간격을 구해요: $$\ell_{ij}=d_{ij}\cdot n_j$$
    local=torch.sum(diff*normals[None,:,:],dim=2)
    # 초기 간격을 구해요: $$f_i=\sum_j a_{ij}\ell_{ij}/D_i$$
    f0=torch.sum(phi*local,dim=1)/D
    # 초기 간격의 공간 기울기를 구해요: $$h_i=\sum_j[a_{ij}n_j+(\ell_{ij}-f_i)G_{ij}]/D_i$$
    grad0=torch.sum(phi[:,:,None]*normals[None,:,:]+(local-f0[:,None])[:,:,None]*grad_phi,dim=1)/D[:,None]
    # 간격 차이의 제곱을 구해요: $$r_{ij}=(f_i-\ell_{ij})^2$$
    residual=(f0[:,None]-local)**2
    # 법선 차이의 제곱을 구해요: $$v_{ij}=\|h_i-n_j\|^2$$
    deviation=torch.sum((grad0[:,None,:]-normals[None,:,:])**2,dim=2)
    # 강건 거리 가중치를 구해요: $$w_{ij}=a_{ij}\exp[-r_{ij}/(\sigma_rR)^2-v_{ij}/\sigma_n^2]$$
    W=phi*torch.exp(-residual/(sigma_r*R)**2-deviation/sigma_n**2)
    # 강건 가중치 합을 구해요: $$Z_i=\sum_jw_{ij}$$
    denominator=W.sum(dim=1)
    if torch.any(denominator<=1e-14).item():raise ValueError('vanishing robust weights')
    # 최종 간격을 구해요: $$s_i=\sum_jw_{ij}\ell_{ij}/Z_i$$
    return torch.sum(W*local,dim=1)/denominator



def _rimls_pair_derivatives(queries, points, normals, R, sigma_r, sigma_n):
    r"""각 질의의 간격을 점 상대좌표와 표면 법선에 대해 미분해요.

    $$\mathrm{d}s_i=\sum_j\left(A_{ij}\cdot\mathrm{d}(q_i-p_j)+B_{ij}\cdot\mathrm{d}n_j\right)$$

    중간변수의 역전파 가중치는 이름 앞에 bar를 붙여 구분해요.
    """
    # 질의와 표면점의 상대좌표를 구해요: $$d_{ij}=q_i-p_j$$
    diff = queries[:, None, :] - points[None, :, :]
    # 유한 지지 커널의 밑을 구해요: $$t_{ij}=\max(1-\|d_{ij}\|^2/R^2,0)$$
    t = torch.clamp(1. - torch.sum(diff * diff, dim=2) / (R * R), min=0.)
    # 거리 가중치를 구해요: $$a_{ij}=t_{ij}^{4}$$
    phi = t ** 4
    # 거리 가중치의 공간 기울기를 구해요: $$G_{ij}=-8t_{ij}^{3}d_{ij}/R^2$$
    grad_phi = (-8. / (R * R)) * t[:, :, None] ** 3 * diff
    # 거리 가중치의 합을 구해요: $$D_i=\sum_j a_{ij}$$
    D = phi.sum(dim=1)
    # 표면 법선 방향의 국부 간격을 구해요: $$\ell_{ij}=d_{ij}\cdot n_j$$
    local = torch.sum(diff * normals[None, :, :], dim=2)
    # 초기 IMLS 간격을 구해요: $$f_i=\sum_j a_{ij}\ell_{ij}/D_i$$
    f0 = torch.sum(phi * local, dim=1) / D
    # 초기 IMLS 간격의 공간 기울기를 구해요: $$h_i=\sum_j[a_{ij}n_j+(\ell_{ij}-f_i)G_{ij}]/D_i$$
    grad0 = torch.sum(phi[:, :, None] * normals[None, :, :] +
                      (local - f0[:, None])[:, :, None] * grad_phi, dim=1) / D[:, None]
    # 초기 간격과 국부 간격의 차이를 구해요: $$u_{ij}=f_i-\ell_{ij}$$
    residual = f0[:, None] - local
    # 초기 기울기와 표면 법선의 차이를 구해요: $$v_{ij}=h_i-n_j$$
    deviation = grad0[:, None, :] - normals[None, :, :]
    # 강건 가중치의 지수 인자를 구해요: $$e_{ij}=\exp[-u_{ij}^2/(\sigma_r R)^2-\|v_{ij}\|^2/\sigma_n^2]$$
    robust = torch.exp(-residual ** 2 / (sigma_r * R) ** 2 -
                       torch.sum(deviation * deviation, dim=2) / sigma_n ** 2)
    # 강건 거리 가중치를 구해요: $$w_{ij}=a_{ij}e_{ij}$$
    W = phi * robust
    # 강건 거리 가중치의 합을 구해요: $$Z_i=\sum_jw_{ij}$$
    denominator = W.sum(dim=1)
    # 최종 RIMLS 간격을 구해요: $$s_i=\sum_jw_{ij}\ell_{ij}/Z_i$$
    value = torch.sum(W * local, dim=1) / denominator
    # 가중치에 대한 역전파 계수를 구해요: $$\bar w_{ij}=(\ell_{ij}-s_i)/Z_i$$
    bar_W = (local - value[:, None]) / denominator[:, None]
    # 분자의 국부 간격에 대한 계수를 구해요: $$\bar\ell_{ij}=w_{ij}/Z_i$$
    bar_local = W / denominator[:, None]
    # 거리 가중치에 대한 첫 기여를 구해요: $$\bar a_{ij}=\bar w_{ij}e_{ij}$$
    bar_phi = bar_W * robust
    # 지수함수 역전파의 공통 계수를 구해요: $$k_{ij}=\bar w_{ij}w_{ij}$$
    coefficient = bar_W * W
    # 간격 차이에 대한 계수를 구해요: $$\bar u_{ij}=-2k_{ij}u_{ij}/(\sigma_rR)^2$$
    bar_residual = (-2. / (sigma_r * R) ** 2) * coefficient * residual
    # 법선 차이에 대한 계수를 구해요: $$\bar v_{ij}=-2k_{ij}v_{ij}/\sigma_n^2$$
    bar_deviation = (-2. / sigma_n ** 2) * coefficient[:, :, None] * deviation
    # 초기 간격에 대한 계수를 모아요: $$\bar f_i=\sum_j\bar u_{ij}$$
    bar_f0 = bar_residual.sum(dim=1)
    # 간격 차이를 통한 국부 간격의 기여를 더해요: $$\bar\ell_{ij}\mathrel{-}=\bar u_{ij}$$
    bar_local = bar_local - bar_residual
    # 초기 기울기에 대한 계수를 모아요: $$\bar h_i=\sum_j\bar v_{ij}$$
    bar_grad0 = bar_deviation.sum(dim=1)
    # 법선 차이를 통한 표면 법선의 기여를 구해요: $$\bar n_{ij}=-\bar v_{ij}$$
    bar_normals = -bar_deviation
    # 초기 기울기 분자의 계수를 구해요: $$c_i=\bar h_i/D_i$$
    scaled_grad0 = bar_grad0 / D[:, None]
    # 초기 기울기 분모의 계수를 구해요: $$\bar D_i=-\bar h_i\cdot h_i/D_i$$
    bar_D = -torch.sum(bar_grad0 * grad0, dim=1) / D
    # 초기 기울기의 가중 법선 항을 반영해요: $$\bar a_{ij}\mathrel{+}=c_i\cdot n_j$$
    bar_phi = bar_phi + torch.sum(scaled_grad0[:, None, :] * normals[None, :, :], dim=2)
    # 가중 법선을 통한 법선 기여를 더해요: $$\bar n_{ij}\mathrel{+}=a_{ij}c_i$$
    bar_normals = bar_normals + phi[:, :, None] * scaled_grad0[:, None, :]
    # 가중치 기울기를 통한 공통 계수를 구해요: $$b_{ij}=c_i\cdot G_{ij}$$
    coupling = torch.sum(scaled_grad0[:, None, :] * grad_phi, dim=2)
    # 국부 간격에 초기 기울기의 기여를 더해요: $$\bar\ell_{ij}\mathrel{+}=b_{ij}$$
    bar_local = bar_local + coupling
    # 초기 간격에 초기 기울기의 기여를 더해요: $$\bar f_i\mathrel{-}=\sum_jb_{ij}$$
    bar_f0 = bar_f0 - coupling.sum(dim=1)
    # 거리 가중치의 공간 기울기에 대한 계수를 구해요: $$\bar G_{ij}=(\ell_{ij}-f_i)c_i$$
    bar_grad_phi = (local - f0[:, None])[:, :, None] * scaled_grad0[:, None, :]
    # 초기 간격 분자의 계수를 구해요: $$b_i=\bar f_i/D_i$$
    scaled_f0 = bar_f0 / D
    # 초기 간격 분자의 거리 가중치 기여를 더해요: $$\bar a_{ij}\mathrel{+}=\ell_{ij}b_i$$
    bar_phi = bar_phi + local * scaled_f0[:, None]
    # 초기 간격 분자의 국부 간격 기여를 더해요: $$\bar\ell_{ij}\mathrel{+}=a_{ij}b_i$$
    bar_local = bar_local + phi * scaled_f0[:, None]
    # 초기 간격 분모의 기여를 더해요: $$\bar D_i\mathrel{-}=\bar f_if_i/D_i$$
    bar_D = bar_D - bar_f0 * f0 / D
    # 가중치 합을 통한 기여를 더해요: $$\bar a_{ij}\mathrel{+}=\bar D_i$$
    bar_phi = bar_phi + bar_D[:, None]
    # 가중치 공간 기울기의 직접 좌표 기여를 구해요: $$\bar d_{ij}=-8t_{ij}^3\bar G_{ij}/R^2$$
    bar_diff = (-8. / (R * R)) * t[:, :, None] ** 3 * bar_grad_phi
    # 커널 밑을 통한 모든 기여를 모아요: $$\bar t_{ij}=4t_{ij}^3\bar a_{ij}-24t_{ij}^2(\bar G_{ij}\cdot d_{ij})/R^2$$
    bar_t = 4. * t ** 3 * bar_phi - (24. / (R * R)) * t ** 2 * torch.sum(bar_grad_phi * diff, dim=2)
    # 커널 밑의 좌표 기여를 더해요: $$\bar d_{ij}\mathrel{-}=2\bar t_{ij}d_{ij}/R^2$$
    bar_diff = bar_diff - (2. / (R * R)) * bar_t[:, :, None] * diff
    # 국부 간격의 좌표 기여를 더해요: $$A_{ij}=\bar d_{ij}+\bar\ell_{ij}n_j$$
    pair_gradient = bar_diff + bar_local[:, :, None] * normals[None, :, :]
    # 국부 간격의 법선 기여를 더해요: $$B_{ij}=\bar n_{ij}+\bar\ell_{ij}d_{ij}$$
    normal_gradient = bar_normals + bar_local[:, :, None] * diff
    return pair_gradient, normal_gradient


class _SelectedRIMLS(torch.autograd.Function):
    """선택 질의의 미분 계수를 한 번 계산한 뒤 모든 역전파 방향에 재사용해요."""

    @staticmethod
    def forward(ctx, queries, points, normals, R, sigma_r, sigma_n):
        """입력을 보관하고 일반 RIMLS 식으로 간격을 평가해요."""
        ctx.save_for_backward(queries, points, normals)
        ctx.scales = R, sigma_r, sigma_n
        return _rimls_values_reference(queries, points, normals, R, sigma_r, sigma_n)

    @staticmethod
    def backward(ctx, cotangent):
        r"""간격의 역전파 가중치를 좌표와 법선의 기울기로 변환해요.

        $$\bar q_i=\bar s_i\sum_j A_{ij},\quad\bar p_j=-\sum_i\bar s_iA_{ij},\quad\bar n_j=\sum_i\bar s_iB_{ij}$$
        """
        queries, points, normals = ctx.saved_tensors
        pair_gradient, normal_gradient = _rimls_pair_derivatives(
            queries, points, normals, *ctx.scales)
        # 질의 좌표에 대한 기여를 모아요: $$\bar q_i=\bar s_i\sum_jA_{ij}$$
        query_gradient = cotangent[:, None] * pair_gradient.sum(dim=1)
        # 표면점의 기여를 행렬곱으로 모아요: $$\bar p_j=-\sum_i\bar s_iA_{ij}$$
        point_gradient = -(cotangent @ pair_gradient.reshape(len(queries), -1)).reshape_as(points)
        # 표면 법선의 기여를 행렬곱으로 모아요: $$\bar n_j=\sum_i\bar s_iB_{ij}$$
        normals_gradient = (cotangent @ normal_gradient.reshape(len(queries), -1)).reshape_as(normals)
        return query_gradient, point_gradient, normals_gradient, None, None, None


def rimls_values_torch(queries, points, normals, R, sigma_r=.5, sigma_n=.5):
    r"""간격의 값과 모든 좌표 미분을 유지하며 선택 질의의 계산을 줄여요.

    $$s_i=\frac{\sum_j w_{ij}\ell_{ij}}{\sum_j w_{ij}}$$
    """
    if R <= 0 or sigma_r <= 0 or sigma_n <= 0:
        raise ValueError('positive R and robust scales required')
    native = try_native_rimls(queries, points, normals, R, sigma_r, sigma_n)
    if native is not None:
        return native
    # 선택 야코비안에서는 묶음 편미분으로 반복되는 계산을 줄여요.
    selected_batch = 0 < len(queries) <= 32 and len(points) >= 48
    # 작은 일반 계산은 별도 미분 모드 검사 없이 기존 식으로 처리해요.
    if not selected_batch and len(queries) * len(points) <= 65536:
        return _rimls_values_reference(queries, points, normals, R, sigma_r, sigma_n)
    # 텐서 매개변수의 미분과 기존 PyTorch 변환은 일반 계산 그래프에 맡겨요.
    tensor_scales = any(torch.is_tensor(scale) for scale in (R, sigma_r, sigma_n))
    transforms_active = getattr(torch._C, '_are_functorch_transforms_active', lambda: False)()
    # ##r## PyTorch 1.9 returns a tuple; newer named tuples also support [1].
    forward_ad = getattr(torch.autograd, 'forward_ad', None)
    dual_inputs = forward_ad is not None and any(
        forward_ad.unpack_dual(value)[1] is not None for value in (queries, points, normals))
    inputs = (queries, points, normals) + tuple(
        scale for scale in (R, sigma_r, sigma_n) if torch.is_tensor(scale))
    needs_reverse = torch.is_grad_enabled() and any(value.requires_grad for value in inputs)
    if not needs_reverse and not transforms_active and not dual_inputs:
        native_value = try_native_values(queries, points, normals, R, sigma_r, sigma_n)
        if native_value is not None:
            return native_value
        # 값만 필요할 때는 질의 축을 나눠 전체 표면 쌍의 임시 배열을 제한해요.
        chunk = max(1, 65536 // max(1, len(points)))
        if len(queries) > chunk:
            return torch.cat([_rimls_values_reference(queries[start:start + chunk], points, normals,
                                                      R, sigma_r, sigma_n)
                              for start in range(0, len(queries), chunk)])
    if selected_batch and not tensor_scales and not transforms_active and not dual_inputs:
        return _SelectedRIMLS.apply(queries, points, normals, R, sigma_r, sigma_n)
    return _rimls_values_reference(queries, points, normals, R, sigma_r, sigma_n)

def make_gaps(S, node_shape, R=.9, points=None, normals=None):
    if points is None and normals is None:
        points, normals, _ = sample_plane()
    elif points is None or normals is None:
        raise ValueError("points and normals must be supplied together")
    St = torch.tensor(S, dtype=torch.float64)
    pt = torch.tensor(points, dtype=torch.float64)
    nt = torch.tensor(normals, dtype=torch.float64)
    def gaps(state):
        samples = St @ state.reshape(node_shape)
        return rimls_values_torch(samples, pt, nt, R, sigma_n=.5)  ##r##
    return gaps
