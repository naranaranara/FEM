"""접촉 에너지와 미분을 계산하고 역전파 배치의 메모리를 제한해요."""
import inspect
import numpy as np
import torch
from .barrier import barrier_first_second, barrier_arrays
from ..geometry.native_rimls import native_full_mode, native_scalar_retry


_BATCHED_GRAD = 'is_grads_batched' in inspect.signature(torch.autograd.grad).parameters
_PUBLIC_VMAP = hasattr(torch, 'vmap') and hasattr(torch.autograd.Function, 'setup_context')
_NATIVE_HESSIAN_BATCH_SIZE = 8


def _jacobian_blocks(values, state, create_graph=False, use_vmap=False):
    r"""기존 계산 그래프의 야코비안 행을 작은 배치로 차례로 반환해요.

    $$J_{ij}=\frac{\partial v_i}{\partial x_j}$$
    """
    values = values.reshape(-1)
    batched = _BATCHED_GRAD
    use_vmap = use_vmap and _PUBLIC_VMAP
    batch_size = _NATIVE_HESSIAN_BATCH_SIZE if use_vmap else 8
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        if not values.requires_grad:
            yield state.new_zeros((stop - start, state.numel()))
            continue
        if use_vmap and stop - start > 1:
            seeds = values.new_zeros((stop - start, len(values)))
            seeds[torch.arange(stop - start), torch.arange(start, stop)] = 1.
            def differentiate(seed):
                """기존 그래프에서 한 역전파 방향의 미분을 구해요."""
                derivative = torch.autograd.grad(
                    values, state, grad_outputs=seed, retain_graph=True,
                    create_graph=create_graph, allow_unused=True)[0]
                return torch.zeros_like(state) if derivative is None else derivative
            try:
                with native_scalar_retry():
                    block = torch.vmap(differentiate)(seeds)
                yield block
                continue
            except RuntimeError as exc:
                if 'vmap' not in str(exc) and 'Batching rule' not in str(exc):
                    raise
                use_vmap = False
        if batched and stop - start > 1:
            seeds = values.new_zeros((stop - start, len(values)))
            seeds[torch.arange(stop - start), torch.arange(start, stop)] = 1.
            try:
                with native_scalar_retry():
                    derivative = torch.autograd.grad(
                        values, state, grad_outputs=seeds, retain_graph=True,
                        create_graph=create_graph, allow_unused=True,
                        is_grads_batched=True)[0]
                yield (state.new_zeros((stop - start, state.numel()))
                       if derivative is None else derivative)
                continue
            except RuntimeError as exc:
                if 'vmap' not in str(exc) and 'Batching rule' not in str(exc):
                    raise
                batched = False
        rows = []
        for value in values[start:stop]:
            derivative = None
            if value.requires_grad:
                derivative = torch.autograd.grad(value, state, retain_graph=True,
                    create_graph=create_graph, allow_unused=True)[0]
            rows.append(torch.zeros_like(state) if derivative is None else derivative)
        yield torch.stack(rows)


def _jacobian_rows(values, state, create_graph=False, use_vmap=False):
    r"""기존 계산 그래프의 야코비안을 조밀 텐서로 반환해요.

    $$J_{ij}=\frac{\partial v_i}{\partial x_j}$$
    """
    rows = list(_jacobian_blocks(values, state, create_graph, use_vmap))
    return torch.cat(rows) if rows else state.new_zeros((0, state.numel()))


def _gap_jacobian_blocks(gaps_autodiff, state, indices, gaps_subset_autodiff=None):
    """선택한 접촉점의 야코비안을 제한된 크기의 배치로 반환해요."""
    for start in range(0, len(indices), 32):
        ids = torch.as_tensor(indices[start:start + 32], dtype=torch.long)
        def selected_gaps(value):
            """요청한 접촉점만 평가하고 출력 크기를 검사해요."""
            selected = (gaps_autodiff(value)[ids] if gaps_subset_autodiff is None
                        else gaps_subset_autodiff(value, ids))
            if selected.ndim != 1 or selected.numel() != ids.numel():
                raise ValueError('selected gaps must match the requested query count')
            return selected
        try:
            block = torch.autograd.functional.jacobian(selected_gaps, state, vectorize=True)
        except RuntimeError as exc:
            if 'vmap' not in str(exc) and 'Batching rule' not in str(exc):raise
            block = torch.autograd.functional.jacobian(selected_gaps, state)
        yield block.detach().numpy()


def _gap_jacobian(gaps_autodiff, state, indices, gaps_subset_autodiff=None,
                  sparse=False):
    r"""접촉 야코비안의 실제 영 성분 비율에 맞춰 저장 방식을 선택해요.

    $$J_{ij}=\frac{\partial s_i}{\partial x_j}$$
    """
    from scipy.sparse import csr_matrix, vstack
    shape = (len(indices), state.numel())
    dense = None if sparse else np.empty(shape)
    blocks = []
    offset = 0
    nonzero = 0
    for block in _gap_jacobian_blocks(gaps_autodiff, state, indices, gaps_subset_autodiff):
        stop = offset + len(block)
        if dense is None:
            nonzero += np.count_nonzero(block)
            if nonzero <= .2 * stop * state.numel():
                blocks.append(csr_matrix(block))
            else:
                dense = np.empty(shape)
                if blocks:
                    dense[:offset] = vstack(blocks, format='csr').toarray()
                    blocks.clear()
                dense[offset:stop] = block
        else:
            dense[offset:stop] = block
        offset = stop
    if dense is not None:
        return dense
    return vstack(blocks, format='csr') if blocks else csr_matrix(shape)


def _selected_force_gradient(state, indices, weights, d_hat, gaps_subset_autodiff):
    r"""활성 접촉점의 작은 묶음마다 스칼라 에너지를 미분해 접촉력을 모아요.

    $$g=\sum_B\nabla_x\left[-\sum_{i\in B}w_i(s_i(x)-\hat d)^2\log(s_i(x)/\hat d)\right]$$
    """
    gradient = np.zeros(state.numel())
    for start in range(0, len(indices), 32):
        selected_indices = indices[start:start + 32]
        ids = torch.as_tensor(selected_indices, dtype=torch.long)
        selected = gaps_subset_autodiff(state, ids)
        if selected.ndim != 1 or selected.numel() != ids.numel():
            raise ValueError('selected gaps must match the requested query count')
        values = selected.detach().numpy()
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError('selected contact gaps must be finite and positive')
        w = torch.as_tensor(weights[selected_indices], dtype=state.dtype)
        # 선택한 활성 접촉점의 에너지를 합해요: $$E_B=-\sum_{i\in B}w_i(s_i-\hat d)^2\log(s_i/\hat d)$$
        scalar_energy = torch.sum(-w * (selected - d_hat)**2 * torch.log(selected / d_hat))
        if scalar_energy.requires_grad:
            # 전체 야코비안 없이 한 스칼라 에너지의 기울기를 구해요: $$g_B=\nabla_x E_B$$
            derivative = torch.autograd.grad(scalar_energy, state, allow_unused=True,
                                             retain_graph=True)[0]
            if derivative is not None:
                # 접촉점 묶음의 기울기를 전체 접촉력에 더해요: $$g\leftarrow g+g_B$$
                gradient += derivative.detach().numpy()
        del selected, scalar_energy, values
    return gradient


def _sparse_full_hessian(gradient, state, use_vmap=False):
    r"""헤시안의 조밀 전체 복사본을 만들지 않고 행 배치를 희소 행렬로 모아요.

    $$H_{ij}=\frac{\partial g_i}{\partial x_j}$$
    """
    from scipy.sparse import csr_matrix, vstack
    blocks = [csr_matrix(block.detach().numpy())
              for block in _jacobian_blocks(gradient, state, use_vmap=use_vmap)]
    return vstack(blocks, format='csr') if blocks else csr_matrix((0, state.numel()))


def _sparse_gn_hessian(J, coefficient):
    r"""야코비안의 실제 지지 영역을 보존해 GN 헤시안을 조립해요.

    $$H=J^T\operatorname{diag}(c)J$$
    """
    from scipy.sparse import csr_matrix, issparse, vstack
    if issparse(J):
        # 행마다 장벽 곡률을 곱해요: $$WJ=\operatorname{diag}(c)J$$
        weighted = J.multiply(coefficient[:, None])
        # 희소 야코비안끼리 곱해요: $$H=J^TWJ$$
        return (J.T @ weighted).tocsr()
    # 행마다 장벽 곡률을 곱해요: $$WJ=\operatorname{diag}(c)J$$
    weighted = coefficient[:, None] * J
    blocks = []
    for start in range(0, J.shape[1], 128):
        # 일부 자유도에 대응하는 헤시안 행만 계산해요: $$H_{I,:}=J_{:,I}^TWJ$$
        block = J[:, start:start + 128].T @ weighted
        blocks.append(csr_matrix(block))
    return vstack(blocks, format='csr') if blocks else csr_matrix((0, 0))


def multiple_contact_terms(state, gaps_autodiff, weights, d_hat=0.2,
                           method='gn', filter_ratio=0.1, compute_hessian=True,
                           gaps_subset_autodiff=None, return_gap_data=False,
                           _sparse_hessian=False, gaps_value_autodiff=None):
    r"""접촉 에너지와 선택한 방식의 기울기 및 헤시안을 계산해요.

    $$E_c = \sum_i -w_i(s_i-\hat d)^2\log(s_i/\hat d)$$
    """
    if return_gap_data and (method != 'gn' or not compute_hessian):
        raise ValueError('gap derivative data requires a GN Hessian evaluation')
    if method not in ('full', 'gn', 'filtered_gn'):
        raise ValueError(method)
    if d_hat <= 0 or not 0 < filter_ratio < 1:
        raise ValueError('invalid barrier/filter parameters')
    z = torch.tensor(np.asarray(state, dtype=float), dtype=torch.float64, requires_grad=True)
    weights = np.asarray(weights, dtype=float)
    native_plan = {'used': False}
    selected_force = (not compute_hessian and z.ndim == 1 and gaps_value_autodiff is not None
                      and gaps_subset_autodiff is not None)
    if selected_force or (compute_hessian and method != 'full' and gaps_value_autodiff is not None):
        with torch.no_grad():
            gap_tensor = gaps_value_autodiff(z)
    else:
        with native_full_mode(method == 'full' and compute_hessian) as native_plan:
            gap_tensor = gaps_autodiff(z)
    # 미분을 별도로 구하는 경로에서는 전체 간격의 계산 그래프를 해제해요.
    if selected_force or (compute_hessian and method != 'full'):
        gap_tensor = gap_tensor.detach()
    gaps = gap_tensor.detach().numpy()
    if gaps.ndim != 1 or weights.shape != gaps.shape:
        raise ValueError('one positive quadrature weight per gap required')
    if not np.all(np.isfinite(gaps)) or np.any(gaps <= 0):
        raise ValueError('all contact gaps must be finite and positive')
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError('quadrature weights must be finite and positive')
    indices = np.flatnonzero(gaps < d_hat)
    active = gaps < filter_ratio * d_hat if method == 'filtered_gn' else np.zeros(len(gaps), dtype=bool)
    gradient = np.zeros(len(z))
    if _sparse_hessian:
        from scipy.sparse import csr_matrix, issparse
    H = ((csr_matrix((len(z), len(z))) if _sparse_hessian
          else np.zeros((len(z), len(z)))) if compute_hessian else None)
    energy = 0.
    J = np.empty((0, len(z))) if return_gap_data else None
    if len(indices):
        s = gap_tensor[torch.as_tensor(indices, dtype=torch.long)]
        w = torch.as_tensor(weights[indices], dtype=torch.float64)
        # 활성 접촉점의 장벽 에너지를 합해요: $$E_c = -\sum_i w_i(s_i-\hat d)^2\log(s_i/\hat d)$$
        scalar_energy = torch.sum(-w * (s - d_hat)**2 * torch.log(s / d_hat))
        energy = float(scalar_energy.detach())
        if selected_force:
            with native_full_mode(False):
                gradient = _selected_force_gradient(z, indices, weights, d_hat, gaps_subset_autodiff)
        elif not compute_hessian or method == 'full':
            g = _jacobian_rows(scalar_energy.reshape(1), z, create_graph=compute_hessian)[0]
            gradient = g.detach().numpy()
            if compute_hessian:
                H = (_sparse_full_hessian(g, z, use_vmap=native_plan['used']) if _sparse_hessian
                     else _jacobian_rows(g, z, use_vmap=native_plan['used']).detach().numpy())
        else:
            J = _gap_jacobian(gaps_autodiff, z, indices, gaps_subset_autodiff,
                              sparse=_sparse_hessian)
            _, b1, coefficient = barrier_arrays(gaps[indices], d_hat)
            if np.any(active):
                coefficient[active[indices]] = barrier_first_second(filter_ratio * d_hat, d_hat)[1]
            # 접촉력의 반대 부호인 에너지 기울기를 조립해요: $$g = J^T(w \odot b')$$
            gradient = J.T @ (weights[indices] * b1)
            # 선택한 장벽 곡률로 GN 헤시안을 조립해요: $$H = J^T\operatorname{diag}(w \odot c)J$$
            H = (_sparse_gn_hessian(J, weights[indices] * coefficient) if _sparse_hessian
                 else J.T @ ((weights[indices] * coefficient)[:, None] * J))
    hessian_values = H.data if _sparse_hessian and H is not None else H
    if not np.all(np.isfinite(gradient)) or (H is not None and not np.all(np.isfinite(hessian_values))):
        raise ValueError('nonfinite multi-contact derivatives')
    result = dict(energy=energy, gradient=gradient, hessian=(H + H.T) / 2 if H is not None else None,
                  gaps=gaps, min_gap=float(gaps.min()) if len(gaps) else float('inf'),
                  filter_count=int(np.count_nonzero(active)))
    # SR1이 활성 접촉점의 야코비안을 그대로 재사용하도록 반환해요.
    if return_gap_data:
        if _sparse_hessian and issparse(J):
            J = J.toarray()
        result.update(gap_indices=indices, gap_jacobian=J, gap_weights=weights,
                      gap_dofs=np.arange(len(z), dtype=int))
    return result


def multiple_contact_terms_sparse(state, gaps_autodiff, weights, dofs, d_hat=.2,
                                  method='gn', compute_hessian=True,
                                  gaps_subset_autodiff=None, return_gap_data=False,
                                  gaps_value_autodiff=None):
    """접촉 자유도에서 계산한 미분을 전체 희소 행렬에 배치해요."""
    from scipy.sparse import csr_matrix
    state = np.asarray(state, dtype=float)
    dofs = np.asarray(dofs, dtype=int)
    if dofs.ndim != 1 or len(dofs) != len(np.unique(dofs)) or np.any(dofs < 0) or np.any(dofs >= len(state)):
        raise ValueError('unique valid contact DOFs required')
    base = torch.tensor(state, dtype=torch.float64)
    ids = torch.tensor(dofs, dtype=torch.long)
    def local_gaps(local):
        """국부 접촉 자유도를 전체 상태에 반영해 간격을 평가해요."""
        return gaps_autodiff(base.index_copy(0, ids, local))
    def local_subset(local, query_ids):
        """국부 상태에서 선택한 접촉점의 간격을 평가해요."""
        return gaps_subset_autodiff(base.index_copy(0, ids, local), query_ids)
    def local_values(local):
        """미분 그래프가 필요 없는 전체 간격 검증을 국부 상태에 연결해요."""
        return gaps_value_autodiff(base.index_copy(0, ids, local))
    c = multiple_contact_terms(state[dofs], local_gaps, weights, d_hat=d_hat,
                               method=method, compute_hessian=compute_hessian,
                               gaps_subset_autodiff=local_subset if gaps_subset_autodiff is not None else None,
                               return_gap_data=return_gap_data, _sparse_hessian=len(dofs) >= 128,
                               gaps_value_autodiff=local_values if gaps_value_autodiff is not None else None)
    gradient = np.zeros(len(state))
    gradient[dofs] = c['gradient']
    H = None
    if compute_hessian:
        local = csr_matrix(c['hessian'])
        counts = np.zeros(len(state), dtype=local.indptr.dtype)
        counts[dofs] = np.diff(local.indptr)
        order = np.argsort(dofs)
        local = local[order]
        indptr = np.r_[0, np.cumsum(counts)]
        H = csr_matrix((local.data, dofs[local.indices], indptr),
                       shape=(len(state), len(state)))
        H.sort_indices()
    if return_gap_data:c['gap_dofs'] = dofs.copy()
    return dict(c, gradient=gradient, hessian=H, contact_dofs=len(dofs))
