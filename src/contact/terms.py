"""Weighted contact derivatives with bounded, batched Jacobian evaluations."""
import numpy as np
import torch
from .barrier import barrier_first_second


def _jacobian_rows(values, state, create_graph=False):
    """Differentiate an existing graph, including constant-output functions."""
    rows = []
    for value in values.reshape(-1):
        derivative = None
        if value.requires_grad:
            derivative = torch.autograd.grad(value, state, retain_graph=True,
                create_graph=create_graph, allow_unused=True)[0]
        rows.append(torch.zeros_like(state) if derivative is None else derivative)
    return torch.stack(rows) if rows else state.new_zeros((0, state.numel()))


def _gap_jacobian(gaps_autodiff, state, indices, gaps_subset_autodiff=None):
    rows = []
    for start in range(0, len(indices), 32):
        ids = torch.as_tensor(indices[start:start + 32], dtype=torch.long)
        def selected_gaps(value):
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
        rows.append(block.detach().numpy())
    return np.concatenate(rows, axis=0) if rows else np.empty((0, state.numel()))


def multiple_contact_terms(state, gaps_autodiff, weights, d_hat=0.2,
                           method='gn', filter_ratio=0.1, compute_hessian=True,
                           gaps_subset_autodiff=None, return_gap_data=False):  # ##r##
    if return_gap_data and (method != 'gn' or not compute_hessian):
        raise ValueError('gap derivative data requires a GN Hessian evaluation')
    if method not in ('full', 'gn', 'filtered_gn'):
        raise ValueError(method)
    if d_hat <= 0 or not 0 < filter_ratio < 1:
        raise ValueError('invalid barrier/filter parameters')
    z = torch.tensor(np.asarray(state, dtype=float), dtype=torch.float64, requires_grad=True)
    weights = np.asarray(weights, dtype=float)
    gap_tensor = gaps_autodiff(z)
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
    H = np.zeros((len(z), len(z))) if compute_hessian else None
    energy = 0.
    J = np.empty((0, len(z))) if return_gap_data else None
    if len(indices):
        s = gap_tensor[torch.as_tensor(indices, dtype=torch.long)]
        w = torch.as_tensor(weights[indices], dtype=torch.float64)
        scalar_energy = torch.sum(-w * (s - d_hat)**2 * torch.log(s / d_hat))
        energy = float(scalar_energy.detach())
        if not compute_hessian or method == 'full':
            g = _jacobian_rows(scalar_energy.reshape(1), z, create_graph=compute_hessian)[0]
            gradient = g.detach().numpy()
            if compute_hessian:
                H = _jacobian_rows(g, z).detach().numpy()
        else:
            J = _gap_jacobian(gaps_autodiff, z, indices, gaps_subset_autodiff)
            derivatives = np.array([barrier_first_second(value, d_hat) for value in gaps[indices]])
            b1, coefficient = derivatives[:, 0], derivatives[:, 1].copy()
            if np.any(active):
                coefficient[active[indices]] = barrier_first_second(filter_ratio * d_hat, d_hat)[1]
            gradient = J.T @ (weights[indices] * b1)
            H = J.T @ ((weights[indices] * coefficient)[:, None] * J)
    if not np.all(np.isfinite(gradient)) or (H is not None and not np.all(np.isfinite(H))):
        raise ValueError('nonfinite multi-contact derivatives')
    result = dict(energy=energy, gradient=gradient, hessian=(H + H.T) / 2 if H is not None else None,
                  gaps=gaps, min_gap=float(gaps.min()) if len(gaps) else float('inf'),
                  filter_count=int(np.count_nonzero(active)))
    ##r## Reuse the active GN Jacobian; SR1 never requests a gap Hessian.
    if return_gap_data:
        result.update(gap_indices=indices, gap_jacobian=J, gap_weights=weights,
                      gap_dofs=np.arange(len(z), dtype=int))
    return result


def multiple_contact_terms_sparse(state, gaps_autodiff, weights, dofs, d_hat=.2,
                                  method='gn', compute_hessian=True,
                                  gaps_subset_autodiff=None, return_gap_data=False):  # ##r##
    from scipy.sparse import coo_matrix
    state = np.asarray(state, dtype=float)
    dofs = np.asarray(dofs, dtype=int)
    if dofs.ndim != 1 or len(dofs) != len(np.unique(dofs)) or np.any(dofs < 0) or np.any(dofs >= len(state)):
        raise ValueError('unique valid contact DOFs required')
    base = torch.tensor(state, dtype=torch.float64)
    ids = torch.tensor(dofs, dtype=torch.long)
    def local_gaps(local):
        return gaps_autodiff(base.index_copy(0, ids, local))
    def local_subset(local, query_ids):
        return gaps_subset_autodiff(base.index_copy(0, ids, local), query_ids)
    c = multiple_contact_terms(state[dofs], local_gaps, weights, d_hat=d_hat,
                               method=method, compute_hessian=compute_hessian,
                               gaps_subset_autodiff=local_subset if gaps_subset_autodiff is not None else None,
                               return_gap_data=return_gap_data)
    gradient = np.zeros(len(state))
    gradient[dofs] = c['gradient']
    H = None
    if compute_hessian:
        row, col = np.nonzero(c['hessian'])
        H = coo_matrix((c['hessian'][row, col], (dofs[row], dofs[col])),
                       shape=(len(state), len(state))).tocsr()
    if return_gap_data:c['gap_dofs'] = dofs.copy()
    return dict(c, gradient=gradient, hessian=H, contact_dofs=len(dofs))
