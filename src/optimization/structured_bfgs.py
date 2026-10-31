import time
import numpy as np
from src.optimization.newton import newton_direction
from src.optimization.line_search import line_search

def structured_bfgs_update(A,G_new,d,z,skip_tol=1e-10, eigen_floor=1e-7):
    A, G_new = np.asarray(A, dtype=float), np.asarray(G_new, dtype=float)
    d, z = np.asarray(d, dtype=float), np.asarray(z, dtype=float)
    if d.ndim != 1 or z.shape != d.shape or A.shape != (d.size, d.size) or G_new.shape != A.shape:
        raise ValueError('incompatible correction, GN, step or secant shapes')
    if not 0 < skip_tol < 1 or not np.isfinite(eigen_floor) or eigen_floor <= 0:
        raise ValueError('invalid skip tolerance or eigenvalue floor')
    info = dict(update_applied=False, skip_reason='', seed_shift=0., secant_error=0.)
    if not all(np.all(np.isfinite(v)) for v in (A, G_new, d, z)):
        info['skip_reason'] = 'nonfinite_input'
        return A.copy(), info
    if np.linalg.norm(d) == 0:
        info['skip_reason'] = 'zero_step'
        return A.copy(), info
    u = G_new @ d + z
    du = float(d @ u)
    if du <= max(1e-30, skip_tol * np.linalg.norm(d) * np.linalg.norm(u)):
        info['skip_reason'] = 'nonpositive_or_small_curvature'
        return A.copy(), info
    M = .5 * (G_new + A + (G_new + A).T)
    # ##r## Cholesky certifies the same eigenvalue floor without a full spectrum.
    from scipy.linalg import cho_factor
    try:
        cho_factor(M-eigen_floor*np.eye(len(d)),check_finite=False)
        shift=0.
    except np.linalg.LinAlgError:
        shift=max(0.,eigen_floor-float(np.linalg.eigvalsh(M)[0]))
    M = M + shift * np.eye(len(d))
    Md = M @ d
    dMd = float(d @ Md)
    if dMd <= max(1e-30, skip_tol * np.linalg.norm(d) * np.linalg.norm(Md)):
        info['skip_reason'] = 'small_seed_curvature'
        return A.copy(), info
    updated = M - G_new - np.outer(Md, Md) / dMd + np.outer(u, u) / du
    updated = .5 * (updated + updated.T)
    if not np.all(np.isfinite(updated)):
        info['skip_reason'] = 'nonfinite_update'
        return A.copy(), info
    info.update(update_applied=True, seed_shift=shift,
                secant_error=float(np.linalg.norm(updated @ d - z)))
    return updated, info

def solve(objective, max_iter=1200, tol=1e-8, secant='total'):
    if secant not in ('total', 'gap'):
        raise ValueError('secant must be total or gap')
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError('max_iter must be nonnegative and tol positive')
    objective.reset()
    start = time.perf_counter()
    x = objective.initial.copy()
    A = np.zeros((len(x), len(x)))
    previous = None
    trace = []
    updates = accepted = skipped = resets = 0
    status = 'max_iter'
    for iteration in range(max_iter + 1):
        E, g, G, info = objective.evaluate(x, 'gn', return_gap_gradient=secant == 'gap')
        gs = info.pop('gap_gradient', None)
        b1 = info.pop('barrier_first', None)
        residual = float(np.linalg.norm(g))
        record = dict(iteration=iteration, energy=E, residual=residual,
                        position=x.tolist(), **info)
        trace.append(record)
        if residual < tol:
            status = 'converged'
            break
        if iteration == max_iter:
            break
        update= dict(update_applied=False, skip_reason='initial', seed_shift=0.)
        if info['gap'] >= .2:
            resets += int(np.any(A))
            A.fill(0)
            update['skip_reason'] = 'inactive_contact'
        elif previous is not None:
            old_x, old_g, old_gs = previous
            d= x-old_x
            z= g- old_g - G@d if secant =='total' else b1*(gs-old_gs)
            A, update = structured_bfgs_update(A,G,d,z)
            accepted += int(update['update_applied'])
            skipped += int(not update['update_applied'])
        record.update(update)
        try:
            direction, shift = newton_direction(G+A,g,eigen_floor=1e-7)
        except np.linalg.LinAlgError:
            status = 'linear_failure'
            break
        trial, ls= line_search(objective.energy,x,E,g,direction,expand=False)
        record.update(regularization=float(shift), directional_slope=float(g@direction),**ls)
        if trial is None:
            status = 'line_search_failed'
            break
        new_x, alpha,_=trial
        record.update(alpha=float(alpha), step_norm=float(np.linalg.norm(new_x-x)))
        previous = (x.copy(),g.copy(), gs.copy() if gs is not None else None)
        x=new_x
        updates +=1
    return dict(status=status, state=x, energy=E, residual=residual, updates=updates,
                elapsed_s=time.perf_counter() - start, trace=trace,
                counts=objective.counts.copy(), bfgs_updates=accepted,
                bfgs_skips=skipped, bfgs_resets=resets)

class MultiGapStructuredBFGS:
    def __init__(self):
        self.A=None;self.previous=None
        self.totals=dict(bfgs_updates=0,bfgs_skips=0,bfgs_resets=0)

    def correction(self,state,contact,G,d_hat):
        from scipy.sparse import csr_matrix,coo_matrix,issparse
        from src.contact.barrier import barrier_first_second
        state=np.asarray(state,dtype=float);n=len(state)
        ids=np.asarray(contact['gap_dofs'],dtype=int)
        indices=np.asarray(contact['gap_indices'],dtype=int)
        J=np.asarray(contact['gap_jacobian'],dtype=float)
        weights=np.asarray(contact['gap_weights'],dtype=float)
        gaps=np.asarray(contact['gaps'],dtype=float)
        if J.shape!=(len(indices),len(ids)) or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=n):
            raise ValueError('invalid structured BFGS contact layout')
        if (indices.ndim!=1 or len(np.unique(indices))!=len(indices) or np.any(indices<0) or np.any(indices>=len(gaps))
                or weights.shape!=gaps.shape or np.any(~np.isfinite(weights)) or np.any(weights<=0)
                or np.any(~np.isfinite(J)) or np.any(~np.isfinite(gaps)) or np.any(gaps<=0)
                or not np.isfinite(d_hat) or d_hat<=0):
            raise ValueError('invalid structured BFGS gap data')
        counts=dict(bfgs_updates=0,bfgs_skips=0,bfgs_resets=0)
        info=dict(update_applied=False,skip_reason='initial',seed_shift=0.)
        x=state[ids]
        if len(indices)==0:
            counts['bfgs_resets']=int(self.previous is not None)
            self.previous=None;self.A=None
        else:
            signature=(tuple(ids),tuple(indices),tuple(weights))
            if self.previous is None or self.previous['signature']!=signature:
                counts['bfgs_resets']=int(self.previous is not None)
                self.A=np.zeros((len(ids),len(ids)))
            elif not np.array_equal(self.previous['x'],x):
                d=x-self.previous['x']
                coefficients=np.array([weights[i]*barrier_first_second(gaps[i],d_hat)[0] for i in indices])
                z=coefficients@(J-self.previous['J'])
                block=G[ids][:,ids].toarray() if issparse(G) else np.asarray(G)[np.ix_(ids,ids)]
                self.A,info=structured_bfgs_update(self.A,block,d,z)
                counts['bfgs_updates']=int(info['update_applied'])
                counts['bfgs_skips']=int(not info['update_applied'])
            self.previous=dict(signature=signature,x=x.copy(),J=J.copy())
        C=csr_matrix((n,n))
        if self.A is not None:
            row,col=np.nonzero(self.A)
            C=coo_matrix((self.A[row,col],(ids[row],ids[col])),shape=(n,n)).tocsr()
        for name,value in counts.items():self.totals[name]+=value
        counts.update(bfgs_seed_shift=float(info.get('seed_shift',0.)),
                      bfgs_correction_norm=float(np.linalg.norm(self.A)) if self.A is not None else 0.)
        return C,counts
