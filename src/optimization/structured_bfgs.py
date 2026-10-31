"""알려진 접촉 곡률을 유지하면서 빠진 곡률을 Structured BFGS로 갱신해요."""
import time
import numpy as np
from src.optimization.newton import newton_direction
from src.optimization.line_search import line_search

def structured_bfgs_update(A,G_new,d,z,skip_tol=1e-10, eigen_floor=1e-7):
    r"""기존 안정화와 곡률 조건을 유지하며 빠진 곡률 보정 행렬을 갱신해요.

    $$A_+=M-G_+-\frac{(Md)(Md)^T}{d^TMd}+\frac{uu^T}{d^Tu},\qquad u=G_+d+z$$
    """
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
    # 새로 알려진 곡률과 빠진 곡률의 시컨트를 합해요: $$u=G_+d+z$$
    u = G_new @ d + z
    # 시컨트 방향의 곡률을 구해요: $$d_u=d^Tu$$
    du = float(d @ u)
    if du <= max(1e-30, skip_tol * np.linalg.norm(d) * np.linalg.norm(u)):
        info['skip_reason'] = 'nonpositive_or_small_curvature'
        return A.copy(), info
    # 알려진 곡률과 이전 보정을 한 번만 합해요: $$M\leftarrow G_++A$$
    M = G_new + A
    # 대칭화의 분자를 같은 작업공간에 모아요: $$M\leftarrow M+M^T$$
    np.add(M,M.T,out=M)
    # 기존 전체 곡률의 대칭화를 마쳐요: $$M\leftarrow M/2$$
    M *= .5
    from scipy.linalg import cho_factor
    probe=np.array(M,order='F',copy=True)
    # 같은 고유값 하한을 대각 성분에서 빼서 검사해요: $$P=M-\lambda_{\mathrm{floor}}I$$
    probe[np.diag_indices_from(probe)]-=eigen_floor
    try:
        cho_factor(probe,overwrite_a=True,check_finite=False)
        shift=0.
    except np.linalg.LinAlgError:
        # 기존 최소 고유값 기준의 안정화 이동량을 구해요: $$\mu=\max(0,\lambda_{\mathrm{floor}}-\lambda_{\min}(M))$$
        shift=max(0.,eigen_floor-float(np.linalg.eigvalsh(M)[0]))
    del probe
    # 전체 곡률의 대각 성분에만 이동량을 더해요: $$M_{ii}\leftarrow M_{ii}+\mu$$
    M[np.diag_indices_from(M)]+=shift
    # 안정화된 전체 곡률의 방향곱을 구해요: $$m_d=Md$$
    Md = M @ d
    # 안정화된 방향 곡률을 구해요: $$d_M=d^TMd$$
    dMd = float(d @ Md)
    if dMd <= max(1e-30, skip_tol * np.linalg.norm(d) * np.linalg.norm(Md)):
        info['skip_reason'] = 'small_seed_curvature'
        return A.copy(), info
    # 알려진 곡률을 제거해 보정 행렬의 시작값을 구해요: $$A_+\leftarrow M-G_+$$
    updated = M - G_new
    W=M
    # 이전 방향곱의 외적을 같은 작업공간에 보관해요: $$W\leftarrow m_dm_d^T$$
    np.multiply(Md[:,None],Md[None,:],out=W)
    # 제거할 기존 방향 곡률을 정규화해요: $$W\leftarrow W/d_M$$
    W /= dMd
    # 기존 방향 곡률을 보정에서 제거해요: $$A_+\leftarrow A_+-W$$
    updated -= W
    # 새 시컨트 외적으로 작업공간을 재사용해요: $$W\leftarrow uu^T$$
    np.multiply(u[:,None],u[None,:],out=W)
    # 추가할 새 방향 곡률을 정규화해요: $$W\leftarrow W/d_u$$
    W /= du
    # 새 방향 곡률을 보정에 더해요: $$A_+\leftarrow A_++W$$
    updated += W
    # 최종 보정의 대칭화 분자를 모아요: $$A_+\leftarrow A_++A_+^T$$
    np.add(updated,updated.T,out=updated)
    # 최종 보정을 기존과 같은 계수로 대칭화해요: $$A_+\leftarrow A_+/2$$
    updated *= .5
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

    def correction(self,state,contact,G,d_hat,*,_dense=False):
        """활성 접촉 집합의 시컨트 보정을 갱신해 전역 행렬로 반환해요."""
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
                if issparse(G):block=G[ids][:,ids].toarray()
                elif len(ids)==n and np.array_equal(ids,np.arange(n)):block=np.asarray(G)
                else:block=np.asarray(G)[np.ix_(ids,ids)]
                self.A,info=structured_bfgs_update(self.A,block,d,z)
                counts['bfgs_updates']=int(info['update_applied'])
                counts['bfgs_skips']=int(not info['update_applied'])
            self.previous=dict(signature=signature,x=x.copy(),J=J.copy())
        if _dense:
            if self.A is not None and len(ids)==n and np.array_equal(ids,np.arange(n)):
                C=self.A.copy()
            else:
                C=np.zeros((n,n))
                if self.A is not None:C[np.ix_(ids,ids)]=self.A
        else:
            C=csr_matrix((n,n))
            if self.A is not None:
                row,col=np.nonzero(self.A)
                C=coo_matrix((self.A[row,col],(ids[row],ids[col])),shape=(n,n)).tocsr()
        for name,value in counts.items():self.totals[name]+=value
        counts.update(bfgs_seed_shift=float(info.get('seed_shift',0.)),
                      bfgs_correction_norm=float(np.linalg.norm(self.A)) if self.A is not None else 0.)
        return C,counts

    def correction_dense(self,state,contact,G,d_hat):
        """조밀 해법에서 희소 형식으로 왕복하지 않고 독립적인 보정 배열을 반환해요."""
        correction=self.correction
        if getattr(correction,'__func__',None) is not MultiGapStructuredBFGS.correction:
            C,info=correction(state,contact,G,d_hat)
            return C.toarray(),info
        return correction(state,contact,G,d_hat,_dense=True)
