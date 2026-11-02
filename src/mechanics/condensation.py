"""상수 선형 탄성체의 내부 자유도를 정확히 축약해 접촉 문제를 풀어요."""
import time
import numpy as np
import torch
from src.mechanics.sparse_factor import factorize_sparse  # ##r##
from scipy.linalg import cho_factor,cho_solve
from src.contact.terms import multiple_contact_terms
from src.contact.barrier import barrier_value


def cholesky_direction(H,g,eigen_floor=1e-7):
    H=np.asarray(H,dtype=float);H=.5*(H+H.T)
    shift=0.
    try:factor=cho_factor(H,check_finite=False)
    except np.linalg.LinAlgError:
        shift=max(0.,eigen_floor-float(np.linalg.eigvalsh(H)[0]))
        factor=cho_factor(H+shift*np.eye(len(g)),check_finite=False)
    p=cho_solve(factor,-g,check_finite=False)
    if not np.all(np.isfinite(p)):raise np.linalg.LinAlgError('nonfinite condensed direction')
    if g@p>=0:p=-g
    return p,shift


def condense_contact_problem(problem,batch=32,progress=None,linear_backend='superlu'):
    r"""접촉 자유도를 남기고 내부 선형 탄성 자유도를 정확히 제거해요.

    $$S=K_{cc}-K_{ci}K_{ii}^{-1}K_{ic}$$
    """
    if not problem['sparse'] or problem.get('prepare') is not None:
        raise ValueError('condensation currently requires fixed contact patches and sparse constant FEM')
    if linear_backend not in ('superlu','pardiso'):
        raise ValueError('linear backend must be superlu or pardiso')
    tick=time.perf_counter()
    K=problem['stiffness_free'].tocsr();n=K.shape[0]
    retained=np.asarray(problem['contact_dofs'],dtype=int)
    interior=np.setdiff1d(np.arange(n),retained)
    if not len(retained):raise ValueError('contact DOFs required')
    # 접촉 강성 블록으로 축약 행렬을 초기화해요: $$S\leftarrow K_{cc}$$
    S=K[retained][:,retained].toarray()
    Kic=K[interior][:,retained].tocsc()
    Kci=Kic.T.tocsr()
    # ##r## Report expensive setup separately from nonlinear iterations.
    if progress:progress('interior factorization: %d DOFs, backend=%s' % (len(interior),linear_backend))
    factor_tick=time.perf_counter()
    factor=factorize_sparse(K[interior][:,interior],linear_backend) if len(interior) else None
    factor_s=time.perf_counter()-factor_tick
    if progress:progress('interior factorization complete: %.2fs; Schur columns=%d' % (factor_s,len(retained)))
    schur_tick=time.perf_counter();last_report=schur_tick
    if factor is not None:
        for start in range(0,len(retained),batch):
            stop=min(start+batch,len(retained))
            # 열 묶음별로 내부 자유도의 영향을 제거해요: $$S_{:,b}\leftarrow S_{:,b}-K_{ci}K_{ii}^{-1}K_{ic,:,b}$$
            S[:,start:stop]-=Kci@factor.solve(Kic[:,start:stop].toarray())
            if progress and (time.perf_counter()-last_report>=10 or stop==len(retained)):
                progress('Schur columns %d/%d, %.2fs' % (stop,len(retained),time.perf_counter()-schur_tick))
                last_report=time.perf_counter()
    schur_s=time.perf_counter()-schur_tick
    # 부동소수점 비대칭을 제거할 분자를 모아요: $$S\leftarrow S+S^T$$
    np.add(S,S.T,out=S)
    # 기존 축약 행렬과 같은 대칭화 계수를 적용해요: $$S\leftarrow S/2$$
    S*=.5
    reference=problem['reference_flat'][problem['free']].copy()
    qc=reference[retained]
    ids_t=torch.tensor(retained,dtype=torch.long)
    reference_t=torch.tensor(reference,dtype=torch.float64)
    cache={};result={}
    def refresh():
        E,g,_=problem['base'](reference)
        fi=g[interior]
        eliminated=factor.solve(fi) if factor is not None else fi
        cache.update(fi=fi,f=g[retained]-Kci@eliminated,
                     constant=float(E-.5*fi@eliminated))
    def set_closure(value):
        problem['set_closure'](value);refresh();result['closure']=float(value)
    refresh()
    def expand(q):
        q=np.asarray(q,dtype=float)
        if q.shape!=qc.shape:raise ValueError('invalid condensed coordinates')
        x=reference.copy();x[retained]=q
        if factor is not None:
            x[interior]-=factor.solve(Kic@(q-qc)+cache['fi'])
        return x
    def full_state(q):return problem['full_state'](expand(q))
    def gaps_ad(q,indices=None):
        full=reference_t.index_copy(0,ids_t,q)
        return problem['gaps_ad'](full,indices=indices)
    def base(q):
        u=q-qc;Su=S@u
        return float(.5*u@Su+cache['f']@u+cache['constant']),Su+cache['f'],S
    def feasible(q):return problem['feasible'](expand(q))
    def contact(q,method,return_gap_data=False):
        """원래 간격 콜백에만 값 전용 경로를 허용해 축약 접촉을 계산해요."""
        safe=problem.get('_no_grad_gaps_ad') is problem['gaps_ad']
        return multiple_contact_terms(q,gaps_ad,problem['weights'],d_hat=problem['d_hat'],
            method=method,gaps_subset_autodiff=gaps_ad,return_gap_data=return_gap_data,
            gaps_value_autodiff=gaps_ad if safe else None)
    old_keys,old_supports=problem['contact_layout']()
    remap=np.full(n,-1,dtype=int);remap[retained]=np.arange(len(retained))
    supports=[remap[np.asarray(ids,dtype=int)] for ids in old_supports]
    if any(np.any(ids<0) for ids in supports):raise ValueError('condensed support misses a geometric dependency')
    def objective(q,method,model=None):
        learned=method in ('sr1_gap','structured_bfgs_gap')
        if learned and model is None:raise ValueError('curvature model required')
        E,g,H=base(q);c=contact(q,'gn' if learned else method,return_gap_data=learned)
        H=H+c['hessian'];info=dict(min_gap=c['min_gap'],filter_count=c['filter_count'])
        if method=='sr1_gap':
            C,data=model.correction(q,c,old_keys,supports,problem['d_hat'])
            H=H+C.toarray();info.update(data)
        elif method=='structured_bfgs_gap':
            dense_correction=getattr(model,'correction_dense',None)
            if callable(dense_correction):
                C,data=dense_correction(q,c,H,problem['d_hat'])
            else:
                C,data=model.correction(q,c,H,problem['d_hat']);C=C.toarray()
            H=H+C;info.update(data)
        return E+c['energy'],g+c['gradient'],H,info
    def energy(q):
        if not feasible(q):return np.inf
        try:gaps=gaps_ad(torch.tensor(q,dtype=torch.float64)).detach().numpy()
        except ValueError:return np.inf
        if np.any(~np.isfinite(gaps)) or np.any(gaps<=0):return np.inf
        return base(q)[0]+float(problem['weights']@np.array([barrier_value(s,problem['d_hat']) for s in gaps]))
    result.update(initial=problem['initial'][retained].copy(),expand=expand,full_state=full_state,
        unpack=lambda q:problem['unpack'](expand(q)),base=base,energy=energy,objective=objective,
        feasible=feasible,gaps_ad=gaps_ad,diagnostics=lambda q:problem['diagnostics'](expand(q)),
        set_closure=set_closure,closure=problem['closure'],direction_solver=cholesky_direction,
        d_hat=problem['d_hat'],weights=problem['weights'],retained=retained,
        original=problem,stiffness_free=S,condensation_s=time.perf_counter()-tick,
        factorization_s=factor_s,schur_assembly_s=schur_s,linear_backend=linear_backend,
        close=getattr(factor,'close',lambda:None))
    return result
