import time
import weakref
from contextlib import nullcontext
from threadpoolctl import threadpool_limits
import numpy as np
import torch
from src.mechanics.sparse_factor import factorize_sparse
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


def condense_contact_problem(problem,batch=32,progress=None,linear_backend='superlu',setup_cache=None):
    if not problem['sparse'] or problem.get('prepare') is not None:
        raise ValueError('condensation currently requires fixed contact patches and sparse constant FEM')
    if linear_backend not in ('superlu','pardiso'):
        raise ValueError('linear backend must be superlu or pardiso')
    tick=time.perf_counter()
    K=problem['stiffness_free'].tocsr();n=K.shape[0]
    retained=np.asarray(problem['contact_dofs'],dtype=int)
    interior=np.setdiff1d(np.arange(n),retained)
    if not len(retained):raise ValueError('contact DOFs required')
    load_tick=time.perf_counter()
    from src.simulation.setup_cache import matrix_signature
    matrix_key=matrix_signature(K) if setup_cache else None
    saved=setup_cache.load('schur_'+linear_backend) if setup_cache else None
    if saved is not None and (saved.get('matrix_key')!=matrix_key
            or not np.array_equal(saved['retained'],retained)
            or not np.array_equal(saved['interior'],interior)
            or saved['S'].shape!=(len(retained),len(retained))
            or not np.all(np.isfinite(saved['S']))):
        saved=None
        setup_cache.hits['schur_'+linear_backend]=False
    schur_load_s=time.perf_counter()-load_tick if setup_cache else 0.
    S=saved['S'] if saved is not None else K[retained][:,retained].toarray()
    Kic=K[interior][:,retained].tocsc()
    Kci=Kic.T.tocsr()
    if progress:progress('interior factorization: %d DOFs, backend=%s' % (len(interior),linear_backend))
    factor_tick=time.perf_counter()
    factor=factorize_sparse(K[interior][:,interior],linear_backend) if len(interior) else None
    factor_s=time.perf_counter()-factor_tick
    if progress:progress('interior factorization complete: %.2fs; Schur columns=%d' % (factor_s,len(retained)))
    schur_tick=time.perf_counter();last_report=schur_tick
    if factor is not None and saved is None:
        for start in range(0,len(retained),batch):
            stop=min(start+batch,len(retained))
            S[:,start:stop]-=Kci@factor.solve(Kic[:,start:stop].toarray())
            if progress and (time.perf_counter()-last_report>=10 or stop==len(retained)):
                progress('Schur columns %d/%d, %.2fs' % (stop,len(retained),time.perf_counter()-schur_tick))
                last_report=time.perf_counter()
    if saved is None:
        np.add(S,S.T,out=S)
        S*=.5
        schur_s=time.perf_counter()-schur_tick
        if setup_cache:setup_cache.store('schur_'+linear_backend,
            dict(S=S,retained=retained,interior=interior,matrix_key=matrix_key))
    else:
        schur_s=0.
        if progress:progress('Schur matrix reused; column solves skipped')
    reference=problem['reference_flat'][problem['free']].copy()
    qc=reference[retained]
    ids_t=torch.tensor(retained,dtype=torch.long)
    reference_t=torch.tensor(reference,dtype=torch.float64)
    cache={};result={}
    # ##r## Cache only exact coordinates within one prescribed load.
    expanded_cache={};gap_cache={};base_cache={};direction_cache={}
    def refresh():
        expanded_cache.clear();gap_cache.clear();base_cache.clear()
        E,g,_=problem['base'](reference)
        fi=g[interior]
        eliminated=factor.solve(fi) if factor is not None else fi
        cache.update(fi=fi,f=g[retained]-Kci@eliminated,
                     constant=float(E-.5*fi@eliminated))
    def set_closure(value):
        problem['set_closure'](value);refresh();result['closure']=float(value)
    refresh()
    def expanded(q):
        q=np.asarray(q,dtype=float)
        if q.shape!=qc.shape:raise ValueError('invalid condensed coordinates')
        if 'q' in expanded_cache and np.array_equal(q,expanded_cache['q']):
            return expanded_cache['x']
        x=reference.copy();x[retained]=q
        if factor is not None:
            x[interior]-=factor.solve(Kic@(q-qc)+cache['fi'])
        expanded_cache.update(q=q.copy(),x=x)
        return x
    def expand(q):return expanded(q).copy()
    def full_state(q):return problem['full_state'](expanded(q))
    def gaps_ad(q,indices=None):
        full=reference_t.index_copy(0,ids_t,q)
        return problem['gaps_ad'](full,indices=indices)
    def gap_values(q):
        a=q.detach().numpy()
        callback=problem['gaps_ad']
        if problem.get('_no_grad_gaps_ad') is not callback:
            return gaps_ad(q).detach()
        if (gap_cache.get('callback') is not callback or 'q' not in gap_cache
                or not np.array_equal(a,gap_cache['q'])):
            with torch.no_grad():values=gaps_ad(q).detach().clone()
            gap_cache.update(q=a.copy(),callback=callback,values=values)
        return gap_cache['values'].clone()
    def gaps_jacobian(q,indices):
        # ##r## Surface dependencies are entirely retained by condensation.
        callback=problem.get('gap_jacobian')
        if callback is None:return None
        full=reference.copy();full[retained]=q
        J=callback(full,indices)
        return None if J is None else J[:,retained]
    def base(q):
        if 'q' not in base_cache or not np.array_equal(q,base_cache['q']):
            u=q-qc;Su=S@u
            base_cache.update(q=np.asarray(q).copy(),energy=float(.5*u@Su+cache['f']@u+cache['constant']),
                              gradient=Su+cache['f'])
        return base_cache['energy'],base_cache['gradient'].copy(),S
    def feasible(q):return problem['feasible'](expanded(q))
    def contact(q,method,return_gap_data=False):
        safe=problem.get('_no_grad_gaps_ad') is problem['gaps_ad']
        return multiple_contact_terms(q,gaps_ad,problem['weights'],d_hat=problem['d_hat'],
            method=method,gaps_subset_autodiff=gaps_ad,return_gap_data=return_gap_data,
            _sparse_hessian=True,
            gaps_value_autodiff=gap_values if safe else None,
            gaps_jacobian=gaps_jacobian if safe else None)
    old_keys,old_supports=problem['contact_layout']()
    remap=np.full(n,-1,dtype=int);remap[retained]=np.arange(len(retained))
    supports=[remap[np.asarray(ids,dtype=int)] for ids in old_supports]
    if any(np.any(ids<0) for ids in supports):raise ValueError('condensed support misses a geometric dependency')
    def objective(q,method,model=None):
        learned=method in ('sr1_gap','structured_bfgs_gap')
        if learned and model is None:raise ValueError('curvature model required')
        E,g,H=base(q);c=contact(q,'gn' if learned else method,return_gap_data=learned)
        # ##r## Preserve constant matrix identity for factor reuse before contact.
        if np.any(c['gaps']<problem['d_hat']):
            # ##r## Assemble only the nonzero contact block into the dense FEM matrix.
            H=H.copy();addition=c['hessian'].tocoo()
            H[addition.row,addition.col]+=addition.data
        info=dict(min_gap=c['min_gap'],filter_count=c['filter_count'])
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
        # ##r## Remember provenance, without keeping obsolete dense matrices alive.
        direction_cache['hessian']=weakref.ref(H) if not learned else None
        return E+c['energy'],g+c['gradient'],H,info
    def energy(q):
        if not feasible(q):return np.inf
        try:
            with torch.no_grad():gaps=gap_values(torch.tensor(q,dtype=torch.float64)).numpy()
        except ValueError:return np.inf
        if np.any(~np.isfinite(gaps)) or np.any(gaps<=0):return np.inf
        return base(q)[0]+float(problem['weights']@np.array([barrier_value(s,problem['d_hat']) for s in gaps]))
    def direction(H,g,eigen_floor=1e-7):
        known=direction_cache.get('hessian')
        if H is not S and (known is None or known() is not H):
            return cholesky_direction(H,g,eigen_floor)
        # ##r## These assembled matrices are symmetric already; avoid a dense copy.
        # Four BLAS threads help large factorizations, and restore the caller's limit.
        limits=threadpool_limits(4,user_api='blas') if len(g)>=2000 else nullcontext()
        with limits:
            if H is S and 'factor' in direction_cache:
                current=direction_cache['factor']
            else:
                try:current=cho_factor(H,check_finite=False)
                except np.linalg.LinAlgError:return cholesky_direction(H,g,eigen_floor)
                if H is S:direction_cache['factor']=current
            p=cho_solve(current,-g,check_finite=False)
        if not np.all(np.isfinite(p)):raise np.linalg.LinAlgError('nonfinite condensed direction')
        return (-g if g@p>=0 else p),0.
    def close():
        direction_cache.clear();expanded_cache.clear();gap_cache.clear();base_cache.clear()
        getattr(factor,'close',lambda:None)()
    result.update(initial=problem['initial'][retained].copy(),expand=expand,full_state=full_state,
        unpack=lambda q:problem['unpack'](expanded(q)),base=base,energy=energy,objective=objective,
        feasible=feasible,gaps_ad=gaps_ad,diagnostics=lambda q:problem['diagnostics'](expanded(q)),
        set_closure=set_closure,closure=problem['closure'],direction_solver=direction,
        d_hat=problem['d_hat'],weights=problem['weights'],retained=retained,
        original=problem,stiffness_free=S,condensation_s=time.perf_counter()-tick,
        factorization_s=factor_s,schur_assembly_s=schur_s,schur_cache_load_s=schur_load_s,linear_backend=linear_backend,
        close=close)
    return result
