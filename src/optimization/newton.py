import numpy as np
from time import perf_counter

def newton_direction(H,g,eigen_floor=1e-8,phase_timer=None):
    tick=perf_counter() if phase_timer else None
    from scipy.sparse import issparse
    if issparse(H):return sparse_newton_direction(H,g,eigen_floor,phase_timer)
    H=np.asarray(H,dtype=float)
    g=np.asarray(g,dtype=float)
    shift=max(0.,float(eigen_floor-np.linalg.eigvalsh(H)[0]))
    if phase_timer:
        phase_timer.record('stabilization_eigenvalues',tick);tick=perf_counter()
    if shift:
        H=H.copy()
        H.flat[::len(g)+1]+=shift
    if phase_timer:
        phase_timer.record('stabilization_shift',tick);tick=perf_counter()
    direction=np.linalg.solve(H,-g)
    if phase_timer:phase_timer.record('linear_solve',tick)
    if g @ direction>=0:
        direction=-g
    return direction,shift


def minimize_static(initial,objective,energy,max_iter=300,tol=1e-8,eigen_floor=1e-7,max_step=None,prepare=None,direction_solver=None):
    import time
    state=np.asarray(initial,dtype=float).copy()
    prepared=prepare(state) if prepare is not None else {}
    if not np.isfinite(energy(state)):raise ValueError('infeasible initial state')
    trace=[];updates=0;status='max_iter';evaluated=False
    start=time.perf_counter()
    for iteration in range(max_iter):
        prepared=prepared if iteration==0 else (prepare(state) if prepare is not None else {})
        E,g,H,info=objective(state)
        evaluated=True
        residual=float(np.linalg.norm(g))
        rec=dict(iteration=iteration,energy=float(E),residual=residual,**info)
        rec.update(prepared or {})
        trace.append(rec)
        if residual<tol:status='converged';break
        try:
            direction,shift=(newton_direction if direction_solver is None else direction_solver)(H,g,eigen_floor)
        except np.linalg.LinAlgError:status='linear_failure';break
        rec['regularization']=float(shift)
        alpha=1.;slope=float(g @ direction)
        if max_step is not None:
            alpha=float(max_step(state,state+direction))
            if not np.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('CCD step bound outside [0,1]')
            rec['step_limit_alpha']=alpha
            rec['ccd_alpha']=alpha
            if alpha<=1e-15:status='ccd_blocked';break
        for backtracks in range(50):
            trial=state+alpha*direction
            if energy(trial)<=E+1e-4*alpha*slope+10*np.finfo(float).eps*max(1.,abs(E)):break
            alpha*=.5
        else:status='line_search_failed';break
        rec.update(alpha=alpha,backtracks=backtracks)
        state=trial;updates+=1;evaluated=False
    elapsed=time.perf_counter()-start
    if not evaluated:
        if prepare is not None:prepare(state)
        E,g,_,info=objective(state)
    return dict(state=state,status=status,updates=updates,residual=float(np.linalg.norm(g)),
                energy=float(E),elapsed_s=elapsed,trace=trace,info=info)

def sparse_newton_direction(H,g,eigen_floor=1e-8,phase_timer=None):
    import warnings
    from scipy.sparse import eye
    from scipy.sparse.linalg import eigsh,spsolve,MatrixRankWarning,ArpackError
    tick=perf_counter() if phase_timer else None
    H=H.tocsr();g=np.asarray(g,dtype=float);n=len(g)
    diagonal=H.diagonal()
    radii=np.asarray(abs(H).sum(axis=1)).ravel()-abs(diagonal)
    lower=float(np.min(diagonal-radii))
    if lower>=eigen_floor:minimum=lower
    elif n==1:minimum=float(diagonal[0])
    elif n==2:
        minimum=float(.5*(diagonal.sum()-np.hypot(diagonal[0]-diagonal[1],2*H[0,1])))
    else:
        try:
            minimum=float(eigsh(H,k=1,which='SA',return_eigenvectors=False,
                v0=np.random.default_rng(0).normal(size=n),tol=1e-9,maxiter=max(1000,10*n))[0])
        except ArpackError:
            minimum=lower
    shift=max(0.,float(eigen_floor-minimum))
    if phase_timer:
        phase_timer.record('stabilization_eigenvalues',tick);tick=perf_counter()
    if shift:H=H+shift*eye(n,format='csr')
    if phase_timer:
        phase_timer.record('stabilization_shift',tick);tick=perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error',MatrixRankWarning)
        try:direction=spsolve(H.tocsc(),-g)
        except (MatrixRankWarning,RuntimeError) as exc:raise np.linalg.LinAlgError(str(exc))
    if not np.all(np.isfinite(direction)):raise np.linalg.LinAlgError('nonfinite sparse direction')
    if phase_timer:phase_timer.record('linear_solve',tick)
    if g@direction>=0:direction=-g
    return direction,shift
