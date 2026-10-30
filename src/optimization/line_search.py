import numpy as np

def line_search(energy,x,E,g,direction,expand=False,max_alpha=64.):
    slope=float(g @ direction)
    if slope>=0:raise ValueError('line search requires a descent direction')
    alpha=1.;trials=0;backtracks=0;expansions=0
    roundoff=10*np.finfo(float).eps*max(1.,abs(E))
    while backtracks<50:
        trial=x+alpha*direction;Et=energy(trial);trials+=1
        if np.isfinite(Et) and Et<=E+1e-4*alpha*slope+roundoff:break
        alpha*=.5;backtracks+=1
    else:return None,dict(trials=trials,backtracks=backtracks,expansions=0)
    best=(trial,alpha,Et)
    if expand:
        while alpha<max_alpha:
            candidate=min(2*alpha,max_alpha)
            trial=x+candidate*direction;Et=energy(trial);trials+=1
            if (not np.isfinite(Et) or Et>E+1e-4*candidate*slope+roundoff
                    or Et>=best[2]):break
            best=(trial,candidate,Et);alpha=candidate;expansions+=1
    return best,dict(trials=trials,backtracks=backtracks,expansions=expansions)
