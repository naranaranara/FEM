import numpy as np
import time
from src.optimization.newton import newton_direction
from src.optimization.line_search import line_search
METHODS=('full','gn','filtered_gn','gn_expand','bfgs','lbfgs')

def bfgs_update(B,s,y):
    ys=float(y @ s)
    if ys<=max(1e-30,1e-10*np.linalg.norm(y)*np.linalg.norm(s)):return B,False
    By=B @ y
    updated=B+(1.+float(y @ By)/ys)*np.outer(s,s)/ys-(np.outer(s,By)+np.outer(By,s))/ys
    return .5*(updated+updated.T),True

def lbfgs_direction(g,history,initial_scale=1/3):
    q=g.copy();alpha=[]
    for s,y in reversed(history):
        rho=1./float(y @ s);a=rho*float(s @ q)
        alpha.append(a);q=q-a*y
    scale=initial_scale
    if history:
        s,y=history[-1];scale=float(s @ y)/float(y @ y)
    r=scale*q
    for (s,y),a in zip(history,reversed(alpha)):
        beta=float(y @ r)/float(y @ s)
        r=r+s*(a-beta)
    return -r

def solve(objective,method,max_iter=1200,tol=1e-8,max_alpha=64.,memory=10):
    if method not in METHODS:raise ValueError('unknown method')
    objective.reset();start=time.perf_counter()
    x=objective.initial.copy();n=len(x)
    B=np.eye(n)/3 if method=='bfgs' else None
    history=[];trace=[];updates=0;status='max_iter'
    skipped=0;resets=0;cached=None
    for iteration in range(max_iter+1):
        mode='gn' if method=='gn_expand' else method
        if method in ('bfgs','lbfgs'):mode='gradient'
        if cached is not None and mode=='gradient':
            E,g,H,info=cached;cached=None
        else:E,g,H,info=objective.evaluate(x,mode)
        residual=float(np.linalg.norm(g))
        rec=dict(iteration=iteration,energy=E,residual=residual,position=x.tolist(),**info)
        trace.append(rec)
        if residual<tol:status='converged';break
        if iteration==max_iter:break
        if method=='bfgs':direction=-B @ g;shift=0.
        elif method=='lbfgs':direction=lbfgs_direction(g,history);shift=0.
        else:
            try:direction,shift=newton_direction(H,g,eigen_floor=1e-7)
            except np.linalg.LinAlgError:status='linear_failure';break
        if not np.all(np.isfinite(direction)) or g @ direction>=0:
            direction=-g;resets+=1;history=[]
            if B is not None:B=np.eye(n)/3
        trial,ls=line_search(objective.energy,x,E,g,direction,method=='gn_expand',max_alpha)
        rec.update(regularization=float(shift),**ls)
        if trial is None:status='line_search_failed';break
        new,alpha,_=trial
        rec.update(alpha=float(alpha),direction_norm=float(np.linalg.norm(direction)))
        if method in ('bfgs','lbfgs'):
            cached=objective.evaluate(new,'gradient')
            s=new-x;y=cached[1]-g
            if method=='bfgs':B,accepted=bfgs_update(B,s,y)
            else:
                accepted=float(y @ s)>max(1e-30,1e-10*np.linalg.norm(y)*np.linalg.norm(s))
                if accepted:history.append((s.copy(),y.copy()));history=history[-memory:]
            skipped+=int(not accepted)
        x=new;updates+=1
    elapsed=time.perf_counter()-start
    return dict(status=status,state=x,energy=E,residual=residual,updates=updates,elapsed_s=elapsed,
                trace=trace,counts=objective.counts.copy(),skipped_updates=skipped,resets=resets,
                alpha_over_one=sum(t.get('alpha',0)>1 for t in trace),
                max_accepted_alpha=max([t.get('alpha',0) for t in trace]+[0.]),
                backtracks=sum(t.get('backtracks',0) for t in trace),
                expansions=sum(t.get('expansions',0) for t in trace))
