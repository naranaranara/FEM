import numpy as np
import time
from src.optimization.timing import PhaseTimer
from src.optimization.newton import newton_direction
from src.optimization.baseline import METHODS,solve as baseline_solve
from src.optimization.line_search import line_search
SR1_METHODS=('sr1_total','sr1_structured','sr1_gap')

def sr1_update(B,d,y,skip_tol=1e-8):
    residual=y-B@d
    denom=float(residual@d)
    scale=float(np.linalg.norm(residual)*np.linalg.norm(d))
    info=dict(update_applied=False,denominator=denom,skip_reason='')
    if scale==0 or abs(denom)<=max(1e-30,skip_tol*scale):
        info['skip_reason']='small_denominator';return B,info
    updated=B+np.outer(residual,residual)/denom
    if not np.all(np.isfinite(updated)):
        info['skip_reason']='nonfinite_update';return B,info
    info['update_applied']=True
    return .5*(updated+updated.T),info

def solve(objective,method,max_iter=1200,tol=1e-8,profile=False,observer=None):
    if (profile or observer) and method not in SR1_METHODS:raise ValueError("hooks require SR1")
    timer=PhaseTimer() if profile else None
    if method in METHODS:return baseline_solve(objective,method,max_iter=max_iter,tol=tol)
    if method not in SR1_METHODS:raise ValueError(method)
    objective.reset();start=time.perf_counter();x=objective.initial.copy();n=len(x)
    B=np.zeros((n,n));previous=None;trace=[];updates=0;accepted=0;skipped=0;resets=0
    status='max_iter'
    for k in range(max_iter+1):
        mode='gradient' if method=='sr1_total' and k>0 else 'gn'
        E,g,G,info=objective.evaluate(x,mode,return_gap_gradient=method=='sr1_gap',phase_timer=timer)
        gs=info.pop('gap_gradient',None);b1=info.pop('barrier_first',None)
        if method=='sr1_total' and k==0:B=G.copy()
        res=float(np.linalg.norm(g))
        record=dict(iteration=k,energy=E,residual=res,position=x.tolist(),**info)
        trace.append(record)
        if res<tol:status='converged';break
        if k==max_iter:break
        tick=time.perf_counter() if timer else None
        inactive=info['gap']>=.2
        update=dict(update_applied=False,skip_reason='initial',denominator=0.)
        if method!='sr1_total' and inactive:
            if np.any(B):resets+=1
            B=np.zeros_like(B);update['skip_reason']='inactive_contact'
        elif previous is not None:
            old,oldg,oldgs=previous;d=x-old
            if method=='sr1_structured':
                y=(g-oldg)-G@d
            elif method=='sr1_gap':
                y=gs-oldgs
            else:y=g-oldg
            B,update=sr1_update(B,d,y)
            accepted+=int(update['update_applied']);skipped+=int(not update['update_applied'])
        if timer:
            timer.record('secant_sr1',tick);tick=time.perf_counter()
        if method=='sr1_gap':H=G+b1*B
        elif method=='sr1_structured':H=G+B
        else:H=B
        if timer:timer.record('hessian_assembly',tick)
        try:p,shift=newton_direction(H,g,eigen_floor=1e-7,phase_timer=timer)
        except np.linalg.LinAlgError:status='linear_failure';break
        tick=time.perf_counter() if timer else None
        trial,ls=line_search(objective.energy,x,E,g,p,expand=False)
        if timer:timer.record('line_search',tick)
        record.update(**update,**ls,regularization=float(shift),directional_slope=float(g@p),
                      learned_matrix_norm=float(np.linalg.norm(B,ord='fro')))
        if trial is None:status='line_search_failed';break
        new,alpha,_=trial
        record.update(alpha=float(alpha),step_norm=float(np.linalg.norm(new-x)))
        if observer is not None:
            if method!='sr1_gap':raise ValueError('observer requires sr1_gap')
            observer(dict(iteration=k,state=x.copy(),gradient=g.copy(),G=G.copy(),
                B=B.copy(),b1=b1,direction=p.copy(),actual_step=new-x,
                incoming_step=x-previous[0] if previous is not None else None))
        previous=(x.copy(),g.copy(),gs.copy() if gs is not None else None)
        x=new;updates+=1
    elapsed=time.perf_counter()-start
    phases=dict(timer.seconds) if timer else {}
    if timer:phases['other']=elapsed-sum(phases.values())
    return dict(phase_seconds=phases,phase_counts=dict(timer.counts) if timer else {},status=status,state=x,energy=E,residual=res,updates=updates,trace=trace,
        elapsed_s=elapsed,counts=objective.counts.copy(),
        sr1_updates=accepted,sr1_skips=skipped,sr1_resets=resets)


##r## Multiple-gap SR1, with local vector histories and bounded memory.
class MultiGapSR1:
    """Rebuild each gap's B from recent accepted secant pairs, starting at zero.

    Keys identify query/target pairs. Supports include every DOF on which a
    gap depends, even if its current gradient component happens to be zero.
    The correction changes the Hessian only; energy and forces stay exact.
    """
    def __init__(self,memory=5,skip_tol=1e-8):
        if not isinstance(memory,(int,np.integer)) or memory<1:
            raise ValueError('SR1 memory must be an integer >= 1')
        if not np.isfinite(skip_tol) or not 0<skip_tol<1:
            raise ValueError('SR1 skip tolerance must lie in (0,1)')
        self.memory,self.skip_tol=memory,skip_tol
        self.models={};self.totals=dict(sr1_updates=0,sr1_skips=0,sr1_resets=0)

    def _rebuild(self,pairs):
        factors=[];last_applied=False
        for d,y in pairs:
            action=np.zeros_like(d)
            for vector,sign in factors:action+=sign*vector*float(vector@d)
            residual=y-action;denom=float(residual@d)
            scale=float(np.linalg.norm(residual)*np.linalg.norm(d))
            last_applied=bool(scale and abs(denom)>max(1e-30,self.skip_tol*scale))
            if last_applied:
                vector=residual/np.sqrt(abs(denom))
                last_applied=bool(np.all(np.isfinite(vector)))
                if last_applied:factors.append((vector,1. if denom>0 else -1.))
        return factors,last_applied

    def correction(self,state,contact,keys,supports,d_hat):
        from scipy.sparse import coo_matrix,csr_matrix,diags
        from src.contact.barrier import barrier_first_second
        state=np.asarray(state,dtype=float);n=len(state)
        gaps=np.asarray(contact['gaps']);weights=np.asarray(contact['gap_weights'])
        indices=np.asarray(contact['gap_indices'],dtype=int)
        J=np.asarray(contact['gap_jacobian']);dofs=np.asarray(contact['gap_dofs'],dtype=int)
        if not np.isfinite(d_hat) or d_hat<=0 or gaps.ndim!=1 or weights.shape!=gaps.shape:
            raise ValueError('invalid gap data or barrier distance')
        if np.any(~np.isfinite(state)) or np.any(~np.isfinite(J)) or np.any(~np.isfinite(gaps)) or np.any(gaps<=0) or np.any(~np.isfinite(weights)) or np.any(weights<=0):
            raise ValueError('finite state/derivatives and positive gaps/weights required')
        if indices.ndim!=1 or len(np.unique(indices))!=len(indices) or np.any(indices<0) or np.any(indices>=len(gaps)):
            raise ValueError('invalid active gap indices')
        if len(keys)!=len(gaps) or len(supports)!=len(gaps) or len(set(keys))!=len(keys):
            raise ValueError('one unique persistent key and support per gap required')
        if J.shape!=(len(indices),len(dofs)) or len(np.unique(dofs))!=len(dofs):
            raise ValueError('invalid gap Jacobian layout')
        if np.any(dofs<0) or np.any(dofs>=n):raise ValueError('invalid Jacobian DOFs')
        lookup=np.full(n,-1,dtype=int);lookup[dofs]=np.arange(len(dofs))
        active_keys={keys[i] for i in indices}
        counts=dict(sr1_updates=0,sr1_skips=0,sr1_resets=0)
        for key in set(self.models)-active_keys:
            del self.models[key];counts['sr1_resets']+=1
        rows=[];cols=[];values=[];coefficients=[]
        for row,index in enumerate(indices):
            key=keys[index];ids=np.asarray(supports[index],dtype=int)
            if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=n):
                raise ValueError('invalid local SR1 support')
            columns=lookup[ids]
            if np.any(columns<0):raise ValueError('SR1 support missing from gap Jacobian')
            gradient=J[row,columns];x=state[ids];model=self.models.get(key)
            reset=model is not None and (not np.array_equal(model['dofs'],ids) or model['weight']!=weights[index])
            if reset:
                del self.models[key];model=None;counts['sr1_resets']+=1
            if model is None:
                model=dict(dofs=ids.copy(),weight=float(weights[index]),x=x.copy(),
                           gradient=gradient.copy(),pairs=[],factors=[])
                self.models[key]=model
            elif not np.array_equal(model['x'],x):
                d=x-model['x'];y=gradient-model['gradient']
                candidate=(model['pairs']+[(d.copy(),y.copy())])[-self.memory:]
                factors,applied=self._rebuild(candidate)
                if applied:
                    model.update(pairs=candidate,factors=factors);counts['sr1_updates']+=1
                else:counts['sr1_skips']+=1
                model.update(x=x.copy(),gradient=gradient.copy())
            coefficient=float(weights[index]*barrier_first_second(gaps[index],d_hat)[0])
            for vector,sign in model['factors']:
                column=len(coefficients)
                rows.append(ids);cols.append(np.full(len(ids),column,dtype=int));values.append(vector)
                coefficients.append(coefficient*sign)
        if coefficients:
            U=coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                         shape=(n,len(coefficients))).tocsc()
            C=(U@diags(coefficients)@U.T).tocsr();C=.5*(C+C.T);C.eliminate_zeros()
            if not np.all(np.isfinite(C.data)):raise ValueError('nonfinite SR1 correction')
        else:C=csr_matrix((n,n))
        for key,value in counts.items():self.totals[key]+=value
        counts.update(sr1_factor_count=len(coefficients),sr1_pairs=sum(len(m['pairs']) for m in self.models.values()),
                      sr1_correction_norm=float(np.sqrt(C.multiply(C).sum())))
        return C,counts
