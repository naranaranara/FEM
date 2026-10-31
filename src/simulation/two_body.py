import numpy as np
import torch
from src.geometry.mesh import make_block
from src.mechanics.solid import assemble_stiffness,tetrahedron_volumes
from src.geometry.surface import bottom_faces,make_surface_sampler,surface_quadrature  # ##r## shared surface utilities
from src.geometry.rimls import rimls_values_torch
from src.geometry.contact_patches import normalize_query_indices  # ##r##
from src.contact.barrier import barrier_value
from src.contact.terms import multiple_contact_terms,multiple_contact_terms_sparse
from src.optimization.newton import minimize_static

def build_problem(n=2,initial_gap=.12,closure=.04,young_lower=30.,young_upper=30.,R=1.5,d_hat=.2,nx=None,ny=None,nz=None,sparse=True,contact_meshes=None,closure_limit=None,poisson=.3):
    limit=initial_gap if closure_limit is None else closure_limit  # ##r## curved indentation may exceed initial clearance
    if not all(np.isfinite(v) for v in (initial_gap,d_hat,closure,limit)) or not 0<initial_gap or not 0<d_hat or not 0<=closure<limit:
        raise ValueError('require positive initial_gap, d_hat and 0 <= closure < closure_limit')
    if min(young_lower,young_upper,R)<=0:raise ValueError('positive stiffness and R required')
    defaults=(n,n,n) if n is not None else (3,3,2)
    counts=tuple(default if value is None else value for value,default in zip((nx,ny,nz),defaults))
    meshes=([make_block(width=1.,depth=1.,height=.4,nx=counts[0],ny=counts[1],nz=counts[2]) for _ in range(2)]
            if contact_meshes is None else list(contact_meshes))
    if len(meshes)!=2 or meshes[0]['nodes'].shape!=meshes[1]['nodes'].shape:
        raise ValueError('two meshes with matching node counts required')
    lower=meshes[0]['nodes'].copy();upper=meshes[1]['nodes'].copy()
    if contact_meshes is None:upper[:,2]+=.4+initial_gap
    else:counts=tuple(meshes[0].get('counts',counts))
    lower_faces=meshes[0]['contact_faces'] if 'contact_faces' in meshes[0] else meshes[0]['top_faces']
    upper_faces=meshes[1]['contact_faces'] if 'contact_faces' in meshes[1] else bottom_faces(meshes[1])
    lower_query=meshes[0].get('query_faces',lower_faces)  # ##r## full cylinders keep a wider normal-support patch
    upper_query=meshes[1].get('query_faces',upper_faces)
    references=[lower,upper];size=lower.size
    reference=np.concatenate([a.ravel() for a in references])
    stiffnesses=[assemble_stiffness(ref,m['tetrahedra'],young=E,poisson=poisson,sparse=sparse)
                 for ref,m,E in zip(references,meshes,(young_lower,young_upper))]
    if sparse:
        from scipy.sparse import block_diag
        K=block_diag(stiffnesses,format='csr')
    else:
        K=np.zeros((2*size,2*size));K[:size,:size]=stiffnesses[0];K[size:,size:]=stiffnesses[1]
    bottom=meshes[0].get('support_nodes',meshes[0]['bottom_nodes']);top=meshes[1].get('support_nodes',np.unique(meshes[1]['top_faces']))
    fixed=np.r_[(3*bottom[:,None]+np.arange(3)).ravel(),size+(3*top[:,None]+np.arange(3)).ravel()]
    free=np.setdiff1d(np.arange(2*size),fixed)
    Kf=K[free][:,free] if sparse else K[np.ix_(free,free)]  # ##r## slice once
    lower_surface=np.unique(lower_faces);upper_surface=np.unique(upper_faces)
    surface_dofs=np.r_[(3*lower_surface[:,None]+np.arange(3)).ravel(),
                       size+(3*upper_surface[:,None]+np.arange(3)).ravel()]
    contact_dofs=np.flatnonzero(np.isin(free,surface_dofs))
    prescribed=reference.copy();prescribed[size+3*top+2]-=closure
    free_t=torch.tensor(free,dtype=torch.long)
    prescribed_t=torch.tensor(prescribed,dtype=torch.float64)
    sample_lower,_,_=make_surface_sampler(lower,lower_faces)
    sample_upper,_,_=make_surface_sampler(upper,upper_faces)
    Sl,wl=surface_quadrature(lower,lower_query);Su,wu=surface_quadrature(upper,upper_query)
    Ql=torch.tensor(Sl,dtype=torch.float64);Qu=torch.tensor(Su,dtype=torch.float64)
    same_lower=np.array_equal(lower_faces,lower_query);same_upper=np.array_equal(upper_faces,upper_query)
    weights=.5*np.r_[wu,wl]
    selection={"model":None};problem={}
    def set_closure(value):
        """Update Dirichlet data while reusing the reference mesh and stiffness."""
        if not np.isfinite(value) or not 0 <= value < limit:
            raise ValueError('require 0 <= closure < closure_limit')
        prescribed[:] = reference
        prescribed[size+3*top+2] -= value
        prescribed_t.copy_(torch.as_tensor(prescribed, dtype=torch.float64))
        problem['closure'] = float(value)

    def full_state(state):
        result=prescribed.copy();result[free]=state;return result
    def unpack(state):
        a=full_state(state);return a[:size].reshape(-1,3),a[size:].reshape(-1,3)
    def gaps_full_ad(state, indices=None):
        if selection["model"] is not None:
            return selection["model"].gaps(state, indices=indices)
        ids=None if indices is None else normalize_query_indices(indices,len(weights),state.device)
        if ids is not None and ids.numel()==0:return state[:0]
        a=state[:size].reshape(-1,3);b=state[size:].reshape(-1,3)
        pl,nl=sample_lower(a);pu,nu=sample_upper(b)
        ql=pl if same_lower else Ql@a;qu=pu if same_upper else Qu@b
        if ids is None:
            return torch.cat((rimls_values_torch(qu,pl,nl,R),rimls_values_torch(ql,pu,nu,R)))
        values=[];positions=[];split=len(wu)
        upper_slots=torch.nonzero(ids<split,as_tuple=False).reshape(-1)
        lower_slots=torch.nonzero(ids>=split,as_tuple=False).reshape(-1)
        if upper_slots.numel():
            values.append(rimls_values_torch(qu[ids[upper_slots]],pl,nl,R));positions.append(upper_slots)
        if lower_slots.numel():
            values.append(rimls_values_torch(ql[ids[lower_slots]-split],pu,nu,R));positions.append(lower_slots)
        return torch.cat(values)[torch.argsort(torch.cat(positions))]
    def gaps_ad(state, indices=None):
        full = prescribed_t.index_copy(0, free_t, state)
        return gaps_full_ad(full, indices=indices)
    def base(state):
        u=full_state(state)-reference
        Ku=K@u
        return float(.5*u@Ku),Ku[free],Kf
    def feasible(state):
        return all(np.all(tetrahedron_volumes(nodes,m['tetrahedra'])>0) for nodes,m in zip(unpack(state),meshes))
    def current_weights():
        return weights if selection['model'] is None else selection['model'].weights
    free_index = np.full(len(reference), -1, dtype=int)
    free_index[free] = np.arange(len(free))
    dof_cache = {'ids': surface_dofs, 'local': contact_dofs}
    def current_dofs(full=False):
        ids=surface_dofs if selection['model'] is None else selection['model'].contact_dofs
        if full:return ids
        if ids is not dof_cache['ids']:
            local = free_index[ids]
            dof_cache.update(ids=ids, local=np.sort(local[local >= 0]))
        return dof_cache['local']
    def contact(state,method,full=False,compute_hessian=True,return_gap_data=False):  # ##r##
        w=current_weights();gap_function=gaps_full_ad if full else gaps_ad
        if not len(w):
            from scipy.sparse import csr_matrix
            empty=dict(energy=0.,gradient=np.zeros(len(state)),
                hessian=(csr_matrix((len(state),len(state))) if sparse else np.zeros((len(state),len(state)))) if compute_hessian else None,
                gaps=np.empty(0),min_gap=float('inf'),filter_count=0)
            if return_gap_data:
                empty.update(gap_indices=np.empty(0,dtype=int),gap_jacobian=np.empty((0,0)),
                             gap_weights=np.empty(0),gap_dofs=np.empty(0,dtype=int))
            return empty
        return (multiple_contact_terms_sparse(state,gap_function,w,current_dofs(full),d_hat=d_hat,method=method,compute_hessian=compute_hessian,gaps_subset_autodiff=gap_function,return_gap_data=return_gap_data)
                if sparse else multiple_contact_terms(state,gap_function,w,d_hat=d_hat,method=method,compute_hessian=compute_hessian,gaps_subset_autodiff=gap_function,return_gap_data=return_gap_data))
    layout_cache={'identity':None}
    def contact_layout():
        model=selection['model']
        identity=('manual',) if model is None else ('auto',id(model),model.signature)
        if layout_cache['identity']==identity:return layout_cache['layout']
        keys=[];supports=[]
        def support(vertices):
            global_ids=(3*np.asarray(vertices)[:,None]+np.arange(3)).ravel()
            ids=free_index[global_ids]
            return np.unique(ids[ids>=0])
        if model is None:
            node_offset=len(lower)
            for query_faces,query_offset,target in (
                (upper_query,node_offset,lower_surface),
                (lower_query,0,upper_surface+node_offset)):
                for face in query_faces:
                    ids=support(np.unique(np.r_[target,face+query_offset]))
                    for _ in range(3):
                        keys.append(('manual',len(keys)));supports.append(ids)
        else:
            for query,gid in model.signature:
                vertices=np.unique(np.r_[model.faces[model.groups[gid]].ravel(),model.faces[query//3]])
                keys.append(('auto',query,gid));supports.append(support(vertices))
        layout=(tuple(keys),supports);layout_cache.update(identity=identity,layout=layout)
        return layout

    def energy(state):
        if not feasible(state):return np.inf
        try:s=gaps_ad(torch.tensor(state,dtype=torch.float64)).detach().numpy()
        except ValueError:return np.inf
        if not np.all(np.isfinite(s)) or np.any(s<=0):return np.inf
        return base(state)[0]+float(current_weights()@np.array([barrier_value(v,d_hat) for v in s]))
    def objective(state,method,curvature_model=None):
        use_sr1=method=='sr1_gap'
        use_bfgs=method=='structured_bfgs_gap'
        if (use_sr1 or use_bfgs) and curvature_model is None:
            raise ValueError('curvature method requires a per-solve model')
        E,g,H=base(state)
        c=contact(state,'gn' if use_sr1 or use_bfgs else method,return_gap_data=use_sr1 or use_bfgs)
        total_H=H+c['hessian']
        info=dict(min_gap=c['min_gap'],filter_count=c['filter_count'])
        if use_sr1:
            keys,supports=contact_layout()
            correction,learned=curvature_model.correction(state,c,keys,supports,d_hat)
            total_H=total_H+(correction if sparse else correction.toarray())
            info.update(learned)
        if use_bfgs:  # ##r## Same GN, energy and forces as other methods.
            correction,learned=curvature_model.correction(state,c,total_H,d_hat)
            total_H=total_H+(correction if sparse else correction.toarray())
            info.update(learned)
        if selection['model'] is not None:info.update(selection['model'].info)
        return E+c['energy'],g+c['gradient'],total_H,info
    def diagnostics(state):
        a=full_state(state)
        c=contact(a,'gn',full=True,compute_hessian=False)
        contact_force=-c['gradient'];total_gradient=K@(a-reference)+c['gradient']
        reactions=np.zeros_like(a);reactions[fixed]=total_gradient[fixed]
        return dict(contact_force=contact_force,reactions=reactions,full_gradient=total_gradient,
                    force_lower=contact_force[:size].reshape(-1,3).sum(axis=0),
                    force_upper=contact_force[size:].reshape(-1,3).sum(axis=0),gaps=c['gaps'])
    def prepare(state):
        model=selection['model']
        was_initialized=model.initialized
        previous=[]
        info=model.refresh(full_state(state).reshape(-1,3), before_change=lambda:previous.append(energy(state)))
        problem['weights']=model.weights;problem['contact_dofs']=current_dofs()
        updated=energy(state) if previous else None
        valid=bool(previous and np.isfinite(previous[0]) and np.isfinite(updated))
        info['patch_energy_jump_valid']=valid if info['patch_changed'] else was_initialized
        info['patch_energy_jump']=float(updated-previous[0]) if valid else 0.
        return info
    def attach_auto_contact(model):
        selection['model']=model;problem['prepare']=prepare
    problem.update(dict(set_closure=set_closure,attach_auto_contact=attach_auto_contact,stiffness=K,stiffness_free=Kf,sparse=sparse,contact_dofs=contact_dofs,mesh_counts=counts,initial=prescribed[free].copy(),reference=references,reference_flat=reference,meshes=meshes,
        full_state=full_state,unpack=unpack,gaps_ad=gaps_ad,gaps_full_ad=gaps_full_ad,base=base,
        feasible=feasible,energy=energy,objective=objective,diagnostics=diagnostics,
        contact=contact,contact_layout=contact_layout,  # ##r## reusable multi-gap data
        free=free,fixed=fixed,size=size,weights=weights,prescribed=prescribed,closure=closure,d_hat=d_hat))
    return problem

def solve(problem,method='gn',initial=None,max_iter=150,sr1_memory=5,tol=1e-8,track_rk=False):
    if method not in ('full','gn','filtered_gn','sr1_gap','structured_bfgs_gap'):raise ValueError(method)
    if track_rk and method!='gn':raise ValueError('r_k recording currently requires GN')
    start=problem['initial'] if initial is None else np.asarray(initial,dtype=float)
    model=None
    if method=='sr1_gap':
        from src.optimization.sr1 import MultiGapSR1
        model=MultiGapSR1(memory=sr1_memory)
    elif method=='structured_bfgs_gap':  # ##r## comparison baseline
        from src.optimization.structured_bfgs import MultiGapStructuredBFGS
        model=MultiGapStructuredBFGS()
    previous=[None]
    def objective(x):
        E,g,H,info=problem['objective'](x,method,model)
        if track_rk:  # ##r## No extra gradient or exact gap Hessian evaluations.
            info=dict(info,rk=None)
            if previous[0] is not None:
                old_x,old_g,old_H=previous[0]
                d=x-old_x;denominator=float(d@(old_H@d))
                numerator=float(d@(g-old_g))
                info.update(rk_numerator=numerator,rk_denominator=denominator,
                            rk_step_norm=float(np.linalg.norm(d)))
                if denominator>1e-30 and np.isfinite(numerator):info['rk']=numerator/denominator
            previous[0]=(x.copy(),g.copy(),H.copy())
        return E,g,H,info
    result=minimize_static(start,objective,problem['energy'],
                           max_iter=max_iter,tol=tol,max_step=problem.get('max_step'),prepare=problem.get('prepare'),
                           direction_solver=problem.get('direction_solver'))
    if model is not None:
        result.update(model.totals)
        if method=='sr1_gap':result['sr1_memory']=sr1_memory
    return result
