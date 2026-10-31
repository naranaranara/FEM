"""##r## Three-stage current-pipeline cylinder study; no barrier strength change.

1. Exact analytic cylinder versus RIMLS zero set, co-refining mesh and R.
2. Fixed selected geometry/R: activation-distance sensitivity, Hertz reference.
3. Identical final load/start: GN, local Gap SR1, aggregate Gap Structured BFGS.
Static condensation is exact for the present constant linear FEM model only.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from src.geometry.mesh import make_cylinder
from src.geometry.surface import make_surface_sampler
from src.geometry.rimls import rimls_values_torch
from src.mechanics.condensation import condense_contact_problem
from src.mechanics.hertz import elliptical_hertz
from src.simulation.test_crossed_cylinders import build_problem
from src.simulation.two_body import solve


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def surface_error(n_angle,n_axis,R,delta=.005,patch_half_width=1.2):
    """Normal-ray zero displacement, measured against the exact cylinder.

    Nine by nine fixed analytic locations on each facing surface. This measures
    local geometry bias, not FEM discretization or full contact-force accuracy.
    """
    rows=[];arrays={}
    for label,radius,facing,angle in (('lower',3.,'up',0.),('upper',5.,'down',45.)):
        m=make_cylinder(radius=radius,n_angle=n_angle,n_axis=n_axis,facing=facing,
            angle_degrees=angle,clearance=.03,patch_half_width=patch_half_width)
        sampler,_,_=make_surface_sampler(m['nodes'],m['contact_faces'])
        with torch.no_grad():points,normals=sampler(torch.tensor(m['nodes'],dtype=torch.float64))
        x,y=np.meshgrid(np.linspace(-.3,.3,9),np.linspace(-.3,.3,9));x=x.ravel();y=y.ravel()
        sign=1. if facing=='up' else -1.
        center=-radius if facing=='up' else .03+radius
        z=center+sign*np.sqrt(radius**2-x*x)
        theta=np.deg2rad(angle);rot=np.array([[np.cos(theta),-np.sin(theta),0.],
            [np.sin(theta),np.cos(theta),0.],[0.,0.,1.]])
        xyz=np.column_stack((x,y,z))@rot.T
        normal=np.column_stack((x/radius,np.zeros_like(x),(z-center)/radius))@rot.T
        def field(offset):
            with torch.no_grad():
                q=torch.tensor(xyz+np.asarray(offset)[:,None]*normal,dtype=torch.float64)
                return rimls_values_torch(q,points,normals,R).numpy()
        values=field(np.zeros(len(x)));lo=np.full(len(x),-.1);hi=-lo
        if np.any(field(lo)>=0) or np.any(field(hi)<=0):raise ValueError('zero-set normal bracket failed')
        for _ in range(36):
            mid=.5*(lo+hi);positive=field(mid)>0
            hi=np.where(positive,mid,hi);lo=np.where(positive,lo,mid)
        offset=.5*(lo+hi);root_residual=float(np.max(np.abs(field(offset))))
        error=float(np.max(np.abs(offset)))
        rows.append(dict(body=label,nodes=len(m['nodes']),target_faces=len(m['contact_faces']),
            samples=len(points),field_mae=float(np.mean(np.abs(values))),
            zero_error_max=error,zero_error_mae=float(np.mean(np.abs(offset))),
            zero_error_over_delta=error/delta,root_residual=root_residual))
        arrays[label+'_locations']=xyz;arrays[label+'_zero_offsets']=offset
    return dict(n_angle=n_angle,n_axis=n_axis,R=R,bodies=rows,
                zero_error_max=max(r['zero_error_max'] for r in rows)),arrays


def activation_converged(row,tolerance=.02):
    """##r## Stop the distance sweep once completed final forces are insensitive."""
    return bool(row['all_loads_converged'] and row['force_change'] is not None
                and row['force_change']<=tolerance)


def benchmark_problem(p,chosen_start,folder,repeats,max_iter):
    """##r## Same final load, initial coordinates, CCD and solver for every method."""
    benchmark=[]
    methods=('gn','sr1_gap','structured_bfgs_gap');states={}
    for method in methods:
        print('warmup '+method,flush=True)
        solve(p,method,initial=chosen_start,max_iter=max_iter)
    for repeat in range(repeats):
        order=methods[repeat%3:]+methods[:repeat%3]
        for method in order:
            tick=time.perf_counter();result=solve(p,method,initial=chosen_start,max_iter=max_iter)
            wall=time.perf_counter()-tick;data=p['diagnostics'](result['state'])
            row=dict(method=method,repeat=repeat,status=result['status'],updates=result['updates'],
                residual=result['residual'],energy=result['energy'],nonlinear_s=result['elapsed_s'],wall_s=wall,
                force=float(data['force_upper'][2]),min_gap=float(np.min(data['gaps'])),
                sr1_updates=result.get('sr1_updates',0),bfgs_updates=result.get('bfgs_updates',0),
                bfgs_resets=result.get('bfgs_resets',0),trace=result['trace'])
            benchmark.append(row);states[method]=result['state'].copy()
            save_json(folder/'stage3_raw.json',benchmark)
            print('repeat=%d %-20s %-12s updates=%d time=%.4fs F=%.6g'%
                (repeat,method,row['status'],row['updates'],wall,row['force']),flush=True)
    summary=[]
    for method in methods:
        runs=[r for r in benchmark if r['method']==method]
        summary.append(dict(method=method,converged=sum(r['status']=='converged' for r in runs),repeats=repeats,
            median_wall_s=float(np.median([r['wall_s'] for r in runs])),
            median_updates=float(np.median([r['updates'] for r in runs])),
            retained_difference_from_gn=float(np.linalg.norm(states[method]-states['gn'])),
            force=runs[-1]['force']))
    save_json(folder/'stage3_summary.json',summary)
    np.savez_compressed(folder/'comparison_states.npz',initial=chosen_start,**states)
    return summary


def run(repeats=3,steps=8,delta=.005,max_iter=120,collision='ipctk',outdir=None):
    if repeats<1 or steps<1 or delta<=0 or max_iter<1:raise ValueError('positive counts and delta required')
    if collision not in ('ipctk','none'):raise ValueError('unknown collision mode')
    folder=Path(outdir) if outdir else Path('experiments/cylinder_validation_results')/datetime.now().strftime('%Y-%m-%d_%H%M%S')
    folder.mkdir(parents=True,exist_ok=False)
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').rglob('*.py')}
    metadata=dict(repeats=repeats,steps=steps,delta=delta,max_iter=max_iter,collision=collision,
        surface_tolerance=.1*delta,force_change_tolerance=.02,hertz_error_tolerance=.1,
        barrier_strength=1.,n_radial=1,volume_mesh_convergence_tested=False,surface_grid='9x9 analytic points, local x/y in [-0.3,0.3]',
        scope='finite cylinders, clamped end disks; Hertz half-space is a reference',
        timing='final load nonlinear solve incl CCD; excludes preparation, diagnostics and common warm-start',
        bfgs='aggregate missing-term S-BFGS-P, reset on any active-set change; not full Petra algorithm',
        sr1='per-gap limited-memory geometry secants; per-gap deactivation resets',source_sha256=sources)
    save_json(folder/'metadata.json',metadata)
    stage1=[];snapshots={};selected=None
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with threadpool_limits(1):
            for level,(na,ny,R) in enumerate(((32,7,.65),(64,13,.325),(128,25,.1625))):
                row,arrays=surface_error(na,ny,R,delta)
                row['passed']=row['zero_error_max']<=metadata['surface_tolerance']
                stage1.append(row);snapshots.update({'level%d_'%level+k:v for k,v in arrays.items()})
                save_json(folder/'stage1_surface.json',stage1)
                print('surface level=%d nodes=%d max_zero_error=%.6g / delta=%.3g pass=%s'%
                    (level,row['bodies'][0]['nodes'],row['zero_error_max'],row['zero_error_max']/delta,row['passed']),flush=True)
                if row['passed'] and selected is None:selected=row
            np.savez_compressed(folder/'surface_locations.npz',**snapshots)
            if selected is None:
                save_json(folder/'conclusion.json',dict(status='surface_gate_failed',message='refine further before force and timing comparison'))
                return folder
            sensitivity=[];benchmark=[];chosen=None;chosen_problem=None;chosen_start=None;chosen_closure=None
            from contextlib import ExitStack
            with ExitStack() as resources:
                for dhat in (.001,.0005,.00025):
                    tick=time.perf_counter()
                    original=build_problem(n_angle=selected['n_angle'],n_axis=selected['n_axis'],
                        R=selected['R'],d_hat=dhat,validation_delta=delta)
                    p=condense_contact_problem(original)
                    collider=None
                    if collision=='ipctk':
                        from src.collision.ipc_toolkit import IPCMeshCollision
                        ref=np.vstack(original['reference']);n=len(original['reference'][0])
                        faces=np.vstack((original['meshes'][0]['boundary_faces'],original['meshes'][1]['boundary_faces']+n))
                        collider=resources.enter_context(IPCMeshCollision(ref,faces,np.r_[np.zeros(n,dtype=int),np.ones(n,dtype=int)]))
                        p['max_step']=lambda x,y,c=collider,cp=p:c.max_step(cp['full_state'](x).reshape(-1,3),cp['full_state'](y).reshape(-1,3))
                    prep=time.perf_counter()-tick
                    print('prepare d_hat=%.6g full_DOFs=%d retained=%d seconds=%.2f'%
                        (dhat,len(original['initial']),len(p['initial']),prep),flush=True)
                    state=p['initial'].copy();loads=[];before_final=None;previous=None
                    # Resolve indentation separately from the much larger initial clearance.
                    closures=np.r_[np.linspace(0.,original['initial_gap']-.002,3),
                        original['initial_gap']+np.linspace(-.001,delta,steps+1)]
                    for step,closure in enumerate(closures):
                        p['set_closure'](float(closure));start=state.copy()
                        if step==len(closures)-1:before_final=start.copy()
                        if collider is not None and previous is not None:
                            if collider.max_step(previous,p['full_state'](start).reshape(-1,3))<1-1e-12:
                                loads.append(dict(step=step,closure=float(closure),status='prescribed_ccd_blocked'));break
                        result=solve(p,'gn',initial=start,max_iter=max_iter)
                        state=result['state'];previous=p['full_state'](state).reshape(-1,3)
                        data=p['diagnostics'](state)
                        row=dict(step=step,closure=float(closure),status=result['status'],updates=result['updates'],
                            residual=result['residual'],energy=result['energy'],nonlinear_s=result['elapsed_s'],
                            force=float(data['force_upper'][2]),min_gap=float(np.min(data['gaps'])),
                            force_balance_error=float(np.linalg.norm(data['force_upper']+data['force_lower'])))
                        loads.append(row)
                        save_json(folder/('loads_dhat_%g.json'%dhat),loads)
                        print('dhat=%g load=%d/%d %-12s updates=%d F=%.6g |g|=%.2g'%
                            (dhat,step,len(closures)-1,row['status'],row['updates'],row['force'],row['residual']),flush=True)
                        if result['status']!='converged':break
                    reference=elliptical_hertz(delta,original['principal_curvatures'])
                    ok=len(loads)==len(closures) and all(r['status']=='converged' for r in loads)
                    final=loads[-1];force=final.get('force')
                    row=dict(d_hat=dhat,all_loads_converged=ok,preparation_s=prep,condensation_s=p['condensation_s'],
                        force=force,hertz_force=reference['force'],force_error=(force/reference['force']-1) if force is not None and ok else None,
                        force_change=None,status=final['status'],load_count=len(loads))
                    if ok and sensitivity and sensitivity[-1]['all_loads_converged']:
                        row['force_change']=abs(force-sensitivity[-1]['force'])/max(abs(force),1e-30)
                    sensitivity.append(row);save_json(folder/'stage2_activation.json',sensitivity)
                    if ok:
                        chosen=row;chosen_problem=p;chosen_start=before_final;chosen_closure=float(closures[-1])
                        np.savez_compressed(folder/('equilibrium_dhat_%g.npz'%dhat),nodes=previous)
                        if activation_converged(row):  # ##r## No unnecessary third load path.
                            metadata['activation_stop']='final force change <= 2 percent'
                            save_json(folder/'metadata.json',metadata)
                            print('Activation sweep stopped: force change %.6g%%'%(100*row['force_change']),flush=True)
                            break
                if chosen is None:
                    save_json(folder/'conclusion.json',dict(status='load_solve_failed',message='no full converged load path; do not compare solver times as successful solves'))
                    return folder
                p=chosen_problem;p['set_closure'](chosen_closure)
                summary=benchmark_problem(p,chosen_start,folder,repeats,max_iter)
                physical=bool(chosen['force_change'] is not None and chosen['force_change']<.02 and abs(chosen['force_error'])<.1)
                save_json(folder/'conclusion.json',dict(status='complete',selected_geometry=selected,
                    selected_d_hat=chosen['d_hat'],hertz_validation_passed=physical,
                    interpretation='solver timing remains diagnostic unless geometry, activation, boundary and FEM convergence are established',summary=summary))
                print('Hertz validation passed:',physical,flush=True)
            return folder
    finally:
        torch.set_num_threads(old_threads)
        print('Saved:',folder.resolve(),flush=True)


def resume_benchmark(folder,repeats=3,max_iter=120,collision='ipctk'):
    """##r## Reuse saved equilibria; reconstruct the common penultimate load once."""
    from contextlib import ExitStack
    folder=Path(folder)
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    levels=json.loads((folder/'stage1_surface.json').read_text(encoding='utf-8'))
    selected=next(row for row in levels if row['passed'])
    rows=json.loads((folder/'stage2_activation.json').read_text(encoding='utf-8'))
    chosen=[row for row in rows if row['all_loads_converged']][-1]
    dhat=chosen['d_hat'];delta=metadata['delta']
    if (folder/'stage3_raw.json').exists():raise ValueError('benchmark already exists; use a separate result folder')
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with threadpool_limits(1),ExitStack() as resources:
            original=build_problem(n_angle=selected['n_angle'],n_axis=selected['n_axis'],
                R=selected['R'],d_hat=dhat,validation_delta=delta)
            p=condense_contact_problem(original)
            if collision=='ipctk':
                from src.collision.ipc_toolkit import IPCMeshCollision
                ref=np.vstack(original['reference']);n=len(original['reference'][0])
                faces=np.vstack((original['meshes'][0]['boundary_faces'],original['meshes'][1]['boundary_faces']+n))
                collider=resources.enter_context(IPCMeshCollision(ref,faces,np.r_[np.zeros(n,dtype=int),np.ones(n,dtype=int)]))
                p['max_step']=lambda x,y:collider.max_step(p['full_state'](x).reshape(-1,3),p['full_state'](y).reshape(-1,3))
            else:collider=None
            nodes=np.load(folder/('equilibrium_dhat_%g.npz'%dhat))['nodes']
            q=nodes.ravel()[original['free']][p['retained']]
            loads=json.loads((folder/('loads_dhat_%g.json'%dhat)).read_text(encoding='utf-8'))
            previous_closure=loads[-2]['closure'];p['set_closure'](previous_closure)
            print('Reconstruct common starting equilibrium at closure=%.6g'%previous_closure,flush=True)
            result=solve(p,'gn',initial=q,max_iter=max_iter)
            save_json(folder/'common_start.json',dict(status=result['status'],closure=previous_closure,
                updates=result['updates'],residual=result['residual'],elapsed_s=result['elapsed_s']))
            if result['status']!='converged':raise RuntimeError('common starting equilibrium did not converge')
            initial=result['state'];previous=p['full_state'](initial).reshape(-1,3)
            p['set_closure'](loads[-1]['closure'])
            if collider is not None and collider.max_step(previous,p['full_state'](initial).reshape(-1,3))<1-1e-12:
                raise RuntimeError('prescribed final step crosses triangles')
            metadata.update(activation_stop='force sensitivity is small; third path stopped at user request',
                benchmark_collision=collision,benchmark_repeats=repeats,
                benchmark_source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('src').rglob('*.py')})
            save_json(folder/'metadata.json',metadata)
            summary=benchmark_problem(p,initial,folder,repeats,max_iter)
            physical=bool(activation_converged(chosen) and abs(chosen['force_error'])<.1)
            save_json(folder/'conclusion.json',dict(status='complete',selected_geometry=selected,
                selected_d_hat=dhat,hertz_validation_passed=physical,
                interpretation='activation sensitivity is small; finite-domain and volume-mesh convergence remain unverified',summary=summary))
            return folder
    finally:
        torch.set_num_threads(old_threads)
        print('Saved:',folder.resolve(),flush=True)


def diagnose_rk(folder,max_iter=60,collision='ipctk'):
    """##r## Accepted-step total-gradient stiffness ratios on the saved GN path."""
    from contextlib import ExitStack
    folder=Path(folder)
    selected=next(row for row in json.loads((folder/'stage1_surface.json').read_text()) if row['passed'])
    chosen=[r for r in json.loads((folder/'stage2_activation.json').read_text()) if r['all_loads_converged']][-1]
    loads=json.loads((folder/('loads_dhat_%g.json'%chosen['d_hat'])).read_text())
    initial=np.load(folder/'comparison_states.npz')['initial']
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with threadpool_limits(1),ExitStack() as resources:
            original=build_problem(n_angle=selected['n_angle'],n_axis=selected['n_axis'],
                R=selected['R'],d_hat=chosen['d_hat'])
            p=condense_contact_problem(original);p['set_closure'](loads[-1]['closure'])
            if collision=='ipctk':
                from src.collision.ipc_toolkit import IPCMeshCollision
                ref=np.vstack(original['reference']);n=len(original['reference'][0])
                faces=np.vstack((original['meshes'][0]['boundary_faces'],original['meshes'][1]['boundary_faces']+n))
                collider=resources.enter_context(IPCMeshCollision(ref,faces,np.r_[np.zeros(n,dtype=int),np.ones(n,dtype=int)]))
                p['max_step']=lambda x,y:collider.max_step(p['full_state'](x).reshape(-1,3),p['full_state'](y).reshape(-1,3))
            result=solve(p,'gn',initial=initial,max_iter=max_iter,track_rk=True)
            records=[]
            for k,row in enumerate(result['trace']):
                if k and 'rk_numerator' in row:
                    records.append(dict(step=k-1,rk=row['rk'],alpha=result['trace'][k-1]['alpha'],
                        step_norm=row['rk_step_norm'],numerator=row['rk_numerator'],denominator=row['rk_denominator']))
            reference=np.load(folder/'comparison_states.npz')['gn']
            data=dict(status=result['status'],updates=result['updates'],residual=result['residual'],
                d_hat=chosen['d_hat'],records=records,
                final_difference_from_timed_gn=float(np.linalg.norm(reference-result['state'])),
                definition='dT delta(total gradient) / dT previous(total GN Hessian) d',
                scope='accepted-step average stiffness; exact linear-FEM condensation; not a generalized eigenvalue',
                diagnostic_excluded_from_timing=True)
            save_json(folder/'rk_diagnostic.json',data)
            for row in records:print('step=%d r_k=%.9g alpha=%.3g step_norm=%.3g'%
                (row['step'],row['rk'] if row['rk'] is not None else float('nan'),row['alpha'],row['step_norm']),flush=True)
            return data
    finally:torch.set_num_threads(old_threads)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--delta',type=float,default=.005);parser.add_argument('--max-iter',type=int,default=120)
    parser.add_argument('--collision',choices=('ipctk','none'),default='ipctk');parser.add_argument('--outdir')
    parser.add_argument('--benchmark-from',help='reuse completed saved load paths; do not rerun the activation sweep')
    parser.add_argument('--rk-from',help='diagnose accepted GN stiffness ratios after saved timing runs')
    args=vars(parser.parse_args());saved=args.pop('benchmark_from');rk=args.pop('rk_from')
    if saved and rk:raise ValueError('choose either benchmark-from or rk-from')
    if rk:diagnose_rk(rk,args['max_iter'],args['collision'])
    elif saved:resume_benchmark(saved,args['repeats'],args['max_iter'],args['collision'])
    else:run(**args)
