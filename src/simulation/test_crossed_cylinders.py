import argparse
import json
from contextlib import ExitStack
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from src.geometry.mesh import make_cylinder
from src.geometry.primitives import crossed_cylinder_curvature,size_based_activation_distance
from src.mechanics.hertz import elliptical_hertz
from src.mechanics.condensation import condense_contact_problem  # ##r##
from src.simulation.two_body import build_problem as build_two_body, solve


# ##r## Both bodies are complete finite cylinders; the end disks are grips.
def build_problem(n_angle=32,n_radial=1,n_axis=7,radius_lower=3.,radius_upper=5.,angle=45.,
                  half_length=3.,query_half_width=.4,patch_half_width=1.2,
                  initial_gap=.03,d_hat=None,d_hat_ratio=1e-3,R=.65,
                  young_lower=30.,young_upper=30.,poisson=.3,validation_delta=None,radial_grading=1.,
                  local_size=None,local_half_width=.4,local_depth=.3):
    K,curvatures,directions=crossed_cylinder_curvature(radius_lower,radius_upper,angle)
    common=dict(half_length=half_length,n_angle=n_angle,n_radial=n_radial,n_axis=n_axis,
                query_half_width=query_half_width,patch_half_width=patch_half_width,radial_grading=radial_grading)  # ##r##
    lower=make_cylinder(radius_lower,facing='up',**common)
    upper=make_cylinder(radius_upper,facing='down',angle_degrees=angle,clearance=initial_gap,**common)
    # ##r## Same conforming local mesh used by the pressure-only diagnostic.
    if local_size is not None:
        from src.geometry.local_refinement import refine_cylinder_contact
        lower,upper=[refine_cylinder_contact(m,local_size,local_half_width,local_depth)
                     for m in (lower,upper)]
    automatic,length=size_based_activation_distance(np.vstack((lower['nodes'],upper['nodes'])),d_hat_ratio)
    activation=automatic if d_hat is None else d_hat
    # ##r## Hertz benchmark guard, before FEM assembly. Not a general IPC rule.
    if validation_delta is not None:
        if not np.isfinite(validation_delta) or validation_delta<=0:
            raise ValueError('Hertz validation requires a positive finite indentation delta')
        if activation>=validation_delta:
            raise ValueError(
                'Hertz validation requires d_hat < indentation delta; '
                'got d_hat=%.8g, delta=%.8g (ratio=%.3g). '
                'Use --d-hat or reduce --d-hat-ratio. Passing this check '
                'does not establish Hertz accuracy; verify parameter convergence.'
                % (activation,validation_delta,activation/validation_delta))
    p=build_two_body(initial_gap=initial_gap,closure=0.,young_lower=young_lower,
        young_upper=young_upper,R=R,d_hat=activation,sparse=True,contact_meshes=(lower,upper),
        closure_limit=initial_gap+.05*min(radius_lower,radius_upper),poisson=poisson)
    p.update(relative_curvature=K,principal_curvatures=curvatures,principal_directions=directions,
             initial_gap=initial_gap,characteristic_length=length,
             parameters=dict(n_angle=n_angle,n_radial=n_radial,n_axis=n_axis,radial_grading=radial_grading,
             radius_lower=radius_lower,radius_upper=radius_upper,angle=angle,half_length=half_length,
             query_half_width=query_half_width,patch_half_width=patch_half_width,
             initial_gap=initial_gap,d_hat=float(activation),d_hat_ratio=d_hat_ratio,
             local_size=local_size,local_half_width=local_half_width,local_depth=local_depth,
             local_refinement=[m.get('local_refinement') for m in (lower,upper)],
             d_hat_mode='scene_scale' if d_hat is None else 'explicit',characteristic_length=length,
             R=R,young_lower=young_lower,young_upper=young_upper,poisson=poisson,
             geometry='complete finite cylinders',boundary_condition='both end disks clamped; upper grips translated down'))
    return p


def plot_results(problem,history):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    methods=list(history)
    fig=plt.figure(figsize=(5*len(methods),5))
    all_nodes=np.vstack(problem['reference']);lo=all_nodes.min(axis=0);hi=all_nodes.max(axis=0)
    artists={}
    for column,method in enumerate(methods,1):
        ax=fig.add_subplot(1,len(methods),column,projection='3d')
        artists[method]=[]
        for nodes,mesh,color in zip(problem['reference'],problem['meshes'],('tab:blue','tab:orange')):
            patch=Poly3DCollection(nodes[mesh['boundary_faces']],facecolors=color,
                edgecolors='k',linewidths=.3,alpha=.5)
            ax.add_collection3d(patch);artists[method].append(patch)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),
               xlabel='x',ylabel='y',zlabel='z',title=method)
        ax.view_init(20,-60)
        if hasattr(ax,'set_box_aspect'):ax.set_box_aspect(hi-lo)
    def animate(frame):
        for method in methods:
            row=history[method][frame]
            for artist,nodes,mesh in zip(artists[method],row['nodes'],problem['meshes']):
                artist.set_verts(nodes[mesh['boundary_faces']])
        fig.suptitle('Crossed full cylinders | closure = %.5f' % history[methods[0]][frame]['closure'])
        return []
    animation=FuncAnimation(fig,animate,frames=min(map(len,history.values())),interval=500,
                            blit=False,cache_frame_data=False)
    fig._crossed_animation=animation
    curves,axes=plt.subplots(1,2,figsize=(10,4))
    for method in methods:
        rows=history[method]
        axes[0].plot([r['nominal_delta'] for r in rows],[r['force'] for r in rows],'-o',label=method)
        axes[1].plot([r['nominal_delta'] for r in rows],[r['updates'] for r in rows],'-o',label=method)
    rows=history[methods[0]]
    axes[0].plot([r['nominal_delta'] for r in rows],[r['hertz_force'] for r in rows],'k--',label='Hertz half-space')
    axes[0].set(xlabel='Nominal approach beyond initial clearance',ylabel='Normal contact force')
    axes[1].set(xlabel='Nominal approach beyond initial clearance',ylabel='Newton updates')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    curves.tight_layout()
    plt.show()


# ##r## Optional local load refinement; no barrier strength or gap floor change.
def closure_schedule(initial_gap, delta, d_hat, steps, contact_steps=None):
    if contact_steps is None:
        return np.linspace(0.,initial_gap+delta,steps+1)
    if not isinstance(contact_steps,(int,np.integer)) or contact_steps < 1:
        raise ValueError('contact_steps must be a positive integer')
    start=max(0.,initial_gap-4*d_hat)
    approach=np.linspace(0.,start,4) if start>0 else np.array([0.])
    return np.r_[approach,np.linspace(start,initial_gap+delta,contact_steps+1)[1:]]


def run(n_angle=32,n_radial=1,n_axis=7,steps=8,delta=.005,methods=('gn','sr1_gap'),collision='ipctk',
        show=False,outdir=None,condense=True,linear_backend='pardiso',contact_steps=None,**kwargs):
    if linear_backend not in ('superlu','pardiso'):
        raise ValueError('linear backend must be superlu or pardiso')
    closure_schedule(.03,delta,.0005,steps,contact_steps)  # validate before setup
    if steps<1 or not np.isfinite(delta) or delta<=0:
        raise ValueError('positive steps and delta required')
    if not methods or len(set(methods))!=len(methods) or any(m not in ('full','gn','filtered_gn','sr1_gap','structured_bfgs_gap') for m in methods):
        raise ValueError('distinct supported methods required')
    if collision not in ('ipctk','none'):raise ValueError('collision must be ipctk or none')
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    rows=[];history={m:[] for m in methods};states={};previous_positions={}
    try:
        with threadpool_limits(1),ExitStack() as resources:
            # ##r## Keep full geometry for diagnostics/CCD; solve retained coordinates.
            prep_tick=time.perf_counter()
            print('Preparing mesh, FEM stiffness and contact geometry...',flush=True)
            original=build_problem(n_angle=n_angle,n_radial=n_radial,n_axis=n_axis,validation_delta=delta,**kwargs)
            assembly_s=time.perf_counter()-prep_tick
            print('Preparation %.2fs; full free DOFs=%d' % (assembly_s,len(original['initial'])),flush=True)
            if condense:
                p=condense_contact_problem(original,linear_backend=linear_backend,
                    progress=lambda message:print(message,flush=True))
                for key in ('reference','meshes','parameters','initial_gap','characteristic_length',
                            'principal_curvatures','principal_directions','relative_curvature'):
                    p[key]=original[key]
                print('Condensation %.2fs: %d -> %d DOFs' %
                      (p['condensation_s'],len(original['initial']),len(p['initial'])),flush=True)
            else:
                p=original
            timings=dict(preparation_s=assembly_s,condensation_s=p.get('condensation_s',0.),
                         factorization_s=p.get('factorization_s',0.),schur_assembly_s=p.get('schur_assembly_s',0.))
            # ##r## Disjoint phase timings; objective includes base + contact E/g/H.
            phases={}
            def measured(name,function):
                def call(*args,**kw):
                    start=time.perf_counter()
                    try:return function(*args,**kw)
                    finally:phases[name]=phases.get(name,0.)+time.perf_counter()-start
                return call
            p['objective']=measured('objective_s',p['objective'])
            p['energy']=measured('energy_checks_s',p['energy'])
            from src.optimization.newton import newton_direction
            p['direction_solver']=measured('direction_s',p.get('direction_solver',newton_direction))
            total_closure=p['initial_gap']+delta
            p['set_closure'](total_closure)  # validate before the load loop
            p['set_closure'](0.)
            collider=None
            if collision=='ipctk':
                from src.collision.ipc_toolkit import IPCMeshCollision
                ref=np.vstack(p['reference']);n=len(p['reference'][0])
                faces=np.vstack((p['meshes'][0]['boundary_faces'],p['meshes'][1]['boundary_faces']+n))
                labels=np.r_[np.zeros(n,dtype=int),np.ones(len(p['reference'][1]),dtype=int)]
                collider=resources.enter_context(IPCMeshCollision(ref,faces,labels))
                p['max_step']=measured('ccd_s',lambda x,y:collider.max_step(p['full_state'](x).reshape(-1,3),p['full_state'](y).reshape(-1,3)))
            print('Crossed full cylinders: nodes=%s free_DOFs=%d samples=%d' %
                  (tuple(len(a) for a in p['reference']),len(p['initial']),len(p['weights'])),flush=True)
            print('L=%.6g d_hat=%.6g mode=%s d_hat/delta=%.3g' %
                  (p['characteristic_length'],p['d_hat'],p['parameters']['d_hat_mode'],p['d_hat']/delta),flush=True)
            print('Relative principal curvatures:',p['principal_curvatures'],flush=True)
            folder=None
            if outdir:
                folder=Path(outdir);folder.mkdir(parents=True,exist_ok=False)
                (folder/'parameters.json').write_text(json.dumps(dict(p['parameters'],steps=steps,delta=delta,
                    methods=list(methods),collision=collision,condensed=condense,contact_steps=contact_steps,
                    linear_backend=linear_backend if condense else 'full_dof_default',
                    full_free_dofs=len(original['initial']),solver_dofs=len(p['initial']),
                    body_node_counts=[len(a) for a in p['reference']],**timings),indent=2),encoding='utf-8')
            closures=closure_schedule(p['initial_gap'],delta,p['d_hat'],steps,contact_steps)
            print('Load increments=%d; contact refinement=%s' % (len(closures)-1,contact_steps),flush=True)
            for step,closure in enumerate(closures):
                p['set_closure'](float(closure))
                nominal_delta=max(0.,closure-p['initial_gap'])
                ref=elliptical_hertz(nominal_delta,p['principal_curvatures'],
                    p['parameters']['young_lower'],p['parameters']['young_upper'],p['parameters']['poisson'])
                for method in methods:
                    initial=states.get(method,p['initial'])
                    if collider is not None and method in previous_positions:
                        moved=p['full_state'](initial).reshape(-1,3)
                        if collider.max_step(previous_positions[method],moved)<1-1e-12:
                            raise RuntimeError('prescribed step crosses triangles; increase --steps')
                    phases.clear()
                    print('Starting step=%d closure=%.5f %s' % (step,closure,method),flush=True)
                    tick=time.perf_counter()
                    result=solve(p,method,initial=initial,max_iter=200)
                    diagnostic_tick=time.perf_counter()
                    data=p['diagnostics'](result['state']);nodes=p['unpack'](result['state'])
                    diagnostics_s=time.perf_counter()-diagnostic_tick
                    force=float(data['force_upper'][2])
                    balance=float(np.linalg.norm(data['force_upper']+data['force_lower']))
                    row=dict(step=step,method=method,closure=float(closure),nominal_delta=float(nominal_delta),
                        status=result['status'],updates=result['updates'],residual=result['residual'],
                        force=force,hertz_force=ref['force'],hertz_a=ref['major_radius'],hertz_b=ref['minor_radius'],
                        force_error=(force/ref['force']-1.) if ref['force'] else None,
                        min_gap=float(np.min(data['gaps'])),force_balance_error=balance,
                        nonlinear_s=result['elapsed_s'],wall_s=time.perf_counter()-tick,
                        sr1_updates=result.get('sr1_updates',0),diagnostics_s=diagnostics_s,**phases)
                    rows.append(row);history[method].append(dict(row,nodes=[a.copy() for a in nodes]))
                    print('step=%d closure=%.5f %-12s %-10s updates=%d F=%.6g Hertz=%.6g gap=%.3g balance=%.1e' %
                        (step,closure,method,result['status'],result['updates'],force,ref['force'],row['min_gap'],balance),flush=True)
                    print('  objective=%.2fs direction=%.2fs energy checks=%.2fs CCD=%.2fs diagnostics=%.2fs' %
                          (phases.get('objective_s',0.),phases.get('direction_s',0.),phases.get('energy_checks_s',0.),phases.get('ccd_s',0.),diagnostics_s),flush=True)
                    if folder is not None:
                        (folder/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
                    # ##r## Keep accepted-step traces and restart/debug coordinates,
                    # including a failed trial, without declaring it an equilibrium.
                    if folder is not None:
                        (folder/('trace_%s_%03d.json'%(method,step))).write_text(
                            json.dumps(result['trace'],indent=2),encoding='utf-8')
                        np.savez_compressed(folder/('checkpoint_%s.npz'%method),
                            state=result['state'],previous_converged_state=initial,
                            nodes=np.vstack(nodes),body_node_offsets=np.cumsum([0]+[len(a) for a in nodes]),
                            closure=float(closure),step=step,
                            status=result['status'],condensed=condense,
                            previous_converged_closure=closures[step-1] if step else np.nan)
                    if result['status']!='converged':raise RuntimeError('failed load step: '+method)
                    states[method]=result['state'].copy();previous_positions[method]=np.vstack(nodes)
            if folder is not None:
                # ##r## Preserve legacy equal-body layout; ragged bodies use stacked nodes.
                equal=len(p['reference'][0])==len(p['reference'][1])
                saved={m:np.array([r['nodes'] if equal else np.vstack(r['nodes']) for r in h])
                       for m,h in history.items()}
                saved['body_node_offsets']=np.cumsum([0]+[len(a) for a in p['reference']])
                saved['layout']=np.array('body_axis' if equal else 'stacked_nodes')
                np.savez_compressed(folder/'states.npz',**saved)
            if show:plot_results(p,history)
            return p,rows,history
    finally:torch.set_num_threads(old_threads)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-angle',type=int,default=32)
    parser.add_argument('--n-radial',type=int,default=1)
    parser.add_argument('--radial-grading',type=float,default=1.)  # ##r## same graded cylinder mesh
    parser.add_argument('--n-axis',type=int,default=7)
    parser.add_argument('--steps',type=int,default=8);parser.add_argument('--delta',type=float,default=.005)
    parser.add_argument('--angle',type=float,default=45.)
    parser.add_argument('--radius-lower',type=float,default=3.);parser.add_argument('--radius-upper',type=float,default=5.)
    parser.add_argument('--half-length',type=float,default=3.)
    parser.add_argument('--query-half-width',type=float,default=.4)
    parser.add_argument('--patch-half-width',type=float,default=1.2)
    parser.add_argument('--initial-gap',type=float,default=.03);parser.add_argument('--d-hat',type=float,default=None,help='explicit distance; default is d_hat_ratio times initial scene diagonal')
    parser.add_argument('--d-hat-ratio',type=float,default=1e-3)
    parser.add_argument('--R',type=float,default=.65)
    parser.add_argument('--methods',nargs='+',choices=('full','gn','filtered_gn','sr1_gap','structured_bfgs_gap'),default=['gn','sr1_gap'])
    parser.add_argument('--collision',choices=('ipctk','none'),default='ipctk')
    parser.add_argument('--linear-backend',choices=('pardiso','superlu'),default='pardiso',
                        help='##r## reusable interior factorization for condensation')
    parser.add_argument('--contact-steps',type=int,default=None,
                        help='##r## refine loads from initial_gap - 4*d_hat to final closure')
    # ##r## Optional local refinement, matching the compliance experiment.
    parser.add_argument('--local-size',type=float,default=None)
    parser.add_argument('--local-half-width',type=float,default=.4)
    parser.add_argument('--local-depth',type=float,default=.3)
    parser.add_argument('--no-condense',dest='condense',action='store_false',
                        help='##r## use the slower full-DOF solver for comparison; default uses exact condensation')
    parser.add_argument('--show',action='store_true');parser.add_argument('--outdir')
    args=vars(parser.parse_args());run(**args)
