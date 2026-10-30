import argparse
from time import perf_counter
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.linalg import solve as dense_solve
from ..geometry.mesh import make_spherical_cap
from ..geometry.surface import surface_lumped_weights
from ..mechanics.solid import assemble_stiffness, tetrahedron_volumes
from ..contact.barrier import barrier_arrays
from ..optimization.newton import minimize_static


def hertz_reference(delta,radius,young,poisson,radial=None):
    """Rigid plane: E*=E/(1-nu^2), a^2=R*delta. Comparison only."""
    if not all(np.isfinite(x) for x in (delta,radius,young,poisson)):
        raise ValueError('Hertz parameters must be finite')
    if delta<0 or radius<=0 or young<=0 or not -1<poisson<.5:
        raise ValueError('require delta>=0, radius>0, E>0, -1<nu<0.5')
    modulus=young/(1-poisson*poisson)
    a=np.sqrt(radius*delta)
    force=(4./3.)*modulus*np.sqrt(radius)*delta**1.5
    p0=2*modulus*a/(np.pi*radius)
    answer=dict(force=float(force),contact_radius=float(a),peak_pressure=float(p0),
                effective_modulus=float(modulus))
    if radial is not None:
        radial=np.asarray(radial,dtype=float)
        if np.any(~np.isfinite(radial)) or np.any(radial<0):
            raise ValueError('radial coordinates must be finite and nonnegative')
        answer['pressure']=p0*np.sqrt(np.maximum(1-(radial/a)**2,0)) if a else np.zeros_like(radial)
    return answer


def make_plane_direction(stiffness,contact_dofs):
    stiffness=stiffness.tocsc();factor=splu(stiffness)
    diagonal=stiffness.diagonal();ids=np.asarray(contact_dofs,dtype=int)
    columns={}
    def direction(H,g,eigen_floor):
        extra=H.diagonal()[ids]-diagonal[ids]
        if np.any(extra<0):
            raise np.linalg.LinAlgError('plane contact diagonal must be nonnegative')
        active=ids[extra>0];d=extra[extra>0]
        p=factor.solve(-g)
        if len(active):
            missing=[int(i) for i in active if int(i) not in columns]
            if missing:
                rhs=np.zeros((len(g),len(missing)))
                rhs[missing,np.arange(len(missing))]=1.
                influences=factor.solve(rhs)
                for j,i in enumerate(missing):columns[i]=influences[:,j].copy()
            C=np.column_stack([columns[int(i)] for i in active])
            W=C[active,:];W=.5*(W+W.T)
            multiplier=dense_solve(W+np.diag(1./d),p[active],assume_a='pos')
            p-=C@multiplier
            p[active]=multiplier/d
        if not np.all(np.isfinite(p)):
            raise np.linalg.LinAlgError('nonfinite plane direction')
        if g@p>=0:p=-g
        return p,0.
    return direction


def build_problem(radius=10.,half_width=6.,height=10.,nxy=25,nz=13,
                  young=1000.,poisson=.3,clearance=.002,d_hat=.001,kappa=None,
                  grading=5.,depth_grading=3.):
    if not np.isfinite(d_hat) or not 0<d_hat<clearance:
        raise ValueError('require 0 < d_hat < initial clearance')
    mesh=make_spherical_cap(radius,half_width,height,nxy,nz,clearance,grading,depth_grading)
    kappa=young/radius if kappa is None else kappa
    if not np.isfinite(kappa) or kappa<=0:
        raise ValueError('kappa must be finite and positive')
    reference=mesh['nodes'];top=np.unique(mesh['top_faces']);bottom=mesh['bottom_nodes']
    fixed=(3*top[:,None]+np.arange(3)).ravel()
    free=np.setdiff1d(np.arange(reference.size),fixed)
    remap=np.full(reference.size,-1,dtype=int);remap[free]=np.arange(len(free))
    contact_dofs=remap[3*bottom+2]
    weights=surface_lumped_weights(reference,mesh['contact_faces'])[bottom]
    K=assemble_stiffness(reference,mesh['tetrahedra'],young,poisson,sparse=True)
    Kff=K[free][:,free].tocsr();prescribed=np.zeros(reference.size)
    relative_offset=np.zeros(len(free))
    ref_volumes=tetrahedron_volumes(reference,mesh['tetrahedra'])
    if np.any(ref_volumes<=0):raise ValueError('invalid curved reference mesh')
    def set_indentation(delta):
        if not np.isfinite(delta) or delta<0 or delta>=height/10:
            raise ValueError('indentation must satisfy 0 <= delta < height/10')
        prescribed[3*top+2]=-(clearance+delta)
        relative_offset[free%3==2]=clearance+delta
    def displacement(state):
        full=prescribed.copy();full[free]=state
        return full
    def gaps(state):
        return reference[bottom,2]+state[contact_dofs]
    def energy(state):
        s=gaps(state)
        if np.any(s<=0) or np.any(~np.isfinite(s)):return np.inf
        U=displacement(state)
        if np.any(tetrahedron_volumes(reference+U.reshape(-1,3),mesh['tetrahedra'])<=0):
            return np.inf
        b,_,_=barrier_arrays(s,d_hat)
        relative=state+relative_offset
        return float(.5*relative@(Kff@relative)+kappa*(weights@b))
    def objective(state):
        relative=state+relative_offset;Ku=Kff@relative
        b,b1,b2=barrier_arrays(gaps(state),d_hat)
        g=Ku.copy();g[contact_dofs]+=kappa*weights*b1
        diagonal=np.zeros(len(free));diagonal[contact_dofs]=kappa*weights*b2
        H=Kff+diags(diagonal,format='csr')
        return float(.5*relative@Ku+kappa*(weights@b)),g,H,{}
    def max_step(state,trial):
        s=gaps(state);ds=gaps(trial)-s;closing=ds<0
        return min(1.,.99*float(np.min(-s[closing]/ds[closing]))) if np.any(closing) else 1.
    def diagnostics(state):
        U=displacement(state);s=gaps(state)
        _,b1,_=barrier_arrays(s,d_hat);nodal_force=-kappa*weights*b1
        force=float(nodal_force.sum())
        relative=np.zeros(reference.size);relative[free]=state+relative_offset
        top_reaction=float((K@relative)[3*top+2].sum())
        nodes=reference+U.reshape(-1,3)
        projected=nodes.copy();projected[:,2]=0.
        pressure_weights=surface_lumped_weights(projected,mesh['contact_faces'])[bottom]
        pressure=nodal_force/pressure_weights
        volumes=tetrahedron_volumes(nodes,mesh['tetrahedra'])
        return dict(force=force,top_reaction=top_reaction,
                    balance_error=abs(force+top_reaction)/max(force,1e-30),
                    gap_min=float(s.min()),pressure=pressure,pressure_weights=pressure_weights,
                    radial=np.linalg.norm(nodes[bottom,:2],axis=1),
                    min_volume_ratio=float(np.min(volumes/ref_volumes)),nodes=nodes)
    return dict(mesh=mesh,reference=reference,free=free,initial=np.zeros(len(free)),
                stiffness=Kff,contact_dofs=contact_dofs,weights=weights,
                set_indentation=set_indentation,displacement=displacement,gaps=gaps,
                objective=objective,energy=energy,max_step=max_step,diagnostics=diagnostics,
                radius=radius,young=young,poisson=poisson,d_hat=d_hat,kappa=kappa,
                parameters=dict(radius=radius,half_width=half_width,height=height,nxy=nxy,nz=nz,
                                young=young,poisson=poisson,clearance=clearance,d_hat=d_hat,
                                kappa=kappa,grading=grading,depth_grading=depth_grading))


def solve(problem,deltas=(.01,.02,.04),tol=1e-7,max_iter=120,verbose=True):
    if not np.isfinite(tol) or tol<=0 or not isinstance(max_iter,(int,np.integer)) or max_iter<1:
        raise ValueError('require tol>0 and integer max_iter>=1')
    deltas=np.asarray(deltas,dtype=float)
    if deltas.ndim!=1 or not len(deltas) or np.any(~np.isfinite(deltas)) or np.any(deltas<=0):
        raise ValueError('deltas must be a nonempty sequence of positive values')
    direction=make_plane_direction(problem['stiffness'],problem['contact_dofs'])
    state=problem['initial'].copy();results=[]
    for delta in deltas:
        problem['set_indentation'](float(delta));start=perf_counter()
        result=minimize_static(state,problem['objective'],problem['energy'],tol=tol,
                               max_iter=max_iter,max_step=problem['max_step'],
                               direction_solver=direction)
        state=result['state'];data=problem['diagnostics'](state)
        expected=hertz_reference(delta,problem['radius'],problem['young'],problem['poisson'],data['radial'])
        near=data['radial']<1.5*expected['contact_radius']
        pressure_error=np.sqrt(data['pressure_weights']@((data['pressure']-expected['pressure'])**2)/
                               np.sum(data['pressure_weights'][near]))/expected['peak_pressure']
        data.update(delta=float(delta),hertz_force=expected['force'],
                    hertz_radius=expected['contact_radius'],hertz_peak=expected['peak_pressure'],
                    force_error=data['force']/expected['force']-1.,pressure_error=float(pressure_error),
                    seconds=perf_counter()-start)
        result.update(data);results.append(result)
        if verbose:
            print('delta={:.4f} {:12s} updates={:3d} FEM={:.6f} Hertz={:.6f} error={:+.2%} '
                  'balance={:.1e} gap_min={:.2e}'.format(
                      delta,result['status'],result['updates'],data['force'],expected['force'],
                      data['force_error'],data['balance_error'],data['gap_min']),flush=True)
        if result['status']!='converged':break
    return results


def plot_results(problem,results):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    delta=np.array([r['delta'] for r in results]);dense=np.linspace(0,delta.max(),200)
    analytical=[hertz_reference(d,problem['radius'],problem['young'],problem['poisson'])['force'] for d in dense]
    axes[0].plot(dense,analytical,'k--',label='Hertz half-space')
    axes[0].plot(delta,[r['force'] for r in results],'o-',label='Finite cap FEM + barrier')
    axes[0].set(xlabel='Nominal indentation',ylabel='Normal force');axes[0].legend()
    last=results[-1];a=last['hertz_radius'];radial=np.linspace(0,1.5*a,200)
    analytical=hertz_reference(last['delta'],problem['radius'],problem['young'],problem['poisson'],radial)
    selected=last['radial']<1.5*a
    axes[1].scatter(last['radial'][selected],last['pressure'][selected],s=14,label='Nodal pressure')
    axes[1].plot(radial,analytical['pressure'],'k--',label='Hertz')
    axes[1].set(xlabel='Current radial coordinate',ylabel='Normal pressure');axes[1].legend()
    fig.suptitle('Spherical cap on rigid plane: finite domain and mesh errors remain')
    fig.tight_layout()
    return fig


def save_results(problem,results,outdir,settings=None):
    import csv,json,hashlib,platform,sys
    from datetime import datetime,timezone
    from pathlib import Path
    import scipy
    directory=Path(outdir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('result directory is not empty; choose a new --outdir')
    directory.mkdir(parents=True,exist_ok=True)
    fields=('delta','status','updates','residual','energy','force','top_reaction','hertz_force',
            'force_error','pressure_error','balance_error','gap_min','min_volume_ratio','seconds')
    with (directory/'summary.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in results)
    with (directory/'pressure.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file);writer.writerow(('delta','node','radius','gap','pressure','plane_area','hertz_pressure'))
        bottom=problem['mesh']['bottom_nodes']
        for r in results:
            exact=hertz_reference(r['delta'],problem['radius'],problem['young'],problem['poisson'],r['radial'])
            gaps=r['nodes'][bottom,2]
            writer.writerows(zip(np.full(len(bottom),r['delta']),bottom,r['radial'],gaps,
                                 r['pressure'],r['pressure_weights'],exact['pressure']))
    np.savez_compressed(directory/'states.npz',
        reference=problem['reference'],tetrahedra=problem['mesh']['tetrahedra'],
        contact_faces=problem['mesh']['contact_faces'],top_faces=problem['mesh']['top_faces'],
        bottom_nodes=problem['mesh']['bottom_nodes'],free_dofs=problem['free'],
        reference_weights=problem['weights'],deltas=np.array([r['delta'] for r in results]),
        states=np.stack([r['state'] for r in results]),nodes=np.stack([r['nodes'] for r in results]))
    traces=[dict(delta=r['delta'],trace=r['trace']) for r in results]
    (directory/'traces.json').write_text(json.dumps(traces,indent=2),encoding='utf-8')
    root=Path(__file__).resolve().parents[2]
    files=('src/simulation/hertz.py','src/geometry/mesh.py','src/geometry/surface.py',
           'src/mechanics/solid.py','src/contact/barrier.py','src/optimization/newton.py')
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        parameters=problem['parameters'],settings=settings or {},
        deltas=[r['delta'] for r in results],
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        code_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
        scope='Finite spherical-bottom cap; rigid infinite plane; linear FEM; nodal reference areas; exact plane gap s=z',
        pressure_definition='Nodal normal force divided by current projected plane area',
        time_definition='seconds per load excludes FEM preparation and the initial cached LU; wall_total_s includes both')
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    try:
        fig=plot_results(problem,results)
    except ModuleNotFoundError as exc:
        if exc.name!='matplotlib':raise
        print('Numeric results saved. Install matplotlib to export the comparison plot.')
    else:
        fig.savefig(directory/'force_pressure.png',dpi=180)
        import matplotlib.pyplot as plt
        plt.close(fig)
    print('Saved Hertz results: '+str(directory.resolve()),flush=True)
    return directory


def show_results(problem,results):
    import matplotlib.pyplot as plt
    plot_results(problem,results);plt.show()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name,default in (('radius',10.),('half-width',6.),('height',10.),('young',1000.),
                         ('poisson',.3),('clearance',.002),('d-hat',.001),('grading',5.),('depth-grading',3.)):
        parser.add_argument('--'+name,type=float,default=default)
    parser.add_argument('--nxy',type=int,default=25);parser.add_argument('--nz',type=int,default=13)
    parser.add_argument('--kappa',type=float,default=None)
    parser.add_argument('--deltas',type=float,nargs='+',default=[.01,.02,.04])
    parser.add_argument('--tol',type=float,default=1e-7);parser.add_argument('--max-iter',type=int,default=120)
    parser.add_argument('--threads',type=int,default=1);parser.add_argument('--show',action='store_true')
    parser.add_argument('--outdir',default=None,help='save tables, states, settings, traces and comparison figure')
    args=parser.parse_args()
    if args.threads<1:parser.error('--threads must be >= 1')
    from threadpoolctl import threadpool_limits
    parameters=vars(args).copy()
    for name in ('deltas','tol','max_iter','threads','show','outdir'):parameters.pop(name)
    with threadpool_limits(args.threads):
        start=perf_counter();problem=build_problem(**parameters)
        print('Hertz verification: {} free DOFs, {} tets, planar gap s=z'.format(
            len(problem['free']),len(problem['mesh']['tetrahedra'])),flush=True)
        results=solve(problem,args.deltas,args.tol,args.max_iter)
        wall_total=perf_counter()-start
        print('Total wall time: {:.3f} s'.format(wall_total),flush=True)
    if args.outdir:
        save_results(problem,results,args.outdir,dict(tol=args.tol,max_iter=args.max_iter,
                                                   threads=args.threads,wall_total_s=wall_total))
    if args.show:show_results(problem,results)
    return 0 if all(r['status']=='converged' for r in results) else 1


if __name__=='__main__':
    raise SystemExit(main())
