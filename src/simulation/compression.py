"""Study 3: incremental prescribed compression of two deformable blocks.
Run: python run.py two-body --show
No output files unless --outdir is supplied; figures are never saved.
"""
import argparse,csv,json
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from src.simulation.two_body import build_problem,solve

def make_animation(problem,histories,fps=4):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button,Slider
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if fps<=0:raise ValueError('fps must be positive')
    methods=list(histories);count=len(histories[methods[0]])
    if count<2 or any(len(histories[m])!=count for m in methods):
        raise ValueError('at least two matching load frames required')
    fig=plt.figure(figsize=(5*len(methods),8))
    fig.subplots_adjust(bottom=.14,top=.90,hspace=.25,wspace=.25)
    artists={};colors=('tab:blue','tab:orange')
    zmax=max(ref[:,2].max() for ref in problem['reference'])+.04
    for column,method in enumerate(methods,1):
        ax=fig.add_subplot(2,len(methods),column,projection='3d')
        side=fig.add_subplot(2,len(methods),len(methods)+column)
        collections=[];lines=[]
        for ref,mesh,color in zip(problem['reference'],problem['meshes'],colors):
            faces=mesh['boundary_faces']
            collection=Poly3DCollection(ref[faces],facecolors=color,edgecolors='black',alpha=.45,linewidths=.5)
            ax.add_collection3d(collection);collections.append(collection)
            on_side=np.isclose(ref[:,1],ref[:,1].min())
            perimeter=(np.isclose(ref[:,0],ref[:,0].min())|np.isclose(ref[:,0],ref[:,0].max())|
                       np.isclose(ref[:,2],ref[:,2].min())|np.isclose(ref[:,2],ref[:,2].max()))
            ids=np.flatnonzero(on_side & perimeter)
            center=ref[ids][:,[0,2]].mean(axis=0)
            order=np.argsort(np.arctan2(ref[ids,2]-center[1],ref[ids,0]-center[0]))
            ids=ids[order];ids=np.r_[ids,ids[0]]
            line,=side.plot(ref[ids,0],ref[ids,2],'-o',color=color,markersize=3)
            side.plot(ref[ids,0],ref[ids,2],'--',color='gray',alpha=.4)
            lines.append((line,ids))
        ax.set(xlim=(-.05,1.05),ylim=(-.05,1.05),zlim=(-.03,zmax),xlabel='x',ylabel='y',zlabel='z',title=method)
        ax.view_init(elev=18,azim=-65)
        if hasattr(ax,'set_box_aspect'):ax.set_box_aspect((1,1,zmax))
        side.set(xlim=(-.05,1.05),ylim=(.30,.4+problem['initial_gap']+.10),xlabel='x',ylabel='z',title='Contact region: actual scale')
        side.axhline(.4,color='gray',alpha=.25);side.grid(alpha=.2)
        text=side.text(.02,.98,'',transform=side.transAxes,va='top',fontsize=9)
        artists[method]=(collections,lines,text)
    title=fig.suptitle('')
    slider=Slider(fig.add_axes([.23,.045,.65,.025]),'Load step',0,count-1,valinit=0,valstep=1)
    button=Button(fig.add_axes([.07,.025,.10,.06]),'Pause')
    state={'frame':0,'playing':True}
    def render(index):
        index=int(index);state['frame']=index
        for method in methods:
            frame=histories[method][index];collections,lines,text=artists[method]
            for nodes,mesh,collection,(line,ids) in zip(frame['nodes'],problem['meshes'],collections,lines):
                collection.set_verts(nodes[mesh['boundary_faces']]);line.set_data(nodes[ids,0],nodes[ids,2])
            text.set_text('closure=%.4f | gap_min=%.4f\n|Fz|=%.4f | %s' %
                (frame['closure'],frame['min_gap'],abs(frame['upper_Fz']),'barrier active' if frame['min_gap']<problem['d_hat'] else 'no contact force'))
        title.set_text('Quasi-static compression | load progress %.0f%% | frame %d/%d' %
            (100.*index/(count-1),index,count-1))
        return []
    def scrub(value):render(value);fig.canvas.draw_idle()
    slider.on_changed(scrub)
    def advance(_):
        if state['playing']:
            index=(state['frame']+1)%count
            slider.set_val(index)
        return []
    animation=FuncAnimation(fig,advance,init_func=lambda:render(0),interval=1000./fps,blit=False,cache_frame_data=False)
    def pause(_):
        state['playing']=not state['playing'];button.label.set_text('Pause' if state['playing'] else 'Play')
    button.on_clicked(pause)
    render(0)
    fig._two_body_controls=(animation,slider,button,render)
    return fig,animation

def show_results(problem,solutions,fps=4):
    import matplotlib.pyplot as plt
    fig,animation=make_animation(problem,problem['history'],fps)
    plt.show()


def run(n=None,steps=20,closure=.14,methods=('full','gn','filtered_gn'),show=False,outdir=None,nx=None,ny=None,nz=None,initial_gap=.24,fps=4,sparse=True,collision="ipctk",contact=None,skin=.1,sr1_memory=5):
    contact=('auto' if collision=='ipctk' else 'manual') if contact is None else contact
    if contact not in ('auto','manual'):raise ValueError('contact must be auto or manual')
    if contact=='auto' and collision!='ipctk':raise ValueError('automatic patches require IPC Toolkit')
    if collision not in ('ipctk','none'):raise ValueError('collision must be ipctk or none')
    if steps<1:raise ValueError('steps must be positive')
    if not methods or len(set(methods))!=len(methods):raise ValueError('unique methods required')
    if fps<=0:raise ValueError('fps must be positive')
    torch.set_num_threads(1);rows=[];solutions={};states={}
    history={m:[] for m in methods}
    with threadpool_limits(limits=1),ExitStack() as resources:
        collider=None;automatic={};previous_positions={}
        # The small-strain reference stiffness is constant across load steps.
        p=build_problem(n=n,closure=closure,nx=nx,ny=ny,nz=nz,initial_gap=initial_gap,sparse=sparse)
        for step,amount in enumerate(np.linspace(0,closure,steps+1)):
            p['set_closure'](float(amount))
            if step==0:
                print('Mesh nx,ny,nz=%s nodes/body=%d tets/body=%d free_DOFs=%d contact_samples=%d' %
                    (p['mesh_counts'],len(p['reference'][0]),len(p['meshes'][0]['tetrahedra']),len(p['initial']),len(p['weights'])),flush=True)
                print('FEM storage:', 'CSR, nnz='+str(p['stiffness'].nnz) if sparse else 'dense (reference)',flush=True)
            if collision=='ipctk' and collider is None:
                from src.collision.ipc_toolkit import IPCMeshCollision
                positions=np.vstack(p['reference']);offset=len(p['reference'][0])
                faces=np.vstack((p['meshes'][0]['boundary_faces'],p['meshes'][1]['boundary_faces']+offset))
                labels=np.r_[np.zeros(offset,dtype=int),np.ones(len(p['reference'][1]),dtype=int)]
                collider=resources.enter_context(IPCMeshCollision(positions,faces,labels))
                print('Collision backend:',collider.info,flush=True)
                print('RIMLS contact selection:',contact,flush=True)
            if collider is not None:
                p['max_step']=lambda x,y,p=p:collider.max_step(p['full_state'](x).reshape(-1,3),p['full_state'](y).reshape(-1,3))
            for method in methods:
                if contact=='auto':
                    if method not in automatic:
                        from src.geometry.contact_patches import AutomaticRIMLSContact
                        automatic[method]=AutomaticRIMLSContact(positions,faces,labels,collider,d_hat=p['d_hat'],skin=skin)
                    model=automatic[method]
                    p['attach_auto_contact'](model)
                    def safe_step(x,y,p=p,model=model):
                        a=p['full_state'](x).reshape(-1,3);b=p['full_state'](y).reshape(-1,3)
                        return min(collider.max_step(a,b),model.max_step(a,b))
                    p['max_step']=safe_step
                if collider is not None and method in previous_positions:
                    start=p['full_state'](states[method]).reshape(-1,3)
                    if collider.max_step(previous_positions[method],start)<1-1e-12:
                        raise RuntimeError('Prescribed boundary step crosses triangles; reduce load step size')
                r=solve(p,method,initial=states.get(method),sr1_memory=sr1_memory);d=p['diagnostics'](r['state'])
                nodes=p['unpack'](r['state'])
                row=dict(contact=contact,collision=collision,sparse=sparse,nx=p['mesh_counts'][0],ny=p['mesh_counts'][1],nz=p['mesh_counts'][2],step=step,closure=float(amount),method=method,status=r['status'],dofs=len(r['state']),
                    contact_samples=len(p['weights']),updates=r['updates'],residual=r['residual'],energy=r['energy'],elapsed_s=r['elapsed_s'],
                    sr1_updates=r.get('sr1_updates',0),sr1_skips=r.get('sr1_skips',0),sr1_resets=r.get('sr1_resets',0),
                    sr1_max_factors=max([t.get('sr1_factor_count',0) for t in r['trace']]+[0]),
                    min_gap=float(d['gaps'].min()) if len(d['gaps']) else float('inf'),max_gap=float(d['gaps'].max()) if len(d['gaps']) else float('inf'),
                    candidate_count=int(r['info'].get('candidate_count',0)),candidate_face_pairs=int(r['info'].get('candidate_face_pairs',0)),
                    patch_updates=sum(bool(t.get('patch_changed',False)) for t in r['trace']),
                    max_patch_energy_jump=max([abs(t.get('patch_energy_jump',0.)) for t in r['trace']]+[0.]),
                    lower_move=float(np.max(np.linalg.norm(nodes[0]-p['reference'][0],axis=1))),
                    upper_move=float(np.max(np.linalg.norm(nodes[1]-p['reference'][1],axis=1))),
                    lower_Fz=float(d['force_lower'][2]),upper_Fz=float(d['force_upper'][2]),
                    force_balance_error=float(np.linalg.norm(d['force_lower']+d['force_upper'])))
                rows.append(row);print('step=%d closure=%.3f %-12s %-12s updates=%d gap=%.6f Fz=(%.6f,%.6f) balance=%.2e' %
                    (step,amount,method,r['status'],r['updates'],row['min_gap'],row['lower_Fz'],row['upper_Fz'],row['force_balance_error']),flush=True)
                if method=='sr1_gap':
                    print('  Gap SR1 updates=%d skips=%d resets=%d max_factors=%d' %
                          (row['sr1_updates'],row['sr1_skips'],row['sr1_resets'],row['sr1_max_factors']),flush=True)
                if contact=='auto':
                    print('  BVH=%d face_pairs=%d samples=%d patch_jump=%.2e' %
                          (row['candidate_count'],row['candidate_face_pairs'],row['contact_samples'],row['max_patch_energy_jump']),flush=True)
                if r['status']!='converged':raise RuntimeError('failed step; stop continuation: '+method)
                states[method]=r['state'].copy();solutions[method]=r
                history[method].append(dict(row,nodes=[a.copy() for a in nodes]))
                previous_positions[method]=np.vstack(nodes)
    p['history']=history;p['initial_gap']=initial_gap
    if outdir:
        folder=Path(outdir);folder.mkdir(parents=True,exist_ok=True)
        with (folder/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (folder/'states.json').write_text(json.dumps({m:r['state'].tolist() for m,r in solutions.items()},indent=2))
    if show:show_results(p,solutions,fps)
    return p,solutions,rows

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--n',type=int,help='uniform node count; default grid is 3 x 3 x 2');
    for axis in ('nx','ny','nz'):a.add_argument('--'+axis,type=int,help='axis node count; overrides --n')
    a.add_argument('--steps',type=int,default=20)
    a.add_argument('--initial-gap',type=float,default=.24);a.add_argument('--fps',type=float,default=4)
    a.add_argument('--closure',type=float,default=.14);a.add_argument('--methods',nargs='+',choices=('full','gn','filtered_gn','sr1_gap'),default=['full','gn','filtered_gn'])
    a.add_argument('--collision',choices=('ipctk','none'),default='ipctk',help='explicit-triangle CCD guard; RIMLS contact faces remain prescribed')
    a.add_argument('--contact',choices=('auto','manual'),help='default: auto with ipctk, manual without it')
    a.add_argument('--skin',type=float,default=.1,help='extra BVH search distance; also bounds nodal movement')
    a.add_argument('--sr1-memory',type=int,default=5,help='recent secant pairs retained per active gap')  # ##r##
    a.add_argument('--dense',action='store_true',help='use old dense assembly/solve for verification')  # ##r##
    a.add_argument('--show',action='store_true');a.add_argument('--outdir');args=a.parse_args()
    run(args.n,args.steps,args.closure,tuple(args.methods),args.show,args.outdir,nx=args.nx,ny=args.ny,nz=args.nz,initial_gap=args.initial_gap,fps=args.fps,sparse=not args.dense,collision=args.collision,contact=args.contact,skin=args.skin,sr1_memory=args.sr1_memory)


if __name__=="__main__":main()
