"""##r## Offline RIMLS sampling/normal audit. Production contact is unchanged.

Factorial position/normal controls use the SAME sample IDs. Area weighting,
vertex sampling, and uniform analytic sampling are diagnostics, not new defaults.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from src.geometry.mesh import make_cylinder
from src.geometry.mesh import refine_cylinder_contact
from src.geometry.surface import make_surface_sampler, surface_lumped_weights
from src.geometry.rimls import rimls_values_torch


def weighted_reference(q,p,n,R,weights):
    """Independent Eq.(4),(12), one robust update; optional density control."""
    diff=q[:,None,:]-p[None,:,:]
    t=torch.clamp(1.-(diff*diff).sum(2)/(R*R),min=0.)
    phi=t**4*weights[None,:]
    G=(-8./(R*R))*t[:,:,None]**3*diff*weights[None,:,None]
    D=phi.sum(1)
    if torch.any(D<1e-14).item():raise ValueError('outside support')
    local=(diff*n[None,:,:]).sum(2)
    f=(phi*local).sum(1)/D
    h=(phi[:,:,None]*n[None,:,:]+(local-f[:,None])[:,:,None]*G).sum(1)/D[:,None]
    W=phi*torch.exp(-((f[:,None]-local)/(.5*R))**2-((h[:,None,:]-n[None,:,:])**2).sum(2)/.5**2)
    Z=W.sum(1)
    if torch.any(Z<=1e-14).item():raise ValueError('vanishing weights')
    return (W*local).sum(1)/Z


class FixedField:
    def __init__(self,p,n,weights=None):
        self.p=torch.tensor(p,dtype=torch.float64);self.n=torch.tensor(n,dtype=torch.float64)
        self.tree=cKDTree(p)
        self.weights=None if weights is None else torch.tensor(weights/np.mean(weights),dtype=torch.float64)

    def evaluate(self,q,R,gradient=True):
        q=np.asarray(q,dtype=float)
        values=[];gradients=[]
        for start in range(0,len(q),32):
            block=q[start:start+32]
            neighbours=self.tree.query_ball_point(block,np.nextafter(R,np.inf),return_sorted=True)
            ids=np.unique(np.concatenate(neighbours)).astype(int)
            if not len(ids):raise ValueError('outside support')
            query=torch.tensor(block,dtype=torch.float64,requires_grad=gradient)
            if self.weights is None:
                s=rimls_values_torch(query,self.p[ids],self.n[ids],R)
            else:
                s=weighted_reference(query,self.p[ids],self.n[ids],R,self.weights[ids])
            if gradient:gradients.append(torch.autograd.grad(s.sum(),query)[0].detach().numpy())
            values.append(s.detach().numpy())
        return np.concatenate(values),np.concatenate(gradients) if gradient else None


def analytic_points_normals(points,radius,facing,angle,clearance):
    a=np.deg2rad(angle)
    rotation=np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
    local=np.asarray(points)@rotation
    center=-radius if facing=='up' else clearance+radius
    radial=local[:,[0,2]].copy();radial[:,1]-=center
    radial/=np.linalg.norm(radial,axis=1)[:,None]
    normal=np.column_stack((radial[:,0],np.zeros(len(points)),radial[:,1]))
    projected=local.copy();projected[:,0]=radius*radial[:,0];projected[:,2]=center+radius*radial[:,1]
    return projected@rotation.T,normal@rotation.T


def numpy_one_pass(q,p,n,R):
    diff=q-p;t=np.maximum(1.-np.sum(diff*diff,axis=1)/R**2,0.)
    phi=t**4;G=-8./R**2*t[:,None]**3*diff;D=phi.sum()
    local=np.einsum('ij,ij->i',diff,n);f=phi@local/D
    h=(phi[:,None]*n+(local-f)[:,None]*G).sum(0)/D
    W=phi*np.exp(-((f-local)/(.5*R))**2-np.sum((h-n)**2,axis=1)/.5**2)
    return float(W@local/W.sum())


def rays(radius,facing,angle,clearance):
    x,y=np.meshgrid([-.3,-.15,0.,.15,.3],[-.2,0.,.2]);x=x.ravel();y=y.ravel()
    sign=1 if facing=='up' else -1;center=-radius if facing=='up' else clearance+radius
    n=np.column_stack((x/radius,np.zeros(len(x)),sign*np.sqrt(1-(x/radius)**2)))
    p=radius*n;p[:,1]=y;p[:,2]+=center
    a=np.deg2rad(angle);rot=np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
    return p@rot.T,n@rot.T,x,y


def uniform_analytic(radius,facing,angle,clearance,spacing):
    theta=np.linspace(-np.arcsin(1.2/radius),np.arcsin(1.2/radius),
                      int(np.ceil(2*radius*np.arcsin(1.2/radius)/spacing))+1)
    axial=np.linspace(-1.2,1.2,int(np.ceil(2.4/spacing))+1)
    theta,y=np.meshgrid(theta,axial);theta=theta.ravel();y=y.ravel()
    sign=1 if facing=='up' else -1;center=-radius if facing=='up' else clearance+radius
    n=np.column_stack((np.sin(theta),np.zeros(len(theta)),sign*np.cos(theta)))
    p=radius*n;p[:,1]=y;p[:,2]+=center
    a=np.deg2rad(angle);rot=np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
    return p@rot.T,n@rot.T


def measure(field,pole,normal,R,shared):
    offsets=np.array([-.001,-.0005,0.,.0005,.001,.005,.03])
    query=(pole[:,None,:]+offsets[None,:,None]*normal[:,None,:]).reshape(-1,3)
    s,g=field.evaluate(query,R)
    s=s.reshape(len(pole),-1);g=g.reshape(len(pole),len(offsets),3)
    records=[];roots=[]
    for i in range(len(pole)):
        def fn(delta):return field.evaluate(pole[i:i+1]+delta*normal[i:i+1],R,gradient=False)[0][0]
        root=brentq(fn,-.01,.01,xtol=1e-13);roots.append(float(root))
        for j,delta in enumerate(offsets):
            records.append(dict(shared,R=R,ray=i,distance=float(delta),gap=float(s[i,j]),
                gap_error=float(s[i,j]-delta),gradient_norm=float(np.linalg.norm(g[i,j])),
                anchored_error=float(s[i,j]-s[i,2]-delta),surface_shift=float(root)))
    near=[r for r in records if abs(r['distance'])<=.001]
    summary=dict(shared,R=R,samples=len(field.p),max_surface_shift=max(abs(r) for r in roots),
        pole_surface_shift=roots[7],shift_min=min(roots),shift_max=max(roots),
        near_eikonal_max=max(abs(r['gradient_norm']-1) for r in near),
        near_anchored_max=max(abs(r['anchored_error']) for r in near))
    return records,summary


def run(outdir):
    folder=Path(outdir);folder.mkdir(parents=True,exist_ok=False)
    rows=[];summary=[];checks=[];start=time.perf_counter()
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with threadpool_limits(1):
            for body,radius,facing,angle,clearance in [('lower',3.,'up',0.,0.),('upper',5.,'down',45.,.03)]:
                pole,normal,_,_=rays(radius,facing,angle,clearance)
                for local_size in (None,.05):
                    mesh=make_cylinder(radius,n_angle=128,n_axis=25,n_radial=8,radial_grading=3.,
                        facing=facing,angle_degrees=angle,clearance=clearance)
                    if local_size is not None:mesh=refine_cylinder_contact(mesh,local_size,.4,.3)
                    faces=mesh['contact_faces'];sample,_,areas=make_surface_sampler(mesh['nodes'],faces,sparse=True)
                    with torch.no_grad():p,n=sample(torch.tensor(mesh['nodes'],dtype=torch.float64))
                    p=p.numpy();n=n.numpy()
                    projected,exact_normals=analytic_points_normals(p,radius,facing,angle,clearance)
                    used=np.unique(faces);vertices=mesh['nodes'][used]
                    remap=np.full(len(mesh['nodes']),-1,dtype=int);remap[used]=np.arange(len(used))
                    local_faces=remap[faces];xyz=vertices[local_faces]
                    area_normals=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
                    vn=np.zeros_like(vertices);unit_vn=np.zeros_like(vertices)
                    for corner in range(3):
                        np.add.at(vn,local_faces[:,corner],area_normals)
                        np.add.at(unit_vn,local_faces[:,corner],area_normals/np.linalg.norm(area_normals,axis=1)[:,None])
                    vn/=np.linalg.norm(vn,axis=1)[:,None];unit_vn/=np.linalg.norm(unit_vn,axis=1)[:,None]
                    variants=[('current',p,n,None),('exact_normals_only',p,exact_normals,None),
                        ('exact_positions_only',projected,n,None),('exact_positions_normals',projected,exact_normals,None),
                        ('area_weighted_current',p,n,areas),
                        ('area_weighted_exact',projected,exact_normals,areas),
                        ('vertex_area_normals',vertices,vn,None),('vertex_equal_face_normals',vertices,unit_vn,None)]
                    for label,points,normals,weights in variants:
                        field=FixedField(points,normals,weights)
                        rr,ss=measure(field,pole,normal,.1625,dict(body=body,local_size=local_size,variant=label))
                        rows.extend(rr);summary.append(ss)
                        print(json.dumps(ss),flush=True)
                    q=pole+normal*.0005
                    actual,_=FixedField(p,n).evaluate(q,.1625)
                    expected=np.array([numpy_one_pass(a,p,n,.1625) for a in q])
                    checks.append(dict(body=body,local_size=local_size,
                        max_native_vs_independent_numpy=float(np.max(np.abs(actual-expected)))))
                for spacing in (.025,.0125):
                    p,n=uniform_analytic(radius,facing,angle,clearance,spacing)
                    field=FixedField(p,n)
                    for R in (.08125,.1625,.325):
                        rr,ss=measure(field,pole,normal,R,dict(body=body,local_size=None,variant='uniform_exact_h'+str(spacing)))
                        rows.extend(rr);summary.append(ss);print(json.dumps(ss),flush=True)
    finally:torch.set_num_threads(old)
    with (folder/'queries.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(description='diagnostic controls only; original contact formulation unchanged',R=.1625,
        sigma_r=.5,sigma_n=.5,n_angle=128,n_axis=25,n_radial=8,radial_grading=3.,
        formula_checks=checks,groups=summary,elapsed_s=time.perf_counter()-start,
        paper_audit=dict(source='E:/NR/IMLS/Paper/IMLS.pdf',pages=[4,12,13],
            matches=['quartic compact-support kernel','initial IMLS gradient Eq.12','one robust update k=1',
                'sigma_r=0.5; sigma_n=0.5 is within the reported range'],
            differences=['current quadrature samples vs paper mesh-vertex samples',
                'current locally nonuniform sample density vs paper isotropic/uniform preprocessing',
                'current R fixed; paper recommends 2-3 times largest edge length'],
            sign_note='Printed Eq.11 reverses the displacement compared with Eq.4 and Eq.12; use Eq.4/outward-positive plane convention, not a blind sign flip.',
            normal_note='Paper describes averaging incident face normals but does not fix area vs equal-face weighting explicitly in this passage.'))
    (folder/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('Saved:',folder.resolve(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',required=True)
    run(parser.parse_args().outdir)
