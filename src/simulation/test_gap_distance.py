"""##r## Fixed-cylinder gap calibration; no FEM solve or contact-model changes.

The mesh is an extrusion of a convex inscribed polygon. Its exact cross-section
signed distance separates faceting error from the RIMLS field/zero-set errors.
Only query coordinates are differentiated; target points/normals remain fixed.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from src.geometry.mesh import make_cylinder
from src.geometry.mesh import refine_cylinder_contact
from src.geometry.surface import make_surface_sampler
from src.geometry.rimls import rimls_values_torch


def cylinder_polygon(radius, n_angle, facing):
    t=np.arange(n_angle//4)/(n_angle//4)
    theta=np.concatenate([q*np.pi/2+(np.pi/4)*(1-np.cos(np.pi*t)) for q in range(4)])
    sign=1 if facing=='up' else -1
    return radius*np.column_stack((np.sin(theta),sign*np.cos(theta)))


def polygon_distance(points, polygon):
    """Exact signed Euclidean distance to a convex polygon (outside positive)."""
    points=np.asarray(points,dtype=float);a=polygon;b=np.roll(a,-1,axis=0);e=b-a
    diff=points[:,None,:]-a
    t=np.clip(np.einsum('qed,ed->qe',diff,e)/np.sum(e*e,axis=1),0.,1.)
    distance=np.linalg.norm(diff-t[:,:,None]*e,axis=2).min(axis=1)
    orientation=np.sign(np.sum(a[:,0]*b[:,1]-a[:,1]*b[:,0]))
    cross=e[None,:,0]*diff[:,:,1]-e[None,:,1]*diff[:,:,0]
    inside=np.all(orientation*cross>=-1e-14,axis=1)
    return np.where(inside,-distance,distance)


def polygon_radial_radius(directions, polygon):
    edge=np.roll(polygon,-1,axis=0)-polygon
    orientation=np.sign(np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1)))
    outward=orientation*np.column_stack((edge[:,1],-edge[:,0]))
    h=np.einsum('ed,ed->e',outward,polygon)
    denom=directions@outward.T
    ratios=np.full_like(denom,np.inf)
    np.divide(h[None,:],denom,out=ratios,where=denom>1e-15)
    return ratios.min(axis=1)


def field_and_gradient(queries, points, normals, R):
    """Independent rows permit one reverse pass of sum(s) for query gradients."""
    queries=np.asarray(queries,dtype=float)
    values=np.full(len(queries),np.nan);gradient=np.full_like(queries,np.nan)
    # Exclude unsupported rows individually; do not hide coverage failures.
    nearest=cKDTree(points).query(queries)[0]
    supported=np.flatnonzero(nearest<R)
    pt=torch.tensor(points,dtype=torch.float64);nt=torch.tensor(normals,dtype=torch.float64)
    for start in range(0,len(supported),32):
        ids=supported[start:start+32]
        q=torch.tensor(queries[ids],dtype=torch.float64,requires_grad=True)
        try:
            s=rimls_values_torch(q,pt,nt,R)
            g=torch.autograd.grad(s.sum(),q)[0]
            values[ids]=s.detach().numpy();gradient[ids]=g.detach().numpy()
        except ValueError:
            for index in ids:
                q=torch.tensor(queries[index:index+1],dtype=torch.float64,requires_grad=True)
                try:
                    s=rimls_values_torch(q,pt,nt,R)
                    g=torch.autograd.grad(s.sum(),q)[0]
                    values[index]=s.item();gradient[index]=g.detach().numpy()[0]
                except ValueError:pass
    return values,gradient


def measure_case(n_angle,n_axis,local_size,R_values,body):
    radius,facing,angle,clearance=(3.,'up',0.,0.) if body=='lower' else (5.,'down',45.,.03)
    mesh=make_cylinder(radius,n_angle=n_angle,n_axis=n_axis,n_radial=8,radial_grading=3.,
                       facing=facing,angle_degrees=angle,clearance=clearance)
    if local_size is not None:mesh=refine_cylinder_contact(mesh,local_size,.4,.3)
    sample,_,_=make_surface_sampler(mesh['nodes'],mesh['contact_faces'],sparse=True)
    with torch.no_grad():pt,nt=sample(torch.tensor(mesh['nodes'],dtype=torch.float64))
    points=pt.numpy();normals=nt.numpy()
    polygon=cylinder_polygon(radius,n_angle,facing)
    x_values=np.array([-.3,-.15,0.,.15,.3])
    axial_values=np.array([-.2,0.,.2])
    x,y=np.meshgrid(x_values,axial_values);x=x.ravel();y=y.ravel()
    sign=1 if facing=='up' else -1
    z_center=-radius if facing=='up' else clearance+radius
    directions=np.column_stack((x/radius,sign*np.sqrt(1-(x/radius)**2)))
    delta_mesh=polygon_radial_radius(directions,polygon)-radius
    normals_local=np.column_stack((directions[:,0],np.zeros(len(x)),directions[:,1]))
    pole=np.column_stack((x,y,z_center+radius*directions[:,1]))
    a=np.deg2rad(angle);rotation=np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
    pole=pole@rotation.T;normal_world=normals_local@rotation.T
    offsets=np.array([-.001,-.0005,0.,.0005,.001,.005,.03])
    records=[];ray_records=[]
    for R in R_values:
        q=(pole[:,None,:]+offsets[None,:,None]*normal_world[:,None,:]).reshape(-1,3)
        values,grad=field_and_gradient(q,points,normals,R)
        values=values.reshape(len(x),-1);grad=grad.reshape(len(x),len(offsets),3)
        at_mesh,_=field_and_gradient(pole+delta_mesh[:,None]*normal_world,points,normals,R)
        for i in range(len(x)):
            def fn(delta):
                query=pole[i]+delta*normal_world[i]
                with torch.no_grad():
                    return rimls_values_torch(torch.tensor(query[None,:],dtype=torch.float64),pt,nt,R).item()
            root=None
            try:
                root=float(brentq(fn,-.01,.01,xtol=1e-13))
            except ValueError:pass
            shared=dict(body=body,n_angle=n_angle,n_axis=n_axis,local_size=local_size,
                        samples=len(points),R=float(R),x=float(x[i]),axis=float(y[i]))
            ray_records.append(dict(shared,s_on_true_surface=float(values[i,2]),
                s_on_mesh_surface=float(at_mesh[i]),mesh_surface_shift=float(delta_mesh[i]),
                rimls_surface_shift=root,
                rimls_shift_from_mesh=None if root is None else root-float(delta_mesh[i])))
            for j,delta in enumerate(offsets):
                radial=(radius+delta)*directions[i]
                mesh_d=float(polygon_distance(radial[None,:],polygon)[0])
                norm=float(np.linalg.norm(grad[i,j]));directional=float(grad[i,j]@normal_world[i])
                value=float(values[i,j]);s0=float(values[i,2])
                records.append(dict(shared,true_distance=float(delta),mesh_distance=mesh_d,
                    rimls_gap=value,gap_error_true=value-delta,gap_error_mesh=value-mesh_d,
                    gradient_norm=norm,eikonal_error=abs(norm-1.),normal_derivative=directional,
                    anchored_error=(value-s0)-delta,
                    gradient_alignment=directional/norm if norm>0 else float('nan'),
                    supported=bool(np.isfinite(value) and np.isfinite(norm))))
        group=records[-len(x)*len(offsets):]
        valid=[r for r in group if r['supported']]
        if valid:
            print('%s angle=%d axis=%d local=%s R=%.5g support=%d/%d max|norm-1|=%.3g max|s-d|=%.3g' %
                  (body,n_angle,n_axis,local_size,R,len(valid),len(group),
                   max(r['eikonal_error'] for r in valid),max(abs(r['gap_error_true']) for r in valid)),flush=True)
    return records,ray_records


def summarize(rows,rays):
    groups={}
    for r in rows:
        key=(r['body'],r['n_angle'],r['n_axis'],r['local_size'],r['R'])
        groups.setdefault(key,[]).append(r)
    result=[]
    for key,group in groups.items():
        valid=[r for r in group if r['supported']]
        near=[r for r in valid if abs(r['true_distance'])<=.001]
        ray=[r for r in rays if (r['body'],r['n_angle'],r['n_axis'],r['local_size'],r['R'])==key]
        roots=[r for r in ray if r['rimls_surface_shift'] is not None]
        result.append(dict(body=key[0],n_angle=key[1],n_axis=key[2],local_size=key[3],R=key[4],
            samples=group[0]['samples'],supported=len(valid),total=len(group),
            gradient_norm_range=[min(r['gradient_norm'] for r in valid),max(r['gradient_norm'] for r in valid)] if valid else None,
            near_max_eikonal_error=max((r['eikonal_error'] for r in near),default=None),
            near_max_gap_error_true=max((abs(r['gap_error_true']) for r in near),default=None),
            near_max_anchored_error=max((abs(r['anchored_error']) for r in near),default=None),
            max_surface_shift=max((abs(r['rimls_surface_shift']) for r in roots),default=None),
            max_mesh_surface_shift=max(abs(r['mesh_surface_shift']) for r in ray),
            max_rimls_shift_from_mesh=max((abs(r['rimls_shift_from_mesh']) for r in roots),default=None),
            zero_roots_found=len(roots),zero_roots_total=len(ray)))
    return result


def run(outdir):
    folder=Path(outdir);folder.mkdir(parents=True,exist_ok=False)
    rows=[];rays=[];start=time.perf_counter()
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with threadpool_limits(1):
            for n_angle,n_axis,local_size in ((64,13,None),(128,25,None),(256,49,None),(128,25,.05)):
                for body in ('lower','upper'):
                    a,b=measure_case(n_angle,n_axis,local_size,(.08125,.1625,.325),body)
                    rows.extend(a);rays.extend(b)
    finally:torch.set_num_threads(old_threads)
    for name,data in (('queries.csv',rows),('surface_shifts.csv',rays)):
        with (folder/name).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    report=dict(description='undeformed fixed surfaces; exact radial normal rays; original surface sampler',
        d_hat=.0005,offsets=[-.001,-.0005,0.,.0005,.001,.005,.03],
        sigma_r=.5,sigma_n=.5,elapsed_s=time.perf_counter()-start,groups=summarize(rows,rays))
    (folder/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('Saved:',folder.resolve(),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',required=True)
    run(parser.parse_args().outdir)
