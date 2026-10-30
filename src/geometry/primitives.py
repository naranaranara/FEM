"""3D surface samples/normals. FEM volume connectivity remains in mesh.py."""
import numpy as np


def sample_sphere(n_samples=512,radius=1.):
    if not isinstance(n_samples,(int,np.integer)) or n_samples<32 or not np.isfinite(radius) or radius<=0:
        raise ValueError('n_samples>=32 and radius>0 required')
    i=np.arange(n_samples);z=1.-2.*(i+.5)/n_samples
    angle=i*np.pi*(3.-np.sqrt(5.));rho=np.sqrt(1.-z*z)
    normals=np.column_stack((rho*np.cos(angle),rho*np.sin(angle),z))
    return radius*normals,normals


def sample_plane(nx=9,ny=7,xlim=(-1.,3.),ylim=(-1.,2.)):
    if nx<2 or ny<2:raise ValueError('plane counts must be >=2')
    points=np.array([[x,y,0.] for y in np.linspace(*ylim,ny) for x in np.linspace(*xlim,nx)])
    faces=[]
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i
            faces.extend(((a,a+1,a+nx+1),(a,a+nx+1,a+nx)))
    return points,np.tile([0.,0.,1.],(len(points),1)),np.asarray(faces,dtype=int)


def sample_cylinder(radius=1.,n_angle=64,n_axis=25,half_length=2.):
    angle=np.arange(n_angle)*2*np.pi/n_angle
    radial=np.column_stack((np.cos(angle),np.sin(angle),np.zeros(n_angle)))
    normals=np.tile(radial,(n_axis,1))
    points=radius*normals.copy()
    points[:,2]=np.repeat(np.linspace(-half_length,half_length,n_axis),n_angle)
    return points,normals


def sample_ellipsoid(axes=(1.4,1.,.7),n_samples=768):
    _,unit=sample_sphere(n_samples)
    axes=np.asarray(axes,dtype=float)
    if np.any(axes<=0):raise ValueError('positive semiaxes required')
    points=unit*axes
    normals=unit/axes
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    return points,normals


def sample_saddle(curvature=.7,n=33,half_width=1.6):
    x,y=np.meshgrid(np.linspace(-half_width,half_width,n),np.linspace(-half_width,half_width,n))
    points=np.column_stack((x.ravel(),y.ravel(),.5*curvature*(x*x-y*y).ravel()))
    normals=np.column_stack((-curvature*x.ravel(),curvature*y.ravel(),np.ones(x.size)))
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    return points,normals

##r## Coordinate warp only: connectivity is supplied by mesh.make_block.
def curve_patch_nodes(nodes,amplitude=.12):
    result=np.asarray(nodes,dtype=float).copy()
    low,high=result.min(axis=0),result.max(axis=0)
    center=(low+high)/2
    x,y=result[:,0]-center[0],result[:,1]-center[1]
    layer=(result[:,2]-low[2])/(high[2]-low[2])
    result[:,:2]-=center[:2]
    result[:,2]-=high[2]
    result[:,2]+=layer*amplitude*(x*x-y*y)
    return result
