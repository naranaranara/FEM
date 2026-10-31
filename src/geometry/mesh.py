from itertools import permutations
import numpy as np

def make_block(width=2., depth=1., height=.5, nx=3, ny=3, nz=3):
    sizes = (width, depth, height)
    counts = (nx, ny, nz)
    if any(not np.isfinite(s) or s <= 0 for s in sizes):
        raise ValueError('dimensions must be finite and positive')
    if any(not isinstance(n, (int, np.integer)) or n < 2 for n in counts):
        raise ValueError('nx, ny, nz must be integers >= 2')
    nodes = np.array([[x, y, z] for z in np.linspace(0., height, nz)
                          for y in np.linspace(0., depth, ny)
                          for x in np.linspace(0., width, nx)])
    def index(i, j, k):
        return i + nx*(j + ny*k)
    tets = []
    for k in range(nz-1):
        for j in range(ny-1):
            for i in range(nx-1):
                for order in permutations(range(3)):
                    corner = np.array([i, j, k])
                    tet = [index(*corner)]
                    for axis in order:
                        corner = corner.copy()
                        corner[axis] += 1
                        tet.append(index(*corner))
                    xyz = nodes[tet]
                    if np.linalg.det(xyz[1:] - xyz[0]) < 0:
                        tet[1], tet[2] = tet[2], tet[1]
                    tets.append(tet)
    tets = np.asarray(tets, dtype=int)
    faces = {}
    for tet in tets:
        for opposite in range(4):
            face = [int(tet[a]) for a in range(4) if a != opposite]
            a, b, c = nodes[face]
            if np.dot(np.cross(b-a, c-a), nodes[tet[opposite]]-a) > 0:
                face[1], face[2] = face[2], face[1]
            key = tuple(sorted(face))
            faces.setdefault(key, []).append(face)
    boundary = np.array([f[0] for f in faces.values() if len(f) == 1], dtype=int)
    top = boundary[np.all(np.isclose(nodes[boundary, 2], height), axis=1)]
    bottom = np.flatnonzero(np.isclose(nodes[:, 2], 0.))
    return dict(nodes=nodes, tetrahedra=tets, boundary_faces=boundary,
                top_faces=top, bottom_nodes=bottom)

def make_spherical_cap(radius=10., half_width=6., height=10., nxy=17, nz=9,
                       clearance=.002, grading=5., depth_grading=3.):
    values=(radius, half_width, height, clearance, grading, depth_grading)
    if not all(np.isfinite(x) for x in values):
        raise ValueError('cap parameters must be finite')
    if half_width<=0 or radius<=np.sqrt(2.)*half_width:
        raise ValueError('require radius > sqrt(2)*half_width > 0')
    if clearance<=0 or not 0<=grading<=10 or not 1<=depth_grading<=6:
        raise ValueError('require clearance>0, grading in [0,10], depth_grading in [1,6]')
    if not isinstance(nxy,(int,np.integer)) or nxy<5 or nxy%2!=1:
        raise ValueError('nxy must be an odd integer >= 5')
    edge_profile=radius-np.sqrt(radius*radius-2*half_width*half_width)
    if height<=edge_profile:
        raise ValueError('flat top must be above the curved lower surface')
    mesh=make_block(width=2.,depth=2.,height=1.,nx=nxy,ny=nxy,nz=nz)
    local=mesh['nodes'];bottom=mesh['bottom_nodes']
    mesh['contact_faces']=mesh['boundary_faces'][
        np.all(np.isin(mesh['boundary_faces'],bottom),axis=1)]
    def graded(t):
        return half_width*(np.sinh(grading*t)/np.sinh(grading) if grading else t)
    x,y=graded(local[:,0]-1.),graded(local[:,1]-1.)
    profile=radius-np.sqrt(radius*radius-x*x-y*y)
    eta=local[:,2]**depth_grading
    mesh['nodes']=np.column_stack((x,y,clearance+profile+(height-profile)*eta))
    return mesh



# ##r## Complete finite cylinder: circular disk meshes extruded along local y.
def make_cylinder(radius=3., half_length=3., n_angle=32, n_radial=1, n_axis=7,
                  facing='up', angle_degrees=0., clearance=0.,
                  query_half_width=.4, patch_half_width=1.2):
    """Outer-ring nodes lie on the circle; triangular faces approximate its side.

    The end disks are the remote grips. The facing central side is the query
    patch; a wider side patch supplies RIMLS positions AND normal neighbors.
    Every part of the cylinder remains in the FEM and collision mesh.
    """
    values=(radius,half_length,angle_degrees,clearance,query_half_width,patch_half_width)
    if not all(np.isfinite(v) for v in values) or min(radius,half_length,query_half_width)<=0 or clearance<0:
        raise ValueError('finite positive cylinder dimensions required')
    if not query_half_width<patch_half_width<min(radius,half_length):
        raise ValueError('require 0 < query_half_width < patch_half_width < radius, half_length')
    if facing not in ('up','down'):raise ValueError('facing must be up or down')
    if not isinstance(n_angle,(int,np.integer)) or n_angle<16 or n_angle%8:
        raise ValueError('n_angle must be divisible by 8 and >= 16')
    if not isinstance(n_radial,(int,np.integer)) or n_radial<1:
        raise ValueError('n_radial must be an integer >= 1')
    if not isinstance(n_axis,(int,np.integer)) or n_axis<7 or n_axis%2!=1:
        raise ValueError('n_axis must be odd and >= 7')
    # Cosine spacing resolves both contact and opposite poles of the disk.
    m=n_angle//4;t=np.arange(m)/m
    theta=np.concatenate([q*np.pi/2+(np.pi/4)*(1-np.cos(np.pi*t)) for q in range(4)])
    disk=[(0.,0.)]
    for ring in range(1,n_radial+1):
        r=radius*ring/n_radial
        disk.extend(zip(r*np.sin(theta),r*np.cos(theta)))
    disk=np.asarray(disk);width=len(disk)
    triangles=[]
    for j in range(n_angle):triangles.append((0,1+j,1+(j+1)%n_angle))
    for ring in range(1,n_radial):
        inner=1+(ring-1)*n_angle;outer=inner+n_angle
        for j in range(n_angle):
            k=(j+1)%n_angle
            triangles.extend(((inner+j,outer+j,outer+k),(inner+j,outer+k,inner+k)))
    axis=half_length*np.sinh(3*np.linspace(-1.,1.,n_axis))/np.sinh(3.)
    local=np.array([(x,y,z) for y in axis for x,z in disk])
    local[:,2]=(local[:,2]-radius if facing=='up' else clearance+radius-local[:,2])
    a=np.deg2rad(angle_degrees)
    nodes=local.copy()
    nodes[:,0]=np.cos(a)*local[:,0]-np.sin(a)*local[:,1]
    nodes[:,1]=np.sin(a)*local[:,0]+np.cos(a)*local[:,1]
    tets=[]
    # Sorted face vertices give the same diagonal on every shared prism side.
    for layer in range(n_axis-1):
        for face in triangles:
            i,j,k=sorted(face);i+=layer*width;j+=layer*width;k+=layer*width
            I,J,K=i+width,j+width,k+width
            tets.extend(((i,j,k,K),(i,j,J,K),(i,I,J,K)))
    tets=np.asarray(tets,dtype=int)
    xyz=nodes[tets];det=np.linalg.det(xyz[:,1:]-xyz[:,:1])
    bad=np.flatnonzero(det<0)
    tets[bad,1],tets[bad,2]=tets[bad,2].copy(),tets[bad,1].copy()
    if np.any(det==0):raise ValueError('degenerate cylinder tetrahedron')
    faces={}
    for tet in tets:
        for opposite in range(4):
            face=[int(tet[c]) for c in range(4) if c!=opposite]
            A,B,C=nodes[face]
            if np.dot(np.cross(B-A,C-A),nodes[tet[opposite]]-A)>0:face[1],face[2]=face[2],face[1]
            faces.setdefault(tuple(sorted(face)),[]).append(face)
    boundary=np.array([v[0] for v in faces.values() if len(v)==1],dtype=int)
    end_mask=np.all(np.isclose(local[boundary,1],-half_length),axis=1)|np.all(np.isclose(local[boundary,1],half_length),axis=1)
    side=boundary[~end_mask]
    normal=np.cross(nodes[side[:,1]]-nodes[side[:,0]],nodes[side[:,2]]-nodes[side[:,0]])
    facing_side=normal[:,2]*(1 if facing=='up' else -1)>0
    def patch(extent):
        return side[facing_side & np.all(np.abs(local[side,0])<=extent+1e-12,axis=1)
                    & np.all(np.abs(local[side,1])<=extent+1e-12,axis=1)]
    contact,query=patch(patch_half_width),patch(query_half_width)
    if len(contact)==0 or len(query)==0:
        raise ValueError('mesh too coarse for the requested facing patch; increase n_angle or n_axis')
    ends=np.r_[np.arange(width),np.arange((n_axis-1)*width,n_axis*width)]
    return dict(nodes=nodes,tetrahedra=tets,boundary_faces=boundary,
                contact_faces=contact,query_faces=query,support_nodes=ends,
                bottom_nodes=np.arange(width),top_faces=boundary[np.all(np.isclose(local[boundary,1],half_length),axis=1)],
                counts=(n_angle,n_radial,n_axis),radius=radius,half_length=half_length,
                facing=facing,angle_degrees=angle_degrees,local_nodes=local,
                side_faces=side,end_faces=boundary[end_mask],axis_nodes=axis)
