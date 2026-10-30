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
