import numpy as np

def bottom_faces(mesh):
    nodes, faces = mesh ['nodes'], mesh['boundary_faces']
    return faces[np.all(np.isclose(nodes[faces, 2], nodes[:, 2].min()), axis=1)]

def surface_quadrature(nodes, faces):
    barycentric = np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    S, weights = [], []
    for face in faces:
        a, b, c = nodes[face]
        area= .5*np.linalg.norm(np.cross(b-a,c-a))
        if area <=0:
            raise ValueError('collapsed surface triangle')
        for shape in barycentric:
            row=np.zeros(len(nodes))
            row[face] = shape
            S.append(row)
            weights.append(area/3)
    return np.asarray(S), np.asarray(weights)

def make_surface_sampler(reference,faces):
    import torch
    S,weights=surface_quadrature(reference,faces)
    used=np.unique(faces)
    remap=np.full(len(reference),-1,dtype=int);remap[used]=np.arange(len(used))
    face_t=torch.tensor(remap[faces],dtype=torch.long)
    used_t=torch.tensor(used,dtype=torch.long)
    St=torch.tensor(S[:,used],dtype=torch.float64)
    def sample(nodes):
        surface=nodes[used_t];xyz=surface[face_t]
        area_normals=torch.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0],dim=1)
        normals=torch.zeros_like(surface)
        for corner in range(3):
            normals=normals.index_add(0,face_t[:,corner],area_normals)
        normals=normals/torch.linalg.norm(normals,dim=1)[:,None]
        positions=St@surface
        interpolated=St@normals
        interpolated=interpolated/torch.linalg.norm(interpolated,dim=1)[:,None]
        return positions,interpolated
    return sample,S,weights


##r## Nodal boundary integration, with every plane-facing vertex protected.
def surface_lumped_weights(nodes,faces):
    """Reference surface areas lumped to vertices; each face contributes A/3."""
    nodes=np.asarray(nodes,dtype=float);faces=np.asarray(faces,dtype=int)
    xyz=nodes[faces]
    areas=.5*np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)
    if np.any(~np.isfinite(areas)) or np.any(areas<=0):
        raise ValueError('collapsed or nonfinite surface triangle')
    weights=np.zeros(len(nodes))
    np.add.at(weights,faces.ravel(),np.repeat(areas/3.,3))
    return weights
