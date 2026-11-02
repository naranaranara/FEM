"""삼각형 표면의 적분점과 면적 가중치 및 미분 가능한 법선을 계산해요."""
import numpy as np
from scipy.sparse import coo_matrix, eye, kron


def bottom_faces(mesh):
    """가장 낮은 높이에 속하는 경계 삼각형을 선택해요."""
    nodes, faces = mesh['nodes'], mesh['boundary_faces']
    return faces[np.all(np.isclose(nodes[faces, 2], nodes[:, 2].min()), axis=1)]


def surface_quadrature(nodes, faces, sparse=False):
    r"""삼각형마다 세 적분점의 보간 행렬과 면적 가중치를 계산해요.

    $$\boldsymbol{x}_q=\sum_{i=1}^{3}N_{qi}\boldsymbol{x}_i,\qquad w_q=A_f/3$$
    """
    barycentric = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
    faces = np.asarray(faces, dtype=int)
    if not len(faces):
        if sparse:
            from scipy.sparse import csr_matrix
            return csr_matrix((0, len(nodes))), np.asarray([])
        return np.asarray([]), np.asarray([])
    xyz = nodes[faces]
    # 삼각형의 넓이를 계산해요: $$A_f=\frac12\| (\boldsymbol{x}_1-\boldsymbol{x}_0)\times(\boldsymbol{x}_2-\boldsymbol{x}_0)\|$$
    areas = .5 * np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    if np.any(areas <= 0):
        raise ValueError('collapsed surface triangle')
    rows = np.arange(len(faces) * 3).reshape(-1, 3, 1)
    if sparse:
        from scipy.sparse import coo_matrix
        shape = (len(faces), 3, 3)
        S = coo_matrix((np.broadcast_to(barycentric, shape).ravel(),
                        (np.broadcast_to(rows, shape).ravel(),
                         np.broadcast_to(faces[:, None, :], shape).ravel())),
                       shape=(len(faces) * 3, len(nodes))).tocsr()
    else:
        S = np.zeros((len(faces) * 3, len(nodes)))
        S[rows, faces[:, None, :]] = barycentric
    # 적분점마다 동일한 면적 가중치를 배분해요: $$w_{fq}=A_f/3$$
    weights = np.repeat(areas / 3, 3)
    return S, weights


def make_surface_sampler(reference, faces, sparse=False):
    """삼각형의 세 절점만 참조하는 미분 가능한 표면 샘플러를 만들어요."""
    import torch
    faces = np.asarray(faces, dtype=int)
    S, weights = surface_quadrature(reference, faces, sparse=sparse)
    used = np.unique(faces)
    remap = np.full(len(reference), -1, dtype=int)
    remap[used] = np.arange(len(used))
    face_t = torch.tensor(remap[faces], dtype=torch.long)
    used_t = torch.tensor(used, dtype=torch.long)
    barycentric = torch.tensor([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]], dtype=torch.float64)

    def sample(nodes):
        r"""변형된 표면의 적분점 좌표와 단위 법선을 계산해요.

        $$\boldsymbol{n}_q=\frac{\sum_iN_{qi}\boldsymbol{n}_i}{\|\sum_iN_{qi}\boldsymbol{n}_i\|}$$
        """
        surface = nodes[used_t]
        xyz = surface[face_t]
        # 방향과 면적을 갖는 삼각형 법선을 구해요: $$\boldsymbol{a}_f=(\boldsymbol{x}_1-\boldsymbol{x}_0)\times(\boldsymbol{x}_2-\boldsymbol{x}_0)$$
        area_normals = torch.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0], dim=1)
        normals = torch.zeros_like(surface)
        for corner in range(3):
            # 이웃 면의 법선을 절점에 더해요: $$\tilde{\boldsymbol{n}}_i=\sum_{f\ni i}\boldsymbol{a}_f$$
            normals = normals.index_add(0, face_t[:, corner], area_normals)
        # 절점 법선을 단위 길이로 정규화해요: $$\boldsymbol{n}_i=\tilde{\boldsymbol{n}}_i/\|\tilde{\boldsymbol{n}}_i\|$$
        normals = normals / torch.linalg.norm(normals, dim=1)[:, None]
        # 삼각형의 절점 좌표를 보간해요: $$\boldsymbol{x}_{fq}=\sum_iN_{qi}\boldsymbol{x}_{fi}$$
        positions = (barycentric @ xyz).reshape(-1, 3)
        # 삼각형의 절점 법선을 보간해요: $$\tilde{\boldsymbol{n}}_{fq}=\sum_iN_{qi}\boldsymbol{n}_{fi}$$
        interpolated = (barycentric @ normals[face_t]).reshape(-1, 3)
        # 적분점 법선을 단위 길이로 정규화해요: $$\boldsymbol{n}_{fq}=\tilde{\boldsymbol{n}}_{fq}/\|\tilde{\boldsymbol{n}}_{fq}\|$$
        interpolated = interpolated / torch.linalg.norm(interpolated, dim=1)[:, None]
        return positions, interpolated
    return sample, S, weights


def surface_lumped_weights(nodes, faces):
    r"""각 삼각형의 면적을 세 절점에 균등하게 배분해요.

    $$w_i=\sum_{f\ni i}A_f/3$$
    """
    nodes = np.asarray(nodes, dtype=float)
    faces = np.asarray(faces, dtype=int)
    xyz = nodes[faces]
    # 삼각형의 넓이를 계산해요: $$A_f=\frac12\| (\boldsymbol{x}_1-\boldsymbol{x}_0)\times(\boldsymbol{x}_2-\boldsymbol{x}_0)\|$$
    areas = .5 * np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    if np.any(~np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError('collapsed or nonfinite surface triangle')
    weights = np.zeros(len(nodes))
    # 각 면의 면적을 절점에 모아요: $$w_i=\sum_{f\ni i}A_f/3$$
    np.add.at(weights, faces.ravel(), np.repeat(areas / 3., 3))
    return weights


# ##r## Exact sparse derivatives of the surface sampler above.
def skew(v):
    result=np.zeros(v.shape[:-1]+(3,3))
    result[...,0,1]=-v[...,2];result[...,0,2]=v[...,1]
    result[...,1,0]=v[...,2];result[...,1,2]=-v[...,0]
    result[...,2,0]=-v[...,1];result[...,2,1]=v[...,0]
    return result


def block_diagonal3(blocks):
    count=len(blocks);base=3*np.arange(count)[:,None,None]
    rows=np.broadcast_to(base+np.arange(3)[None,:,None],blocks.shape)
    columns=np.broadcast_to(base+np.arange(3)[None,None,:],blocks.shape)
    return coo_matrix((blocks.ravel(),(rows.ravel(),columns.ravel())),
                      shape=(3*count,3*count)).tocsr()


def normalized(v):
    length=np.linalg.norm(v,axis=1)
    if np.any(length<=0) or np.any(~np.isfinite(length)):
        raise ValueError('collapsed or nonfinite surface normals')
    unit=v/length[:,None]
    derivative=(np.eye(3)[None,:,:]-unit[:,:,None]*unit[:,None,:])/length[:,None,None]
    return unit,block_diagonal3(derivative)


class PointMap:
    def __init__(self,reference,faces):
        self.used=np.unique(faces)
        remap=np.full(len(reference),-1,dtype=int);remap[self.used]=np.arange(len(self.used))
        self.faces=remap[np.asarray(faces,dtype=int)]
        S,_=surface_quadrature(np.asarray(reference)[self.used],self.faces,sparse=True)
        self.position_jacobian=kron(S,eye(3,format='csr'),format='csr')
        self.barycentric=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
        self.dofs=(3*self.used[:,None]+np.arange(3)).ravel()

    def positions(self,nodes):
        return (self.barycentric@np.asarray(nodes)[self.used][self.faces]).reshape(-1,3)


class SurfaceMap(PointMap):
    def __init__(self,reference,faces):
        super().__init__(reference,faces)
        face_ids=np.repeat(np.arange(len(self.faces)),3)
        incidence=coo_matrix((np.ones(len(face_ids)),(self.faces.ravel(),face_ids)),
                             shape=(len(self.used),len(self.faces))).tocsr()
        self.incidence=kron(incidence,eye(3,format='csr'),format='csr')

    def linearize(self,nodes):
        surface=np.asarray(nodes)[self.used];xyz=surface[self.faces]
        area=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
        raw=np.zeros_like(surface)
        for corner in range(3):np.add.at(raw,self.faces[:,corner],area)
        vn,Nv=normalized(raw)
        interpolated=(self.barycentric@vn[self.faces]).reshape(-1,3)
        normals,Ns=normalized(interpolated)
        blocks=np.stack((skew(xyz[:,2]-xyz[:,1]),-skew(xyz[:,2]-xyz[:,0]),
                         skew(xyz[:,1]-xyz[:,0])),axis=1)
        rows=np.broadcast_to(3*np.arange(len(xyz))[:,None,None,None]+np.arange(3)[None,None,:,None],blocks.shape)
        columns=np.broadcast_to(3*self.faces[:,:,None,None]+np.arange(3)[None,None,None,:],blocks.shape)
        face_jacobian=coo_matrix((blocks.ravel(),(rows.ravel(),columns.ravel())),
            shape=(3*len(xyz),3*len(surface))).tocsr()
        vertex_jacobian=Nv@(self.incidence@face_jacobian)
        normal_jacobian=Ns@(self.position_jacobian@vertex_jacobian)
        points=(self.barycentric@xyz).reshape(-1,3)
        return points,normals,normal_jacobian.tocsr()
