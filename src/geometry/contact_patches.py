"""물체 표면의 접촉 패치를 선택하고 미분 가능한 간격을 계산해요."""
import numpy as np
import torch
from .surface import surface_quadrature,make_surface_sampler
from .rimls import rimls_values_torch

def normalize_query_indices(indices, count, device):
    ids = torch.as_tensor(indices, device=device)
    if ids.ndim != 1:
        raise ValueError('query indices must be a one-dimensional integer array')
    if ids.numel() == 0:
        return ids.to(dtype=torch.long)
    if ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError('query indices must be integers')
    if torch.any((ids < 0) | (ids >= count)).item():
        raise IndexError('query index outside the current gap vector')
    return ids.to(dtype=torch.long)

def face_normals(vertices,faces):
    xyz=vertices[faces];n=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    lengths=np.linalg.norm(n,axis=1)
    if np.any(lengths<1e-14):raise ValueError('collapsed collision triangle')
    return n/lengths[:,None]

def smooth_face_components(reference,faces,body_ids,angle_degrees=30.):
    normals=face_normals(reference,faces);edges={};adj=[set() for _ in faces]
    for i,face in enumerate(faces):
        for a,b in zip(face,np.roll(face,-1)):edges.setdefault(tuple(sorted((a,b))),[]).append(i)
    threshold=np.cos(np.deg2rad(angle_degrees))
    for incident in edges.values():
        for a in incident:
            for b in incident:
                if body_ids[faces[a,0]]==body_ids[faces[b,0]] and normals[a]@normals[b]>=threshold:
                    adj[a].add(b)
    unseen=set(range(len(faces)));groups=[]
    while unseen:
        seed=min(unseen);unseen.remove(seed);stack=[seed];group=[]
        while stack:
            a=stack.pop();group.append(a)
            for b in sorted(adj[a]&unseen):
                if normals[seed]@normals[b]>=threshold:
                    unseen.remove(b);stack.append(b)
        groups.append(np.asarray(sorted(group),dtype=int))
    return groups

def points_triangle_distances(points,triangles):
    points=np.asarray(points,dtype=float)
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    u=b-a;v=c-a;w=points[:,None,:]-a[None,:,:]
    uu=np.sum(u*u,axis=1);vv=np.sum(v*v,axis=1);uv=np.sum(u*v,axis=1)
    wu=np.sum(w*u,axis=2);wv=np.sum(w*v,axis=2);det=uu*vv-uv*uv
    if np.any(det<=1e-28):raise ValueError('degenerate patch triangle')
    s=(wu*vv-wv*uv)/det;t=(wv*uu-wu*uv)/det
    projection=a+s[:,:,None]*u+t[:,:,None]*v
    distances=np.where((s>=0)&(t>=0)&(s+t<=1),np.linalg.norm(points[:,None,:]-projection,axis=2),np.inf)
    for start,end in ((a,b),(b,c),(c,a)):
        edge=end-start
        alpha=np.clip(np.sum((points[:,None,:]-start)*edge,axis=2)/np.sum(edge*edge,axis=1),0,1)
        distances=np.minimum(distances,np.linalg.norm(points[:,None,:]-(start+alpha[:,:,None]*edge),axis=2))
    return distances

def point_triangle_distances(point,triangles):
    return points_triangle_distances(np.asarray(point)[None,:],triangles)[0]

class AutomaticRIMLSContact:
    def __init__(self,reference,faces,body_ids,collider,R=1.5,d_hat=.2,skin=.1):
        """표면 연결과 희소 적분 정보를 준비해요."""
        if skin<=0 or d_hat<=0 or R<=0:raise ValueError('positive skin, barrier distance and support required')
        self.reference=np.asarray(reference,dtype=float);self.faces=np.asarray(faces,dtype=int)
        self.body_ids=np.asarray(body_ids,dtype=int);self.collider=collider
        self.R,self.d_hat,self.skin=R,d_hat,skin
        self._sparse_S,self.reference_weights=surface_quadrature(self.reference,self.faces,sparse=True)
        self._dense_S=None;self._dense_blocks=None;self._query_blocks=[]
        # 작은 보간 행렬은 조밀 곱셈을 사용해 질의별 인덱싱 비용을 줄여요.
        if self._sparse_S.shape[0]*self._sparse_S.shape[1]<=16384:
            self._dense_S=self._sparse_S.toarray()
        self._barycentric=torch.tensor([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]],dtype=torch.float64)
        self.groups=smooth_face_components(self.reference,self.faces,self.body_ids)
        self.group_ids=np.empty(len(self.faces),dtype=int);self.samplers=[]
        for gid,group in enumerate(self.groups):
            self.group_ids[group]=gid
            sample,_,_=make_surface_sampler(self.reference,self.faces[group],sparse=True);self.samplers.append(sample)
        self.signature=();self.weights=np.empty(0);self.contact_dofs=np.empty(0,dtype=int)
        self.query_ids=np.empty(0,dtype=int);self.updates=0;self.initialized=False
        self.info={}
    @property
    def S(self):
        """명시적으로 요청한 경우에만 기존 조밀 보간 행렬을 만들어요."""
        if self._dense_S is None:
            self._dense_S=self._sparse_S.toarray()
        return self._dense_S

    @S.setter
    def S(self,value):
        """외부에서 지정한 보간 행렬을 기존 공개 인터페이스대로 보관해요."""
        self._dense_S=value

    @property
    def blocks(self):
        """명시적으로 요청한 경우에만 기존 조밀 접촉 블록을 만들어요."""
        if self._dense_blocks is None:
            self._dense_blocks=[]
            for gid,vertices,shape in self._query_blocks:
                matrix=shape.new_zeros((len(vertices),len(self.reference)))
                matrix.scatter_(1,vertices,shape)
                self._dense_blocks.append((gid,matrix))
        return self._dense_blocks

    @blocks.setter
    def blocks(self,value):
        """외부에서 지정한 접촉 블록을 기존 공개 인터페이스대로 보관해요."""
        self._dense_blocks=value

    def refresh(self,positions,before_change=None):
        """현재 위치에서 접촉 패치를 갱신하고 바뀐 질의만 다시 구성해요."""
        positions=np.asarray(positions,dtype=float);normals=face_normals(positions,self.faces)
        centers=positions[self.faces].mean(axis=1);distance=self.d_hat+self.skin
        pairs,data=self.collider.face_pairs(positions,distance)
        if len(pairs):
            a,b=pairs[:,0],pairs[:,1];delta=centers[b]-centers[a]
            valid=(np.sum(normals[a]*normals[b],axis=1)<=-.5)
            valid&=(np.sum(normals[a]*delta,axis=1)>=-1e-10)
            valid&=(np.sum(normals[b]*(-delta),axis=1)>=-1e-10)
            facing=pairs[valid]
        else:facing=pairs
        requests={}
        for a,b in facing:
            requests.setdefault(int(self.group_ids[b]),set()).update(range(3*a,3*a+3))
            requests.setdefault(int(self.group_ids[a]),set()).update(range(3*b,3*b+3))
        interpolation=self._sparse_S if self._dense_S is None else self._dense_S
        query_positions=interpolation@positions;options={}
        for gid,queries in sorted(requests.items()):
            group=self.groups[gid];triangles=positions[self.faces[group]]
            ids=np.asarray(sorted(queries),dtype=int)
            for start in range(0,len(ids),64):
                batch=ids[start:start+64]
                distances=points_triangle_distances(query_positions[batch],triangles)
                best=np.argmin(distances,axis=1)
                for row,query in enumerate(batch):
                    d=float(distances[row,best[row]]);target_face=group[best[row]]
                    signed=float((query_positions[query]-positions[self.faces[target_face,0]])@normals[target_face])
                    if d>distance or signed<-1e-10:continue
                    target_body=int(self.body_ids[self.faces[target_face,0]])
                    candidates=options.setdefault(int(query),{});candidate=(d,gid)
                    if target_body not in candidates or candidate<candidates[target_body]:candidates[target_body]=candidate
        chosen=[(query,gid) for query,candidates in sorted(options.items())
                for target_body,(_,gid) in sorted(candidates.items())]
        signature=tuple(sorted(chosen,key=lambda x:(x[1],x[0])))
        changed=signature!=self.signature
        if changed or not self.initialized:
            if self.initialized and before_change is not None:before_change()
            self.signature=signature;self.query_ids=np.asarray([q for q,g in signature],dtype=int)
            self.weights=.5*self.reference_weights[self.query_ids]
            self._query_blocks=[]
            self._dense_blocks=[] if self._dense_S is not None else None
            vertices=set()
            for gid in sorted({g for q,g in signature}):
                ids=np.asarray([q for q,g in signature if g==gid],dtype=int)
                if self._dense_S is None:
                    query_vertices=torch.tensor(self.faces[ids//3],dtype=torch.long)
                    shape=self._barycentric[torch.tensor(ids%3,dtype=torch.long)]
                    self._query_blocks.append((gid,query_vertices,shape))
                else:
                    self._dense_blocks.append((gid,torch.tensor(self._dense_S[ids],dtype=torch.float64)))
                vertices.update(self.faces[self.groups[gid]].ravel())
                vertices.update(self.faces[ids//3].ravel())
            self.contact_dofs=(3*np.asarray(sorted(vertices),dtype=int)[:,None]+np.arange(3)).ravel()
            self.updates+=int(changed);self.initialized=True
        self.info=dict(candidate_count=int(data['total']),candidate_face_pairs=len(pairs),
            contact_samples=len(self.weights),patch_changed=changed,patch_updates=self.updates)
        return self.info.copy()
    def gaps(self, state, indices=None):
        r"""각 질의의 세 절점 보간으로 선택한 RIMLS 간격을 계산해요.

        $$\boldsymbol{x}_q=\sum_{i=1}^{3}N_{qi}\boldsymbol{x}_{v_{qi}}$$
        """
        if not self.initialized:raise RuntimeError('refresh contact patches before evaluating gaps')
        ids = None if indices is None else normalize_query_indices(indices, len(self.weights), state.device)
        if ids is not None and ids.numel() == 0:return state[:0]
        nodes=state.reshape(-1,3);values=[];positions=[];offset=0
        compact=self._dense_blocks is None
        blocks=self._query_blocks if compact else self._dense_blocks
        for block in blocks:
            gid=block[0];interpolation=block[1]
            shape=block[2] if compact else None
            if ids is not None:
                next_offset=offset+len(interpolation)
                slots=torch.nonzero((ids>=offset)&(ids<next_offset),as_tuple=False).reshape(-1)
                local_ids=ids[slots]-offset;offset=next_offset
                if slots.numel()==0:continue
                interpolation=interpolation.index_select(0,local_ids)
                if compact:shape=shape.index_select(0,local_ids)
                positions.append(slots)
            if compact:
                # 질의 좌표를 세 절점에서 보간해요: $$\boldsymbol{x}_q=\sum_iN_{qi}\boldsymbol{x}_{v_{qi}}$$
                queries=torch.sum(shape[:,:,None]*nodes[interpolation],dim=1)
            else:
                # 공개 조밀 블록이 지정되면 기존 보간을 사용해요: $$X_q=S_qX$$
                queries=interpolation@nodes
            points,normals=self.samplers[gid](nodes)
            values.append(rimls_values_torch(queries,points,normals,self.R))
        if not values:return state[:0]
        result=torch.cat(values)
        return result if ids is None else result[torch.argsort(torch.cat(positions))]
    def max_step(self,start,end):
        """패치 탐색 여유보다 작은 절점 이동을 허용해요."""
        movement=float(np.linalg.norm(np.asarray(end)-np.asarray(start),axis=1).max())
        return min(1.,self.skin/(4*movement)) if movement>0 else 1.
