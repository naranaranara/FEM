"""독립 프로세스의 IPC Toolkit으로 충돌 후보와 안전한 이동 보폭을 계산해요."""
import atexit,os,pickle,subprocess,sys
from pathlib import Path
import numpy as np

def _pack_incidence(incidence):
    """면 인접 집합을 연속 배열과 시작 위치로 묶어요."""
    counts = np.fromiter(map(len, incidence), dtype=np.intp, count=len(incidence))
    offsets = np.r_[0, np.cumsum(counts)]
    values = np.fromiter((face for group in incidence for face in sorted(group)),
                         dtype=np.intp, count=int(offsets[-1]))
    return offsets, values


def _incidence_pairs(candidates, first_incidence, second_incidence):
    """후보 양쪽의 인접 면 조합을 한 번에 펼쳐요."""
    first, second = candidates.T
    second_offsets, second_values = second_incidence
    second_counts = second_offsets[second + 1] - second_offsets[second]
    if first_incidence is None:
        first_counts = np.ones(len(first), dtype=np.intp)
    else:
        first_offsets, first_values = first_incidence
        first_counts = first_offsets[first + 1] - first_offsets[first]
    counts = first_counts * second_counts
    starts = np.cumsum(counts) - counts
    local = np.arange(counts.sum()) - np.repeat(starts, counts)
    widths = np.repeat(second_counts, counts)
    b = second_values[np.repeat(second_offsets[second], counts) + local % widths]
    if first_incidence is None:
        a = np.repeat(first, counts)
    else:
        a = first_values[np.repeat(first_offsets[first], counts) + local // widths]
    return np.column_stack((a, b))


class IPCMeshCollision:
    def __init__(self,positions,faces,body_ids,self_contact=False,python=None):
        positions=np.asarray(positions,dtype=float);faces=np.asarray(faces,dtype=np.int32)
        labels=np.asarray(body_ids,dtype=int)
        if positions.ndim!=2 or positions.shape[1]!=3 or labels.shape!=(len(positions),):
            raise ValueError('3D positions and one body id per vertex required')
        if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or np.any(faces<0) or np.any(faces>=len(positions)):
            raise ValueError('valid triangular boundary faces required')
        if not np.all(np.isfinite(positions)):raise ValueError('finite positions required')
        self.vertex_ids=np.unique(faces);remap=np.full(len(positions),-1,dtype=int)
        remap[self.vertex_ids]=np.arange(len(self.vertex_ids));local=remap[faces].astype(np.int32)
        edges=sorted({tuple(sorted((int(a),int(b)))) for face in local for a,b in zip(face,np.roll(face,-1))})
        self.faces=faces.copy();self.body_ids=labels.copy()
        if np.any(labels[faces]!=labels[faces[:,0]][:,None]):raise ValueError('each face must belong to one body')
        self.edges=self.vertex_ids[np.asarray(edges,dtype=int)]
        self.vertex_faces=[set() for _ in range(len(positions))]
        for index,face in enumerate(self.faces):
            for vertex in face:self.vertex_faces[vertex].add(index)
        self.edge_faces=[self.vertex_faces[a]&self.vertex_faces[b] for a,b in self.edges]
        self._vertex_incidence = _pack_incidence(self.vertex_faces)
        self._edge_incidence = _pack_incidence(self.edge_faces)
        self.vertex_count=len(positions)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        self.process=subprocess.Popen([python or sys.executable,'-u',str(Path(__file__).with_name('_worker.py'))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        atexit.register(self.close)
        try:self.info=self._request('init',positions[self.vertex_ids],np.asarray(edges,dtype=np.int32),local,labels[self.vertex_ids],bool(self_contact))
        except Exception:
            self.close();raise
    def _request(self,operation,*args):
        try:
            pickle.dump((operation,args),self.process.stdin,protocol=4);self.process.stdin.flush()
            ok,result=pickle.load(self.process.stdout)
        except (EOFError,BrokenPipeError,OSError) as exc:
            raise RuntimeError('IPC worker exited; check ipctk installation in its Python environment') from exc
        if not ok:raise RuntimeError('IPC Toolkit: '+result)
        return result
    def _surface(self,positions):
        positions=np.asarray(positions,dtype=float)
        if positions.shape!=(self.vertex_count,3) or not np.all(np.isfinite(positions)):
            raise ValueError('positions must match the original full mesh')
        return positions[self.vertex_ids]
    def max_step(self,start,end):
        return self._request('max_step',self._surface(start),self._surface(end))
    def candidates(self,positions,inflation_radius=0.):
        if not np.isfinite(inflation_radius) or inflation_radius<0:raise ValueError('nonnegative inflation radius required')
        return self._request('candidates',self._surface(positions),float(inflation_radius))
    def candidate_topology(self,positions,inflation_radius=0.):
        """Return original full-mesh vertex ids, face-row ids and edge-row ids."""
        if not np.isfinite(inflation_radius) or inflation_radius<0:raise ValueError('nonnegative inflation radius required')
        result=self._request('candidate_indices',self._surface(positions),float(inflation_radius))
        fv=result['face_vertex_pairs'].copy()
        if len(fv):fv[:,1]=self.vertex_ids[fv[:,1]]
        result['face_vertex_pairs']=fv
        result['edges']=self.edges.copy()
        return result
    def face_pairs(self,positions,search_distance):
        """충돌 후보를 물체 간의 중복 없는 면 쌍으로 보수적으로 확장해요."""
        data=self.candidate_topology(positions,float(search_distance)/2)
        fv = data['face_vertex_pairs']
        ee = data['edge_edge_pairs']
        labels = self.body_ids[self.faces[:, 0]]
        if len(fv) + len(ee) < 32:
            pairs=set()
            def add(a,b):
                """서로 다른 물체에 속하는 면 쌍을 추가해요."""
                if labels[a] != labels[b]:
                    pairs.add(tuple(sorted((int(a),int(b)))))
            for face,vertex in fv:
                for adjacent in self.vertex_faces[vertex]:add(face,adjacent)
            for first,second in ee:
                for a in self.edge_faces[first]:
                    for b in self.edge_faces[second]:add(a,b)
            return np.asarray(sorted(pairs),dtype=int).reshape(-1,2),data
        chunks = []
        for candidates, first_incidence, second_incidence in (
                (fv, None, self._vertex_incidence),
                (ee, self._edge_incidence, self._edge_incidence)):
            for start in range(0, len(candidates), 4096):
                pairs = _incidence_pairs(candidates[start:start + 4096],
                                         first_incidence, second_incidence)
                pairs = pairs[labels[pairs[:, 0]] != labels[pairs[:, 1]]]
                if len(pairs):
                    first = np.minimum(pairs[:, 0], pairs[:, 1])
                    second = np.maximum(pairs[:, 0], pairs[:, 1])
                    chunks.append(np.unique(first * len(self.faces) + second))
        if not chunks:
            return np.empty((0, 2), dtype=int), data
        keys = np.unique(np.concatenate(chunks))
        return np.column_stack((keys // len(self.faces), keys % len(self.faces))), data
    def close(self):
        process=getattr(self,'process',None)
        if process is None:return
        if process.poll() is None:
            try:
                pickle.dump(('close',()),process.stdin,protocol=4);process.stdin.flush();process.wait(timeout=5)
            except (OSError,subprocess.TimeoutExpired):
                process.kill();process.wait()
        for stream in (process.stdin,process.stdout,process.stderr):
            if stream:stream.close()
        self.process=None;atexit.unregister(self.close)
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
