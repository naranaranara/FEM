import atexit,os,pickle,subprocess,sys
from pathlib import Path
import numpy as np

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
        """Expand FV/EE incidence to unique cross-body face pairs (conservative)."""
        data=self.candidate_topology(positions,float(search_distance)/2)
        pairs=set()
        def add(a,b):
            if self.body_ids[self.faces[a,0]]!=self.body_ids[self.faces[b,0]]:
                pairs.add(tuple(sorted((int(a),int(b)))))
        for face,vertex in data['face_vertex_pairs']:
            for adjacent in self.vertex_faces[vertex]:add(face,adjacent)
        for first,second in data['edge_edge_pairs']:
            for a in self.edge_faces[first]:
                for b in self.edge_faces[second]:add(a,b)
        return np.asarray(sorted(pairs),dtype=int).reshape(-1,2),data
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
