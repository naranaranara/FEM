import pickle,sys
import numpy as np
import ipctk

def main():
    ipctk.set_num_threads(1);mesh=None
    while True:
        try:operation,args=pickle.load(sys.stdin.buffer)
        except EOFError:break
        if operation=='close':break
        try:
            if operation=='init':
                vertices,edges,faces,labels,self_contact=args
                mesh=ipctk.CollisionMesh(vertices,edges,faces)
                mesh.can_collide=lambda i,j:bool(self_contact or labels[i]!=labels[j])
                result=dict(version=ipctk.__version__,broad_phase='BOUNDING_VOLUME_HIERARCHY')
            elif operation=='max_step':
                result=float(ipctk.compute_collision_free_stepsize(mesh,*args,
                    broad_phase_method=ipctk.BroadPhaseMethod.BOUNDING_VOLUME_HIERARCHY))
            elif operation in ('candidates','candidate_indices'):
                positions,radius=args;c=ipctk.Candidates()
                c.build(mesh,positions,float(radius),ipctk.BroadPhaseMethod.BOUNDING_VOLUME_HIERARCHY)
                result=dict(total=len(c),face_vertex=len(c.fv_candidates),edge_edge=len(c.ee_candidates))
                if operation=='candidate_indices':
                    result['face_vertex_pairs']=np.asarray([(x.face_id,x.vertex_id) for x in c.fv_candidates],dtype=np.int64).reshape(-1,2)
                    result['edge_edge_pairs']=np.asarray([(x.edge0_id,x.edge1_id) for x in c.ee_candidates],dtype=np.int64).reshape(-1,2)
                    if len(c.ev_candidates) or len(c.vv_candidates):
                        raise ValueError('automatic patches require triangle-only FV/EE candidates')
            else:raise ValueError('unknown operation: '+operation)
            response=(True,result)
        except Exception as exc:response=(False,type(exc).__name__+': '+str(exc))
        pickle.dump(response,sys.stdout.buffer,protocol=4);sys.stdout.buffer.flush()
if __name__=='__main__':main()
