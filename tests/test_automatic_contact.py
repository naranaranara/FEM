"""Automatic surface discovery, fixed-patch derivatives and deduplication."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.simulation.two_body import build_problem
from src.geometry.contact_patches import AutomaticRIMLSContact,point_triangle_distances
from src.collision.ipc_toolkit import IPCMeshCollision

class AutomaticContactChecks(unittest.TestCase):
    def setup_model(self):
        p=build_problem(n=2);vertices=np.vstack(p['reference']);n=len(p['reference'][0])
        faces=np.vstack((p['meshes'][0]['boundary_faces'],p['meshes'][1]['boundary_faces']+n))
        labels=np.r_[np.zeros(n,dtype=int),np.ones(n,dtype=int)]
        collider=IPCMeshCollision(vertices,faces,labels);self.addCleanup(collider.close)
        model=AutomaticRIMLSContact(vertices,faces,labels,collider)
        return p,vertices,faces,collider,model
    def test_point_triangle_distance_interior_edge_vertex(self):
        tri=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
        self.assertAlmostEqual(point_triangle_distances(np.array([.2,.2,.3]),tri)[0],.3)
        self.assertAlmostEqual(point_triangle_distances(np.array([2.,0.,0.]),tri)[0],1.)
        self.assertAlmostEqual(point_triangle_distances(np.array([.5,-.2,0.]),tri)[0],.2)
    def test_automatic_faces_and_no_duplicate_weights(self):
        p,vertices,faces,c,model=self.setup_model();model.refresh(vertices)
        self.assertEqual(len(model.groups),12)
        self.assertEqual(len(model.weights),12);self.assertEqual(len(set(model.signature)),12)
        self.assertAlmostEqual(model.weights.sum(),1.,places=13)
        query_faces=np.unique(model.query_ids//3)
        z=vertices[faces[query_faces],2]
        self.assertTrue(np.all(np.isclose(z,.4)|np.isclose(z,.52)))
        pairs,data=c.face_pairs(vertices,.3);original=c.face_pairs
        c.face_pairs=lambda *args:(np.repeat(pairs,3,axis=0),data)
        signature=model.signature;weights=model.weights.copy();model.refresh(vertices)
        self.assertEqual(signature,model.signature);np.testing.assert_array_equal(weights,model.weights)
        c.face_pairs=original
    def test_patch_updates_after_lateral_separation_and_return(self):
        p,vertices,faces,c,model=self.setup_model();model.refresh(vertices);initial=model.signature
        moved=vertices.copy();moved[len(p['reference'][0]):,0]+=2.
        model.refresh(moved);self.assertEqual(len(model.weights),0)
        model.refresh(vertices);self.assertEqual(model.signature,initial)
        self.assertGreaterEqual(model.updates,3)
    def test_auto_terms_match_manual_and_finite_difference(self):
        p,vertices,faces,c,model=self.setup_model();manual=build_problem(n=2)
        p['attach_auto_contact'](model);x=p['initial'].copy();p['prepare'](x)
        for method in ('gn','filtered_gn','full'):
            E,g,H,_=p['objective'](x,method);E0,g0,H0,_=manual['objective'](x,method)
            self.assertAlmostEqual(E,E0,places=12)
            np.testing.assert_allclose(g,g0,atol=1e-12)
            np.testing.assert_allclose(H.toarray(),H0.toarray(),atol=1e-11)
        # Freeze selection during this derivative check, exactly as in line search.
        v=np.random.default_rng(4).normal(size=len(x));v/=np.linalg.norm(v);h=1e-5
        E,g,H,_=p['objective'](x,'full')
        ep,gp,_,_=p['objective'](x+h*v,'gn');em,gm,_,_=p['objective'](x-h*v,'gn')
        self.assertAlmostEqual((ep-em)/(2*h),float(g@v),places=7)
        np.testing.assert_allclose((gp-gm)/(2*h),H@v,rtol=3e-6,atol=3e-7)
    def test_solver_uses_independent_patch_state_for_each_method(self):
        import contextlib,io
        from src.simulation.compression import run
        with contextlib.redirect_stdout(io.StringIO()):
            _,solutions,rows=run(n=2,steps=1,methods=('gn','full'),contact='auto')
        self.assertTrue(all(r['status']=='converged' for r in solutions.values()))
        np.testing.assert_allclose(solutions['gn']['state'],solutions['full']['state'],atol=2e-8)
        self.assertTrue(all(r['contact_samples']==12 for r in rows))
        self.assertTrue(all(r['max_patch_energy_jump']<1e-12 for r in rows))
        self.assertTrue(all(t.get('patch_updates',0)==1 for r in solutions.values() for t in r['trace']))
    def test_candidate_indices_restore_noncontiguous_full_vertex_ids(self):
        vertices=np.zeros((12,3));ids=np.array([1,3,5,7,9,11])
        vertices[ids]=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,.05],[1.,0.,.05],[0.,1.,.05]])
        faces=np.array([ids[:3],ids[3:]]);labels=np.zeros(12,dtype=int);labels[ids[3:]]=1
        with IPCMeshCollision(vertices,faces,labels) as c:
            data=c.candidate_topology(vertices,.05)
            self.assertGreater(len(data['face_vertex_pairs']),0)
            self.assertTrue(np.all(np.isin(data['face_vertex_pairs'][:,1],ids)))
            self.assertTrue(np.all(np.isin(data['edges'],ids)))
            pairs,_=c.face_pairs(vertices,.1);np.testing.assert_array_equal(pairs,[[0,1]])
if __name__=='__main__':unittest.main()
