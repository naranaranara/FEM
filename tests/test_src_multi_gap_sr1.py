"""##r## Local multi-gap SR1, coupled FEM and automatic-contact regression checks."""
import contextlib,io,unittest
from unittest.mock import patch
import numpy as np
import torch
from scipy.sparse import issparse
from threadpoolctl import threadpool_limits
from src.optimization.sr1 import MultiGapSR1
from src.contact.barrier import barrier_first_second
from src.contact.terms import multiple_contact_terms,multiple_contact_terms_sparse
from src.simulation.two_body import build_problem,solve


class MultiGapSR1Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.threads=threadpool_limits(1)
    @classmethod
    def tearDownClass(cls):cls.threads.restore_original_limits()

    def test_weighted_signed_curvature_recovers_quadratic_gap_hessians(self):
        matrices=[np.diag([2.,-1.,.5]),np.array([[3.,.4,.2],[.4,2.,.3],[.2,.3,1.]])]
        weights=np.array([.3,.7]);model=MultiGapSR1(memory=3);keys=('a','b');support=[np.arange(3)]*2
        x=np.zeros(3)
        def data():
            return dict(gaps=np.array([.1+.5*x@M@x for M in matrices]),gap_weights=weights,
                        gap_indices=np.arange(2),gap_jacobian=np.array([M@x for M in matrices]),gap_dofs=np.arange(3))
        model.correction(x,data(),keys,support,.2)
        for j in range(3):
            x=x.copy();x[j]+=.001
            C,info=model.correction(x,data(),keys,support,.2)
        expected=sum(w*barrier_first_second(s,.2)[0]*M for w,s,M in zip(weights,data()['gaps'],matrices))
        np.testing.assert_allclose(C.toarray(),expected,rtol=2e-11,atol=2e-12)
        self.assertTrue(issparse(C));self.assertGreater(model.totals['sr1_updates'],0)
        self.assertTrue(any(sign<0 for m in model.models.values() for _,sign in m['factors']))
        before=model.totals.copy();again,_=model.correction(x,data(),keys,support,.2)
        self.assertEqual(before,model.totals);np.testing.assert_array_equal(again.toarray(),C.toarray())

    def test_local_mapping_and_bounded_vector_history(self):
        model=MultiGapSR1(memory=2);x=np.zeros(11);support=np.array([1,4,8]);columns=np.array([8,1,4])
        M=np.diag([2.,3.,4.]);keys=(('query',5),)
        for k in range(9):
            if k:x[support[(k-1)%3]]+=.001
            local_gradient=M@x[support]
            J=local_gradient[[2,0,1]][None,:]
            data=dict(gaps=np.array([.1]),gap_weights=np.ones(1),gap_indices=np.array([0]),
                      gap_jacobian=J,gap_dofs=columns)
            C,info=model.correction(x,data,keys,[support],.2)
        self.assertLessEqual(info['sr1_factor_count'],2)
        m=model.models[keys[0]];self.assertLessEqual(len(m['pairs']),2)
        self.assertTrue(all(d.shape==(3,) and y.shape==(3,) for d,y in m['pairs']))
        B=sum(sign*np.outer(v,v) for v,sign in m['factors'])
        d,y=m['pairs'][-1];np.testing.assert_allclose(B@d,y,atol=2e-15)
        outside=np.setdiff1d(np.arange(11),support)
        np.testing.assert_array_equal(C.toarray()[outside],0.)

    def test_release_target_support_and_weight_changes_reset_history(self):
        model=MultiGapSR1();x=np.zeros(3);ids=np.arange(3)
        def data(gap=.1,weight=1.):
            active=np.array([0]) if gap<.2 else np.empty(0,dtype=int)
            return dict(gaps=np.array([gap]),gap_weights=np.array([weight]),gap_indices=active,
                        gap_jacobian=(x[None,:] if len(active) else np.empty((0,3))),gap_dofs=ids)
        model.correction(x,data(),('old',),[ids],.2);x+=.001
        model.correction(x,data(),('old',),[ids],.2)
        C,info=model.correction(x,data(),('new',),[ids],.2)
        self.assertEqual(C.nnz,0);self.assertEqual(info['sr1_resets'],1)
        _,info=model.correction(x,data(weight=2.),('new',),[ids],.2)
        self.assertEqual(info['sr1_resets'],1)
        _,info=model.correction(x,data(weight=2.),('new',),[ids[:2]],.2)
        self.assertEqual(info['sr1_resets'],1)
        _,info=model.correction(x,data(gap=.3),('new',),[ids],.2)
        self.assertEqual(info['sr1_resets'],1);self.assertFalse(model.models)
        C,info=model.correction(x,data(),('new',),[ids],.2)
        self.assertEqual(C.nnz,0);self.assertEqual(info['sr1_factor_count'],0)
        with self.assertRaises(ValueError):MultiGapSR1(memory=0)
        with self.assertRaises(ValueError):model.correction(x,data(),('a','a'),[ids,ids],.2)

    def test_affine_plane_gaps_learn_no_curvature(self):
        model=MultiGapSR1();J=np.array([[0.,0.,1.],[0.,0.,-1.]])
        for k in range(4):
            x=np.array([.001*k,-.001*k,.002*k])
            data=dict(gaps=.1+J@x,gap_weights=np.array([.3,.7]),gap_indices=np.arange(2),
                      gap_jacobian=J,gap_dofs=np.arange(3))
            C,info=model.correction(x,data,('up','down'),[np.arange(3)]*2,.2)
            self.assertEqual(C.nnz,0);self.assertEqual(info['sr1_factor_count'],0)
        self.assertEqual(model.totals['sr1_updates'],0)

    def test_active_jacobian_data_are_reused_and_sparse_columns_are_global(self):
        ids=np.array([1,4,6]);A=torch.tensor([[.2,.3,.1],[-.1,.1,.4]],dtype=torch.float64)
        def gaps(x):return torch.tensor([.1,.3],dtype=torch.float64)+A@x[ids]
        x=np.zeros(8)
        plain=multiple_contact_terms_sparse(x,gaps,[.3,.7],ids)
        extra=multiple_contact_terms_sparse(x,gaps,[.3,.7],ids,return_gap_data=True)
        np.testing.assert_array_equal(extra['gap_indices'],[0])
        np.testing.assert_array_equal(extra['gap_dofs'],ids)
        np.testing.assert_array_equal(extra['gap_jacobian'],A[:1].numpy())
        np.testing.assert_array_equal(extra['gradient'],plain['gradient'])
        np.testing.assert_array_equal(extra['hessian'].toarray(),plain['hessian'].toarray())
        with self.assertRaises(ValueError):multiple_contact_terms(x,gaps,[.3,.7],method='full',return_gap_data=True)

    def test_manual_layout_contains_all_gap_dependencies(self):
        p=build_problem(n=3);x=p['initial']+np.random.default_rng(4).normal(size=len(p['initial']))*1e-4
        keys,supports=p['contact_layout']()
        J=torch.autograd.functional.jacobian(p['gaps_ad'],torch.tensor(x,dtype=torch.float64)).numpy()
        self.assertEqual(len(keys),len(J))
        for row,support in zip(J,supports):
            outside=np.setdiff1d(np.arange(len(x)),support)
            np.testing.assert_allclose(row[outside],0.,atol=1e-14)

    def test_two_body_sr1_matches_full_without_exact_gap_hessians(self):
        states=[]
        for sparse in (False,True):
            p=build_problem(n=2,closure=.04,sparse=sparse)
            initial=p['initial']+np.random.default_rng(6).normal(size=len(p['initial']))*1e-4
            full=solve(p,'full',initial=initial)
            with patch('src.contact.terms._jacobian_rows',side_effect=AssertionError('exact Hessian path')):
                sr1=solve(p,'sr1_gap',initial=initial,sr1_memory=4)
            self.assertEqual(full['status'],'converged');self.assertEqual(sr1['status'],'converged')
            np.testing.assert_allclose(sr1['state'],full['state'],atol=3e-8,rtol=0)
            self.assertAlmostEqual(sr1['energy'],full['energy'],places=11)
            self.assertGreater(sr1['sr1_updates'],0)
            self.assertGreater(max(t['sr1_correction_norm'] for t in sr1['trace']),1e-8)
            data=p['diagnostics'](sr1['state'])
            np.testing.assert_allclose(data['force_lower']+data['force_upper'],0.,atol=2e-12)
            np.testing.assert_allclose(data['full_gradient'][p['free']],0.,atol=1e-8)
            self.assertTrue(p['feasible'](sr1['state']));self.assertTrue(np.all(data['gaps']>0))
            states.append(sr1['state'])
        np.testing.assert_allclose(*states,atol=2e-9,rtol=0)

    def test_auto_contact_and_ccd_continuation_use_sr1(self):
        from src.simulation.compression import run
        with contextlib.redirect_stdout(io.StringIO()):
            p,solutions,rows=run(n=2,steps=2,methods=('full','sr1_gap'),collision='ipctk',contact='auto')
        self.assertEqual(len(rows),6)
        for method,r in solutions.items():
            self.assertEqual(r['status'],'converged')
            self.assertTrue(any('ccd_alpha' in t for t in r['trace']))
        np.testing.assert_allclose(solutions['full']['state'],solutions['sr1_gap']['state'],atol=3e-8,rtol=0)
        self.assertTrue(any(r['sr1_updates']>0 for r in rows if r['method']=='sr1_gap'))

    def test_automatic_contact_release_and_recontact_clear_sr1_history(self):
        from src.collision.ipc_toolkit import IPCMeshCollision
        from src.geometry.contact_patches import AutomaticRIMLSContact
        p=build_problem(n=2);vertices=np.vstack(p['reference']);offset=len(p['reference'][0])
        faces=np.vstack((p['meshes'][0]['boundary_faces'],p['meshes'][1]['boundary_faces']+offset))
        labels=np.r_[np.zeros(offset,dtype=int),np.ones(offset,dtype=int)]
        with IPCMeshCollision(vertices,faces,labels) as collider:
            surface=AutomaticRIMLSContact(vertices,faces,labels,collider)
            curvature=MultiGapSR1()
            def evaluate(nodes):
                surface.refresh(nodes)
                contact=multiple_contact_terms_sparse(nodes.ravel(),surface.gaps,surface.weights,
                    surface.contact_dofs,gaps_subset_autodiff=surface.gaps,return_gap_data=True)
                supports=[]
                for query,gid in surface.signature:
                    used=np.unique(np.r_[faces[surface.groups[gid]].ravel(),faces[query//3]])
                    supports.append((3*used[:,None]+np.arange(3)).ravel())
                return curvature.correction(nodes.ravel(),contact,surface.signature,supports,.2)
            evaluate(vertices)
            perturbed=vertices.copy();perturbed[offset]+=[.001,-.0003,-.001]
            _,info=evaluate(perturbed)
            self.assertGreater(info['sr1_factor_count'],0)
            far=perturbed.copy();far[offset:,0]+=3.
            C,info=evaluate(far)
            self.assertEqual(C.nnz,0);self.assertFalse(curvature.models)
            self.assertGreater(info['sr1_resets'],0)
            C,info=evaluate(perturbed)
            self.assertTrue(curvature.models);self.assertEqual(C.nnz,0)
            self.assertEqual(info['sr1_factor_count'],0)


if __name__=='__main__':unittest.main()
