"""##r## Exact direct surface/RIMLS chain-rule regression checks."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from src.geometry.surface import make_surface_sampler
from src.geometry.surface import SurfaceMap
from src.geometry.rimls import TwoBodyGapJacobian
from src.geometry import native_rimls
from src.simulation.two_body import build_problem


class DirectJacobianChecks(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,self.threads)
        pool=threadpool_limits(1);self.addCleanup(pool.restore_original_limits)

    def test_surface_values_and_normals_derivatives_match_autograd(self):
        reference=np.array([[0.,0.,0.],[1.,0.,.1],[0.,1.,-.1],[1.,1.,.2],[2.,0.,.3],[2.,1.,-.2]])
        faces=np.array([[0,1,2],[1,3,2],[1,4,3],[4,5,3]])
        sample,_,_=make_surface_sampler(reference,faces,sparse=True)
        x=torch.tensor(reference+np.random.default_rng(7).normal(0,.02,reference.shape),dtype=torch.float64,requires_grad=True)
        mapper=SurfaceMap(reference,faces)
        p,n,Jn=mapper.linearize(x.detach().numpy());pt,nt=sample(x)
        np.testing.assert_allclose(p,pt.detach(),atol=1e-15)
        np.testing.assert_allclose(n,nt.detach(),atol=1e-15)
        expected=torch.autograd.functional.jacobian(lambda value:sample(value)[1],x,vectorize=True)
        np.testing.assert_allclose(Jn.toarray(),expected.reshape(len(n)*3,-1),atol=2e-14,rtol=2e-13)
        expectedp=torch.autograd.functional.jacobian(lambda value:sample(value)[0],x,vectorize=True)
        np.testing.assert_allclose(mapper.position_jacobian.toarray(),expectedp.reshape(len(p)*3,-1),atol=1e-15)

    def test_coupled_gap_jacobian_duplicates_fixed_dofs_and_deformation(self):
        if not hasattr(native_rimls._rimls_native,'local_evaluate'):self.skipTest('native extension missing')
        problem=build_problem(n=3)
        meshes=problem['meshes'];reference=problem['reference']
        faces=[meshes[0]['top_faces'],meshes[1]['boundary_faces'][
            np.all(np.isclose(reference[1][meshes[1]['boundary_faces'],2],reference[1][:,2].min()),axis=1)]]
        engine=TwoBodyGapJacobian(reference,faces,faces,1.5)
        state=problem['initial']+np.random.default_rng(5).normal(0,.001,len(problem['initial']))
        full=problem['full_state'](state);nodes=problem['unpack'](state)
        mapping=np.full(len(full),-1,dtype=int);mapping[problem['free']]=np.arange(len(state))
        ids=np.array([0,3,31,2,0,47])
        direct=engine.evaluate(nodes,ids,mapping,len(state)).toarray()
        tensor=torch.tensor(state,dtype=torch.float64,requires_grad=True)
        expected=torch.autograd.functional.jacobian(lambda x:problem['gaps_ad'](x,indices=ids),tensor,vectorize=True)
        np.testing.assert_allclose(direct,expected,atol=2e-13,rtol=1e-10)
        np.testing.assert_array_equal(direct[0],direct[4])
        self.assertGreater(np.linalg.norm(direct[:,:len(state)//2]),0)
        self.assertGreater(np.linalg.norm(direct[:,len(state)//2:]),0)
        changed=state.copy();changed[problem['contact_dofs']]+=np.random.default_rng(8).normal(0,.001,len(problem['contact_dofs']))
        updated=engine.evaluate(problem['unpack'](changed),ids,mapping,len(state)).toarray()
        self.assertGreater(np.linalg.norm(direct-updated),1e-7)
        self.assertEqual(engine.evaluate(nodes,[],mapping,len(state)).shape,(0,len(state)))
        with patch('src.geometry.rimls.native_rimls._rimls_native',None):
            self.assertIsNone(engine.evaluate(nodes,ids,mapping,len(state)))


    def test_dense_sparse_contact_and_condensed_jacobian_match(self):
        from src.geometry.rimls import direct_jacobian_mode
        from src.mechanics.condensation import condense_contact_problem
        from src.contact import terms
        if not hasattr(native_rimls._rimls_native,'local_evaluate'):self.skipTest('native extension missing')
        p=build_problem(n=4)
        state=p['initial']+np.random.default_rng(9).normal(0,1e-4,len(p['initial']))
        ids=np.arange(len(p['weights']))
        direct=p['gap_jacobian'](state,ids)
        self.assertIsNotNone(direct)
        z=torch.tensor(state,dtype=torch.float64,requires_grad=True)
        expected=torch.autograd.functional.jacobian(p['gaps_ad'],z,vectorize=True)
        np.testing.assert_allclose(direct.toarray(),expected,rtol=1e-10,atol=2e-13)
        for method in ('gn','filtered_gn'):
            with direct_jacobian_mode(False):old=p['contact'](state,method)
            with patch.object(terms,'_gap_jacobian_blocks',side_effect=AssertionError('autograd Jacobian repeated')):
                new=p['contact'](state,method)
            np.testing.assert_allclose(new['gradient'],old['gradient'],rtol=1e-11,atol=1e-13)
            np.testing.assert_allclose(new['hessian'].toarray(),old['hessian'].toarray(),rtol=1e-10,atol=1e-13)
        reduced=condense_contact_problem(p)
        self.addCleanup(reduced['close'])
        x=state[p['contact_dofs']]
        with direct_jacobian_mode(False):old=reduced['objective'](x,'gn')
        with patch.object(terms,'_gap_jacobian_blocks',side_effect=AssertionError('condensed AD repeated')):
            new=reduced['objective'](x,'gn')
        self.assertEqual(new[0],old[0])
        np.testing.assert_allclose(new[1],old[1],rtol=1e-11,atol=1e-13)
        np.testing.assert_allclose(new[2],old[2],rtol=1e-10,atol=1e-13)


    def test_gn_sr1_bfgs_equilibria_match_original_derivatives(self):
        from src.geometry.rimls import direct_jacobian_mode
        from src.mechanics.condensation import condense_contact_problem
        from src.simulation.two_body import solve
        if not hasattr(native_rimls._rimls_native,'local_evaluate'):self.skipTest('native extension missing')
        original=build_problem(n=4)
        p=condense_contact_problem(original)
        self.addCleanup(p['close'])
        for method in ('gn','sr1_gap','structured_bfgs_gap'):
            with direct_jacobian_mode(False):old=solve(p,method,max_iter=100)
            new=solve(p,method,max_iter=100)
            self.assertEqual(old['status'],'converged',method)
            self.assertEqual(new['status'],'converged',method)
            np.testing.assert_allclose(new['state'],old['state'],rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(p['diagnostics'](new['state'])['force_upper'],
                                       p['diagnostics'](old['state'])['force_upper'],rtol=1e-8,atol=1e-10)
            if method=='sr1_gap':self.assertGreater(new['sr1_updates'],0)


if __name__=='__main__':unittest.main()
