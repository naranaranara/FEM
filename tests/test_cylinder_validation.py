import unittest
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from src.simulation.two_body import build_problem,solve
from src.mechanics.condensation import condense_contact_problem
from src.optimization.structured_bfgs import MultiGapStructuredBFGS
from src.simulation.test_cylinder_validation import surface_error,activation_converged

class ValidationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=torch.get_num_threads();torch.set_num_threads(1);cls.pool=threadpool_limits(1)
    @classmethod
    def tearDownClass(cls):
        cls.pool.restore_original_limits();torch.set_num_threads(cls.old)
    def test_condensed_base_and_full_contact_match(self):
        p=build_problem(n=2,sparse=True);c=condense_contact_problem(p)
        c['set_closure'](.02)
        q=c['initial'].copy();q+=np.random.default_rng(2).normal(0,1e-4,len(q))
        x=c['expand'](q);E,g,K=p['base'](x);Ec,gc,S=c['base'](q)
        retained=c['retained'];interior=np.setdiff1d(np.arange(len(x)),retained)
        self.assertAlmostEqual(E,Ec,places=12)
        np.testing.assert_allclose(g[interior],0,atol=1e-12)
        np.testing.assert_allclose(g[retained],gc,atol=1e-12)
        eo,go,ho,_=p['objective'](x,'gn');ec,gc,hc,_=c['objective'](q,'gn')
        self.assertAlmostEqual(eo,ec,places=12)
        np.testing.assert_allclose(go[retained],gc,atol=1e-11)
        expected=ho[retained][:,retained].toarray()
        if len(interior):expected-=ho[retained][:,interior].toarray()@np.linalg.solve(ho[interior][:,interior].toarray(),ho[interior][:,retained].toarray())
        np.testing.assert_allclose(hc,expected,atol=1e-10)
    def test_condensed_and_original_equilibria(self):
        p=build_problem(n=2,sparse=True);c=condense_contact_problem(p);c['set_closure'](.03)
        full=solve(p,'gn');small=solve(c,'gn')
        self.assertEqual(full['status'],'converged');self.assertEqual(small['status'],'converged')
        np.testing.assert_allclose(c['expand'](small['state']),full['state'],atol=1e-7)
        for method in ('sr1_gap','structured_bfgs_gap'):
            result=solve(c,method)
            self.assertEqual(result['status'],'converged')
            np.testing.assert_allclose(result['state'],small['state'],atol=1e-7)
    def test_multi_bfgs_negative_missing_curvature_and_reset(self):
        model=MultiGapStructuredBFGS();G=np.diag([2.,2.])
        def data(J,indices=(0,)):
            return dict(gap_dofs=np.array([0,1]),gap_indices=np.array(indices),
                gap_jacobian=np.array(J).reshape(len(indices),2),gap_weights=np.array([1.]),gaps=np.array([.1]))
        old=data([[1.,0.]])
        C,_=model.correction(np.zeros(2),old,G,.2)
        new=data([[1.01,0.]])
        C,info=model.correction(np.array([.01,0.]),new,G,.2)
        from src.contact.barrier import barrier_first_second
        z=barrier_first_second(.1,.2)[0]*np.array([.01,0.])
        np.testing.assert_allclose(C@np.array([.01,0.]),z,atol=1e-12)
        self.assertLess(C[0,0],0);self.assertEqual(info['bfgs_updates'],1)
        C,info=model.correction(np.array([.02,0.]),data([],()),G,.2)
        self.assertEqual(C.nnz,0);self.assertEqual(info['bfgs_resets'],1)
    def test_activation_stops_only_after_converged_force_comparison(self):
        self.assertTrue(activation_converged(dict(all_loads_converged=True,force_change=1e-4)))
        self.assertFalse(activation_converged(dict(all_loads_converged=True,force_change=None)))
        self.assertFalse(activation_converged(dict(all_loads_converged=False,force_change=1e-4)))
        self.assertFalse(activation_converged(dict(all_loads_converged=True,force_change=.1)))
    def test_rk_is_one_for_exact_quadratic_and_does_not_change_solution(self):
        H=np.diag([2.,3.]);initial=np.array([1.,.5])
        p=dict(initial=initial,energy=lambda x:float(.5*x@H@x),
            objective=lambda x,method,model:(float(.5*x@H@x),H@x,H,{}))
        a=solve(p,'gn');b=solve(p,'gn',track_rk=True)
        self.assertEqual(b['status'],'converged')
        self.assertEqual(a['updates'],b['updates'])
        np.testing.assert_array_equal(a['state'],b['state'])
        self.assertAlmostEqual(b['trace'][1]['rk'],1.,places=12)
    def test_normal_ray_root_is_resolved(self):
        row,arrays=surface_error(32,7,.65)
        self.assertGreater(row['zero_error_max'],.0005)
        self.assertTrue(all(r['root_residual']<1e-10 for r in row['bodies']))
        self.assertEqual(arrays['lower_zero_offsets'].shape,(81,))

if __name__=='__main__':unittest.main()
