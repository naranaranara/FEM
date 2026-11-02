"""##r## PARDISO/SuperLU equivalence and refined contact loading checks."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import json
import numpy as np
import torch
from scipy.sparse import diags
from threadpoolctl import threadpool_limits
from src.mechanics.sparse_factor import factorize_sparse
from src.mechanics.condensation import condense_contact_problem
from src.simulation.test_crossed_cylinders import build_problem, closure_schedule, run
from src.simulation.two_body import solve


@unittest.skipUnless(importlib.util.find_spec('pypardiso'), 'optional pypardiso missing')
class PardisoChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.pool=threadpool_limits(1)
    @classmethod
    def tearDownClass(cls):
        cls.pool.restore_original_limits();torch.set_num_threads(cls.threads)

    def test_multiple_rhs_reuse_and_release(self):
        A=diags([-np.ones(29),3*np.ones(30),-np.ones(29)],[-1,0,1],format='csr')
        rhs=np.random.default_rng(3).normal(size=(30,4))
        p=factorize_sparse(A,'pardiso');self.addCleanup(p.close)
        lu=factorize_sparse(A,'superlu')
        for b in (rhs,rhs[:,0],rhs*2):
            x=p.solve(b)
            np.testing.assert_allclose(x,lu.solve(b),rtol=1e-12,atol=1e-13)
            self.assertLess(np.linalg.norm(A@x-b)/np.linalg.norm(b),1e-12)
        p.close();p.close()
        with self.assertRaises(RuntimeError):p.solve(rhs)

    def test_condensed_graded_cylinder_equivalence(self):
        original=build_problem(n_radial=2,radial_grading=3.)
        a=condense_contact_problem(original,linear_backend='superlu')
        b=condense_contact_problem(original,linear_backend='pardiso')
        self.addCleanup(b['close'])
        a['set_closure'](.0175);b['set_closure'](.0175)
        np.testing.assert_allclose(a['stiffness_free'],b['stiffness_free'],atol=1e-10,rtol=1e-10)
        x=solve(a,'gn',max_iter=60);y=solve(b,'gn',max_iter=60)
        self.assertEqual(x['status'],'converged');self.assertEqual(y['status'],'converged')
        np.testing.assert_allclose(a['full_state'](x['state']),b['full_state'](y['state']),atol=1e-8)
        np.testing.assert_allclose(a['diagnostics'](x['state'])['force_upper'],
                                   b['diagnostics'](y['state'])['force_upper'],atol=1e-9)

    def test_runner_saves_trace_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'run'
            p,rows,_=run(n_radial=2,radial_grading=3.,delta=.025,d_hat=.02,
                         methods=('gn',),collision='none',contact_steps=3,outdir=str(folder))
            try:
                self.assertTrue(all(r['status']=='converged' for r in rows))
                trace=json.loads((folder/'trace_gn_003.json').read_text())
                self.assertIn('iteration',trace[-1])
                with np.load(folder/'checkpoint_gn.npz') as data:
                    self.assertEqual(str(data['status']),'converged')
                    self.assertAlmostEqual(float(data['closure']),.055)
                self.assertEqual(json.loads((folder/'parameters.json').read_text())['linear_backend'],'pardiso')
            finally:p['close']()


class LoadScheduleChecks(unittest.TestCase):
    def test_refinement_preserves_final_load(self):
        schedule=closure_schedule(.03,.005,.0005,12,40)
        self.assertEqual(schedule[0],0.)
        self.assertAlmostEqual(schedule[-1],.035)
        self.assertTrue(np.all(np.diff(schedule)>0))
        self.assertTrue(np.all(np.diff(schedule[3:])<.0005))
        np.testing.assert_allclose(closure_schedule(.03,.005,.0005,12),np.linspace(0,.035,13))
        with self.assertRaises(ValueError):closure_schedule(.03,.005,.0005,12,0)


if __name__=='__main__':unittest.main()
