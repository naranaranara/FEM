"""##r## Unequal body sizes and locally refined contact runner regression checks."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from src.geometry.mesh import make_block
from src.simulation.two_body import build_problem, solve
from src.simulation.test_crossed_cylinders import run
from src.mechanics.condensation import condense_contact_problem


class UnequalBodyContactChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.pool=threadpool_limits(1)
    @classmethod
    def tearDownClass(cls):
        cls.pool.restore_original_limits();torch.set_num_threads(cls.threads)

    def test_unequal_dense_sparse_condensation_force_and_bc(self):
        lo=make_block(width=1.,depth=1.,height=.4,nx=2,ny=2,nz=2)
        hi=make_block(width=1.,depth=1.,height=.4,nx=3,ny=2,nz=2)
        hi['nodes'][:,2]+=.52
        args=dict(contact_meshes=(lo,hi),initial_gap=.12,closure=.04)
        sparse=build_problem(sparse=True,**args)
        dense=build_problem(sparse=False,**args)
        np.testing.assert_allclose(sparse['stiffness'].toarray(),dense['stiffness'],atol=1e-12)
        self.assertEqual(sparse['stiffness'].shape,(60,60))
        self.assertEqual(sparse['body_node_counts'],(8,12))
        c=condense_contact_problem(sparse,linear_backend='pardiso')
        self.addCleanup(c['close'])
        full=solve(sparse,'gn');reduced=solve(c,'gn')
        self.assertEqual(full['status'],'converged')
        self.assertEqual(reduced['status'],'converged')
        np.testing.assert_allclose(sparse['full_state'](full['state']),c['full_state'](reduced['state']),atol=1e-8)
        nodes=c['unpack'](reduced['state'])
        self.assertEqual(nodes[0].shape,(8,3));self.assertEqual(nodes[1].shape,(12,3))
        data=c['diagnostics'](reduced['state'])
        self.assertGreater(data['force_upper'][2],0)
        np.testing.assert_allclose(data['force_upper']+data['force_lower'],0.,atol=1e-12)
        np.testing.assert_allclose(c['full_state'](reduced['state'])[sparse['fixed']],sparse['prescribed'][sparse['fixed']])

    def test_local_runner_ccd_and_saved_unequal_bodies(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'run'
            p,rows,_=run(n_radial=2,radial_grading=3.,local_size=.4,local_half_width=.2,
                local_depth=.15,delta=.025,d_hat=.02,methods=('gn',),collision='ipctk',
                contact_steps=3,outdir=str(folder))
            try:
                self.assertTrue(all(r['status']=='converged' for r in rows))
                counts=[len(x) for x in p['reference']]
                self.assertNotEqual(*counts)
                with np.load(folder/'states.npz') as saved:
                    self.assertEqual(str(saved['layout']),'stacked_nodes')
                    self.assertEqual(saved['gn'].shape,(len(rows),sum(counts),3))
                    np.testing.assert_array_equal(saved['body_node_offsets'],np.cumsum([0]+counts))
                with np.load(folder/'checkpoint_gn.npz') as saved:
                    self.assertEqual(saved['nodes'].shape,(sum(counts),3))
                self.assertEqual(json.loads((folder/'parameters.json').read_text())['local_size'],.4)
            finally:p['close']()


if __name__=='__main__':unittest.main()
