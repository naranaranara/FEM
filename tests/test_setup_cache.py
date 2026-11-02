"""##r## Setup reuse must preserve equilibrium and invalidate stale data."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from src.simulation.setup_cache import SetupCache
from src.simulation.test_crossed_cylinders import build_problem, run
from src.simulation.two_body import solve
from src.mechanics.condensation import condense_contact_problem


class SetupCacheChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'cache'
        self.threads=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,self.threads)
        pool=threadpool_limits(1);self.addCleanup(pool.restore_original_limits)

    def test_cold_warm_and_disabled_equilibrium(self):
        args=dict(n_radial=2,radial_grading=3.,d_hat=.02)
        plain=build_problem(**args)
        cold=build_problem(cache_dir=self.root,**args)
        a=condense_contact_problem(cold,setup_cache=cold['setup_cache'])
        self.addCleanup(a['close'])
        with patch('src.simulation.test_crossed_cylinders.make_cylinder',side_effect=AssertionError('mesh rebuilt')), \
             patch('src.simulation.two_body.assemble_stiffness',side_effect=AssertionError('K rebuilt')):
            warm=build_problem(cache_dir=self.root,**args)
        from src.mechanics import condensation
        factorize=condensation.factorize_sparse
        rhs_shapes=[]
        def tracked(*args):
            factor=factorize(*args)
            class Proxy:
                def solve(self,rhs):
                    rhs_shapes.append(rhs.shape)
                    return factor.solve(rhs)
            return Proxy()
        with patch.object(condensation,'factorize_sparse',side_effect=tracked):
            b=condense_contact_problem(warm,setup_cache=warm['setup_cache'])
        self.addCleanup(b['close'])
        self.assertTrue(all(warm['setup_cache'].hits.values()))
        self.assertEqual(b['schur_assembly_s'],0.)
        self.assertFalse(any(len(shape)==2 for shape in rhs_shapes))
        np.testing.assert_array_equal(plain['stiffness'].data,cold['stiffness'].data)
        np.testing.assert_array_equal(a['stiffness_free'],b['stiffness_free'])
        for closure in (.01,.045):
            a['set_closure'](closure);b['set_closure'](closure)
            x=solve(a,'gn');y=solve(b,'gn')
            self.assertEqual(x['status'],'converged');self.assertEqual(y['status'],'converged')
            np.testing.assert_allclose(x['state'],y['state'],atol=1e-11)
            np.testing.assert_allclose(a['full_state'](x['state']),b['full_state'](y['state']),atol=1e-11)
            np.testing.assert_allclose(a['diagnostics'](x['state'])['force_upper'],
                                       b['diagnostics'](y['state'])['force_upper'],atol=1e-11)

    def test_pardiso_warm_cache(self):
        import importlib.util
        if importlib.util.find_spec('pypardiso') is None:
            self.skipTest('optional pypardiso missing')
        matrices=[]
        for index in range(2):
            p=build_problem(cache_dir=self.root,n_radial=2,d_hat=.02)
            c=condense_contact_problem(p,setup_cache=p['setup_cache'],linear_backend='pardiso')
            try:
                c['set_closure'](.045)
                result=solve(c,'gn')
                self.assertEqual(result['status'],'converged')
                matrices.append((c['stiffness_free'].copy(),result['state'].copy()))
                self.assertEqual(p['setup_cache'].hits['schur_pardiso'],bool(index))
            finally:c['close']()
        for first,second in zip(*matrices):
            np.testing.assert_allclose(first,second,atol=1e-11)

    def test_keys_corruption_and_partial_cache(self):
        first=SetupCache(self.root,dict(young=30,mesh=2))
        first.store('entry',dict(a=np.arange(5)))
        self.assertIsNotNone(first.load('entry'))
        changed=SetupCache(self.root,dict(young=31,mesh=2))
        self.assertNotEqual(first.key,changed.key);self.assertIsNone(changed.load('entry'))
        with patch('src.simulation.setup_cache.source_signature',return_value='changed'):
            source=SetupCache(self.root,dict(young=30,mesh=2))
        self.assertNotEqual(first.key,source.key)
        (first.folder/'entry.npz').write_bytes(b'interrupted write')
        self.assertIsNone(first.load('entry'))
        first.store('entry',dict(a=np.arange(5)))
        (first.folder/'entry.sha256').unlink()
        self.assertIsNone(first.load('entry'))

    def test_changed_matrix_rebuilds_schur(self):
        p=build_problem(cache_dir=self.root,n_radial=2)
        a=condense_contact_problem(p,setup_cache=p['setup_cache']);self.addCleanup(a['close'])
        p['stiffness_free']=2*p['stiffness_free']
        b=condense_contact_problem(p,setup_cache=p['setup_cache']);self.addCleanup(b['close'])
        self.assertFalse(p['setup_cache'].hits['schur_superlu'])
        np.testing.assert_allclose(b['stiffness_free'],2*a['stiffness_free'],atol=1e-11)

    def test_runner_records_cache_usage(self):
        args=dict(n_radial=2,d_hat=.02,delta=.025,methods=('gn',),collision='none',
                  contact_steps=2,linear_backend='superlu',cache_dir=self.root)
        for index in range(2):
            folder=Path(self.tmp.name)/('run%d'%index)
            p,rows,_=run(outdir=folder,**args)
            try:
                self.assertTrue(all(row['status']=='converged' for row in rows))
                data=json.loads((folder/'parameters.json').read_text())
                self.assertEqual(all(data['setup_cache_hits'].values()),bool(index))
                if index:self.assertEqual(data['schur_assembly_s'],0.)
            finally:p['close']()


if __name__=='__main__':unittest.main()
