"""##r## Local-neighbour GN acceleration preserves geometry derivatives."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.geometry.rimls import rimls_values_torch, _rimls_values_reference
from src.geometry.local_rimls import local_rimls_mode
from src.geometry import native_rimls


class LocalRIMLSChecks(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(1)
        rng=np.random.default_rng(42)
        p=rng.uniform(-1,1,(96,3));p[:,2]*=.08
        n=rng.normal(scale=.1,size=p.shape);n[:,2]+=1.
        q=rng.uniform(-.35,.35,(24,3));q[:,2]=.07
        self.arrays=tuple(torch.tensor(x,dtype=torch.float64,requires_grad=True) for x in (q,p,n))
        self.scales=(.6,.5,.5)
    def tearDown(self):torch.set_num_threads(self.threads)

    def test_values_all_input_jacobians_and_second_derivatives(self):
        expected=_rimls_values_reference(*self.arrays,*self.scales)
        jac=torch.autograd.functional.jacobian(lambda *a:_rimls_values_reference(*a,*self.scales),self.arrays,vectorize=True)
        for cpp in (False,True):
            stats={}
            with local_rimls_mode(native_first=cpp,stats=stats):
                actual=rimls_values_torch(*self.arrays,*self.scales)
                got=torch.autograd.functional.jacobian(lambda *a:rimls_values_torch(*a,*self.scales),self.arrays,vectorize=True)
            torch.testing.assert_close(actual,expected,rtol=2e-11,atol=1e-13)
            for a,b in zip(got,jac):torch.testing.assert_close(a,b,rtol=2e-10,atol=2e-12)
            self.assertLess(stats['local_pairs'],stats['dense_pairs'])
        # Full Hessian of query coordinates through the create_graph fallback.
        def function(q,reference):
            fn=_rimls_values_reference if reference else rimls_values_torch
            return fn(q,*self.arrays[1:],*self.scales).square().sum()
        a=torch.autograd.functional.hessian(lambda q:function(q,False),self.arrays[0])
        b=torch.autograd.functional.hessian(lambda q:function(q,True),self.arrays[0])
        torch.testing.assert_close(a,b,rtol=2e-10,atol=2e-11)

    def test_refresh_moving_support_and_value_only(self):
        q,p,n=self.arrays
        for shift in (0.,.1,-.12):
            moved=p.detach().clone();moved[:,0]+=shift
            with torch.no_grad():
                a=rimls_values_torch(q,moved,n,*self.scales)
                b=_rimls_values_reference(q,moved,n,*self.scales)
            torch.testing.assert_close(a,b,rtol=2e-11,atol=1e-13)
        with self.assertRaisesRegex(ValueError,'outside kernel support'):
            rimls_values_torch(q+20.,p,n,*self.scales)

    def test_surface_nodes_normals_and_contact_gn(self):
        from src.geometry.mesh import make_block
        from src.geometry.surface import make_surface_sampler
        from src.contact.terms import multiple_contact_terms
        mesh=make_block(nx=5,ny=5,nz=2)
        ref=mesh['nodes'];ref[:,2]+=.04*ref[:,0]**2
        sampler,_,w=make_surface_sampler(ref,mesh['top_faces'],sparse=True)
        xyz=torch.tensor(ref,dtype=torch.float64)
        points,_=sampler(xyz)
        q=(points[:24]+torch.tensor([.01,.01,.08],dtype=torch.float64)).numpy()
        state=np.r_[q.ravel(),ref.ravel()]
        def gaps(z):
            p,n=sampler(z[q.size:].reshape(-1,3))
            return rimls_values_torch(z[:q.size].reshape(-1,3),p,n,.65)
        def evaluate():
            return multiple_contact_terms(state,gaps,np.ones(len(q)),d_hat=.2,method='gn',
                    gaps_subset_autodiff=lambda z,ids:gaps(z)[ids],gaps_value_autodiff=gaps)
        with local_rimls_mode(False):old=evaluate()
        new=evaluate()
        for key in ('gaps','gradient','hessian','energy'):
            np.testing.assert_allclose(new[key],old[key],rtol=2e-10,atol=2e-11)

    def test_native_csr_rejects_invalid_indices(self):
        if not hasattr(native_rimls._rimls_native,'local_evaluate'):self.skipTest('native extension missing')
        q,p,n=(x.detach().numpy() for x in self.arrays)
        offsets=np.zeros(len(q)+1,dtype=np.int64);offsets[1:]=1
        ids=np.array([len(p)],dtype=np.int64)
        with self.assertRaises(ValueError):
            native_rimls._rimls_native.local_evaluate(q,p,n,offsets,ids,np.empty(len(q)),np.empty((1,3)),np.empty((1,3)),*self.scales)


if __name__=='__main__':unittest.main()
