"""##r## Selected-query values, coupled derivatives and sparse scatter."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.geometry import contact_patches
from src.geometry.contact_patches import AutomaticRIMLSContact, normalize_query_indices
from src.simulation.two_body import build_problem
from src.contact.terms import multiple_contact_terms, multiple_contact_terms_sparse


class AllCrossBodyFaces:
    """Conservative candidate fixture; actual native BVH/CCD has separate tests."""
    def __init__(self, faces, labels):
        bodies=labels[faces[:, 0]]
        self.pairs=np.array([(a,b) for a in range(len(faces))
            for b in range(a+1,len(faces)) if bodies[a]!=bodies[b]],dtype=int)
    def face_pairs(self, positions, distance):
        return self.pairs, {'total':len(self.pairs)}


class SubsetContactChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def automatic_problem(self, n=2):
        p=build_problem(n=n)
        ref=np.vstack(p['reference']);offset=len(p['reference'][0])
        faces=np.vstack((p['meshes'][0]['boundary_faces'],p['meshes'][1]['boundary_faces']+offset))
        labels=np.r_[np.zeros(offset,dtype=int),np.ones(offset,dtype=int)]
        collider=AllCrossBodyFaces(faces,labels)
        model=AutomaticRIMLSContact(ref,faces,labels,collider)
        p['attach_auto_contact'](model);p['prepare'](p['initial'])
        return p,model

    def test_auto_and_manual_subsets_keep_request_order_and_duplicates(self):
        for automatic in (False, True):
            p,model=self.automatic_problem() if automatic else (build_problem(),None)
            x=torch.tensor(p['initial']+np.random.default_rng(3).normal(size=len(p['initial']))*1e-4,dtype=torch.float64)
            full=p['gaps_ad'](x)
            for ids in ([7,1,11,0,7], [0], [], list(range(len(full)-1,-1,-1))):
                selected=p['gaps_ad'](x,indices=ids)
                expected=full[torch.tensor(ids,dtype=torch.long)]
                torch.testing.assert_allclose(selected,expected,rtol=1e-13,atol=1e-14)

    def test_selected_jacobian_and_hessian_keep_two_body_coupling(self):
        p,model=self.automatic_problem()
        x=torch.tensor(p['prescribed']+np.random.default_rng(12).normal(size=len(p['prescribed']))*1e-4,dtype=torch.float64)
        ids=torch.tensor([7,1,11,0,7],dtype=torch.long)
        def direct(z):return p['gaps_full_ad'](z,indices=ids)
        def reference(z):return p['gaps_full_ad'](z)[ids]
        J=torch.autograd.functional.jacobian(direct,x)
        J0=torch.autograd.functional.jacobian(reference,x)
        H=torch.autograd.functional.hessian(lambda z:torch.sum(direct(z)**2),x)
        H0=torch.autograd.functional.hessian(lambda z:torch.sum(reference(z)**2),x)
        np.testing.assert_allclose(J.numpy(),J0.numpy(),rtol=1e-12,atol=2e-13)
        np.testing.assert_allclose(H.numpy(),H0.numpy(),rtol=1e-11,atol=2e-12)
        self.assertGreater(float(torch.norm(J[:,:p['size']])),.1)
        self.assertGreater(float(torch.norm(J[:,p['size']:])),.1)
        self.assertGreater(float(torch.norm(H[:p['size'],p['size']:])),1e-4)

    def test_subset_evaluates_only_requested_queries_and_target_patch(self):
        p,model=self.automatic_problem(n=None)
        x=torch.tensor(p['prescribed'],dtype=torch.float64)
        original=contact_patches.rimls_values_torch;counts=[];sampler_calls=[]
        def counted(queries,*args,**kwargs):
            counts.append(len(queries));return original(queries,*args,**kwargs)
        for gid,sampler in enumerate(model.samplers):
            def counted_sampler(nodes,gid=gid,sampler=sampler):
                sampler_calls.append(gid);return sampler(nodes)
            model.samplers[gid]=counted_sampler
        with patch.object(contact_patches,'rimls_values_torch',side_effect=counted):
            result=model.gaps(x,indices=[0,2,0])
        self.assertEqual(sum(counts),3)
        self.assertEqual(sampler_calls,[model.blocks[0][0]])
        self.assertEqual(tuple(result.shape),(3,))
        self.assertLess(sum(counts),len(model.weights))

    def test_dense_and_sparse_selected_terms_match_full_evaluation(self):
        # More than 32 queries, plus active filtering and moving normals.
        p=build_problem(n=None,initial_gap=.012,closure=.003)
        x=p['initial']+np.random.default_rng(5).normal(size=len(p['initial']))*1e-5
        for method in ('gn','filtered_gn','full'):
            reference=multiple_contact_terms(x,p['gaps_ad'],p['weights'],method=method)
            dense=multiple_contact_terms(x,p['gaps_ad'],p['weights'],method=method,gaps_subset_autodiff=p['gaps_ad'])
            sparse=multiple_contact_terms_sparse(x,p['gaps_ad'],p['weights'],p['contact_dofs'],method=method,gaps_subset_autodiff=p['gaps_ad'])
            self.assertAlmostEqual(dense['energy'],reference['energy'],places=13)
            np.testing.assert_allclose(dense['gradient'],reference['gradient'],rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(dense['hessian'],reference['hessian'],rtol=2e-11,atol=2e-11)
            np.testing.assert_allclose(sparse['gradient'],reference['gradient'],rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(sparse['hessian'].toarray(),reference['hessian'],rtol=2e-11,atol=2e-11)
            self.assertEqual(dense['filter_count'],reference['filter_count'])
            if method=='filtered_gn':self.assertGreater(dense['filter_count'],0)

    def test_empty_patch_release_and_recontact(self):
        p,model=self.automatic_problem();x=torch.tensor(p['prescribed'],dtype=torch.float64)
        initial=model.signature
        moved=np.vstack(p['reference']);moved[len(p['reference'][0]):,0]+=2.
        model.refresh(moved)
        self.assertEqual(len(model.weights),0)
        self.assertEqual(tuple(model.gaps(x,indices=[]).shape),(0,))
        with self.assertRaises(IndexError):model.gaps(x,indices=[0])
        model.refresh(p['full_state'](p['initial']).reshape(-1,3))
        self.assertEqual(model.signature,initial)
        torch.testing.assert_allclose(model.gaps(x,indices=[0,7]),model.gaps(x)[[0,7]])

    def test_query_indices_reject_invalid_input(self):
        p,model=self.automatic_problem();x=torch.tensor(p['initial'],dtype=torch.float64)
        for ids in ([-1],[len(model.weights)],[1.5],[True],[[0,1]]):
            with self.assertRaises((ValueError,IndexError)):
                p['gaps_ad'](x,indices=ids)
        self.assertEqual(normalize_query_indices([],0,x.device).dtype,torch.long)

    def test_all_gap_validation_is_preserved(self):
        for invalid in (float('nan'),0.,-.1):
            calls=[]
            def gaps(z):return torch.stack((z[0]+.1,z[0]*0+invalid))
            def selected(z,ids):calls.append(1);return gaps(z)[ids]
            with self.assertRaises(ValueError):
                multiple_contact_terms(np.zeros(1),gaps,[1.,1.],gaps_subset_autodiff=selected)
            self.assertEqual(calls,[])

    def test_full_and_force_only_do_not_request_subset_jacobian(self):
        p=build_problem();x=p['initial']
        def unused(*args):raise AssertionError('subset Jacobian should not be needed')
        for method,compute in (('full',True),('gn',False)):
            a=multiple_contact_terms(x,p['gaps_ad'],p['weights'],method=method,compute_hessian=compute,gaps_subset_autodiff=unused)
            self.assertTrue(np.isfinite(a['energy']))

    def test_selected_callback_shape_is_checked(self):
        def gaps(x):return x[:2]+.1
        with self.assertRaises(ValueError):
            multiple_contact_terms(np.zeros(2),gaps,[1.,1.],gaps_subset_autodiff=lambda x,ids:x[:1]+.1)


if __name__=='__main__':unittest.main()
