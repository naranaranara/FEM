"""##r## Geometry, independent Hertz limits, and coupled contact checks."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from src.geometry.mesh import make_cylinder
from src.geometry.primitives import crossed_cylinder_curvature,size_based_activation_distance
from src.mechanics.solid import tetrahedron_volumes
from src.mechanics.hertz import elliptical_hertz
from src.simulation.test_crossed_cylinders import build_problem
from src.simulation.two_body import solve


class CrossedCylinderChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.pool=threadpool_limits(1)
    @classmethod
    def tearDownClass(cls):
        cls.pool.restore_original_limits();torch.set_num_threads(cls.threads)

    def test_full_cylinder_geometry_volume_and_manifold(self):
        for facing,angle in (('up',0.),('down',45.)):
            m=make_cylinder(facing=facing,angle_degrees=angle,clearance=.03)
            nodes=m['nodes'];local=m['local_nodes'];radius=m['radius']
            center=-radius if facing=='up' else .03+radius
            rr=local[:,0]**2+(local[:,2]-center)**2
            outer=np.unique(m['side_faces'])
            np.testing.assert_allclose(rr[outer],radius**2,atol=1e-13)
            self.assertAlmostEqual(nodes[:,2].max()-nodes[:,2].min(),2*radius)
            volumes=tetrahedron_volumes(nodes,m['tetrahedra'])
            self.assertTrue(np.all(volumes>0))
            # Polygonal cylinder volume: cross-section shoelace area times length.
            ring=local[:33][1:33][:,[0,2]]
            area=.5*abs(np.sum(ring[:,0]*np.roll(ring[:,1],-1)-ring[:,1]*np.roll(ring[:,0],-1)))
            self.assertAlmostEqual(float(volumes.sum()),area*2*m['half_length'],places=11)
            edges={}
            for face in m['boundary_faces']:
                for i,j in zip(face,np.roll(face,-1)):
                    key=tuple(sorted((i,j)));edges[key]=edges.get(key,0)+1
            self.assertTrue(all(v==2 for v in edges.values()))
            self.assertEqual(len(m['boundary_faces']),2*32+2*32*6)
            np.testing.assert_allclose(np.abs(local[m['support_nodes'],1]),m['half_length'])
            self.assertTrue(set(map(tuple,m['query_faces'])).issubset(set(map(tuple,m['contact_faces']))))
            xyz=nodes[m['contact_faces']]
            normals=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
            self.assertTrue(np.all(normals[:,2]*(1 if facing=='up' else -1)>0))

    def test_size_based_activation_and_override(self):
        p=build_problem()
        diagonal=np.linalg.norm(np.ptp(np.vstack(p['reference']),axis=0))
        self.assertAlmostEqual(p['d_hat'],1e-3*diagonal,places=14)
        old=p['d_hat'];p['set_closure'](.01)
        self.assertEqual(p['d_hat'],old)
        explicit=build_problem(d_hat=.001)
        self.assertEqual(explicit['d_hat'],.001)
        self.assertEqual(explicit['parameters']['d_hat_mode'],'explicit')
        points=np.vstack(p['reference'])
        scaled,L=size_based_activation_distance(2*points+np.array([5.,8.,10.]))
        self.assertAlmostEqual(scaled,2*old,places=14)
        with self.assertRaises(ValueError):size_based_activation_distance(points,ratio=0.)

    def test_hertz_guard_rejects_large_activation_before_fem_assembly(self):
        with patch('src.simulation.test_crossed_cylinders.build_two_body') as assembly:
            for kwargs in ({}, {'d_hat':.005}, {'d_hat':.01}):
                with self.subTest(kwargs=kwargs):
                    with self.assertRaisesRegex(ValueError,'Hertz validation requires d_hat <'):
                        build_problem(validation_delta=.005,**kwargs)
            assembly.assert_not_called()

    def test_hertz_guard_allows_smaller_explicit_distance(self):
        p=build_problem(d_hat=.0005,validation_delta=.005)
        self.assertEqual(p['d_hat'],.0005)
        self.assertLess(p['d_hat'],.005)

    def test_relative_curvature_and_parallel_exclusion(self):
        K,k,Q=crossed_cylinder_curvature(3.,5.,45.)
        np.testing.assert_allclose(Q@np.diag(k)@Q.T,K,atol=1e-15)
        self.assertAlmostEqual(np.trace(K),1./3+1./5)
        self.assertAlmostEqual(np.linalg.det(K),np.sin(np.pi/4)**2/15)
        with self.assertRaises(ValueError):crossed_cylinder_curvature(3.,5.,0.)

    def test_circular_limit_matches_sphere_hertz_force_radius_pressure(self):
        # Equal-radius perpendicular cylinders have the same relative quadratic
        # gap as a sphere of radius R against a plane.
        radius=4.;delta=.01;young=30.;nu=.3
        _,k,_=crossed_cylinder_curvature(radius,radius,90.)
        r=elliptical_hertz(delta,k,young,young,nu)
        effective=young/(2*(1-nu**2))
        a=np.sqrt(radius*delta)
        self.assertAlmostEqual(r['major_radius'],a,places=12)
        self.assertAlmostEqual(r['minor_radius'],a,places=12)
        self.assertAlmostEqual(r['force'],4./3*effective*np.sqrt(radius)*delta**1.5,places=12)
        self.assertAlmostEqual(r['peak_pressure'],2*effective*a/(np.pi*radius),places=12)

    def test_elliptical_scaling_and_swap_symmetry(self):
        _,k,_=crossed_cylinder_curvature(3.,5.,45.)
        _,swapped,_=crossed_cylinder_curvature(5.,3.,45.)
        np.testing.assert_allclose(k,swapped)
        r=elliptical_hertz(.01,k);twice=elliptical_hertz(.04,k)
        self.assertGreater(r['major_radius'],r['minor_radius'])
        self.assertAlmostEqual(twice['force']/r['force'],8.,places=11)
        self.assertAlmostEqual(twice['major_radius']/r['major_radius'],2.,places=12)
        self.assertEqual(elliptical_hertz(0.,k)['force'],0.)

    def test_small_coupled_methods_agree_and_balance(self):
        p=build_problem()
        p['set_closure'](.0175)
        reference=solve(p,'full',max_iter=60)
        self.assertEqual(reference['status'],'converged')
        for method in ('gn','sr1_gap'):
            r=solve(p,method,max_iter=60)
            self.assertEqual(r['status'],'converged')
            np.testing.assert_allclose(r['state'],reference['state'],atol=2e-8)
            d=p['diagnostics'](r['state'])
            self.assertTrue(np.all(d['gaps']>0))
            self.assertGreater(d['force_upper'][2],0.)
            np.testing.assert_allclose(d['force_upper']+d['force_lower'],0.,atol=1e-12)
            self.assertTrue(p['feasible'](r['state']))
        full=p['contact'](reference['state'],'full')
        gn=p['contact'](reference['state'],'gn')
        self.assertGreater(np.linalg.norm((full['hessian']-gn['hessian']).toarray()),1e-5)


if __name__=='__main__':unittest.main()
