"""##r## Independent checks for the fixed-surface distance diagnostic."""
import unittest
import numpy as np
import torch
from src.geometry.mesh import make_cylinder
from src.geometry.surface import make_surface_sampler
from src.simulation.test_gap_distance import (polygon_distance,polygon_radial_radius,
    cylinder_polygon,field_and_gradient)


class GapDistanceChecks(unittest.TestCase):
    def test_polygon_sign_euclidean_distance_and_ray(self):
        square=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        points=np.array([[0.,0.],[2.,0.],[2.,2.],[1.,.4]])
        expected=np.array([-1.,1.,np.sqrt(2),0.])
        for vertices in (square,square[::-1]):
            np.testing.assert_allclose(polygon_distance(points,vertices),expected,atol=1e-14)
            directions=np.array([[1.,0.],[1/np.sqrt(2),1/np.sqrt(2)]])
            np.testing.assert_allclose(polygon_radial_radius(directions,vertices),[1.,np.sqrt(2)],atol=1e-14)

    def test_polygon_matches_actual_side_triangle_samples(self):
        for facing,clearance in (('up',0.),('down',.03)):
            mesh=make_cylinder(radius=3.,n_angle=32,n_axis=7,facing=facing,clearance=clearance)
            samples=mesh['nodes'][mesh['side_faces']].mean(axis=1)
            center=-3. if facing=='up' else clearance+3.
            section=samples[:,[0,2]].copy();section[:,1]-=center
            polygon=cylinder_polygon(3.,32,facing)
            np.testing.assert_allclose(polygon_distance(section,polygon),0.,atol=2e-15)

    def test_plane_eikonal_and_fixed_query_derivative(self):
        old=torch.get_num_threads();torch.set_num_threads(1)
        try:
            x,y=np.meshgrid(np.linspace(-1.,1.,11),np.linspace(-1.,1.,11))
            points=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
            normals=np.tile([0.,0.,1.],(len(points),1))
            q=np.array([[.1,.05,.001],[-.1,.2,-.001],[.2,0.,.03]])
            value,gradient=field_and_gradient(q,points,normals,.65)
            np.testing.assert_allclose(value,q[:,2],atol=1e-15)
            np.testing.assert_allclose(gradient,np.tile([0.,0.,1.],(len(q),1)),atol=1e-13)
            mesh=make_cylinder()
            sample,_,_=make_surface_sampler(mesh['nodes'],mesh['contact_faces'],sparse=True)
            with torch.no_grad():p,n=sample(torch.tensor(mesh['nodes'],dtype=torch.float64))
            q=np.array([[.1,.02,.02]])
            _,gradient=field_and_gradient(q,p.numpy(),n.numpy(),.65)
            for axis in range(3):
                step=np.zeros((1,3));step[0,axis]=1e-6
                plus,_=field_and_gradient(q+step,p.numpy(),n.numpy(),.65)
                minus,_=field_and_gradient(q-step,p.numpy(),n.numpy(),.65)
                np.testing.assert_allclose(gradient[:,axis],(plus-minus)/2e-6,atol=1e-8)
        finally:torch.set_num_threads(old)


class SamplingAuditChecks(unittest.TestCase):
    def test_independent_weighted_reference_matches_production(self):
        from src.simulation.test_rimls_sampling_audit import weighted_reference
        from src.geometry.rimls import _rimls_values_reference
        rng=np.random.default_rng(123)
        p=torch.tensor(rng.uniform(-.2,.2,(40,3)),dtype=torch.float64)
        p[:,2]*=.1
        n=torch.tensor(rng.normal(0,.05,(40,3)),dtype=torch.float64);n[:,2]+=1.
        n=n/torch.linalg.norm(n,dim=1)[:,None]
        q=torch.tensor([[.01,.02,.01],[-.03,.01,.02]],dtype=torch.float64,requires_grad=True)
        a=weighted_reference(q,p,n,.4,torch.ones(len(p),dtype=torch.float64))
        b=_rimls_values_reference(q,p,n,.4)
        np.testing.assert_allclose(a.detach(),b.detach(),atol=1e-15)
        ga=torch.autograd.grad(a.sum(),q,retain_graph=True)[0]
        gb=torch.autograd.grad(b.sum(),q)[0]
        np.testing.assert_allclose(ga,gb,atol=1e-13)

    def test_density_compensation_under_sample_duplication(self):
        from src.simulation.test_rimls_sampling_audit import weighted_reference
        p=torch.tensor([[-.1,0.,-.002],[0.,0.,0.],[.1,0.,-.002]],dtype=torch.float64)
        n=torch.tensor([[-.04,0.,1.],[0.,0.,1.],[.04,0.,1.]],dtype=torch.float64)
        n=n/torch.linalg.norm(n,dim=1)[:,None]
        q=torch.tensor([[.01,0.,.01]],dtype=torch.float64)
        base=weighted_reference(q,p,n,.3,torch.ones(3,dtype=torch.float64))
        ids=torch.tensor([0,0,0,1,2])
        balanced=weighted_reference(q,p[ids],n[ids],.3,
            torch.tensor([1/3,1/3,1/3,1.,1.],dtype=torch.float64))
        np.testing.assert_allclose(base,balanced,atol=1e-15)

    def test_analytic_projection_and_normals_on_both_cylinders(self):
        from src.simulation.test_rimls_sampling_audit import analytic_points_normals
        for radius,facing,angle,clearance in [(3.,'up',0.,0.),(5.,'down',45.,.03)]:
            points=np.array([[.01,.05,.001],[.1,-.03,.01]])
            projected,n=analytic_points_normals(points,radius,facing,angle,clearance)
            projected2,n2=analytic_points_normals(projected,radius,facing,angle,clearance)
            np.testing.assert_allclose(projected2,projected,atol=2e-15)
            np.testing.assert_allclose(n,n2,atol=2e-15)
            np.testing.assert_allclose(np.linalg.norm(n,axis=1),1.,atol=1e-15)


if __name__=='__main__':unittest.main()
