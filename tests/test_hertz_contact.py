"""##r## Independent physical and numerical checks for the Hertz benchmark."""
import unittest
import tempfile,json,csv,contextlib,io
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits
from src.geometry.mesh import make_spherical_cap
from src.geometry.surface import surface_lumped_weights
from src.geometry.rimls import rimls_values_torch
from src.mechanics.solid import tetrahedron_volumes
from src.contact.barrier import barrier_value,barrier_first_second,barrier_arrays
from src.contact.terms import multiple_contact_terms
from src.optimization.newton import minimize_static
from src.simulation.hertz import build_problem,solve,hertz_reference,make_plane_direction,save_results


class HertzContactChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.threads=threadpool_limits(1)
    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_analytical_scaling_and_pressure_integral(self):
        h=hertz_reference(.02,10.,1000.,.3)
        self.assertAlmostEqual(h['contact_radius']**2,.2)
        self.assertAlmostEqual(hertz_reference(.08,10.,1000.,.3)['force']/h['force'],8.)
        self.assertAlmostEqual(hertz_reference(.02,10.,2000.,.3)['force']/h['force'],2.)
        r=np.linspace(0,h['contact_radius'],10001)
        p=hertz_reference(.02,10.,1000.,.3,r)['pressure']
        integrated=2*np.pi*np.trapz(r*p,r)
        self.assertLess(abs(integrated/h['force']-1),3e-6)
        zero=hertz_reference(0,10.,1000.,.3,r)
        self.assertEqual(zero['force'],0.)
        np.testing.assert_array_equal(zero['pressure'],0.)

    def test_curved_mesh_keeps_contact_topology_and_positive_volumes(self):
        m=make_spherical_cap(nxy=9,nz=5)
        xyz=m['nodes'];bottom=m['bottom_nodes'];x,y=xyz[bottom,:2].T
        np.testing.assert_allclose(xyz[bottom,2],.002+10-np.sqrt(100-x*x-y*y),atol=1e-14)
        np.testing.assert_allclose(xyz[np.unique(m['top_faces']),2],10.002)
        np.testing.assert_array_equal(np.unique(m['contact_faces']),bottom)
        self.assertTrue(np.all(tetrahedron_volumes(xyz,m['tetrahedra'])>0))
        with self.assertRaises(ValueError):make_spherical_cap(nxy=8)
        with self.assertRaises(ValueError):make_spherical_cap(radius=1)
        with self.assertRaises(ValueError):make_spherical_cap(height=1)

    def test_lumped_surface_area_and_linear_integration(self):
        nodes=np.array([[0.,0.,0.],[2.,0.,0.],[0.,3.,0.],[7.,7.,7.]])
        w=surface_lumped_weights(nodes,[[0,1,2]])
        np.testing.assert_allclose(w,[1.,1.,1.,0.])
        self.assertAlmostEqual(w.sum(),3.)
        np.testing.assert_allclose(w@nodes,3*nodes[:3].mean(axis=0))
        with self.assertRaises(ValueError):surface_lumped_weights(nodes,[[0,0,1]])

    def test_affine_plane_identity_matches_actual_rimls_derivatives(self):
        xy=np.array([[x,y,0.] for x in np.linspace(-1,1,5) for y in np.linspace(-1,1,5)])
        points=torch.tensor(xy,dtype=torch.float64)
        normals=torch.tensor(np.tile([0.,0.,1.],(len(xy),1)))
        q=torch.tensor([[.1,.2,.0007],[-.7,.4,.15],[.8,-.3,.3]],dtype=torch.float64).ravel()
        fn=lambda x:rimls_values_torch(x.reshape(-1,3),points,normals,1.3)
        np.testing.assert_allclose(fn(q).numpy(),q.reshape(-1,3)[:,2].numpy(),atol=1e-15)
        J=torch.autograd.functional.jacobian(fn,q).numpy()
        expected=np.zeros_like(J);expected[np.arange(3),3*np.arange(3)+2]=1.
        np.testing.assert_allclose(J,expected,atol=2e-14)
        H=torch.autograd.functional.hessian(lambda x:fn(x).sum(),q).numpy()
        np.testing.assert_allclose(H,0.,atol=3e-13)

    def test_vector_barrier_matches_scalar(self):
        s=np.array([1e-7,.0002,.0009,.001,.002])
        b,b1,b2=barrier_arrays(s,.001)
        np.testing.assert_allclose(b,[barrier_value(x,.001) for x in s],rtol=2e-15)
        np.testing.assert_allclose(np.column_stack((b1,b2)),[barrier_first_second(x,.001) for x in s],rtol=2e-15)
        self.assertTrue(np.all(b2>=0))
        for invalid in ([0.],[-1.],[np.nan]):
            with self.assertRaises(ValueError):barrier_arrays(invalid,.001)

    def test_specialized_contact_matches_general_autodiff(self):
        p=build_problem(nxy=5,nz=3)
        inactive=build_problem(nxy=5,nz=3,d_hat=1e-8)
        p['set_indentation'](.02);inactive['set_indentation'](.02)
        state=p['initial'].copy();state[p['contact_dofs']]=-.0015
        base,g0,H0,_=inactive['objective'](state)
        E,g,H,_=p['objective'](state)
        z=torch.tensor(p['reference'][p['mesh']['bottom_nodes'],2])
        ids=torch.tensor(p['contact_dofs'],dtype=torch.long)
        gaps=lambda x:z+x[ids]
        for method in ('gn','full'):
            c=multiple_contact_terms(state,gaps,p['weights']*p['kappa'],d_hat=p['d_hat'],method=method)
            self.assertAlmostEqual(E-base,c['energy'],places=11)
            np.testing.assert_allclose(g-g0,c['gradient'],atol=2e-12)
            np.testing.assert_allclose((H-H0).toarray(),c['hessian'],rtol=1e-13,atol=2e-12)

    def test_cached_plane_direction_is_an_exact_linear_solve(self):
        rng=np.random.default_rng(12);A=rng.normal(size=(14,14))
        K=csr_matrix(A.T@A+2*np.eye(14));ids=np.array([1,5,11])
        direction=make_plane_direction(K,ids);g=rng.normal(size=14)
        for values in ([0.,0.,0.],[10.,0.,1e8],[1.,7.,3.],[0.,0.,0.]):
            diagonal=np.zeros(14);diagonal[ids]=values;H=K+diags(diagonal)
            p,shift=direction(H,g,1e-7)
            np.testing.assert_allclose(p,spsolve(H.tocsc(),-g),rtol=2e-12,atol=2e-13)
            self.assertEqual(shift,0.)
            self.assertLess(g@p,0.)

    def test_newton_direction_hook_and_plane_step_limit(self):
        calls=[]
        def direction(H,g,floor):
            calls.append(1);return np.linalg.solve(H,-g),0.
        obj=lambda x:(float(x@x),2*x,2*np.eye(len(x)),{})
        result=minimize_static(np.ones(2),obj,lambda x:float(x@x),direction_solver=direction)
        np.testing.assert_array_equal(result['state'],0.)
        self.assertEqual(len(calls),1)
        p=build_problem(nxy=5,nz=3);state=p['initial'];trial=state.copy()
        trial[p['contact_dofs']]-=20.
        alpha=p['max_step'](state,trial)
        self.assertGreater(alpha,0.);self.assertLess(alpha,1.)
        self.assertTrue(np.all(p['gaps'](state+alpha*(trial-state))>0))
        self.assertFalse(np.isfinite(p['energy'](trial)))

    def test_stable_energy_gradient_and_full_newton_linearization(self):
        p=build_problem(nxy=9,nz=5);p['set_indentation'](.02)
        x=p['initial'].copy();x[p['contact_dofs']]=-.0015
        E,g,H,_=p['objective'](x)
        self.assertAlmostEqual(E,p['energy'](x),places=13)
        v=np.random.default_rng(4).normal(size=len(x));v/=np.linalg.norm(v)
        h=1e-7
        numerical=(p['energy'](x+h*v)-p['energy'](x-h*v))/(2*h)
        self.assertAlmostEqual(numerical,float(g@v),delta=2e-7*max(1.,abs(g@v)))
        gp=p['objective'](x+h*v)[1];gm=p['objective'](x-h*v)[1]
        np.testing.assert_allclose((gp-gm)/(2*h),H@v,rtol=3e-7,atol=3e-6)

    def test_real_fem_contact_equilibrium(self):
        p=build_problem(nxy=9,nz=5)
        r=solve(p,[.01,.02],verbose=False)
        self.assertEqual(len(r),2)
        for x in r:
            self.assertEqual(x['status'],'converged')
            self.assertLess(x['residual'],1e-7)
            self.assertLess(x['balance_error'],2e-6)
            self.assertGreater(x['gap_min'],0.)
            self.assertGreater(x['min_volume_ratio'],0.)
            self.assertGreater(x['force'],0.)
            self.assertAlmostEqual(float(x['pressure_weights']@x['pressure']),x['force'],places=11)
            self.assertLess(x['top_reaction'],0.)
        self.assertGreater(r[1]['force'],r[0]['force'])
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'results'
            with contextlib.redirect_stdout(io.StringIO()):save_results(p,r,output,{'tol':1e-7})
            meta=json.loads((output/'metadata.json').read_text())
            self.assertEqual(meta['parameters']['nxy'],9)
            self.assertEqual(meta['deltas'],[.01,.02])
            self.assertEqual(len(meta['code_sha256']),6)
            with (output/'summary.csv').open(encoding='utf-8',newline='') as file:
                table=list(csv.DictReader(file))
            self.assertEqual(len(table),2)
            self.assertAlmostEqual(float(table[1]['force']),r[1]['force'])
            with np.load(output/'states.npz') as saved:
                np.testing.assert_array_equal(saved['states'][0],r[0]['state'])
                np.testing.assert_array_equal(saved['nodes'][1],r[1]['nodes'])
            with self.assertRaises(FileExistsError):save_results(p,r,output)



if __name__=='__main__':unittest.main()
