import unittest
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from src.contact.objective import ContactObjective
from src.optimization.structured_bfgs import structured_bfgs_update, solve
from src.optimization.baseline import solve as baseline_solve


class StructuredBFGSChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.pool = threadpool_limits(1)

    @classmethod
    def tearDownClass(cls):
        cls.pool.restore_original_limits()
        torch.set_num_threads(cls.old_threads)

    def test_changing_known_hessian_and_signed_missing_secant(self):
        A = np.array([[.3, .1], [.1, -.2]])
        G_new = np.array([[3., .2], [.2, 2.]])
        d = np.array([.3, -.4])
        C = np.array([[-1., .1], [.1, .2]])
        z = C @ d
        updated, info = structured_bfgs_update(A, G_new, d, z)
        self.assertTrue(info['update_applied'])
        np.testing.assert_allclose(updated @ d, z, atol=1e-14)
        np.testing.assert_allclose(updated, updated.T, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(G_new + updated)[0], 0.)
        self.assertLess(d @ updated @ d, 0.)

    def test_matches_standard_bfgs_on_total_with_constant_known_part(self):
        G = np.diag([3., 4.])
        A = np.eye(2)
        d = np.array([1., .2])
        y = np.array([2., 1.])
        updated, _ = structured_bfgs_update(A, G, d, y - G @ d)
        M = G + A
        Md = M @ d
        expected = M - np.outer(Md, Md) / (d @ Md) + np.outer(y, y) / (d @ y)
        np.testing.assert_allclose(G + updated, expected)

    def test_skip_bad_curvature_without_changing_model(self):
        A = np.eye(2)
        for d, z in [(np.zeros(2), np.zeros(2)), (np.ones(2), -3 * np.ones(2))]:
            updated, info = structured_bfgs_update(A, np.eye(2), d, z)
            self.assertFalse(info['update_applied'])
            np.testing.assert_array_equal(updated, A)

    def test_indefinite_total_seed_is_stabilized_but_secant_preserved(self):
        G = np.eye(2)
        A = np.diag([-2., 0.])
        d = np.array([1., 1.])
        z = np.zeros(2)
        updated, info = structured_bfgs_update(A, G, d, z)
        self.assertTrue(info['update_applied'])
        self.assertGreater(info['seed_shift'], 0.)
        np.testing.assert_allclose(updated @ d, z, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(G + updated)[0], 0.)

    @staticmethod
    def objective(curved):
        initial = np.array([.2, .12])
        target = np.array([0., -.05])
        def gap(x):
            return x[1] + (.1 * x[0] ** 2 if curved else 0. * x[0])
        def base(x):
            d = x - target
            return 1.5 * d @ d, 3. * d, 3. * np.eye(2)
        return ContactObjective(initial, gap, base)

    def test_contact_equilibrium_and_no_exact_gap_hessian(self):
        objective = self.objective(True)
        reference = baseline_solve(objective, 'full')
        self.assertEqual(reference['status'], 'converged')
        for mode in ('total', 'gap'):
            r = solve(objective, secant=mode)
            self.assertEqual(r['status'], 'converged')
            np.testing.assert_allclose(r['state'], reference['state'], atol=1e-8)
            self.assertEqual(r['counts']['gap_hessian_calls'], 0)
            self.assertGreater(r['bfgs_updates'], 0)
            self.assertTrue(all(t.get('directional_slope', -1.) < 0 for t in r['trace']))
            self.assertTrue(all(b['energy'] <= a['energy'] + 3e-14 for a, b in zip(r['trace'], r['trace'][1:])))

    def test_affine_gap_geometry_secant_does_not_invent_curvature(self):
        objective = self.objective(False)
        gn = baseline_solve(objective, 'gn')
        result = solve(objective, secant='gap')
        self.assertEqual(result['status'], 'converged')
        self.assertEqual(result['updates'], gn['updates'])
        np.testing.assert_allclose(result['state'], gn['state'], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
