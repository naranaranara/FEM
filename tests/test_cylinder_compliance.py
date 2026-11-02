"""##r## Checks for prescribed projected Hertz pressure, independent of contact."""
import unittest
import numpy as np
from src.simulation.test_cylinder_compliance import hertz_quadrature, pressure_load


class PressureLoadChecks(unittest.TestCase):
    def setUp(self):
        self.reference = dict(major_radius=.3, minor_radius=.1, peak_pressure=2.)
        self.reference['force'] = 2*np.pi*.3*.1*2/3
        a = .4
        self.directions = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        self.nodes = np.array([[-1., -1., 0.], [1., -1., .1],
                               [1., 1., .2], [-1., 1., .1]])
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])

    def test_exact_total_pressure_and_zero_first_moments(self):
        xy, weights = hertz_quadrature(self.reference, self.directions, 12)
        self.assertAlmostEqual(weights.sum(), self.reference['force'], places=14)
        np.testing.assert_allclose(weights @ xy, 0., atol=1e-15)

    def test_nodal_transfer_preserves_force_and_linear_virtual_work(self):
        f = pressure_load(self.nodes, self.faces, self.reference, self.directions, 32, -1.).reshape(-1, 3)
        self.assertAlmostEqual(f[:, 2].sum(), -self.reference['force'], places=14)
        np.testing.assert_allclose(np.cross(self.nodes, f).sum(0), 0., atol=1e-14)
        # A linear virtual vertical displacement is reproduced by triangle N_i.
        virtual = 2.+3*self.nodes[:, 0]-4*self.nodes[:, 1]
        self.assertAlmostEqual(virtual @ f[:, 2], -2*self.reference['force'], places=13)
        opposite = pressure_load(self.nodes, self.faces, self.reference, self.directions, 32, 1.)
        np.testing.assert_allclose(opposite, -f.ravel(), atol=1e-15)

    def test_uncovered_ellipse_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not covered'):
            pressure_load(self.nodes, self.faces[:1], self.reference, self.directions, 16, -1.)


# ##r## Check that grading changes only interior ring positions.
class RadialGradingChecks(unittest.TestCase):
    def test_surface_volume_and_default_are_preserved(self):
        from src.geometry.mesh import make_cylinder
        from src.mechanics.solid import tetrahedron_volumes
        for facing, angle in (('up', 0.), ('down', 45.)):
            args = dict(n_radial=4, facing=facing, angle_degrees=angle, clearance=.03)
            uniform = make_cylinder(**args)
            explicit = make_cylinder(radial_grading=1., **args)
            graded = make_cylinder(radial_grading=3., **args)
            np.testing.assert_array_equal(uniform['nodes'], explicit['nodes'])
            for key in ('tetrahedra', 'side_faces', 'contact_faces', 'query_faces', 'support_nodes'):
                np.testing.assert_array_equal(uniform[key], graded[key])
            surface = np.unique(uniform['side_faces'])
            np.testing.assert_array_equal(uniform['nodes'][surface], graded['nodes'][surface])
            v = tetrahedron_volumes(graded['nodes'], graded['tetrahedra'])
            self.assertTrue(np.all(v > 0))
            self.assertAlmostEqual(v.sum(), tetrahedron_volumes(uniform['nodes'], uniform['tetrahedra']).sum(), places=10)
            # Local contact pole has radius r along z from cylinder center.
            center_z = -3. if facing == 'up' else 3.03
            ids = 1+np.arange(4)*32
            r = np.abs(graded['local_nodes'][ids, 2]-center_z)
            thickness = np.diff(np.r_[0., r])
            self.assertTrue(np.all(np.diff(thickness) < 0))
            self.assertAlmostEqual(thickness[-1], 3./4**3)

    def test_invalid_grading(self):
        from src.geometry.mesh import make_cylinder
        for value in (0., .5, 7., float('nan')):
            with self.assertRaises(ValueError):
                make_cylinder(radial_grading=value)


if __name__ == '__main__':
    unittest.main()
