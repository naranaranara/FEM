"""##r## Conformity and geometry checks for local cylinder refinement."""
import unittest
import numpy as np
from src.geometry.mesh import make_cylinder
from src.geometry.local_refinement import refine_cylinder_contact
from src.mechanics.solid import tetrahedron_volumes


def face_keys(faces):
    return set(map(tuple, np.sort(faces, axis=1)))


class LocalRefinementChecks(unittest.TestCase):
    def test_conforming_positive_geometry_and_fixed_ends(self):
        for facing, angle in (('up', 0.), ('down', 45.)):
            old = make_cylinder(n_angle=32, n_axis=13, n_radial=4,
                                radial_grading=3., facing=facing,
                                angle_degrees=angle, clearance=.03)
            new = refine_cylinder_contact(old, .3, half_width=.2, depth=.15)
            v0 = tetrahedron_volumes(old['nodes'], old['tetrahedra'])
            v1 = tetrahedron_volumes(new['nodes'], new['tetrahedra'])
            self.assertTrue(np.all(v1 > 0))
            self.assertAlmostEqual(v0.sum(), v1.sum(), places=10)
            np.testing.assert_array_equal(old['nodes'], new['nodes'][:len(old['nodes'])])
            np.testing.assert_array_equal(old['support_nodes'], new['support_nodes'])
            np.testing.assert_array_equal(old['end_faces'], new['end_faces'])
            # Exact volume-boundary agreement detects cracks / hanging faces.
            faces = new['tetrahedra'][:, [[1,2,3],[0,3,2],[0,1,3],[0,2,1]]].reshape(-1,3)
            unique, count = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
            self.assertTrue(np.all((count == 1) | (count == 2)))
            self.assertEqual(face_keys(unique[count == 1]), face_keys(new['boundary_faces']))
            self.assertTrue(face_keys(new['query_faces']) <= face_keys(new['contact_faces']))
            self.assertTrue(face_keys(new['contact_faces']) <= face_keys(new['side_faces']))
            self.assertEqual(face_keys(new['boundary_faces']),
                             face_keys(new['side_faces']) | face_keys(new['end_faces']))
            def area(m, key):
                xyz=m['nodes'][m[key]]
                return np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1).sum()/2
            for key in ('side_faces','query_faces','contact_faces'):
                self.assertAlmostEqual(area(old,key),area(new,key),places=10)
            self.assertLessEqual(new['local_refinement']['max_local_edge'],.3*(1+1e-12))
            self.assertTrue(set(map(tuple,old['tetrahedra'])) & set(map(tuple,new['tetrahedra'])))

    def test_noop_and_invalid_size(self):
        m=make_cylinder(n_radial=2)
        n=refine_cylinder_contact(m,100.)
        np.testing.assert_array_equal(m['nodes'],n['nodes'])
        np.testing.assert_array_equal(m['tetrahedra'],n['tetrahedra'])
        for h in (0.,-1.,np.nan):
            with self.assertRaises(ValueError):refine_cylinder_contact(m,h)


if __name__=='__main__':unittest.main()
