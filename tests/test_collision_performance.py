"""면 후보의 배열 확장이 기존 보수적 인접 집합과 일치하는지 검사해요."""
import unittest
import numpy as np
from src.collision.ipc_toolkit import IPCMeshCollision, _pack_incidence


class CollisionIncidenceChecks(unittest.TestCase):
    """빈 인접 집합과 중복 후보까지 동일한 면 쌍을 만드는지 확인해요."""

    def make_collider(self, count):
        """다양한 차수와 물체 번호를 가진 충돌 후보를 만들어요."""
        rng = np.random.default_rng(91)
        collider = IPCMeshCollision.__new__(IPCMeshCollision)
        collider.faces = np.arange(300).reshape(-1, 3)
        collider.body_ids = np.repeat(np.arange(100) % 3, 3)
        collider.vertex_faces = [set(rng.integers(100, size=i % 8)) for i in range(300)]
        collider.edge_faces = [set(rng.integers(100, size=i % 4)) for i in range(150)]
        collider._vertex_incidence = _pack_incidence(collider.vertex_faces)
        collider._edge_incidence = _pack_incidence(collider.edge_faces)
        fv = np.column_stack((rng.integers(100, size=count), rng.integers(300, size=count)))
        ee = rng.integers(150, size=(count, 2))
        data = dict(face_vertex_pairs=np.repeat(fv, 2, axis=0),
                    edge_edge_pairs=np.repeat(ee, 2, axis=0), total=4 * count)
        collider.candidate_topology = lambda *args: data
        return collider, data

    def test_ragged_expansion_matches_conservative_sets(self):
        """작은 경로와 여러 배치 경로가 동일한 정렬 결과를 내는지 검사해요."""
        for count in (0, 3, 40, 2100):
            with self.subTest(count=count):
                collider, data = self.make_collider(count)
                expected = set()
                candidates = []
                for face, vertex in data['face_vertex_pairs']:
                    candidates.extend((face, adjacent) for adjacent in collider.vertex_faces[vertex])
                for first, second in data['edge_edge_pairs']:
                    candidates.extend((a, b) for a in collider.edge_faces[first]
                                      for b in collider.edge_faces[second])
                for a, b in candidates:
                    if collider.body_ids[collider.faces[a, 0]] != collider.body_ids[collider.faces[b, 0]]:
                        expected.add(tuple(sorted((int(a), int(b)))))
                actual, returned = collider.face_pairs(None, .2)
                np.testing.assert_array_equal(actual, np.asarray(sorted(expected), dtype=int).reshape(-1, 2))
                self.assertIs(returned, data)

    def test_empty_incidence_and_same_body_candidates(self):
        """확장 결과가 없어도 빈 배열의 형상과 자료형을 유지하는지 검사해요."""
        collider, _ = self.make_collider(40)
        collider.body_ids[:] = 0
        pairs, _ = collider.face_pairs(None, .2)
        self.assertEqual(pairs.shape, (0, 2))
        self.assertEqual(pairs.dtype, np.dtype(int))


if __name__ == '__main__':
    unittest.main()
