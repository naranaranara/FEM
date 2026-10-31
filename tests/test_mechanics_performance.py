"""묶음 조립과 표면 보간이 기존 기하 및 탄성 미분을 보존하는지 검사해요."""
import unittest
from itertools import permutations
import numpy as np
from src.geometry.mesh import make_block, make_spherical_cap
from src.geometry.surface import surface_quadrature, make_surface_sampler
from src.mechanics.solid import assemble_stiffness, material_matrix, tetrahedron_volumes


def reference_topology(nodes, nx, ny, nz):
    """기존 순회 방식으로 사면체와 경계면의 기준 순서를 만들어요."""
    tets = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                for order in permutations(range(3)):
                    corner = np.array([i, j, k])
                    tet = [int(corner @ np.array([1, nx, nx * ny]))]
                    for axis in order:
                        corner[axis] += 1
                        tet.append(int(corner @ np.array([1, nx, nx * ny])))
                    xyz = nodes[tet]
                    if np.linalg.det(xyz[1:] - xyz[0]) < 0:
                        tet[1], tet[2] = tet[2], tet[1]
                    tets.append(tet)
    faces = {}
    for tet in tets:
        for opposite in range(4):
            face = [tet[a] for a in range(4) if a != opposite]
            a, b, c = nodes[face]
            if np.dot(np.cross(b - a, c - a), nodes[tet[opposite]] - a) > 0:
                face[1], face[2] = face[2], face[1]
            faces.setdefault(tuple(sorted(face)), []).append(face)
    return np.asarray(tets), np.asarray([f[0] for f in faces.values() if len(f) == 1])


def reference_stiffness(nodes, tetrahedra, young, poisson):
    r"""개별 요소의 형상함수에서 기준 강성 행렬을 조립해요.

    $$K=\sum_eP_e^TV_eB_e^TDB_eP_e$$
    """
    D = material_matrix(young, poisson)
    K = np.zeros((nodes.size, nodes.size))
    for tet in tetrahedra:
        A = np.column_stack((np.ones(4), nodes[tet]))
        # 형상함수의 기울기를 구해요: $$G=(A^{-1})_{1:,:}^T$$
        gradients = np.linalg.inv(A)[1:, :].T
        B = np.zeros((6, 12))
        for i, (dx, dy, dz) in enumerate(gradients):
            B[:, 3*i:3*i+3] = [[dx, 0., 0.], [0., dy, 0.], [0., 0., dz],
                                [dy, dx, 0.], [0., dz, dy], [dz, 0., dx]]
        # 절점 번호를 자유도로 변환해요: $$d_{ia}=3t_i+a$$
        dofs = (3 * tet[:, None] + np.arange(3)).ravel()
        # 요소 부피를 계산해요: $$V=\det(A)/6$$
        volume = np.linalg.det(A) / 6
        # 요소 강성을 전역 행렬에 더해요: $$K\mathrel{+}=P_e^TV_eB_e^TDB_eP_e$$
        K[np.ix_(dofs, dofs)] += volume * (B.T @ D @ B)
    return K


class MechanicsEquivalenceChecks(unittest.TestCase):
    """배열화 전후의 연결 순서와 물리적 성질을 검사해요."""

    def test_topology_order_and_outward_normals(self):
        """직육면체 격자의 연결과 경계면 순서가 정확히 같은지 검사해요."""
        for nx, ny, nz in ((2, 2, 2), (3, 4, 2), (4, 2, 3)):
            mesh = make_block(width=2.3, depth=.8, height=1.7, nx=nx, ny=ny, nz=nz)
            tets, faces = reference_topology(mesh['nodes'], nx, ny, nz)
            np.testing.assert_array_equal(mesh['tetrahedra'], tets)
            np.testing.assert_array_equal(mesh['boundary_faces'], faces)
            self.assertTrue(np.all(tetrahedron_volumes(mesh['nodes'], tets) > 0))

    def test_dense_sparse_assembly_and_rigid_motions(self):
        """변형된 격자의 조립과 강체 운동에 대한 영에너지를 검사해요."""
        mesh = make_block(nx=3, ny=3, nz=2)
        nodes = mesh['nodes'].copy()
        nodes += np.random.default_rng(92).normal(scale=.01, size=nodes.shape)
        reference = reference_stiffness(nodes, mesh['tetrahedra'], 43., .27)
        for sparse in (False, True):
            actual = assemble_stiffness(nodes, mesh['tetrahedra'], 43., .27, sparse=sparse)
            actual = actual.toarray() if sparse else actual
            np.testing.assert_allclose(actual, reference, rtol=2e-14, atol=1e-13)
            np.testing.assert_allclose(actual, actual.T, atol=1e-13)
            for direction in np.eye(3):
                translation = np.tile(direction, len(nodes))
                rotation = np.cross(direction, nodes).ravel()
                np.testing.assert_allclose(actual @ translation, 0., atol=2e-13)
                np.testing.assert_allclose(actual @ rotation, 0., atol=2e-13)

    def test_assembly_batches_and_invalid_elements(self):
        """배치 경계를 넘는 조립과 빈 입력 및 뒤집힌 요소를 검사해요."""
        mesh = make_block(nx=10, ny=10, nz=10)
        nodes, tets = mesh['nodes'], mesh['tetrahedra']
        total = assemble_stiffness(nodes, tets, sparse=True)
        split = assemble_stiffness(nodes, tets[:4096], sparse=True)
        split += assemble_stiffness(nodes, tets[4096:], sparse=True)
        difference = total - split
        self.assertLess(np.max(np.abs(difference.data), initial=0.), 1e-13)
        for sparse in (False, True):
            empty = assemble_stiffness(nodes[:4], [], sparse=sparse)
            self.assertEqual(empty.shape, (12, 12))
            self.assertEqual(empty.nnz if sparse else np.count_nonzero(empty), 0)
            with self.assertRaises(ValueError):
                assemble_stiffness(nodes, tets[:1, [0, 2, 1, 3]], sparse=sparse)

    def test_curved_graded_mesh_stiffness(self):
        """곡면 등급 격자에서도 개별 요소 조립과 일치하는지 검사해요."""
        mesh = make_spherical_cap(nxy=5, nz=3)
        expected = reference_stiffness(mesh['nodes'], mesh['tetrahedra'], 1000., .3)
        actual = assemble_stiffness(mesh['nodes'], mesh['tetrahedra'], 1000., .3, sparse=True)
        np.testing.assert_allclose(actual.toarray(), expected, rtol=3e-14, atol=2e-10)

    def test_surface_quadrature_and_differentiable_gather(self):
        """표면 보간의 값과 일차 및 이차 미분이 조밀 보간과 같은지 검사해요."""
        import torch
        mesh = make_block(nx=3, ny=2, nz=2)
        faces, nodes = mesh['top_faces'], mesh['nodes']
        sampler, S, weights = make_surface_sampler(nodes, faces)
        S0, weights0 = surface_quadrature(nodes, faces)
        sparse_sampler, Ss, ws = make_surface_sampler(nodes, faces, sparse=True)
        np.testing.assert_array_equal(Ss.toarray(), S)
        np.testing.assert_array_equal(ws, weights)
        self.assertEqual(Ss.nnz, 9 * len(faces))
        np.testing.assert_array_equal(S, S0)
        np.testing.assert_array_equal(weights, weights0)
        self.assertAlmostEqual(weights.sum(), 2.)
        rng = np.random.default_rng(31)
        x = torch.tensor(nodes + rng.normal(scale=.03, size=nodes.shape), dtype=torch.float64).ravel()
        St = torch.tensor(S, dtype=torch.float64)
        ft = torch.tensor(faces, dtype=torch.long)
        used = torch.tensor(np.unique(faces), dtype=torch.long)

        def dense(state):
            """조밀 보간 행렬을 사용하는 기준 표면 값을 계산해요."""
            vertices = state.reshape(-1, 3)
            xyz = vertices[ft]
            area_normals = torch.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0], dim=1)
            normals = torch.zeros_like(vertices)
            for corner in range(3):
                normals = normals.index_add(0, ft[:, corner], area_normals)
            local = normals[used]
            local = local / torch.linalg.norm(local, dim=1)[:, None]
            normals = normals.index_copy(0, used, local)
            interpolated = St @ normals
            interpolated = interpolated / torch.linalg.norm(interpolated, dim=1)[:, None]
            return torch.cat(((St @ vertices).ravel(), interpolated.ravel()))

        def gathered(state):
            """세 절점 보간을 사용하는 표면 값을 계산해요."""
            position, normal = sampler(state.reshape(-1, 3))
            return torch.cat((position.ravel(), normal.ravel()))

        np.testing.assert_allclose(gathered(x), dense(x), rtol=1e-14, atol=1e-14)
        for actual, expected in zip(sparse_sampler(x.reshape(-1, 3)), sampler(x.reshape(-1, 3))):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(torch.autograd.functional.jacobian(gathered, x),
                                   torch.autograd.functional.jacobian(dense, x), rtol=2e-13, atol=2e-13)
        coefficients = torch.tensor(rng.normal(size=dense(x).numel()), dtype=torch.float64)
        np.testing.assert_allclose(torch.autograd.functional.hessian(lambda z: gathered(z) @ coefficients, x),
                                   torch.autograd.functional.hessian(lambda z: dense(z) @ coefficients, x),
                                   rtol=2e-12, atol=2e-12)


if __name__ == '__main__':
    unittest.main()
