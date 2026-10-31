"""자동 접촉의 희소 보관과 세 절점 보간이 기존 미분을 보존하는지 검사해요."""
import unittest
import numpy as np
import torch

from src.geometry.contact_patches import AutomaticRIMLSContact
from src.geometry.surface import surface_quadrature
from src.simulation.two_body import build_problem


class InterfacePairs:
    """서로 마주 보는 두 접촉면의 모든 면 쌍을 제공해요."""

    def __init__(self, faces, labels, reference):
        """초기 높이가 같은 두 접촉면을 선택해요."""
        centers = reference[faces].mean(axis=1)
        lower = np.flatnonzero((labels[faces[:, 0]] == 0) & np.isclose(centers[:, 2], .4))
        upper_height = reference[labels == 1, 2].min()
        upper = np.flatnonzero((labels[faces[:, 0]] == 1) & np.isclose(centers[:, 2], upper_height))
        self.pairs = np.array([(a, b) for a in lower for b in upper], dtype=int).reshape(-1, 2)

    def face_pairs(self, positions, distance):
        """동일한 후보 집합과 개수를 반환해요."""
        return self.pairs, {'total': len(self.pairs)}


def make_model(n=3):
    """공유 기준 격자 위에 독립적인 자동 접촉 모델을 만들어요."""
    problem = build_problem(n=n)
    reference = np.vstack(problem['reference'])
    offset = len(problem['reference'][0])
    faces = np.vstack((problem['meshes'][0]['boundary_faces'],
                       problem['meshes'][1]['boundary_faces'] + offset))
    labels = np.r_[np.zeros(offset, dtype=int), np.ones(offset, dtype=int)]
    model = AutomaticRIMLSContact(reference, faces, labels, InterfacePairs(faces, labels, reference))
    return model, reference


class PatchStorageChecks(unittest.TestCase):
    """큰 보간 행렬을 만들지 않는 경로와 기존 공개 속성을 검사해요."""

    @classmethod
    def setUpClass(cls):
        """작은 미분 문제의 스레드 비용을 제한해요."""
        torch.set_num_threads(1)

    def test_sparse_storage_and_lazy_legacy_properties(self):
        """희소 보관량과 명시적인 조밀 속성 접근을 검사해요."""
        model, reference = make_model(n=5)
        self.assertIsNone(model._dense_S)
        model.refresh(reference)
        self.assertIsNone(model._dense_blocks)
        self.assertEqual(model._sparse_S.nnz, 9 * len(model.faces))
        compact_bytes = sum(x.nbytes for x in (model._sparse_S.data,
                            model._sparse_S.indices, model._sparse_S.indptr))
        dense_bytes = len(model.faces) * 3 * len(reference) * 8
        self.assertLess(compact_bytes, dense_bytes / 20)
        for _, vertices, shape in model._query_blocks:
            self.assertEqual(tuple(vertices.shape), tuple(shape.shape))
            self.assertEqual(vertices.shape[1], 3)
        expected, _ = surface_quadrature(reference, model.faces)
        np.testing.assert_array_equal(model.S, expected)
        self.assertIs(model.S, model._dense_S)
        for gid, matrix in model.blocks:
            ids = [query for query, group in model.signature if group == gid]
            np.testing.assert_array_equal(matrix.numpy(), expected[ids])
        self.assertIs(model.blocks, model._dense_blocks)

    def test_compact_queries_match_dense_values_and_derivatives(self):
        """움직이는 양쪽 표면의 간격과 일차 및 이차 미분을 비교해요."""
        compact, reference = make_model(n=4)
        dense, _ = make_model(n=4)
        dense.S = surface_quadrature(reference, dense.faces)[0]
        compact.refresh(reference)
        dense.refresh(reference)
        self.assertEqual(compact.signature, dense.signature)
        state = torch.tensor(reference + np.random.default_rng(14).normal(
            scale=2e-4, size=reference.shape), dtype=torch.float64).ravel()
        indices = [7, 1, len(compact.weights) - 1, 0, 7]
        first = lambda x: compact.gaps(x, indices=indices)
        second = lambda x: dense.gaps(x, indices=indices)
        np.testing.assert_allclose(first(state), second(state), rtol=2e-13, atol=1e-14)
        np.testing.assert_allclose(torch.autograd.functional.jacobian(first, state),
                                   torch.autograd.functional.jacobian(second, state), rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(torch.autograd.functional.hessian(lambda x: first(x).square().sum(), state),
                                   torch.autograd.functional.hessian(lambda x: second(x).square().sum(), state),
                                   rtol=2e-10, atol=2e-12)
        self.assertIsNone(compact._dense_S)
        self.assertIsNone(compact._dense_blocks)

    def test_small_dense_path_matches_compact_queries(self):
        """작은 격자의 조밀 경로가 세 절점 보간과 같은 간격을 내는지 검사해요."""
        dense, reference = make_model(n=3)
        compact, _ = make_model(n=3)
        self.assertIsNotNone(dense._dense_S)
        self.assertLessEqual(dense._dense_S.nbytes, 128 * 1024)
        compact._dense_S = None
        dense.refresh(reference)
        compact.refresh(reference)
        self.assertIsNotNone(dense._dense_blocks)
        self.assertIsNone(compact._dense_blocks)
        self.assertEqual(dense.signature, compact.signature)
        state = torch.tensor(reference + np.random.default_rng(23).normal(
            scale=2e-4, size=reference.shape), dtype=torch.float64).ravel()
        np.testing.assert_allclose(dense.gaps(state), compact.gaps(state), rtol=2e-13, atol=1e-14)
        indices = [7, 1, len(dense.weights) - 1, 0, 7]
        np.testing.assert_allclose(dense.gaps(state, indices=indices),
                                   compact.gaps(state, indices=indices), rtol=2e-13, atol=1e-14)

    def test_materialized_block_mutation_and_patch_release(self):
        """공개 블록을 수정한 동작과 접촉 해제 후 재생성을 검사해요."""
        model, reference = make_model(n=2)
        model.refresh(reference)
        state = torch.tensor(reference, dtype=torch.float64).ravel()
        original = model.gaps(state).clone()
        blocks = model.blocks
        blocks[0][1][:] = blocks[0][1].flip(0)
        changed = model.gaps(state)
        expected = torch.cat((original[:len(blocks[0][1])].flip(0), original[len(blocks[0][1]):]))
        np.testing.assert_allclose(changed, expected, atol=1e-14)
        model.collider.pairs = np.empty((0, 2), dtype=int)
        model.refresh(reference)
        self.assertEqual(model.signature, ())
        self.assertEqual(tuple(model.gaps(state).shape), (0,))
        self.assertEqual(model.blocks, [])


if __name__ == '__main__':
    unittest.main()
