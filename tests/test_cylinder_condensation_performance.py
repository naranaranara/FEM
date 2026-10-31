"""축약 접촉의 조밀 곡률 전달이 기존 희소 경로와 같은 상태와 결과를 유지하는지 검사해요."""
import unittest

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.mechanics.condensation import condense_contact_problem
from src.optimization.structured_bfgs import MultiGapStructuredBFGS
from src.simulation.two_body import build_problem


class CondensedCurvatureChecks(unittest.TestCase):
    """보정 행렬의 표현과 사용자 정의 곡률 모델의 호환성을 검사해요."""

    @classmethod
    def setUpClass(cls):
        """작은 수치 검사를 단일 스레드에서 실행해요."""
        cls.old_threads=torch.get_num_threads()
        torch.set_num_threads(1)
        cls.pool=threadpool_limits(1)

    @classmethod
    def tearDownClass(cls):
        """검사 전 스레드 설정을 복구해요."""
        cls.pool.restore_original_limits()
        torch.set_num_threads(cls.old_threads)

    def test_dense_sparse_layout_reset_and_buffer_independence(self):
        """전체 및 순서가 섞인 접촉 자유도에서 보정과 재설정 및 배열 소유권을 보존해요."""
        rng=np.random.default_rng(613)
        n=6
        G=np.diag(np.arange(n)+2.)
        for ids in (np.arange(n),np.array([4,0,2])):
            dense=MultiGapStructuredBFGS()
            sparse=MultiGapStructuredBFGS()
            initial_J=rng.normal(scale=.01,size=(2,len(ids)))
            for step in range(4):
                indices=np.arange(2) if step<3 else np.empty(0,dtype=int)
                J=(initial_J+step*.001) if len(indices) else np.empty((0,len(ids)))
                contact=dict(gap_dofs=ids,gap_indices=indices,gap_jacobian=J.copy(),
                             gap_weights=np.array([.7,1.2]),gaps=np.array([.1,.13]))
                state=step*.01*np.arange(n)
                actual,info=dense.correction_dense(state,contact,G,.2)
                expected,reference=sparse.correction(state,contact,G,.2)
                np.testing.assert_array_equal(actual,expected.toarray())
                self.assertEqual(info,reference)
                self.assertEqual(dense.totals,sparse.totals)
                if dense.A is not None:
                    matrix=dense.A.copy()
                    previous_J=dense.previous['J'].copy()
                    actual.fill(123.)
                    contact['gap_jacobian'].fill(456.)
                    np.testing.assert_array_equal(dense.A,matrix)
                    np.testing.assert_array_equal(dense.previous['J'],previous_J)
            self.assertIsNone(dense.A)
            self.assertIsNone(dense.previous)
            self.assertEqual(dense.totals['bfgs_resets'],1)

    def test_condensed_objective_keeps_sparse_only_model_compatibility(self):
        """조밀 반환을 지원하지 않는 기존 곡률 모델도 같은 축약 목적함수를 계산해요."""
        class SparseOnly:
            """기존 희소 보정 인터페이스만 제공하는 사용자 정의 모델을 흉내 내요."""

            correction_dense=None

            def __init__(self):
                """기존 Structured BFGS 모델을 보관해요."""
                self.model=MultiGapStructuredBFGS()

            def correction(self,*args):
                """기존 호출 규약대로 희소 보정을 반환해요."""
                return self.model.correction(*args)

        class LegacySubclass(MultiGapStructuredBFGS):
            """조밀 반환 도입 이전의 인자 목록으로 보정을 재정의해요."""

            def __init__(self):
                """기본 모델과 재정의된 메서드의 호출 횟수를 준비해요."""
                super().__init__()
                self.calls=0

            def correction(self,state,contact,G,d_hat):
                """추가 키워드 없이 기존 보정 호출을 관찰해요."""
                self.calls+=1
                return super().correction(state,contact,G,d_hat)

        problem=build_problem(n=2,sparse=True)
        condensed=condense_contact_problem(problem)
        condensed['set_closure'](.025)
        dense=MultiGapStructuredBFGS()
        legacy=SparseOnly()
        subclass=LegacySubclass()
        rng=np.random.default_rng(807)
        for _ in range(3):
            q=condensed['initial']+rng.normal(scale=1e-4,size=len(condensed['initial']))
            actual=condensed['objective'](q,'structured_bfgs_gap',dense)
            for model in (legacy,subclass):
                expected=condensed['objective'](q,'structured_bfgs_gap',model)
                self.assertEqual(actual[0],expected[0])
                np.testing.assert_array_equal(actual[1],expected[1])
                np.testing.assert_array_equal(actual[2],expected[2])
                self.assertEqual(actual[3],expected[3])
        self.assertEqual(subclass.calls,3)


if __name__=='__main__':
    unittest.main()
