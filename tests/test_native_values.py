"""값 전용 C++ RIMLS 계산의 수치 일치와 미분·오류 처리의 기존 경로를 검사해요."""
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.geometry import native_rimls, rimls


def _cloud(seed=712, queries=257, points=256):
    """겹치는 지지 영역 안에 작은 곡면 점군과 질의점을 만들어요."""
    rng=np.random.default_rng(seed)
    q=rng.normal(scale=.035,size=(queries,3))
    q[:,2]+=.13
    p=rng.normal(scale=.12,size=(points,3))
    p[:,2]*=.1
    n=rng.normal(scale=.025,size=(points,3))
    n[:,2]+=1.
    return tuple(torch.tensor(value,dtype=torch.float64) for value in (q,p,n))


def _outcome(function,arrays,scales=(1.2,.5,.5)):
    """기준 계산과 비교할 값 또는 예외 종류와 메시지를 받아요."""
    try:
        return function(*arrays,*scales),None
    except (ValueError,RuntimeError,OverflowError) as error:
        return None,(type(error),str(error))


@unittest.skipUnless(native_rimls.available() and hasattr(native_rimls._rimls_native,'values'),
                     '선택적인 값 전용 C++ 확장이 없어요.')
class NativeValueChecks(unittest.TestCase):
    """네이티브 값 계산이 기준 수식과 기존 입력 지원 범위를 보존하는지 검사해요."""

    def setUp(self):
        """독립적인 작은 점군과 단일 계산 스레드를 준비해요."""
        torch.set_num_threads(1)
        self.arrays=_cloud()

    def assert_reference_outcome(self,arrays,scales=(1.2,.5,.5)):
        """계산값 또는 예외가 일반 텐서 식과 일치하는지 확인해요."""
        with torch.no_grad():
            actual,error=_outcome(rimls.rimls_values_torch,arrays,scales)
            expected,reference_error=_outcome(rimls._rimls_values_reference,arrays,scales)
        self.assertEqual(error,reference_error)
        if error is None:torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-14,equal_nan=True)

    def test_curved_and_planar_values_varied_scales_and_noncontiguous_inputs(self):
        """평면과 곡면 및 비연속 입력에서 스칼라 척도별 간격을 비교해요."""
        with torch.no_grad():
            for scales in ((1.2,.5,.5),(.9,.23,.7),(2.,.8,.3)):
                with self.subTest(scales=scales):
                    actual=native_rimls.try_native_values(*self.arrays,*scales)
                    self.assertIsNotNone(actual)
                    expected=rimls._rimls_values_reference(*self.arrays,*scales)
                    torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-14)
            q,p,n=(value.clone() for value in self.arrays)
            p[:,2]=0.
            n[:]=torch.tensor([0.,0.,1.])
            q[:,2]=.13
            q=torch.stack((q,q),dim=-1)[:,:,0]
            self.assertFalse(q.is_contiguous())
            actual=native_rimls.try_native_values(q,p,n,1.2,.5,.5)
            torch.testing.assert_close(actual,torch.full((len(q),),.13,dtype=torch.float64),rtol=1e-13,atol=1e-14)

    def test_near_support_and_robust_weight_thresholds_use_reference(self):
        r"""지지 영역과 강건 가중치의 소멸 경계에서 기준 텐서 판정을 사용해요.

        $$D=P t^4,\qquad Z=P\exp(-1/\sigma_n^2)$$
        """
        count=256
        p=torch.zeros((count,3),dtype=torch.float64)
        n=p.clone()
        n[:,2]=1.
        for ratio in (1.-1e-9,1.,1.+1e-9):
            # 지정한 지지 가중치 합에 맞는 커널 밑을 구해요: $$t=(D/P)^{1/4}$$
            t=(1e-14*ratio/count)**.25
            q=torch.tensor([[np.sqrt(1.-t),0.,0.]],dtype=torch.float64).repeat(257,1)
            with torch.no_grad():
                self.assertIsNone(native_rimls.try_native_values(q,p,n,1.,.5,.5))
            self.assert_reference_outcome((q,p,n),(1.,.5,.5))
        q=torch.zeros((257,3),dtype=torch.float64)
        n[::2,2]=-1.
        for ratio in (1.-1e-9,1.,1.+1e-9):
            # 지정한 강건 가중치 합에 맞는 척도를 구해요: $$\sigma_n=1/\sqrt{\log(P/Z)}$$
            sigma_n=np.sqrt(1./np.log(count/(1e-14*ratio)))
            with torch.no_grad():
                self.assertIsNone(native_rimls.try_native_values(q,p,n,1.,.5,sigma_n))
            self.assert_reference_outcome((q,p,n),(1.,.5,sigma_n))

    def test_errors_nonfinite_values_and_empty_queries_preserve_reference(self):
        """영역 밖 질의와 비정상 값 및 빈 입력의 기존 결과를 보존해요."""
        q,p,n=(value.clone() for value in self.arrays)
        self.assert_reference_outcome((q+10.,p,n))
        self.assert_reference_outcome((q[:0],p,n))
        self.assert_reference_outcome((q,p[:0],n[:0]))
        n[0,0]=float('nan')
        self.assert_reference_outcome((q,p,n))
        for scales in ((float('inf'),.5,.5),(float('nan'),.5,.5),(1.2,float('nan'),.5),(1e200,.5,.5)):
            self.assert_reference_outcome(self.arrays,scales)
        for scales in ((0.,.5,.5),(1.,0.,.5),(1.,.5,-1.)):
            self.assert_reference_outcome(self.arrays,scales)

    def test_float32_tensor_scales_and_missing_extension_preserve_fallback(self):
        """자료형과 텐서 척도 및 구형 확장에서 기존 텐서 경로를 유지해요."""
        with torch.no_grad(),patch.object(native_rimls._rimls_native,'values',side_effect=AssertionError('unsupported native route')):
            q,p,n=(value.float() for value in self.arrays)
            actual=rimls.rimls_values_torch(q,p,n,1.2)
            expected=rimls._rimls_values_reference(q,p,n,1.2)
            torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-7)
            R=torch.tensor(1.2,dtype=torch.float64)
            actual=rimls.rimls_values_torch(*self.arrays,R)
            expected=rimls._rimls_values_reference(*self.arrays,R)
            torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-14)
        if hasattr(torch,'_neg_view'):
            negative=torch._neg_view(self.arrays[0])
            with patch.object(native_rimls._rimls_native,'values',side_effect=AssertionError('unsupported array view')):
                self.assert_reference_outcome((negative,*self.arrays[1:]))
        for extension in (None,types.SimpleNamespace()):
            with patch.object(native_rimls,'_rimls_native',extension):
                self.assert_reference_outcome(self.arrays)

    def test_reverse_and_third_derivatives_do_not_use_value_only_kernel(self):
        """좌표와 텐서 척도의 고차 미분을 값 전용 경로로 분리하지 않아요."""
        direction=torch.tensor([.2,-.3,.1],dtype=torch.float64)
        derivatives=[]
        with patch.object(native_rimls._rimls_native,'values',side_effect=AssertionError('gradient entered value kernel')):
            for function in (rimls.rimls_values_torch,rimls._rimls_values_reference):
                shift=torch.zeros(3,dtype=torch.float64,requires_grad=True)
                radius=torch.tensor(1.2,dtype=torch.float64,requires_grad=True)
                values=function(self.arrays[0]+shift,*self.arrays[1:],radius)
                first,scale_gradient=torch.autograd.grad(values.sum(),(shift,radius),create_graph=True)
                second=torch.autograd.grad(first@direction,shift,create_graph=True)[0]
                third=torch.autograd.grad(second@direction,shift)[0]
                derivatives.append((values,first,scale_gradient,second,third))
        for actual,expected in zip(*derivatives):torch.testing.assert_close(actual,expected,rtol=2e-10,atol=2e-11)

    def test_forward_ad_and_function_transforms_keep_reference_derivatives(self):
        """전방 자동미분과 함수 변환에서 기준 미분 경로를 유지해요."""
        q,p,n=self.arrays
        tangent=torch.full_like(q,.02)
        with patch.object(native_rimls._rimls_native,'values',side_effect=AssertionError('transformed input entered native value kernel')):
            if hasattr(torch.autograd,'forward_ad'):
                with torch.autograd.forward_ad.dual_level():
                    dual=torch.autograd.forward_ad.make_dual(q,tangent)
                    actual=torch.autograd.forward_ad.unpack_dual(rimls.rimls_values_torch(dual,p,n,1.2))
                    expected=torch.autograd.forward_ad.unpack_dual(rimls._rimls_values_reference(dual,p,n,1.2))
                    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=2e-10,atol=2e-12)
            if hasattr(torch,'func'):
                actual=torch.func.jvp(lambda x:rimls.rimls_values_torch(x,p,n,1.2),(q,),(tangent,))
                expected=torch.func.jvp(lambda x:rimls._rimls_values_reference(x,p,n,1.2),(q,),(tangent,))
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=2e-10,atol=2e-12)

    def test_native_value_buffer_contract(self):
        """잘못된 버퍼 자료형과 크기 및 읽기 전용 출력을 거부해요."""
        arrays=tuple(value.numpy() for value in self.arrays)
        output=np.empty(len(arrays[0]))
        kernel=native_rimls._rimls_native.values
        with self.assertRaises(TypeError):kernel(arrays[0].astype(np.float32),*arrays[1:],output,1.2,.5,.5)
        with self.assertRaises(ValueError):kernel(arrays[0].ravel(),*arrays[1:],output,1.2,.5,.5)
        with self.assertRaises(ValueError):kernel(*arrays,output[:-1],1.2,.5,.5)
        with self.assertRaises(ValueError):kernel(*arrays,output[:,None],1.2,.5,.5)
        output.flags.writeable=False
        with self.assertRaises((ValueError,BufferError)):kernel(*arrays,output,1.2,.5,.5)

    def test_available_gpu_preserves_tensor_fallback(self):
        """사용 가능한 GPU에서는 기존 장치의 텐서 계산을 유지해요."""
        if torch.cuda.is_available():
            device,dtype='cuda',torch.float64
        elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
            device,dtype='mps',torch.float32
        else:
            self.skipTest('사용 가능한 GPU가 없어요.')
        arrays=tuple(value.to(device=device,dtype=dtype) for value in self.arrays)
        with torch.no_grad(),patch.object(native_rimls._rimls_native,'values',side_effect=AssertionError('GPU entered CPU native kernel')):
            actual=rimls.rimls_values_torch(*arrays,1.2)
            expected=rimls._rimls_values_reference(*arrays,1.2)
        torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-7)


if __name__=='__main__':
    unittest.main()
