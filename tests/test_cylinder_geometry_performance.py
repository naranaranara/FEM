"""원통 메시의 연결 순서와 좁은 질의면·넓은 법선 지지면의 미분을 검사해요."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.geometry.mesh import make_cylinder
from src.geometry.surface import make_surface_sampler, surface_quadrature
from src.geometry.rimls import _rimls_values_reference
from src.simulation import two_body


def _scalar_tetrahedra(mesh):
    r"""순차적인 삼각기둥 분할로 기준 사면체 연결을 만들어요.

    $$6V=\det(\boldsymbol{x}_1-\boldsymbol{x}_0,\boldsymbol{x}_2-\boldsymbol{x}_0,\boldsymbol{x}_3-\boldsymbol{x}_0)$$
    """
    n_angle,n_radial,n_axis=mesh['counts']
    width=1+n_angle*n_radial
    triangles=[(0,1+j,1+(j+1)%n_angle) for j in range(n_angle)]
    for ring in range(1,n_radial):
        inner=1+(ring-1)*n_angle
        outer=inner+n_angle
        for j in range(n_angle):
            k=(j+1)%n_angle
            triangles.extend(((inner+j,outer+j,outer+k),(inner+j,outer+k,inner+k)))
    tetrahedra=[]
    for layer in range(n_axis-1):
        for face in triangles:
            i,j,k=(index+layer*width for index in sorted(face))
            I,J,K=i+width,j+width,k+width
            tetrahedra.extend(((i,j,k,K),(i,j,J,K),(i,I,J,K)))
    tetrahedra=np.asarray(tetrahedra)
    vertices=mesh['nodes'][tetrahedra]
    # 방향을 판별하는 부호 체적을 구해요: $$6V=\det(\boldsymbol{x}_1-\boldsymbol{x}_0,\boldsymbol{x}_2-\boldsymbol{x}_0,\boldsymbol{x}_3-\boldsymbol{x}_0)$$
    determinant=np.linalg.det(vertices[:,1:]-vertices[:,:1])
    negative=np.flatnonzero(determinant<0)
    tetrahedra[negative,1],tetrahedra[negative,2]=tetrahedra[negative,2].copy(),tetrahedra[negative,1].copy()
    return tetrahedra


def _scalar_boundary(nodes,tetrahedra):
    r"""각 사면체를 순회하며 반대 절점에서 멀어지는 경계면을 모아요.

    $$((\boldsymbol{x}_b-\boldsymbol{x}_a)\times(\boldsymbol{x}_c-\boldsymbol{x}_a))\cdot(\boldsymbol{x}_o-\boldsymbol{x}_a)<0$$
    """
    faces={}
    for tet in tetrahedra:
        for opposite in range(4):
            face=[int(tet[index]) for index in range(4) if index!=opposite]
            a,b,c=nodes[face]
            # 법선과 내부 절점 사이의 방향을 검사해요: $$d=((\boldsymbol{x}_b-\boldsymbol{x}_a)\times(\boldsymbol{x}_c-\boldsymbol{x}_a))\cdot(\boldsymbol{x}_o-\boldsymbol{x}_a)$$
            direction=np.dot(np.cross(b-a,c-a),nodes[tet[opposite]]-a)
            if direction>0:face[1],face[2]=face[2],face[1]
            faces.setdefault(tuple(sorted(face)),[]).append(face)
    return np.array([values[0] for values in faces.values() if len(values)==1])


class CylinderGeometryPerformanceChecks(unittest.TestCase):
    """원통의 벡터화와 국소 질의 보간이 기존 기하학을 유지하는지 검사해요."""

    def setUp(self):
        """작은 수치 검사의 계산 스레드를 제한해요."""
        torch.set_num_threads(1)

    def test_scalar_connectivity_boundary_order_and_orientation(self):
        """다중 반지름과 세밀한 원통에서도 기준 연결 및 경계면 순서를 보존해요."""
        for na,nr,ny,facing,angle in ((32,1,7,'up',0.),(64,3,13,'down',45.),(128,1,25,'up',73.)):
            with self.subTest(n_angle=na,n_radial=nr,n_axis=ny):
                mesh=make_cylinder(n_angle=na,n_radial=nr,n_axis=ny,facing=facing,
                                   angle_degrees=angle,clearance=.03)
                expected=_scalar_tetrahedra(mesh)
                np.testing.assert_array_equal(mesh['tetrahedra'],expected)
                np.testing.assert_array_equal(mesh['boundary_faces'],_scalar_boundary(mesh['nodes'],expected))
                np.testing.assert_array_equal(mesh['side_faces'],
                    mesh['boundary_faces'][~np.all(np.isin(mesh['boundary_faces'],mesh['support_nodes']),axis=1)])

    def test_distinct_query_and_normal_support_match_dense_values_and_derivatives(self):
        """넓은 면의 법선을 유지한 채 좁은 질의면의 값과 이차 미분을 비교해요."""
        lower=make_cylinder(radius=3.,facing='up')
        upper=make_cylinder(radius=5.,facing='down',angle_degrees=45.,clearance=.03)
        self.assertLess(len(lower['query_faces']),len(lower['contact_faces']))
        p=two_body.build_problem(initial_gap=.03,closure=0.,contact_meshes=(lower,upper),R=.65)
        sampler_l,_,_=make_surface_sampler(lower['nodes'],lower['contact_faces'])
        sampler_u,_,_=make_surface_sampler(upper['nodes'],upper['contact_faces'])
        sl,wl=surface_quadrature(lower['nodes'],lower['query_faces'])
        su,wu=surface_quadrature(upper['nodes'],upper['query_faces'])
        ql=torch.tensor(sl,dtype=torch.float64)
        qu=torch.tensor(su,dtype=torch.float64)
        np.testing.assert_array_equal(p['weights'],.5*np.r_[wu,wl])
        rng=np.random.default_rng(170)
        x=torch.tensor(np.r_[lower['nodes'].ravel(),upper['nodes'].ravel()]
                       +rng.normal(scale=2e-4,size=lower['nodes'].size+upper['nodes'].size),requires_grad=True)
        split=lower['nodes'].size

        def reference(state):
            """조밀 질의 보간과 넓은 지지면의 RIMLS 식으로 기준 간격을 계산해요."""
            a,b=state[:split].reshape(-1,3),state[split:].reshape(-1,3)
            pl,nl=sampler_l(a)
            pu,nu=sampler_u(b)
            return torch.cat((_rimls_values_reference(qu@b,pl,nl,.65),
                              _rimls_values_reference(ql@a,pu,nu,.65)))

        actual=p['gaps_full_ad'](x)
        expected=reference(x)
        torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-13)
        ids=np.array([len(actual)-1,0,3,1])
        torch.testing.assert_close(p['gaps_full_ad'](x,indices=ids),expected[ids],rtol=1e-11,atol=1e-13)
        weights=torch.tensor(rng.normal(size=len(actual)))
        direction=torch.tensor(rng.normal(size=len(x)))
        derivatives=[]
        for gaps in (actual,expected):
            first=torch.autograd.grad(gaps,x,weights,create_graph=True)[0]
            second=torch.autograd.grad(first,x,direction)[0]
            derivatives.append((first,second))
        for a,b in zip(*derivatives):torch.testing.assert_close(a,b,rtol=2e-9,atol=3e-11)

    def test_query_setup_uses_sparse_metadata_and_block_reuses_samples(self):
        """질의용 조밀 행렬을 만들지 않고 동일한 블록 질의는 기존 샘플을 재사용해요."""
        with patch.object(two_body,'surface_quadrature',side_effect=AssertionError('unexpected extra query matrix')):
            two_body.build_problem(n=2)
        mesh=make_cylinder()
        with patch.object(two_body,'surface_quadrature',wraps=surface_quadrature) as quadrature:
            sampler,weights=two_body._make_query_sampler(mesh['nodes'],mesh['query_faces'])
        self.assertTrue(quadrature.call_args.kwargs['sparse'])
        self.assertEqual(sampler(torch.tensor(mesh['nodes'])).shape,(len(weights),3))


if __name__=='__main__':
    unittest.main()
