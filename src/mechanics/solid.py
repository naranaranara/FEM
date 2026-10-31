"""선형 사면체의 탄성과 강성을 계산하고 묶음 연산으로 조립해요."""
import numpy as np


def material_matrix(young=30., poisson=.3):
    r"""등방성 재료의 공학 전단변형률에 대한 탄성 행렬을 계산해요.

    $$\boldsymbol{\sigma}=D\boldsymbol{\varepsilon}$$
    """
    if not np.isfinite(young) or young <= 0 or not np.isfinite(poisson) or not -1 < poisson < .5:
        raise ValueError('require E>0 and -1<nu<0.5')
    # 첫 번째 라메 상수를 계산해요: $$\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}$$
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    # 전단 탄성계수를 계산해요: $$\mu=\frac{E}{2(1+\nu)}$$
    mu = young / (2 * (1 + poisson))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    # 수직 성분의 대각항을 완성해요: $$D_{ii}=\lambda+2\mu,\quad i<3$$
    D[np.arange(3), np.arange(3)] += 2 * mu
    # 공학 전단변형률에 대한 성분을 채워요: $$D_{ss}=\mu I_3$$
    D[3:, 3:] = mu * np.eye(3)
    return D


def _element_kinematics_batch(xyz):
    r"""사면체 묶음의 부피와 변형률 행렬을 계산해요.

    $$V_e=\det(A_e)/6,\qquad \nabla N_e=(A_e^{-1})_{1:,:}^{T}$$
    """
    A = np.ones((len(xyz), 4, 4))
    A[:, :, 1:] = xyz
    # 각 사면체의 방향과 크기를 구해요: $$d_e=\det(A_e)$$
    determinant = np.linalg.det(A)
    if np.any(determinant <= 0):
        raise ValueError('tetrahedra must have positive volume')
    # 형상함수의 공간 기울기를 구해요: $$G_e=(A_e^{-1})_{1:,:}^{T}$$
    gradients = np.linalg.inv(A)[:, 1:, :].transpose(0, 2, 1)
    B = np.zeros((len(xyz), 6, 12))
    B[:, 0, 0::3] = gradients[:, :, 0]
    B[:, 1, 1::3] = gradients[:, :, 1]
    B[:, 2, 2::3] = gradients[:, :, 2]
    B[:, 3, 0::3] = gradients[:, :, 1]
    B[:, 3, 1::3] = gradients[:, :, 0]
    B[:, 4, 1::3] = gradients[:, :, 2]
    B[:, 4, 2::3] = gradients[:, :, 1]
    B[:, 5, 0::3] = gradients[:, :, 2]
    B[:, 5, 2::3] = gradients[:, :, 0]
    # 행렬식에서 부피를 계산해요: $$V_e=d_e/6$$
    volumes = determinant / 6
    return volumes, B


def element_kinematics(xyz):
    r"""한 사면체의 부피와 변형률 행렬을 계산해요.

    $$\boldsymbol{\varepsilon}_e=B_e\boldsymbol{u}_e$$
    """
    volumes, matrices = _element_kinematics_batch(np.asarray(xyz)[None, :, :])
    return volumes[0], matrices[0]


def assemble_stiffness(nodes, tetrahedra, young=30., poisson=.3, sparse=False):
    r"""사면체 강성을 묶음 계산해 전역 강성 행렬에 조립해요.

    $$K=\sum_e P_e^T\left(V_eB_e^TDB_e\right)P_e$$
    """
    D = material_matrix(young, poisson)
    tetrahedra = np.asarray(tetrahedra, dtype=int).reshape(-1, 4)
    size = nodes.size
    if sparse:
        index_type = np.int32 if max(size, len(tetrahedra) * 144) < np.iinfo(np.int32).max else np.int64
        rows = np.empty((len(tetrahedra), 12, 12), dtype=index_type)
        cols = np.empty_like(rows)
        values = np.empty((len(tetrahedra), 12, 12))
    else:
        K = np.zeros((size, size))
    for start in range(0, len(tetrahedra), 4096):
        stop = min(start + 4096, len(tetrahedra))
        tets = tetrahedra[start:stop]
        volumes, B = _element_kinematics_batch(nodes[tets])
        # 절점 번호를 세 방향 자유도로 변환해요: $$d_{eia}=3t_{ei}+a$$
        dofs = (3 * tets[:, :, None] + np.arange(3)).reshape(-1, 12)
        # 변형률 행렬의 전치에 재료 행렬을 곱해요: $$B_e^TD=B_e^T D$$
        BTD = B.transpose(0, 2, 1) @ D
        # 요소 강성을 계산해요: $$K_e=V_eB_e^TDB_e$$
        Ke = volumes[:, None, None] * (BTD @ B)
        if sparse:
            rows[start:stop] = dofs[:, :, None]
            cols[start:stop] = dofs[:, None, :]
            values[start:stop] = Ke
        else:
            np.add.at(K, (dofs[:, :, None], dofs[:, None, :]), Ke)
    if sparse:
        from scipy.sparse import coo_matrix
        K = coo_matrix((values.ravel(), (rows.ravel(), cols.ravel())), shape=(size, size)).tocsr()
        K.eliminate_zeros()
    return K


def elastic_terms(nodes, reference, stiffness):
    r"""변위에서 탄성 에너지와 기울기 및 강성을 반환해요.

    $$E=\tfrac12\boldsymbol{u}^TK\boldsymbol{u}$$
    """
    # 기준 위치에 대한 변위를 구해요: $$\boldsymbol{u}=\boldsymbol{x}-\boldsymbol{X}$$
    u = (nodes - reference).ravel()
    # 탄성 에너지의 기울기를 구해요: $$\boldsymbol{g}=K\boldsymbol{u}$$
    gradient = stiffness @ u
    # 탄성 에너지를 계산해요: $$E=\tfrac12\boldsymbol{u}^T\boldsymbol{g}$$
    energy = float(.5 * u @ gradient)
    return energy, gradient, stiffness


def tetrahedron_volumes(nodes, tetrahedra):
    r"""사면체의 부호 있는 부피를 계산해요.

    $$V_e=\det([\boldsymbol{x}_1-\boldsymbol{x}_0,\boldsymbol{x}_2-\boldsymbol{x}_0,\boldsymbol{x}_3-\boldsymbol{x}_0])/6$$
    """
    xyz = nodes[tetrahedra]
    # 모서리 행렬의 행렬식에서 부피를 구해요: $$V_e=\det(X_{e,1:}-X_{e,0})/6$$
    return np.linalg.det(xyz[:, 1:] - xyz[:, :1]) / 6
