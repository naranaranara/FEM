import numpy as np

def material_matrix(young=30., poisson=.3):
    if not np.isfinite(young) or young <=0 or not np.isfinite(poisson) or not -1< poisson <.5:
        raise ValueError('require E>0 and -1<nu<0.5')
    lam = young*poisson/((1+poisson)*(1-2*poisson))
    mu = young/(2*(1+poisson))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2*mu
    D[3:, 3:] = mu*np.eye(3)
    return D

def element_kinematics(xyz):
    A = np.column_stack((np.ones(4), xyz))
    determinant = np.linalg.det(A)
    if determinant <= 0:
        raise ValueError('tetrahedra must have positive volume')
    gradients = np.linalg.inv(A)[1:, :].T
    B = np.zeros((6, 12))
    for i, (dx, dy, dz) in enumerate(gradients):
        c = 3*i
        B[0, c] = dx
        B[1, c+1] = dy
        B[2, c+2] = dz
        B[3, c:c+3] = [dy, dx, 0.]
        B[4, c:c+3] = [0., dz, dy]
        B[5, c:c+3] = [dz, 0., dx]
    return determinant/6, B

def assemble_stiffness(nodes, tetrahedra, young=30., poisson=.3, sparse=False):
    D = material_matrix(young, poisson)
    if sparse:rows,cols,values=[],[],[]
    else:K = np.zeros((nodes.size, nodes.size))
    for tet in tetrahedra:
        volume, B = element_kinematics(nodes[tet])
        dofs = (3*tet[:, None] + np.arange(3)).ravel()
        Ke=volume*(B.T @ D @ B)
        if sparse:
            rows.append(np.repeat(dofs,12));cols.append(np.tile(dofs,12));values.append(Ke.ravel())
        else:K[np.ix_(dofs, dofs)] += Ke
    if sparse:
        from scipy.sparse import coo_matrix
        K=coo_matrix((np.concatenate(values) if values else np.empty(0),
                      (np.concatenate(rows) if rows else np.empty(0,dtype=int),
                       np.concatenate(cols) if cols else np.empty(0,dtype=int))),shape=(nodes.size,nodes.size)).tocsr()
        K.eliminate_zeros()
    return K

def elastic_terms(nodes, reference, stiffness):
    u = (nodes-reference).ravel()
    gradient = stiffness @ u
    return float(.5*u @ gradient), gradient, stiffness

def tetrahedron_volumes(nodes,tetrahedra):
    xyz=nodes[tetrahedra]
    return np.linalg.det(xyz[:,1:]-xyz[:,:1])/6
