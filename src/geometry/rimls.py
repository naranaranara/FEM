"""Differentiable implicit moving least squares gaps; no FEM dependencies."""
import numpy as np
import torch
from .primitives import sample_plane

def imls_value_gradient_torch(query, points, normals, R):
    import torch
    if R <= 0:
        raise ValueError('R must be positive')
    diff = query - points
    t = torch.clamp(1.0 - torch.sum(diff * diff, dim=1) / (R * R), min=0.0)
    phi = t ** 4
    grad_phi = (-8.0 / (R * R)) * (t ** 3)[:, None] * diff
    D = phi.sum()
    if D.item() < 1e-14:
        raise ValueError('query outside kernel support')
    local_gap = torch.sum(diff * normals, dim=1)
    f0 = torch.sum(phi * local_gap) / D
    grad0 = torch.sum(phi[:, None] * normals +
                      (local_gap - f0)[:, None] * grad_phi, dim=0) / D
    return f0, grad0, phi, local_gap


def rimls_value_torch(query, points, normals, R, sigma_r=0.5, sigma_n=1.0):
    import torch
    if sigma_r <= 0 or sigma_n <= 0:
        raise ValueError('sigma_r and sigma_n must be positive')
    f0, grad0, phi, local_gap = imls_value_gradient_torch(query, points, normals, R)
    # Algebraically identical squared residuals, without norm/abs singularities.
    residual_squared = (f0 - local_gap) ** 2
    normal_difference_squared = torch.sum((grad0[None, :] - normals) ** 2, dim=1)
    W = phi * torch.exp(-residual_squared / (sigma_r * R) ** 2)
    W = W * torch.exp(-normal_difference_squared / sigma_n ** 2)
    denominator = W.sum()
    if denominator.item() <= 1e-14:
        raise ValueError('vanishing robust kernel weights')
    return torch.sum(W * local_gap) / denominator


def rimls_values_torch(queries,points,normals,R,sigma_r=.5,sigma_n=.5):
    if R<=0 or sigma_r<=0 or sigma_n<=0:
        raise ValueError('positive R and robust scales required')
    diff=queries[:,None,:]-points[None,:,:]
    t=torch.clamp(1.-torch.sum(diff*diff,dim=2)/(R*R),min=0.)
    phi=t**4
    grad_phi=(-8./(R*R))*t[:,:,None]**3*diff
    D=phi.sum(dim=1)
    if torch.any(D<1e-14).item():raise ValueError('query outside kernel support')
    local=torch.sum(diff*normals[None,:,:],dim=2)
    f0=torch.sum(phi*local,dim=1)/D
    grad0=torch.sum(phi[:,:,None]*normals[None,:,:]+(local-f0[:,None])[:,:,None]*grad_phi,dim=1)/D[:,None]
    residual=(f0[:,None]-local)**2
    deviation=torch.sum((grad0[:,None,:]-normals[None,:,:])**2,dim=2)
    W=phi*torch.exp(-residual/(sigma_r*R)**2-deviation/sigma_n**2)
    denominator=W.sum(dim=1)
    if torch.any(denominator<=1e-14).item():raise ValueError('vanishing robust weights')
    return torch.sum(W*local,dim=1)/denominator


def make_gaps(S, node_shape, R=.9, points=None, normals=None):
    if points is None and normals is None:
        points, normals, _ = sample_plane()
    elif points is None or normals is None:
        raise ValueError("points and normals must be supplied together")
    St = torch.tensor(S, dtype=torch.float64)
    pt = torch.tensor(points, dtype=torch.float64)
    nt = torch.tensor(normals, dtype=torch.float64)
    def gaps(state):
        samples = St @ state.reshape(node_shape)
        return rimls_values_torch(samples, pt, nt, R, sigma_n=.5)  ##r##
    return gaps
