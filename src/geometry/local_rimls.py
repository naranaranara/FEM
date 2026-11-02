"""##r## Current-coordinate radius neighbours and cached C++ GN derivatives.

Only sample pairs inside R contribute to the compact-support kernel. Refresh
neighbours on EVERY call (including line-search trials); retain the upstream
sample/normal autograd graph so all incident FEM nodes still receive forces.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np
import torch
from scipy.spatial import cKDTree
from . import native_rimls as native

_mode = ContextVar('local_rimls_mode', default=(True, True))
_stats = ContextVar('local_rimls_stats', default=None)


@contextmanager
def local_rimls_mode(enabled=True, native_first=True, stats=None):
    """##r## Validation switch: legacy / neighbours+Torch / neighbours+C++."""
    token=_mode.set((enabled,native_first));st=_stats.set(stats)
    try:yield
    finally:_mode.reset(token);_stats.reset(st)


def _local_reference(q,p,n,rows,ids,R,sr,sn):
    """Differentiable sparse-pair reference/fallback; no detached normals."""
    def accumulate(v):
        return v.new_zeros((len(q),)+v.shape[1:]).index_add(0,rows,v)
    diff=q[rows]-p[ids]
    t=torch.clamp(1.-(diff*diff).sum(1)/(R*R),min=0.)
    phi=t**4;G=(-8./(R*R))*t[:,None]**3*diff
    D=accumulate(phi)
    if torch.any(D<1e-14).item():raise ValueError('query outside kernel support')
    local=(diff*n[ids]).sum(1)
    f=accumulate(phi*local)/D
    grad=accumulate(phi[:,None]*n[ids]+(local-f[rows])[:,None]*G)/D[:,None]
    W=phi*torch.exp(-((f[rows]-local)/(sr*R))**2-((grad[rows]-n[ids])**2).sum(1)/(sn*sn))
    Z=accumulate(W)
    if torch.any(Z<=1e-14).item():raise ValueError('vanishing robust weights')
    return accumulate(W*local)/Z


class _LocalNative(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,p,n,offsets,ids,rows,R,sr,sn):
        inputs=tuple(native._array(x) for x in (q,p,n))
        values=np.empty(len(q));A=np.empty((len(ids),3));B=np.empty_like(A)
        native._rimls_native.local_evaluate(*inputs,offsets,ids,values,A,B,R,sr,sn)
        if not all(np.isfinite(x).all() for x in (values,A,B)):
            raise ValueError('nonfinite local RIMLS output')
        ctx.save_for_backward(q,p,n)
        ctx.scales=R,sr,sn
        ctx.coefficients=(torch.from_numpy(A),torch.from_numpy(B),
                          torch.from_numpy(rows),torch.from_numpy(ids))
        ctx.batched=None
        # ##r## Small GN blocks reuse compact dense coefficients in batched VJP.
        # C++ still computes only true neighbours; zero padding avoids repeating
        # scatter-add for every query/sample pair in each Jacobian seed.
        if len(q)<=32:
            used,inverse=np.unique(ids,return_inverse=True)
            da=np.zeros((len(q),len(used),3));db=np.zeros_like(da)
            da[rows,inverse]=A;db[rows,inverse]=B
            ctx.batched=(torch.from_numpy(da.reshape(len(q),-1)),
                         torch.from_numpy(db.reshape(len(q),-1)),
                         torch.from_numpy(da.sum(axis=1)),torch.from_numpy(used))
        return torch.from_numpy(values)

    @staticmethod
    def backward(ctx,cotangent):
        q,p,n=ctx.saved_tensors
        if torch.is_grad_enabled():
            # Preserve exact higher derivatives: never differentiate frozen A/B.
            gradients=native._torch_vjp(q,p,n,cotangent,*ctx.scales)
        elif ctx.batched is not None:
            A,B,total,used=ctx.batched
            gradients=(cotangent[:,None]*total,
                       torch.zeros_like(p).index_add(0,used,-(cotangent@A).reshape(-1,3)),
                       torch.zeros_like(n).index_add(0,used,(cotangent@B).reshape(-1,3)))
        else:
            A,B,rows,ids=ctx.coefficients
            weighted=cotangent[rows,None]
            ga=weighted*A;gb=weighted*B
            gradients=(torch.zeros_like(q).index_add(0,rows,ga),
                       torch.zeros_like(p).index_add(0,ids,-ga),
                       torch.zeros_like(n).index_add(0,ids,gb))
        return (*gradients,None,None,None,None,None,None)


def try_local_rimls(q,p,n,R,sr,sn):
    """Return None for unsupported inputs so the original dispatch remains valid."""
    enabled,use_native=_mode.get()
    if (not enabled or native._full_mode.get() or len(p)<48 or len(q)*len(p)<2048
            or not native._supported(q,p,n,(R,sr,sn))):
        return None
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        scales=np.asarray((R,sr*R,sn),dtype=float)**2
    if not np.all(np.isfinite(scales)) or np.any(scales<=0):return None
    arrays=tuple(native._array(x) for x in (q,p,n))
    if not all(np.isfinite(x).all() for x in arrays):return None
    # No stale topology cache: safe for arbitrary deformation and trial positions.
    neighbours=cKDTree(arrays[1]).query_ball_point(arrays[0],np.nextafter(float(R),np.inf),
                                                  eps=0.,workers=1,return_sorted=True)
    counts=np.fromiter((len(x) for x in neighbours),dtype=np.int64,count=len(q))
    offsets=np.r_[np.int64(0),np.cumsum(counts)]
    ids=np.concatenate(neighbours).astype(np.int64) if offsets[-1] else np.empty(0,dtype=np.int64)
    rows=np.repeat(np.arange(len(q),dtype=np.int64),counts)
    cpp=use_native and hasattr(native._rimls_native,'local_evaluate')
    stats=_stats.get()
    if stats is not None:
        for key,value in (('calls',1),('dense_pairs',len(q)*len(p)),('local_pairs',len(ids)),('native_calls',int(cpp))):
            stats[key]=stats.get(key,0)+value
    if not cpp:
        return _local_reference(q,p,n,torch.from_numpy(rows),torch.from_numpy(ids),R,sr,sn)
    try:
        if torch.is_grad_enabled() and any(x.requires_grad for x in (q,p,n)):
            return _LocalNative.apply(q,p,n,offsets,ids,rows,R,sr,sn)
        values=np.empty(len(q));empty=np.empty((0,3))
        native._rimls_native.local_evaluate(*arrays,offsets,ids,values,empty,empty,R,sr,sn)
        return torch.from_numpy(values) if np.isfinite(values).all() else None
    except ValueError:
        # Keep reference decisions at the support/robust-weight thresholds.
        return None
