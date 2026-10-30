"""Backward-Euler inertial potential helper; quasi-static CLI does not use it.
Lumped positive masses are repeated for coordinate DOFs by the caller.
"""
import numpy as np
from scipy.sparse import diags

def inertial_terms(position,predicted,mass,dt):
    position=np.asarray(position,dtype=float);predicted=np.asarray(predicted,dtype=float)
    mass=np.asarray(mass,dtype=float)
    if position.ndim!=1 or position.shape!=predicted.shape or mass.shape!=position.shape:
        raise ValueError('matching position, predicted and lumped mass vectors required')
    if not np.isfinite(dt) or dt<=0 or not np.all(np.isfinite(mass)) or np.any(mass<=0):
        raise ValueError('positive finite dt and masses required')
    delta=position-predicted;diagonal=mass/dt**2
    return float(.5*delta@(diagonal*delta)),diagonal*delta,diags(diagonal,format='csr')
