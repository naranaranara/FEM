import math

def barrier_value(s: float, d_hat: float) -> float:
    if d_hat <= 0:
        raise ValueError('d_hat must be positive')
    if s<=0:
        return math.inf
    if s>=d_hat:
        return 0.0
    return -(s-d_hat)**2*math.log(s/d_hat)

def barrier_first_second(s: float, d_hat: float):
    if s<=0 or d_hat <= 0:
        raise ValueError('barriet dericatives requires s>0 and d_hat>0')
    if s>= d_hat:
        return 0.0, 0.0
    r = s - d_hat
    b1 = -2*r*math.log(s/d_hat) - r*r/s
    b2 = -2*math.log(s/d_hat) -4*r/s + (r*r)/(s*s)
    return b1, b2

def barrier_arrays(gaps,d_hat):
    import numpy as np
    gaps=np.asarray(gaps,dtype=float)
    if not np.isfinite(d_hat) or d_hat<=0:
        raise ValueError('d_hat must be finite and positive')
    if np.any(~np.isfinite(gaps)) or np.any(gaps<=0):
        raise ValueError('gaps must be finite and positive')
    b=np.zeros_like(gaps);b1=np.zeros_like(gaps);b2=np.zeros_like(gaps)
    active=gaps<d_hat;s=gaps[active];r=s-d_hat;log=np.log(s/d_hat)
    b[active]=-r*r*log
    b1[active]=-2*r*log-r*r/s
    b2[active]=-2*log-4*r/s+r*r/(s*s)
    return b,b1,b2
