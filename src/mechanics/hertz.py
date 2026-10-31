import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def elliptical_hertz(delta, curvatures, young_lower=30., young_upper=30., poisson=.3):
    k = np.sort(np.asarray(curvatures,dtype=float))
    if k.shape != (2,) or np.any(~np.isfinite(k)) or k[0] <= 0:
        raise ValueError('two positive principal relative curvatures required')
    if not all(np.isfinite(v) for v in (delta,young_lower,young_upper,poisson)) or delta < 0 or min(young_lower,young_upper)<=0 or not -1<poisson<.5:
        raise ValueError('invalid Hertz parameters')
    modulus = 1./((1-poisson**2)/young_lower+(1-poisson**2)/young_upper)
    def integrals(ratio):
        def common(u):
            return 2./np.sqrt((1+u*u)*(ratio*ratio+u*u))
        return np.array([quad(lambda u: common(u),0,np.inf,epsabs=1e-10)[0],
                         quad(lambda u: common(u)/(1+u*u),0,np.inf,epsabs=1e-10)[0],
                         quad(lambda u: common(u)/(ratio*ratio+u*u),0,np.inf,epsabs=1e-10)[0]])
    target = k[1]/k[0]
    ratio = 1. if np.isclose(target,1.,rtol=1e-12,atol=0.) else brentq(
        lambda r: integrals(r)[2]/integrals(r)[1]-target,1e-5,1.,xtol=1e-12)
    J0,J1,J2 = integrals(ratio)
    a = np.sqrt(2*delta*J1/(k[0]*J0))
    b = ratio*a
    p0 = modulus*a*k[0]/(ratio*J1)
    force = 2*np.pi*a*b*p0/3
    return dict(force=float(force),major_radius=float(a),minor_radius=float(b),
                peak_pressure=float(p0),effective_modulus=float(modulus),
                ellipse_ratio=float(ratio),curvatures=k.tolist())
