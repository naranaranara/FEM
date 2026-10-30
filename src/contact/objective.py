import numpy as np
import torch
from time import perf_counter
from .barrier import barrier_value,barrier_first_second

class ContactObjective:
    def __init__(self,initial,gap_ad,base,feasible=None):
        self.initial=np.asarray(initial,dtype=float)
        self.gap_ad,self.base=gap_ad,base
        self.feasible=feasible or (lambda x:np.all(np.isfinite(x)))
        self.reset()
    def reset(self):
        self.counts=dict(gap_calls=0,gradient_calls=0,gap_hessian_calls=0,energy_calls=0)
    def tensor_gap(self,x):
        self.counts['gap_calls']+=1
        return self.gap_ad(x)
    def energy(self,x):
        self.counts['energy_calls']+=1
        if not self.feasible(x):return np.inf
        try:s=float(self.tensor_gap(torch.tensor(x,dtype=torch.float64)).detach())
        except ValueError:return np.inf
        if not np.isfinite(s) or s<=0:return np.inf
        return float(self.base(x)[0]+barrier_value(s,.2))
    def evaluate(self,x,mode='gradient',return_gap_gradient=False,phase_timer=None):
        self.counts['gradient_calls']+=1
        tick=perf_counter() if phase_timer else None  # ##r## optional profiling
        z=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        s_tensor=self.tensor_gap(z)
        s=float(s_tensor.detach())
        if not np.isfinite(s) or s<=0:raise ValueError('invalid model gap')
        if phase_timer:
            phase_timer.record('gap_forward',tick);tick=perf_counter()
        gs=torch.autograd.grad(s_tensor,z)[0].detach().numpy()
        if phase_timer:
            phase_timer.record('gap_backward',tick);tick=perf_counter()
        b1,b2=barrier_first_second(s,.2)
        Eb,gb,Hb=self.base(x)
        E=float(Eb+barrier_value(s,.2));g=gb+b1*gs
        info=dict(gap=s,filter_active=mode=='filtered_gn' and s<.02)
        if return_gap_gradient:
            info.update(gap_gradient=gs.copy(),barrier_first=float(b1))
        H=None
        if mode!='gradient':
            coefficient=barrier_first_second(max(s,.02),.2)[1] if info['filter_active'] else b2
            H=Hb+coefficient*np.outer(gs,gs)
            if mode=='full':
                self.counts['gap_hessian_calls']+=1
                hs=torch.autograd.functional.hessian(self.tensor_gap,z).detach().numpy()
                H=H+b1*hs
        if phase_timer:phase_timer.record('objective_assembly',tick)
        return E,g,H,info
