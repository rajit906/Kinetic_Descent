import torch
from torch.optim import Optimizer
import math
import numpy as np

class KineticDescent(Optimizer):
    def __init__(self, params):
        super(KineticDescent, self).__init__(params, {})
        self.p = {}
        self.momentum_magnitude_history = []
        self.t = [0.]

    def c_and_cprime(self, c_init, gamma, t):
        c = c_init * np.exp(-gamma * t)
        cp = -gamma * c_init * np.exp(-gamma * t)
        return c, cp
    
    def simp(self, f, a, b, n):
        if n % 2 == 1:
            n += 1  # Ensure n is even
        h = (b - a) / n
        x = torch.linspace(a, b, n + 1, device=f.device)
        y = f(x)
        return h / 3 * (y[0] + 2 * torch.sum(y[2:n:2]) + 4 * torch.sum(y[1:n:2]) + y[n])

    def initialize_momentum(self, param, c_init):
        p = torch.randn_like(param, dtype = torch.float64)
        p = p / torch.norm(p, p=2, dtype=torch.float64)
        p = p * math.sqrt(2 * c_init)
        return p        
    
    def U_step(self, param, p, h, t, gamma, c_init):
        #c, _ = self.c_and_cprime(c_init, gamma, t)
        #c_next, _ = self.c_and_cprime(c_init, gamma, t + h)
        #c_sqrt = lambda t: torch.sqrt(c_init * torch.exp(-gamma * t))
        param.data += 2 * (1 - math.exp(-0.5 * gamma * h)) * p / gamma #(p / c_sqrt(t)) * self.simp(c_sqrt, t, t + h, 16)
        p *= math.exp(-0.5 * gamma * h)
        return p

    def RK4(self,t,y,h,f):
        F0 = f(t,y)
        Y1 = y + (h/2)*F0
        F1 = f(t+h/2, Y1)
        Y2 = y + (h/2)*F1
        F2 = f(t+h/2, Y2)
        Y3 = y + h*F2
        F3 = f(t+h, Y3)
        y1 = y + (h/6)*(F0 + 2*F1 + 2*F2 + F3)
        return y1
    
    def B_step(self, F, p, h):
        c = torch.dot(p.flatten(), p.flatten())
        np0 = torch.sqrt(c)
        nF = torch.norm(F, p=2)
        eta = (torch.dot(p.flatten(), F.flatten()) / (nF * np0)).numpy().item()
        u = 1-eta
        alpha = (nF / np0).numpy().item()
        if(u > 1e-16):
            kappa = np.arctanh(eta)
            A = np.cosh(kappa) / np.cosh(alpha * h + kappa)
            #B = (np.sinh(alpha * h + kappa) - np.sinh(kappa)) / (alpha * np.cosh(alpha * h + kappa))
            sech = lambda x: 2/(np.exp(x) + np.exp(-x))
            B = (np.tanh(alpha * h + kappa) - np.sinh(kappa) * sech(alpha * h + kappa)) / alpha
            #print(f"A: {A}, B: {B}, Alpha: {alpha}, Kappa: {kappa}")
        else:
            vf = lambda t,y: np.array([-alpha*(1 - np.exp(-2*alpha*t)*u)*y[0],  1 - alpha*(1 - np.exp(-2*alpha*t)*u)*y[1]])
            y0 = np.array([1,0])
            y1 = self.RK4(0,y0,h,vf)
            A, B = y1[0], y1[1]
        p = A * p + B * F
        return p

    def kd_step(self, param, p, h, t, gamma, c_init):
        p = self.U_step(param, p, h, t, gamma, c_init)
        F = -param.grad
        p = self.B_step(F, p, h)
        return p

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        momentum_magnitude = 0
        for group in self.param_groups:
            h = group['lr']
            gamma = group['gamma']
            c_init = group['c_init']
            for param in group['params']:
                if param.grad is None:
                    continue
                if len(param.shape) == 1:
                    param.data.add_(-h * param.grad)
                else:
                    if param not in self.p:
                        self.p[param] = self.initialize_momentum(param, c_init)
                    momentum_magnitude += torch.dot(self.p[param].flatten(), self.p[param].flatten()).item()
                    self.p[param] = self.kd_step(param, self.p[param], h, self.t[-1] + h, gamma, c_init)
        self.momentum_magnitude_history.append(0.5 * momentum_magnitude)
        self.t.append(self.t[-1] + h)
        return loss