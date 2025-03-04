import torch
import numpy as np
import math
from torch.optim import Optimizer

class KineticDescent(Optimizer):
    """
    Implements Kinetic Descent optimizer with weight decay.
    """
    def __init__(self, params, lr: float, gamma: float, c_init: float, eps: float, weight_decay: float = 0.0):
        defaults = {'lr': lr, 'gamma': gamma, 'c_init': 2 * c_init, 'eps': eps, 'weight_decay': weight_decay}
        super().__init__(params, defaults)
        self.p = {}
        self.momentum_magnitude_history = []
        self.t = [0.0]
        self.eps = eps

    def c_and_cprime(self, c_init: float, gamma: float, t: float):
        """
        Calculates c(t) and its derivative c'(t) based on the logarithmic decay function.
        """
        #c = c_init / np.log(np.e + gamma * t)  # New logarithmic decay function
        #cp = -c_init * gamma / ((np.log(np.e + gamma * t))**2 * (np.e + gamma * t))  # Derivative of the new function
        
        # Previous implementation with power-law decay
        #c = c_init * (1 + t) ** -gamma
        #cp = -gamma * c_init * (1 + t) ** -(gamma + 1)

        # Alternative implementation with exponential decay
        c = c_init * np.exp(-gamma * t)
        cp = -c_init * gamma * np.exp(-gamma * t)

        #c = self.eps + (c_init - self.eps) * (1 + np.cos(np.pi * (t % gamma) / gamma)) / 2 * (0.5 ** (t // gamma))
        #cp = -0.5 * (c_init - self.eps) * (np.pi / gamma) * np.sin(np.pi * (t % gamma) / gamma) * (0.5 ** (t // gamma))

        return c, cp

    def simp(self, f, a: float, b: float, n: int):
        if n % 2 == 1:
            n += 1  # Ensure n is even
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        return h / 3 * (y[0] + 2 * np.sum(y[2:n:2]) + 4 * np.sum(y[1:n:2]) + y[n])
    
    def RK4(self, t: float, y: np.ndarray, h: float, f):
        F0 = f(t, y)
        Y1 = y + (h / 2) * F0
        F1 = f(t + h / 2, Y1)
        Y2 = y + (h / 2) * F1
        F2 = f(t + h / 2, Y2)
        Y3 = y + h * F2
        F3 = f(t + h, Y3)
        y1 = y + (h / 6) * (F0 + 2 * F1 + 2 * F2 + F3)
        return y1

    def initialize_momentum(self, param: torch.Tensor, c_init: float):
        p = -param.grad
        p = p / torch.norm(p, p=2, dtype=torch.float64)
        p = p * math.sqrt(2 * c_init)
        return p

    def U_step(self, param: torch.Tensor, p: torch.Tensor, h: float, t: float, gamma: float, c_init: float):
        c, _ = self.c_and_cprime(c_init, gamma, t)
        c_next, _ = self.c_and_cprime(c_init, gamma, t + h)
        c_sqrt = lambda t: np.sqrt(self.c_and_cprime(c_init, gamma, t)[0])
        param.data += 2 * (p / c_sqrt(t)) * self.simp(c_sqrt, t, t + h, 16)
        p *= np.sqrt(c_next / c)
        return p

    def B_step(self, F: torch.Tensor, p: torch.Tensor, h: float, eps: float):
        c = torch.dot(p.flatten(), p.flatten())
        if c < eps:
            c += eps
        np0 = torch.sqrt(c)
        nF = torch.norm(F, p=2)
        eta = (torch.dot(p.flatten(), F.flatten()) / (nF * np0)).item()
        u = 1 - eta
        alpha = (nF / np0).item()
        if u > 1e-16:
            kappa = np.arctanh(eta)
            A = np.cosh(kappa) / np.cosh(alpha * h + kappa)
            B = (np.sinh(alpha * h + kappa) - np.sinh(kappa)) / (alpha * np.cosh(alpha * h + kappa))
        else:
            vf = lambda t, y: np.array([
                -alpha * (1 - np.exp(-2 * alpha * t) * u) * y[0],
                1 - alpha * (1 - np.exp(-2 * alpha * t) * u) * y[1]
            ])
            y0 = np.array([1, 0])
            y1 = self.RK4(0, y0, h, vf)
            A, B = y1[0], y1[1]
        p = A * p + B * F
        return p

    def kd_step(self, param: torch.Tensor, p: torch.Tensor, h: float, t: float, gamma: float, c_init: float, eps: float, weight_decay: float):
        F = -param.grad

        # Apply weight decay: F = F + weight_decay * param
        if weight_decay > 0:
            F += weight_decay * param

        p = self.U_step(param, p, h, t, gamma, c_init)
        p = self.B_step(F, p, h, eps)
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
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                if param not in self.p:
                    self.p[param] = self.initialize_momentum(param, c_init)
                
                momentum_magnitude += torch.dot(self.p[param].flatten(), self.p[param].flatten()).item()
                self.p[param] = self.kd_step(param, self.p[param], h, self.t[-1] + h, gamma, c_init, eps, weight_decay)

        self.momentum_magnitude_history.append(0.5 * momentum_magnitude)
        self.t.append(self.t[-1] + h)
        return loss
