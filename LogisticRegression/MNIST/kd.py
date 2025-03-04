import math
import numpy as np
import torch
from torch.optim import Optimizer

class KineticDescentUB(Optimizer):
    """
    Implements Kinetic Descent optimizer.
    """
    def __init__(self, params, lr: float, gamma: float, c_init: float, eps: float):
        defaults = {'lr': lr, 'gamma': gamma, 'c_init': 2 * c_init, 'eps': eps}
        super().__init__(params, defaults)
        self.p = {}
        self.momentum_magnitude_history = []
        self.t = [0.0]
        self.eps = eps

    def c_and_cprime(self, c_init: float, gamma: float, t: float):
        """
        Calculates c(t) and its derivative c'(t) based on the logarithmic decay function.
        """
        c = c_init / np.log(np.e + gamma * t)  # New logarithmic decay function
        cp = -c_init * gamma / ((np.log(np.e + gamma * t))**2 * (np.e + gamma * t))  # Derivative of the new function
        
        # Previous implementation with power-law decay
        #c = c_init * (1 + t) ** -gamma
        #cp = -gamma * c_init * (1 + t) ** -(gamma + 1)

        # Alternative implementation with exponential decay
        # c = c_init * np.exp(-gamma * t)
        # cp = -c_init * gamma * np.exp(-gamma * t)

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

    def B_step(self, F: torch.Tensor, p: torch.Tensor, h: float, eps: float):
        c = torch.dot(p.flatten(), p.flatten())
        if c < eps:
            c += 0 #eps
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

    def kd_step(self, param: torch.Tensor, p: torch.Tensor, h: float, t: float, gamma: float, c_init: float, eps: float):
        F = -param.grad
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
            for param in group['params']:
                if param.grad is None:
                    continue
                if param not in self.p:
                    self.p[param] = self.initialize_momentum(param, c_init)
                momentum_magnitude += torch.dot(self.p[param].flatten(), self.p[param].flatten()).item()
                self.p[param] = self.kd_step(param, self.p[param], h, self.t[-1] + h, gamma, c_init, eps)

        self.momentum_magnitude_history.append(0.5 * momentum_magnitude)
        self.t.append(self.t[-1] + h)
        return loss

    

class KineticDescentRK4(Optimizer):
    def __init__(self, params, lr, gamma, c_init):
        defaults = {'lr': lr, 'gamma': gamma, 'c_init': c_init}
        super(KineticDescentRK4, self).__init__(params, defaults)
        self.p = {}
        self.momentum_magnitude_history = []
        self.t = [0.]

    def initialize_momentum(self, param, c_init):
        p = -param.grad
        p = p / torch.norm(p, p=2, dtype=torch.float64)
        p = p * math.sqrt(2 * c_init)
        return p

    def solve_p(self, p, F, h, gamma):
        eps_reg = 1e-16  # Regularization term
        def compute_g(p, F):
            pdotF = torch.dot(p.flatten(), F.flatten())
            pdotp = torch.dot(p.flatten(), p.flatten()) + eps_reg
            return F - (pdotF / pdotp) * p - 0.5 * gamma * p
        g1 = compute_g(p, F)
        g2 = compute_g(p + 0.5 * h * g1, F)
        g3 = compute_g(p + 0.5 * h * g2, F)
        g4 = compute_g(p + h * g3, F)
        p1 = p + (h / 6) * (g1 + 2 * g2 + 2 * g3 + g4)
        return p1

    def kd_step(self, param, p, h, gamma):
        param.data.add_(0.5 * h * p)
        F = -param.grad
        p = self.solve_p(p, F, h, gamma)
        param.data.add_(0.5 * h * p)
        return p

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            h = group['lr']
            gamma = group['gamma']
            c_init = group['c_init']
            momentum_magnitude = 0
            for param in group['params']:
                if param.grad is None:
                    continue
                if param not in self.p:
                    self.p[param] = self.initialize_momentum(param, c_init)
                momentum_magnitude += torch.dot(self.p[param].flatten(), self.p[param].flatten())
                self.p[param] = self.kd_step(param, self.p[param], h, gamma)
        self.momentum_magnitude_history.append(momentum_magnitude)
        self.t.append(self.t[-1] + h)
        return loss
    

class KineticDescentRed(Optimizer):
    def __init__(self, params, lr, gamma, c_init):
        defaults = {'lr': lr, 'gamma': gamma, 'c_init': c_init}
        super(KineticDescentRed, self).__init__(params, defaults)
        self.p = {}
        self.momentum_magnitude_history = []
        self.t = [0.]

    def c_and_cprime(self, t, gamma, c_init):
        c = c_init * np.exp(-gamma * t)
        cp = -gamma * c_init * np.exp(-gamma * t)
        return c, cp

    def initialize_momentum(self, param, c_init):
        p = -param.grad
        p = p / torch.norm(p, p=2, dtype=torch.float64)
        p = p * math.sqrt(2 * c_init)
        return p
    
    def solve_w_eqn(self, w, t, h, alpha, gamma, c_init):
        w = w / (1 + (w * (np.exp(gamma * h/2) - 1) / (2 * c_init * gamma))) # Quadratic half-step
        w = (w - 2 * alpha / gamma) * np.exp(-0.5 * gamma * h) + 2 * alpha / gamma # Linear full-step
        w = w / (1 + (w * (np.exp(gamma * h/2) - 1) / (2 * c_init * gamma))) # Quadratic half-step
        return w

    def solve_reduced_system(self, w0, t0, h, alpha, gamma, c_init):
        ns = 2
        w = w0
        t = t0
        hs = h / ns
        A = 1.0
        B = 0.0
        for _ in range(ns):
            wh = self.solve_w_eqn(w, t, hs / 2, alpha, gamma, c_init)
            w = self.solve_w_eqn(wh, t + 0.5 * hs, hs / 2, alpha, gamma, c_init)
            ch, _ = self.c_and_cprime(t + 0.5 * hs, gamma, c_init)
            kh = 0.5 * (-gamma - (wh / ch))
            A = A * (1 + 0.5 * hs * kh) / (1 - 0.5 * hs * kh)
            B = (hs + B * (1 + 0.5 * hs * kh)) / (1 - 0.5 * hs * kh)
            t = t + hs
        return A, B

    def kd_step(self, param, p, t, h, gamma, c_init):
        param.data.add_(0.5 * h * p)
        F = -param.grad
        alpha = torch.dot(F.flatten(), F.flatten())
        w0 = torch.dot(p.flatten(), F.flatten())
        A, B = self.solve_reduced_system(w0, t, h, alpha, gamma, c_init)
        p = A * p + B * F
        param.data.add_(0.5 * h * p)
        return p

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            h = group['lr']
            gamma = group['gamma']
            c_init = group['c_init']
            momentum_magnitude = 0
            for param in group['params']:
                if param.grad is None:
                    continue
                if param not in self.p:
                    self.p[param] = self.initialize_momentum(param, c_init)
                momentum_magnitude += torch.dot(self.p[param].flatten(), self.p[param].flatten())
                self.p[param] = self.kd_step(param, self.p[param], self.t[-1], h, gamma, c_init)
        self.momentum_magnitude_history.append(momentum_magnitude)
        self.t.append(self.t[-1] + h)
        return loss