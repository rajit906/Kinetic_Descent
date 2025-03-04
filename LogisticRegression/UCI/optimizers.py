import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


#### UB

def simp(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    return h / 3 * (y[0] + 2 * np.sum(y[2:n:2]) + 4 * np.sum(y[1:n:2]) + y[n])

def U_step(x, p, t, h, pars):
    c0 = pars['decay']['c'](t, pars)
    csqrt = lambda t: pars['decay']['csqrt'](t, pars)
    x += (2 * p / np.sqrt(c0)) * simp(csqrt, t, t + h, 16) # 2 * (1 - np.exp(-0.5 * pars['gamma'] * h)) * p / pars['gamma']
    cnext = pars['decay']['c'](t + h, pars)
    p *= np.sqrt(cnext / c0)
    return x, p

def myrk4step(t,y,h,f):
    F0 = f(t,y)
    Y1 = y + (h/2)*F0
    F1 = f(t+h/2, Y1)
    Y2 = y + (h/2)*F1
    F2 = f(t+h/2, Y2)
    Y3 = y + h*F2
    F3 = f(t+h, Y3)
    y1 = y + (h/6)*(F0 + 2*F1 + 2*F2 + F3)
    return y1

def stable_AB(alpha, h, kappa):
    #A = np.cosh(kappa) / np.cosh(alpha * h + kappa)
    #B = (np.sinh(alpha * h + kappa) - np.sinh(kappa)) / (alpha * np.cosh(alpha * h + kappa))
    if abs(alpha * h) < 1e-5:  # Small alpha * h approximation
        A = 1.0
        B = h
    elif (kappa > 50) or (alpha * h > 50):  # Large kappa or alpha * h approximation
        A = np.exp(-alpha * h)
        exp_alpha_h = np.exp(alpha * h)
        B = (exp_alpha_h - 1) / (alpha * exp_alpha_h)
    else:  # General case
        num_A, num_B  = 1 + np.exp(-2 * kappa), 2 * np.cosh(kappa + alpha * h / 2) * np.sinh(alpha * h / 2)
        denom_A, denom_B = np.exp(alpha * h) + np.exp(-alpha * h) * np.exp(-2 * kappa), alpha * np.cosh(alpha * h + kappa)
        A, B = num_A / denom_A, num_B / denom_B
    return A, B


def KB_step(x, p, h, F, P):
    c = np.dot(p, p)
    if c < P['eps']:
        c += max(P['eps'], 1e-8)
    np0 = np.sqrt(c)
    nF = np.linalg.norm(F)
    alpha = nF / np0
    eta0 = np.dot(p, F) / (nF * np0)
    u0 = 1-eta0
    #print('np0: ' + str(np0), 'nF: ' + str(nF), 'alpha: ' + str(alpha), 'eta: '+ str(eta0))
    if(u0 > 1e-8):
        kappa = np.arctanh(eta0)
        A = np.cosh(kappa) / np.cosh(alpha * h + kappa)
        B = (np.sinh(alpha * h + kappa) - np.sinh(kappa)) / (alpha * np.cosh(alpha * h + kappa))
        #A, B = stable_AB(alpha, h, kappa)
    else:
        vf = lambda t,y: np.array([-alpha*(1 - np.exp(-2*alpha*t)*u0)*y[0], 
                                    1 - alpha*(1 - np.exp(-2*alpha*t)*u0)*y[1]])
        y0 = np.array([1,0])
        y1 = myrk4step(0,y0,h,vf)
        A = y1[0]
        B = y1[1]
    p = A * p + B * F
    return x,p

def kd_step_ub(x0, p0, t, h, pars, Force):
    x, p, Force = x0.reshape(-1), p0.reshape(-1), Force.reshape(-1)
    x, p = U_step(x, p, t, h, pars)
    x, p = KB_step(x, p, h, Force, pars)
    return x.reshape(-1, 1), p.reshape(-1, 1)

### Hamiltonian Descent

def hd_step(w, p, h, pars, Force):
    gamma = pars['gamma']
    delta = 1/(1 + gamma * h)
    p = delta * p + h * delta * Force
    w = w + h * p
    return w, p

# Gradient Descent without and with momentum

def gd_step(w, p, h, Force):
    w = w + h * Force
    return w, p

def gdm_step(w, p, h, pars, Force):
    p = pars['beta'] * p + h * Force
    w = w + p
    return w, p

### RK4
def solve_p(p0, F, h, pars):
    p = p0.copy()
    eps_reg = 1e-6
    # First RK4 step (g1)
    pdotF = np.dot(p, F)
    pdotp = np.dot(p, p) + eps_reg
    g1 = F - (pdotF / pdotp) * p - 0.5 * pars['gamma'] * p
    # Second RK4 step (g2)
    p = p0 + 0.5 * h * g1
    pdotF = np.dot(p, F)
    pdotp = np.dot(p, p) + eps_reg
    g2 = F - (pdotF / pdotp) * p - 0.5 * pars['gamma'] * p
    # Third RK4 step (g3)
    p = p0 + 0.5 * h * g2
    pdotF = np.dot(p, F)
    pdotp = np.dot(p, p) + eps_reg
    g3 = F - (pdotF / pdotp) * p - 0.5 * pars['gamma'] * p
    # Fourth RK4 step (g4)
    p = p0 + h * g3
    pdotF = np.dot(p, F)
    pdotp = np.dot(p, p) + eps_reg
    g4 = F - (pdotF / pdotp) * p - 0.5 * pars['gamma'] * p
    # Final RK4 update for p1
    p1 = p0 + (h / 6) * (g1 + 2 * g2 + 2 * g3 + g4)
    return p1

def kd_step_rk4(x, p, t, h, pars, Force):
    x, p, Force = x.reshape(-1), p.reshape(-1), Force.reshape(-1)
    x = x + 0.5 * h * p
    p = solve_p(p, Force, h, pars)
    x = x + 0.5 * h * p
    return x.reshape(-1,1), p.reshape(-1,1)

### REDUCED

def c_and_cprime(t, pars):
    c = pars['c0'] * np.exp(-pars['gamma'] * t)
    cp = -pars['gamma'] * pars['c0'] * np.exp(-pars['gamma'] * t)
    return c, cp

def solve_w_eqn_RK4(w, t, h, alpha, pars):
    w = w / (1 + (w * (np.exp(pars['gamma'] * h/2) - 1) / (2 * pars['c0'] * pars['gamma']))) # Quadratic half-step
    w = (w - 2 * alpha / pars['gamma']) * np.exp(-0.5 * pars['gamma'] * h) + 2 * alpha / pars['gamma'] # Linear full-step
    w = w / (1 + (w * (np.exp(pars['gamma'] * h/2) - 1) / (2 * pars['c0'] * pars['gamma']))) # Quadratic half-step
    return w

def solve_reduced_system(w0, t0, h, alpha, pars):
    ns = pars.get('ns', 2)
    w = w0
    t = t0
    hs = h / ns
    A = 1.0
    B = 0.0
    for _ in range(ns):
        wh = solve_w_eqn_RK4(w, t, hs / 2, alpha, pars)
        w = solve_w_eqn_RK4(wh, t + 0.5 * hs, hs / 2, alpha, pars)
        ch, _ = c_and_cprime(t + 0.5 * hs, pars)
        kh = 0.5 * (-pars['gamma'] - (wh / ch))
        A = A * (1 + 0.5 * hs * kh) / (1 - 0.5 * hs * kh)
        B = (hs + B * (1 + 0.5 * hs * kh)) / (1 - 0.5 * hs * kh)
        t = t + hs
    return A, B

def kd_step_red(x, p, t, h, pars, Force):
    x, p, Force = x.reshape(-1), p.reshape(-1), Force.reshape(-1)
    x = x + 0.5 * h * p
    alpha = np.linalg.norm(Force)**2
    w0 = np.dot(p, Force)
    A, B = solve_reduced_system(w0, t, h, alpha, pars)
    p = A * p + B * Force
    x = x + 0.5 * h * p
    return x.reshape(-1, 1), p.reshape(-1, 1)