import numpy as np
import matplotlib.pyplot as plt

def kd_traj(x0, p0, h, Ns, Pars, Model, tol = 1e-8):
    T = [0]
    X = [x0]
    P = [p0]
    KE = [np.dot(p0, p0)]
    PE = [Model['U'](x0, Pars)]
    x = np.array(x0)
    p = np.array(p0)
    t = 0

    for _ in range(Ns):
        x, p = kd_step(x, p, t, h, Pars, Model, tol = tol)
        t += h
        T.append(t)
        X.append(x)
        P.append(p)
        KE.append(np.dot(p, p))
        PE.append(Model['U'](x, Pars))

    return np.array(T), np.array(X), np.array(P), np.array(KE), np.array(PE)

def U_step(x0, p0, t0, h, Pars):
    c0 = Pars['decay']['c'](t0, Pars)
    csqrt = lambda t: Pars['decay']['csqrt'](t, Pars)
    analytical = 2 * (1 - np.exp(-0.5 * Pars['gamma'] * h)) * p0 / Pars['gamma']
    x1 = x0 + (2 * p0 / np.sqrt(c0)) * simp(csqrt, t0, t0 + h, 16) # analytical
    p1 = p0 * np.sqrt(Pars['decay']['c'](t0 + h, Pars) / c0)
    return x1, p1

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
    
def KB_step(x, p, h, Pars, Model, tol):
    c = np.dot(p, p)
    if c < tol:
        c += tol
    F = Model['F'](x, Pars)
    nF = np.linalg.norm(F)
    F_tilde, p_tilde = F/nF, p/np.sqrt(c)
    eta = np.dot(p_tilde, F_tilde)
    alpha = nF / np.sqrt(c)
    u = 1-eta
    if (u > tol):
        kappa = np.arctanh(eta)
        #print('np0: ' + str(np.sqrt(c)), 'nF: ' + str(nF), 'alpha: ' + str(alpha), 'eta: '+ str(eta), 'kappa: ' + str(kappa))
        A = np.cosh(kappa) / np.cosh(alpha * h + kappa)
        B = (np.sinh(alpha * h + kappa) - np.sinh(kappa)) / (alpha * np.cosh(alpha * h + kappa))
        #A, B = stable_AB(alpha, h, kappa)
    else:
        vf = lambda t,y: np.array([-alpha*(1 - np.exp(-2*alpha*t)*u)*y[0], 1 - alpha*(1 - np.exp(-2*alpha*t)*u)*y[1]])
        y0 = np.array([1,0])
        y1 = myrk4step(0,y0,h,vf)
        A, B = y1[0], y1[1]
    p = A * p + B * F 
    return x,p,F

def simp(f, a, b, n):
    if n % 2 == 1:
        n += 1  # Ensure n is even
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    return h / 3 * (y[0] + 2 * np.sum(y[2:n:2]) + 4 * np.sum(y[1:n:2]) + y[n])

def kd_step(x0, p0, t0, h, Pars, Model, tol):
    x1, p1, F = KB_step(x0, p0, h, Pars, Model, tol)
    x1, p1 = U_step(x1, p1, t0, h, Pars)
    return x1, p1


def gd_traj(x0, p0, h, Ns, Pars, Model):
    T = [0]  # Time steps
    X = [x0]  # Positions
    P = [p0]  # Momentum/Velocity
    KE = [np.dot(p0, p0)]  # Kinetic energy
    PE = [Model['U'](x0, Pars)]  # Potential energy

    x = np.array(x0)
    v = np.zeros_like(p0)  # Velocity initialized to zero
    t = 0

    for _ in range(Ns):
        x, v = gd_step_with_momentum(x, v, h, Pars, Model)
        t += h
        T.append(t)
        X.append(x)
        P.append(v)  # Store velocity as "momentum"
        KE.append(np.dot(v, v))  # Kinetic energy is based on velocity
        PE.append(Model['U'](x, Pars))

    return np.array(T), np.array(X), np.array(P), np.array(KE), np.array(PE)


def gd_step_with_momentum(x, v, h, Pars, Model):
    grad_U = -Model['F'](x, Pars)  # Force is the negative gradient
    v = Pars['beta'] * v - h * grad_U  # Update velocity with momentum
    x = x + v  # Update position
    return x, v

def gd_step(x, p, t, h, Pars, Model):
    return x + h * Model['F'](x, Pars), p