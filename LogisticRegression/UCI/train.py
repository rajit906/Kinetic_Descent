import numpy as np
import math
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from optimizers import *


def sigmoid(X):
    if X >= 0:
        return 1 / (1 + math.exp(-X))
    else:
        return math.exp(X) / (1 + math.exp(X))

def log_loss(y, y_):
    # Numerically stable computation of log(sigmoid) and log(1 - sigmoid)
    if y_ >= 0:
        log_sigmoid = -math.log1p(math.exp(-y_))  # log1p(x) computes log(1 + x) accurately
        log_one_minus_sigmoid = -y_ + log_sigmoid
    else:
        log_one_minus_sigmoid = -math.log1p(math.exp(y_))
        log_sigmoid = y_ + log_one_minus_sigmoid
    
    # Compute log loss
    return -1 * (y * log_sigmoid + (1 - y) * log_one_minus_sigmoid)

def cal_grad(y,y_,x):
    return (y - sigmoid(y_)) * (x.reshape(-1, 1)) 

def predict(w, X):
    y_pred = []
    for i in range(len(X)):
        y_pred.append(np.round(sigmoid(np.dot(w.T, X[i].reshape(X.shape[1], 1)))))
    return y_pred

def train(w_init, X, Y, Xt, Yt, epochs, eta, pars, optimizer, batch_size):
    w = np.copy(w_init)
    N = X.shape[1]
    p = np.zeros_like(w)
    losses = []
    train_acc = []
    test_acc = []
    ps = [p]
    ws = [w]
    t = 0

    epoch_loss = 0
    for i in range(0, len(X), batch_size):
        gt = 0
        batch_loss = 0
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]

        for j in range(len(X_batch)):
            y = Y_batch[j]
            x = X_batch[j].reshape(N, 1)
            y_ = np.dot(w.T, x)
            gt += cal_grad(y, y_, x)
            batch_loss += log_loss(y, y_)

        gt /= batch_size
        batch_loss /= batch_size
        epoch_loss += batch_loss
    losses.append(epoch_loss.item())
    train_acc.append(accuracy_score(Y, predict(w, X)))
    test_acc.append(accuracy_score(Yt, predict(w, Xt)))
    for epoch in range(epochs):
        epoch_loss = 0
        perm = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            gt = 0
            batch_loss = 0
            X_batch = X[perm[i:i + batch_size]]
            Y_batch = Y[perm[i:i + batch_size]]

            for j in range(len(X_batch)):
                y = Y_batch[j]
                x = X_batch[j].reshape(N, 1)
                y_ = np.dot(w.T, x)
                gt += cal_grad(y, y_, x)
                batch_loss += log_loss(y, y_)

            gt /= batch_size
            batch_loss /= batch_size
            if optimizer == 'hd':
                if epoch == 0:
                    p = gt
                w, p = hd_step(w, p, eta, pars, gt)
            elif optimizer == 'kd_rk4':
                if epoch == 0:
                    p = gt
                    p = (p / np.linalg.norm(p)) * np.sqrt(2 * pars['c0'])
                    ps[0] = p
                w, p = kd_step_rk4(w, p, t, eta, pars, gt)
            elif optimizer == 'kd_ub':
                if epoch == 0:
                    p = gt #+ 0.1 * np.random.randn(*p.shape)
                    p = (p / np.linalg.norm(p)) * np.sqrt(2 * pars['c0'])
                    ps[0] = p
                w, p = kd_step_ub(w, p, t, eta, pars, gt)
            elif optimizer == 'kd_red':
                if epoch == 0:
                    p = gt
                    p = (p / np.linalg.norm(p)) * np.sqrt(2 * pars['c0'])
                    ps[0] = p
                w, p = kd_step_red(w, p, t, eta, pars, gt)
            elif optimizer == 'gdm':
                w, p = gdm_step(w, p, eta, pars, gt)
            epoch_loss += batch_loss
            t += eta
            ps.append(p)
            ws.append(w)
        losses.append(np.array(epoch_loss).item())
        train_acc.append(accuracy_score(Y, predict(w, X)))
        test_acc.append(accuracy_score(Yt, predict(w, Xt)))
    return ws, np.array(ps), losses, np.array(train_acc), np.array(test_acc)