import numpy as np
import numpy.random as rnd
from fakedata import GaussianBall, GaussianLine, GaussianXOR
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
# rnd.seed(120)


def logi(x):
    return 1 / (1 + np.exp(-x))


def forward2_(X, w_):
    x0_ = np.ones((1, X.shape[1]))
    X = np.vstack((x0_, X))
    a = X.T.dot(w_)
    return logi(a)


def forward1_(X, W):
    x0_ = np.ones((1, X.shape[1]))
    X = np.vstack((x0_, X))
    a = W.dot(X)
    return logi(a)


def forward_(X, W, w_):
    _X = forward1_(X, W)
    out_ = forward2_(_X, w_)
    return out_


def backward_(y, X, phat, W, w_):
    _X = forward1_(X, W)
    x0_ = np.ones((1, _X.shape[1]))
    _X0 = np.vstack((x0_, _X))
    Err2 = y - phat.T
    dw_ = _X0.dot(Err2.T)
    Err1 = w_[1:,:].dot(Err2) * (_X * (1 - _X))
    x0_ = np.ones((1, X.shape[1]))
    X = np.vstack((x0_, X))
    dW = Err1.dot(X.T)
    return dw_, dW


def backward1_(y, X, phat, W):
    _X = forward1_(X, W)
    x0_ = np.ones((1, _X.shape[1]))
    _X = np.vstack((x0_, _X))
    dw_ = _X.dot((y - phat).T)
    return np.ravel(dw_)

N_groups = 2
dim = 2
N_sample_e = 100
max_iter = 10000
N_sample = N_sample_e * N_groups
data = GaussianBall(dim, N_sample_e, N_groups)
data = GaussianLine(dim, N_sample_e, N_groups, 100)
data = GaussianXOR(dim, N_sample_e, N_groups)
X = data.X
y = data.y

Nu_2 = 5

w_ = rnd.random((Nu_2 + 1, 1))
alpha = 0.01*np.ones(w_.shape)
W = rnd.random((Nu_2, dim + 1))
Alpha = 0.01*np.ones(W.shape)

dw_ = np.ones(w_.shape)
dw__ = dw_
dW = np.ones(W.shape)
dW_ = dW
i = 0
Err = []
while abs(np.sum(dw_)) >= 0.0001:
    phat = forward_(X, W, w_)

    dw_, dW = backward_(y, X, phat, W, w_)
    Err.append(sum(dw_**2))
    for j in range(alpha.size):
        if np.sign(dw_[j]) == -np.sign(dw__[j]):
            alpha[j] *= 0.5
        else:
            alpha[j] *= 1.2

    for i in range(Alpha.shape[0]):
        for j in range(Alpha.shape[1]):
            if np.sign(dW[i, j]) == -np.sign(dW_[i, j]):
                Alpha[i, j] *= 0.5
            else:
                Alpha[i, j] *= 1.2

    dw__ = dw_
    dW_ = dW

    w_ += alpha * dw_
    W += Alpha * dW
    i += 1
    if i >= max_iter:
        break
x_ = np.ravel([1, 0.5])
pbar = forward_(X, W, w_)
print pbar
print y
plt.plot(Err)
plt.show()

x0 = np.linspace(np.min(X[0,:]), np.max(X[0,:]), 20)
x1 = np.linspace(np.min(X[1,:]), np.max(X[1,:]), 20)
x0_mesh, x1_mesh = np.meshgrid(x0, x1)
x_mesh = np.vstack((np.ravel(x0_mesh), np.ravel(x1_mesh)))
p_mesh = forward_(x_mesh, W, w_).reshape((20,20))

Xs = data.Xs
color = iter(cm.rainbow(np.linspace(0,1,N_groups)))
h = plt.contour(x0_mesh, x1_mesh, p_mesh)
for g in range(N_groups):
    c = next(color)
    plt.scatter(Xs[g][0,:], Xs[g][1,:], color=c)
plt.clabel(h, inline=1, fontsize=10)
plt.show()
