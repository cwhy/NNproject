import numpy as np
import numpy.random as rnd
from fakedata import GaussianBall, GaussianLine, GaussianXOR
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
# rnd.seed(120)


def logi(x):
    return 1 / (1 + np.exp(-x))


def forward(x_, w_):
    x_ = np.hstack((1, x_))
    return logi(x_.dot(w_))


def forward_(X, w_):
    x0_ = np.ones((1, X.shape[1]))
    X = np.vstack((x0_, X))
    a = X.T.dot(w_)
    return logi(a)


def backward_(y, X, phat):
    x0_ = np.ones((1, X.shape[1]))
    X = np.vstack((x0_, X))
    return np.sum((y - phat) * X, 1)

N_groups = 2
dim = 2
N_sample_e = 100
max_iter = 100000
N_sample = N_sample_e * N_groups
data = GaussianBall(dim, N_sample_e, N_groups)
data = GaussianLine(dim, N_sample_e, N_groups, 100)
data = GaussianXOR(dim, N_sample_e, N_groups)
alpha = [0.01]*3
X = data.X
y = data.y

w_ = np.ravel(rnd.random((1,dim + 1)))

dw_ = [1, 1, 1]
dw__ = dw_
i = 0
Err = []
while abs(np.sum(dw_)) >= 0.0001:
    phat = forward_(X, w_)

    dw_ = backward_(y, X, phat)
    Err.append(dw_)
    for j in range(3):
        if np.sign(dw_[j]) == -np.sign(dw__[j]):
            alpha[j] *= 0.5
        else:
            alpha[j] *= 1.2
    dw__ = dw_
    w_ += alpha * dw_
    i += 1
    if i >= max_iter:
        break
x_ = np.ravel([1, 0.5])
pbar = forward_(X, w_)
print w_
print pbar
print y

x0 = np.linspace(np.min(X[0,:]), np.max(X[0,:]), 20)
x1 = np.linspace(np.min(X[1,:]), np.max(X[1,:]), 20)
x0_mesh, x1_mesh = np.meshgrid(x0, x1)
x_mesh = np.vstack((np.ravel(x0_mesh), np.ravel(x1_mesh)))
p_mesh = forward_(x_mesh, w_).reshape((20,20))

Xs = data.Xs
color = iter(cm.rainbow(np.linspace(0,1,N_groups)))
h = plt.contour(x0_mesh, x1_mesh, p_mesh)
for g in range(N_groups):
    c = next(color)
    plt.scatter(Xs[g][0,:], Xs[g][1,:], color=c)
plt.clabel(h, inline=1, fontsize=10)
plt.show()
plt.plot(Err)
plt.show()
