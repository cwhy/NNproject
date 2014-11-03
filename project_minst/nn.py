import pandas as pd
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
# rnd.seed(120)

train_raw = pd.read_csv('./data/train.csv')
test_raw = pd.read_csv('./data/test.csv')

train_all = train_raw.values


def dat_get(dataset):
    labels = dataset[:,-1]
    features = dataset[:,0:-1]
    return features, labels


def compare(lb1, lb2):
    acc = np.sum(lb1 == lb2)
    return acc


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

N_groups = 10
dim = 784
N_sample = 100
to_train = train_all[0:N_sample,:]
to_validate = train_all[1001:2000,:]

tr_ft, tr_lb = dat_get(to_train)
va_ft, va_lb = dat_get(to_validate)
X = tr_ft.T
y = tr_lb

Nu_2 = 5
max_iter = 10000

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
while abs(np.sum(dw_)) >= 0.01:
    phat = forward_(X, W, w_)

    dw_, dW = backward_(y, X, phat, W, w_)
    print sum(dw_**2)
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
plt.plot(Err)
plt.show()
