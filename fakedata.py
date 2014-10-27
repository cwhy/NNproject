from __future__ import division
import numpy as np
import numpy.random as rnd


class GaussianBall:
    def __init__(self, dim, N_samples, N_groups):
        self.Xs = []
        self.ys = []
        for c in range(N_groups):
            center = rnd.choice(N_groups*2, size=dim, replace=False)
            _std = N_groups/2
            _X = np.zeros((dim, N_samples))
            for d in range(dim):
                _X[d,:] = rnd.normal(center[d], _std, size=N_samples)
            self.Xs.append(_X)
            self.ys.append(np.ones((1, N_samples))*c)
        self.X = np.hstack(self.Xs)
        self.y = np.hstack(self.ys)


class GaussianLine:
    def __init__(self, dim, N_samples, N_groups, k):
        self.Xs = []
        self.ys = []
        for c in range(N_groups):
            _b = rnd.rand()*N_groups*k
            x0 = rnd.rand(N_samples)*10
            _std = N_groups
            _X = np.zeros((dim, N_samples))
            _X[0,:] = x0
            for d in range(1, dim):
                y0 = _b + k*x0
                _X[d,:] = rnd.normal(y0, _std, size=N_samples)
            self.Xs.append(_X)
            self.ys.append(np.ones((1, N_samples))*c)
        self.X = np.hstack(self.Xs)
        self.y = np.hstack(self.ys)


class GaussianXOR:
    def __init__(self, dim, N_samples, N_groups):
        self.Xs = []
        self.ys = []
        for c in range(N_groups):
            center = rnd.choice(N_groups*4, size=dim*2, replace=False)
            _std = N_groups/2
            _X = np.zeros((dim, N_samples))
            for d in range(dim):
                N_0 = int(N_samples/(1 + rnd.rand()))
                _X[d,:N_0] = rnd.normal(center[d], _std, size=N_0)
                _X[d,N_0:] = rnd.normal(center[d + dim],
                                        _std,
                                        size=N_samples - N_0)
            self.Xs.append(_X)
            self.ys.append(np.ones((1, N_samples))*c)
        self.X = np.hstack(self.Xs)
        self.y = np.hstack(self.ys)
