import numpy as np
from fakedata import FakeGausianBall
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
N_groups = 2
color = iter(cm.rainbow(np.linspace(0,1,N_groups)))

data = FakeGausianBall(2, 30, N_groups)
X = data.X
y = data.y

Xs = data.Xs
for g in range(N_groups):
    c = next(color)
    plt.scatter(Xs[g][1,:], Xs[g][0,:], color=c)
plt.show()
