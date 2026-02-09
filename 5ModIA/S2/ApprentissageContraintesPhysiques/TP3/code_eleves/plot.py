import matplotlib.pyplot as plt
import filters 
import numpy as np
import torch 
import lorenz_exp as le
import lin2d_exp as l2d_exp

x_dim = 2
N = 1
dt = 0
init = "0"
mb = 2 
T = 50
b_size = l2d_exp.b_size

lin2d = filters.Lin2d(x_dim, N, dt, init)

exp_params = l2d_exp.get_params()

prop_params = exp_params["prop_kwargs"]
obs_params = exp_params["obs_kwargs"]

prop = filters.ConstructorProp(**prop_params)
obs = filters.ConstructorObs(**obs_params)

xtps = torch.empty(T,b_size, x_dim)

yts = torch.empty(T,b_size,x_dim)

xtp = lin2d.x0

X = [lin2d.x0]

for t in range(T):
    #Solution parfaite
    X.append(lin2d.forward(X[t]))
    #Solution perturbée
    xtp = prop(xtp).sample()
    yt = obs(xtp).sample()
    xtps[t] = xtp 
    yts[t] = yt


X = np.array(X)

print(xtps.shape)

plt.scatter(X[:,0,0],X[:,0,1], label='solution parfaite')
plt.scatter(xtps[:,0,0], xtps[:,0,1],label="x solution perturbée")
plt.scatter(yts[:,0,0], yts[:,0,1],label="y solution perturbée")

plt.legend()
plt.show()

