"""
Author: minhhai
"""
# %%
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
# %% Question 1


def generate_data(d: int, n: int, theta: ndarray, sigma: float) -> ndarray:
    X = np.random.randn(d, n)
    noise = np.random.randn(n) * sigma
    Y = np.matmul(X.T, theta) + noise
    return X, Y

# %% Question 2: mean risk


def E(w: ndarray, theta: ndarray, sigma: float):
    return 0.5 * np.sum((w - theta) ** 2) + 0.5 * sigma ** 2

# %% Question 3: empirical risk


def En(w: ndarray, X: ndarray, Y: ndarray):
    n = X.shape[1]
    return 0.5 * np.sum((np.matmul(X.T, w) - Y) ** 2) / n

# %% Question 4: grad of empirical risk


def grad_En(w: ndarray, X: ndarray, Y: ndarray):
    n = X.shape[1]
    return np.matmul(X, np.matmul(X.T, w) - Y) / n

# %% Question 5: stochastic grad of empirical risk


def grad_sto_En(w, X, Y, batch_size):
    n = X.shape[1]
    idx = np.random.randint(0, n, size=batch_size)
    X_batch = X[:, idx]
    Y_batch = Y[idx]

    return grad_En(w, X_batch, Y_batch)

# %% Question 6: Lipschitz constant


def lipschitz_constant(X: ndarray):
    n = X.shape[1]
    L = np.max(np.linalg.svd(X, compute_uv=False))

    return L / np.sqrt(n)

# %% Question 7: conjugate gradient for solving linear system

def conjugate_gradient(X, Y, eps=1e-8):
    A = np.matmul(X, X.T)
    b = np.matmul(X, Y)

    d = X.shape[0]
    w = np.random.randn(d)

    r_old = b - np.matmul(A, w)
    p = np.copy(r_old)

    for i in range(d**2):
        Ap = np.matmul(A, p)
        norm_r_old = np.sum(r_old * r_old)
        alpha = norm_r_old / np.sum(p * Ap)
        w = w + alpha * p
        r_new = r_old - alpha * Ap
        norm_r_new = np.sum(r_new * r_new)
        if norm_r_new < eps:
            print("Conjugate gradient converged at iteration {}".format(i))
            return w
        beta = norm_r_new / norm_r_old
        p = r_new + beta * p
        r_old = r_new

    return w



# %% Question 8: gradient descent on En
d = 5
n = 1000
sigma = 5.

theta = np.ones(d)
n_iter = 1000
X, Y = generate_data(d=d, n=n, theta=theta, sigma=sigma)
L = lipschitz_constant(X)

# Verify that the Lipschitz constant is correct
w = np.random.rand(d)
w_prime = np.random.rand(d)
if np.linalg.norm(grad_En(w, X, Y) - grad_En(w_prime, X, Y)) < L * np.linalg.norm(w - w_prime):
    print("Lipschitz constant is correct")
else:
    print("Lipschitz constant is not correct")

# Verify that the conjugate gradient converges to the true solution
print("Solution of conjugate gradient:" ,conjugate_gradient(X, Y, eps=1e-16))
print("Solution of numpy: ", np.linalg.solve(np.matmul(X, X.T), np.matmul(X, Y)))

# You have to tune this lr carefully
lr = 1e-3
