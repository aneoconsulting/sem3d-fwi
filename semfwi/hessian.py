"""Hessian of the misfit J(x) of a Problem, from gradients only (no second-order adjoint, no change
to SEM3D): Hessian-vector products, point-spread functions and a low-rank approximation.

H v ~ (g(x + eps v) - g(x)) / eps, with eps = dx / max|v|: the model changes by at most dx (relative
perturbation, x is relative to the reference model). Each product costs one gradient (forward and
adjoint simulations of every source). At the true model with noise-free data, the residuals and
g(x) vanish and H is the Gauss-Newton Hessian, the one used to judge an acquisition design.

A point-spread function is H v for a bump v around one node of one parameter: its spread shows the
resolution there, and its part in the other parameter's block the lambda-mu crosstalk.
"""
import numpy as np


class Hessian:
    """v -> H v at model x. g0: gradient at x if already known (otherwise computed by the first
    product). The simulations go to workdir/tag/hv{k}."""

    def __init__(self, pb, x, tag, g0=None, dx=1e-3):
        self.pb, self.x, self.tag, self.dx = pb, x, tag, dx
        self.g0 = g0
        self.count = 0

    def gradient(self):
        if self.g0 is None:
            _, _, self.g0 = self.pb.evaluate(self.x, f"{self.tag}/g0")
        return self.g0

    def __call__(self, v):
        g0 = self.gradient()
        eps = self.dx / np.max(np.abs(v))
        _, _, g = self.pb.evaluate(self.x + eps * v, f"{self.tag}/hv{self.count:03d}")
        self.count += 1
        return (g - g0) / eps


def bump(pb, block, point, width=None):
    """Vector of unknowns: a Gaussian bump of the parameter block ("a" or "b") centred on the grid
    node closest to point (x, y, z), standard deviation width (one grid step by default)."""
    assert block in pb.active, block
    grid = pb.cfg.grid
    width = width or grid.h
    node = [grid.axis(c)[np.argmin(np.abs(grid.axis(c) - point[c]))] for c in range(3)]
    X, Y, Z = grid.mesh()
    f = np.exp(-((X - node[0]) ** 2 + (Y - node[1]) ** 2 + (Z - node[2]) ** 2) / (2 * width ** 2))
    blocks = {p: (f[pb.mask] if p == block else np.zeros(pb.n)) for p in pb.active}
    return np.concatenate([blocks[p] for p in pb.active])


def low_rank(H, n, k, oversample=5, seed=0):
    """Dominant eigenpairs of H (symmetric, size n) by randomized range finding (Halko, Martinsson
    and Tropp 2011): 2 (k + oversample) products. Return the k largest eigenvalues (descending)
    and the eigenvectors as columns."""
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n, k + oversample))
    Q, _ = np.linalg.qr(np.column_stack([H(w) for w in omega.T]))
    B = Q.T @ np.column_stack([H(q) for q in Q.T])
    w, U = np.linalg.eigh(0.5 * (B + B.T))
    order = np.argsort(w)[::-1][:k]
    return w[order], Q @ U[:, order]
