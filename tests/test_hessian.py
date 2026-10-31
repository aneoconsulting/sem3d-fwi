"""Hessian tools (semfwi.hessian) on a quadratic problem standing in for Problem: no SEM3D."""
from types import SimpleNamespace

import numpy as np

from semfwi import hessian
from semfwi.config import Grid

RNG = np.random.default_rng(1)


class Quadratic:
    """J(x) = 1/2 (x - x0)^T A (x - x0), exact gradient A (x - x0)."""

    def __init__(self, A, x0):
        self.A, self.x0, self.tags = A, x0, []

    def evaluate(self, x, tag, grad=True):
        self.tags.append(tag)
        r = x - self.x0
        return 0.5 * r @ self.A @ r, None, self.A @ r


def spd(n, rank=None):
    M = RNG.standard_normal((n, rank or n))
    return M @ M.T


def test_hessian_vector_product():
    n = 30
    A = spd(n)
    pb = Quadratic(A, RNG.standard_normal(n))
    x = RNG.standard_normal(n)
    H = hessian.Hessian(pb, x, "h")
    v = RNG.standard_normal(n)
    assert np.allclose(H(v), A @ v, rtol=1e-8)
    assert np.allclose(H(2 * v), 2 * A @ v, rtol=1e-8)
    # one gradient at x, then one per product
    assert pb.tags == ["h/g0", "h/hv000", "h/hv001"]


def test_hessian_vector_product_central():
    """Central differences: two gradients per product, no gradient at x; exact on a cubic J up to
    second order, where the one-sided product is off by O(eps)."""
    n = 20
    A = spd(n)
    c = RNG.standard_normal(n)

    class Cubic(Quadratic):
        def evaluate(self, x, tag, grad=True):
            self.tags.append(tag)
            r = x - self.x0
            return None, None, self.A @ r + 3 * (c @ r) ** 2 * c  # gradient of 1/2 r^T A r + (c.r)^3

    x0 = RNG.standard_normal(n)
    x = x0 + RNG.standard_normal(n)
    exact = lambda v: A @ v + 6 * (c @ (x - x0)) * (c @ v) * c
    v = RNG.standard_normal(n)
    pb = Cubic(A, x0)
    Hc = hessian.Hessian(pb, x, "c", dx=1e-2, central=True)
    Hf = hessian.Hessian(Cubic(A, x0), x, "f", dx=1e-2)
    ec = np.linalg.norm(Hc(v) - exact(v)) / np.linalg.norm(exact(v))
    ef = np.linalg.norm(Hf(v) - exact(v)) / np.linalg.norm(exact(v))
    assert ec < 1e-10 < ef
    assert pb.tags == ["c/hv000p", "c/hv000m"]


def test_low_rank():
    n, k = 40, 6
    A = spd(n, rank=k)
    pb = Quadratic(A, np.zeros(n))
    w, U = hessian.low_rank(hessian.Hessian(pb, np.zeros(n), "h", g0=np.zeros(n)), n, k)
    assert np.allclose(w, np.sort(np.linalg.eigvalsh(A))[::-1][:k], rtol=1e-6)
    assert np.allclose(U @ np.diag(w) @ U.T, A, atol=1e-6 * np.abs(A).max())


def test_bump():
    g = Grid(xmin=(0., 0., -4.), h=1., n=(5, 5, 5))
    mask = np.ones(g.shape, bool)
    mask[0] = False
    pb = SimpleNamespace(cfg=SimpleNamespace(grid=g), mask=mask, n=int(mask.sum()), active=("a", "b"))
    v = hessian.bump(pb, "b", (2.2, 1.9, -2.))
    a, b = v[:pb.n], v[pb.n:]
    assert not a.any()
    f = np.zeros(g.shape)
    f[mask] = b
    assert np.unravel_index(np.argmax(f), g.shape) == (2, 2, 2) and np.isclose(f.max(), 1.)
