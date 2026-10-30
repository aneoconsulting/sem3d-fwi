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
