"""Level 0 tests of semfwi. The test marked `sem3d` runs two simulations (about 15 s):
    pytest -q tests            # all
    pytest -q tests -m "not sem3d"
"""
import os

import h5py
import numpy as np
import pytest

from semfwi import misfit, model, optim, sem_io
from semfwi.config import Config, Grid
from semfwi.runner import Runner

RNG = np.random.default_rng(0)


def test_adjoint_history_convention():
    dt = 1e-3
    res = RNG.standard_normal((2, 50, 3))
    tau, hist = misfit.adjoint_history(res, dt)
    assert np.allclose(tau, dt * (np.arange(50) + 0.5))
    assert np.array_equal(hist[:, 0], res[:, -1])
    assert np.array_equal(hist[:, ::-1], res)


def test_cost_and_residual():
    u, d = RNG.standard_normal((2, 3, 40, 3))
    dt = 2e-3
    assert np.isclose(misfit.cost(u, d, dt), 0.5 * dt * np.sum(misfit.residual(u, d) ** 2))
    # derivative of J with respect to u: residual * dt
    du = RNG.standard_normal(u.shape)
    h = 1e-6
    fd = (misfit.cost(u + h * du, d, dt) - misfit.cost(u - h * du, d, dt)) / (2 * h)
    assert np.isclose(fd, dt * np.sum(misfit.residual(u, d) * du))


def test_trilinear_exact_on_linear_fields():
    g = Grid()
    pts = np.array(g.xmin) + RNG.random((200, 3)) * (np.array(g.xmax) - np.array(g.xmin))
    idx, w = model.trilinear(g, pts)
    assert np.allclose(w.sum(1), 1)
    X, Y, Z = g.mesh()
    f = 1. + 2 * X - 3 * Y + 0.5 * Z
    assert np.allclose(model.interpolate(g, f, idx, w), 1. + pts @ [2., -3., 0.5])


def test_trilinear_dot_product():
    """<P f, v> = <f, P^T v>: project is indeed the transpose of interpolate."""
    g = Grid()
    pts = np.array(g.xmin) + RNG.random((500, 3)) * (np.array(g.xmax) - np.array(g.xmin))
    idx, w = model.trilinear(g, pts)
    f = RNG.standard_normal(g.shape)
    v = RNG.standard_normal(len(pts))
    lhs = model.interpolate(g, f, idx, w) @ v
    rhs = np.sum(f * model.project(g, v, idx, w))
    assert np.isclose(lhs, rhs, rtol=1e-12)


def test_write_model_layout(tmp_path):
    g = Grid()
    lam = 4e9 + RNG.random(g.shape)
    mu = 2e9 + RNG.random(g.shape)
    rho = 2000. + RNG.random(g.shape)
    model.write_model(tmp_path, g, lam, mu, rho)
    with h5py.File(tmp_path / "Mu.h5") as f:
        assert f["Mu/samples"].shape == g.shape[::-1]
        assert np.allclose(f["Mu"].attrs["xMinGlob"], g.xmin)
        assert np.allclose(f["Mu"].attrs["xMaxGlob"], g.xmax)
        assert f["Mu/samples"][3, 2, 1] == mu[1, 2, 3]
    for a, b in zip(model.read_model(tmp_path), (lam, mu, rho)):
        assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        model.write_model(tmp_path, g, lam, -mu, rho)


def test_interior_mask():
    g = Grid()
    m = model.interior_mask(g, 400., 400.)
    assert m.shape == g.shape
    X, Y, Z = g.mesh()
    assert X[m].min() == 25. and X[m].max() == 375. and Z[m].min() == -375. and Z[m].max() == 0.


def test_lbfgs_quadratic():
    """On an ill-conditioned quadratic, L-BFGS with exact step converges in few iterations."""
    n = 30
    A = np.diag(np.logspace(0, 3, n))
    b = RNG.standard_normal(n)
    x = np.zeros(n)
    opt = optim.LBFGS(m=10)
    g = A @ x - b
    for _ in range(60):
        p = opt.direction(g)
        a = -(g @ p) / (p @ A @ p)
        xn = x + a * p
        gn = A @ xn - b
        opt.update(xn - x, gn - g)
        x, g = xn, gn
    assert np.linalg.norm(g) < 1e-8 * np.linalg.norm(b)


@pytest.mark.sem3d
@pytest.mark.skipif(not os.environ.get("SEM3D_BIN"), reason="SEM3D_BIN not set")
def test_file_model_equals_material_input(tmp_path):
    """A homogeneous model read via material.spec gives the same traces as material.input."""
    cfg = Config(workdir=str(tmp_path), sim_time=0.2, stations=[(262., 233., -141.), (121., 297., 0.)],
                 sources=[sem_io.point_source((163., 181., -77.), (1., 0., 0.), func="ricker", tau=0.1,
                                              freq=8.)])
    runner = Runner(cfg)
    g = cfg.grid
    tr = []
    for name, with_file in (("file", True), ("input", False)):
        d = str(tmp_path / name)
        runner.prepare(d)
        if with_file:
            one = np.ones(g.shape)
            model.write_model(d, g, cfg.lam0 * one, cfg.mu0 * one, cfg.rho0 * one)
        sem_io.write_input(d, cfg, cfg.sources)
        runner.run(d)
        tr.append(sem_io.read_traces(d, 2))
    assert np.max(np.abs(tr[0] - tr[1])) <= 1e-10 * np.max(np.abs(tr[1]))


@pytest.mark.parametrize("param", ["lame", "vel"])
def test_chain_rule(param):
    """For J(lambda, mu) = <a, lambda> + <b, mu> + 1/2 <c, mu^2>, the gradient in x derived from
    dJ/dlambda and dJ/dmu by Problem.chain matches a centered finite difference."""
    from semfwi.problem import Problem
    cfg = Config(workdir="/nonexistent")
    g = cfg.grid
    one = np.ones(g.shape)
    pb = Problem(cfg, None, None, cfg.lam0 * one, cfg.mu0 * one, cfg.rho0 * one, param=param)
    a, b, c = (RNG.standard_normal(g.shape) for _ in range(3))
    c *= 1e-9

    def J(x):
        lam, mu, _ = pb.to_model(x)
        return np.sum(a * lam) + np.sum(b * mu) + 0.5 * np.sum(c * mu ** 2)

    x = 0.05 * RNG.standard_normal(pb.size)
    lam, mu, _ = pb.to_model(x)
    gx = pb.chain(x, a, b + c * mu)
    dx = RNG.standard_normal(pb.size)
    h = 1e-6
    fd = (J(x + h * dx) - J(x - h * dx)) / (2 * h)
    assert np.isclose(gx @ dx, fd, rtol=1e-6)
    assert np.allclose(pb.to_x(lam, mu), x)
