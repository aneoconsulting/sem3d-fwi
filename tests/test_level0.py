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


def test_config_roundtrip():
    from semfwi import config
    cfg = Config(workdir="/a", stations=[(1., 2., 0.)], layers=[(100., 2), (300., 6)], adjoint="extended",
                 receiver_grid=(np.array([0., 50.]), np.array([0., 50.]), 0.), nprocs=2,
                 sources=[sem_io.point_source((1., 2., 3.), (0., 0., 1.))])
    d = config.to_dict(cfg)
    import json
    back = config.from_dict(json.loads(json.dumps(d)), "/b", sem3d_bin="/bin")
    assert back.workdir == "/b" and back.sources == [] and back.sem3d_bin == "/bin"
    assert back.grid == cfg.grid and back.nt == cfg.nt and back.layer_list == [[100., 2], [300., 6]]
    assert config.to_dict(back) == d


def test_armonik_io_roundtrip(tmp_path):
    from semfwi import armonik_io
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_text("x")
    (tmp_path / "a" / "y.h5").write_bytes(b"\0\1")
    armonik_io.unpack(armonik_io.pack_tree(tmp_path / "a"), tmp_path / "b")
    assert (tmp_path / "b" / "x.txt").read_text() == "x" and (tmp_path / "b" / "y.h5").read_bytes() == b"\0\1"
    a = RNG.standard_normal((3, 4, 3))
    assert np.array_equal(armonik_io.decode_array(armonik_io.encode_array(a)), a)
    r = armonik_io.decode_result(armonik_io.encode_result(dict(u=a, gl=a[0])))
    assert np.array_equal(r["u"], a) and np.array_equal(r["gl"], a[0])


class _FakeRunner:
    """Prepares directories without a mesh; input_hash as in Runner."""
    input_hash = Runner.input_hash
    input_files = Runner.input_files

    def prepare(self, d):
        os.makedirs(d, exist_ok=True)


class _FakeBackend:
    """u = sum of the Mu model read from the forward directory; gl = gm = 1."""
    def __init__(self, cfg):
        self.cfg, self.calls = cfg, []

    def run(self, jobs):
        self.calls.append([(j.index, j.grad) for j in jobs])
        out = []
        for j in jobs:
            _, mu, _ = model.read_model(j.fwd)
            r = dict(u=np.full((len(self.cfg.stations), self.cfg.nt, 3), mu.sum() * 1e-12 + j.index))
            if j.grad:
                r.update(gl=np.ones(self.cfg.grid.shape), gm=np.ones(self.cfg.grid.shape))
            out.append(r)
        return out


def test_evaluate_cache(tmp_path):
    """evaluate resubmits only the sources whose inputs have changed."""
    from semfwi.problem import Problem
    cfg = Config(workdir=str(tmp_path), sim_time=0.01, stations=[(1., 2., 0.)],
                 sources=[sem_io.point_source((1., 2., -3.), (0., 0., 1.), func="ricker", tau=0.1, freq=8.)] * 2)
    one = np.ones(cfg.grid.shape)
    be = _FakeBackend(cfg)
    dobs = np.zeros((2, 1, cfg.nt, 3))
    pb = Problem(cfg, _FakeRunner(), dobs, cfg.lam0 * one, cfg.mu0 * one, cfg.rho0 * one, backend=be)
    x = np.zeros(pb.size)
    J, u, g = pb.evaluate(x, "t")
    assert be.calls == [[(0, True), (1, True)]] and g.shape == x.shape and pb.nsim == 4
    J2, u2, g2 = pb.evaluate(x, "t")
    assert be.calls[1:] == [] and J2 == J and np.array_equal(g2, g)
    J3, _, g3 = pb.evaluate(x, "t", grad=False)          # input.spec changed (no strains)
    assert be.calls[1:] == [[(0, False), (1, False)]] and g3 is None and J3 == J
    pb.evaluate(x + 0.01, "t")                             # model changed
    assert be.calls[2:] == [[(0, True), (1, True)]]


@pytest.mark.sem3d
@pytest.mark.skipif(not os.environ.get("SEM3D_BIN"), reason="SEM3D_BIN not set")
def test_worker_compute_matches_local(tmp_path, monkeypatch):
    """The ArmoniK worker path (archives, payload, cached mesh) gives the same result as
    LocalBackend, without ArmoniK."""
    import json
    from semfwi import armonik_io, armonik_worker, config
    from semfwi.backend import Job, LocalBackend
    cfg = Config(workdir=str(tmp_path / "local"), sim_time=0.1, stations=[(262., 233., 0.), (121., 297., 0.)],
                 sources=[sem_io.point_source((163., 181., -77.), (1., 0., 0.), func="ricker", tau=0.05,
                                              freq=10.)])
    runner = Runner(cfg)
    d = str(tmp_path / "local" / "fwd")
    runner.prepare(d)
    one = np.ones(cfg.grid.shape)
    model.write_model(d, cfg.grid, cfg.lam0 * one, cfg.mu0 * (one + 0.05), cfg.rho0 * one)
    sem_io.write_input(d, cfg, cfg.sources, save_strain=True)
    dobs = RNG.standard_normal((2, cfg.nt, 3)) * 1e-9
    deps = {"m": armonik_io.pack_tree(os.path.realpath(f"{d}/sem")),
            "i": armonik_io.pack(d, Runner.input_files(d)), "d": armonik_io.encode_array(dobs)}
    payload = json.dumps(dict(version=1, cfg=config.to_dict(cfg), grad=True, mesh="m", inputs="i", dobs="d"))
    monkeypatch.setattr(armonik_worker, "SCRATCH", str(tmp_path / "scratch"))
    (tmp_path / "w").mkdir()
    remote = armonik_io.decode_result(armonik_worker.compute(payload, deps, str(tmp_path / "w")))
    local = LocalBackend(cfg, runner).run([Job(0, d, str(tmp_path / "local" / "adj"), dobs, True)])[0]
    for k in ("u", "gl", "gm"):
        assert np.array_equal(remote[k], local[k]), k
    assert np.abs(local["gm"]).max() > 0
