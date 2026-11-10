"""Level 0 tests of semfwi. The test marked `sem3d` runs two simulations (about 15 s):
    pytest -q tests            # all
    pytest -q tests -m "not sem3d"
"""
import dataclasses
import json
import os

import h5py
import numpy as np
import pytest

from semfwi import hexmesh, misfit, model, optim, sem_io
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


def test_weighted_cost_and_residual():
    u, d = RNG.standard_normal((2, 3, 40, 3))
    dt = 2e-3
    w = RNG.random((3, 3))
    w[1] = 0.
    # residual weighted for the adjoint sources: derivative of the weighted J with respect to u
    du = RNG.standard_normal(u.shape)
    h = 1e-6
    fd = (misfit.cost(u + h * du, d, dt, w) - misfit.cost(u - h * du, d, dt, w)) / (2 * h)
    assert np.isclose(fd, dt * np.sum(misfit.residual(u, d, w) * du))
    # a zero weight removes the station; weights 1 change nothing; one source or several
    u2 = u.copy()
    u2[1] += 1.
    assert np.isclose(misfit.cost(u2, d, dt, w), misfit.cost(u, d, dt, w))
    assert np.isclose(misfit.cost(u, d, dt, np.ones((3, 3))), misfit.cost(u, d, dt))
    assert np.isclose(misfit.cost(np.stack([u, u]), np.stack([d, d]), dt, w), 2 * misfit.cost(u, d, dt, w))


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


def test_mesh_dir_per_ranks(tmp_path):
    """A mesh partitioned for another number of ranks is kept; the runner uses another one beside it."""
    assert Runner(Config(workdir=str(tmp_path), nprocs=4)).mesh_dir == str(tmp_path / "mesh")
    os.makedirs(tmp_path / "mesh")
    (tmp_path / "mesh" / "mesh.input").write_text("4\n1\n")
    assert Runner(Config(workdir=str(tmp_path), nprocs=4)).mesh_dir == str(tmp_path / "mesh")
    assert Runner(Config(workdir=str(tmp_path), nprocs=2)).mesh_dir == str(tmp_path / "mesh_n2")
    # a directory prepared with the first mesh, prepared again with 2 ranks: linked to the other one
    for m in ("mesh", "mesh_n2"):
        os.makedirs(tmp_path / m / "sem", exist_ok=True)
        (tmp_path / m / "sem" / "mesh4spec.0000.h5").write_text("")
        (tmp_path / m / "material.input").write_text("")
    d = str(tmp_path / "trial")
    Runner(Config(workdir=str(tmp_path), nprocs=4)).prepare(d)
    Runner(Config(workdir=str(tmp_path), nprocs=2)).prepare(d)
    assert os.readlink(f"{d}/sem") == str(tmp_path / "mesh_n2" / "sem")


@pytest.mark.sem3d
@pytest.mark.skipif(not os.environ.get("SEM3D_BIN"), reason="SEM3D_BIN not set")
def test_resume_with_other_ranks(tmp_path):
    """Same workdir, 1 then 2 ranks (a run resumed with more ranks): its own mesh, the same traces."""
    tr = []
    for n in (1, 2):
        cfg = Config(workdir=str(tmp_path), sim_time=0.2, nprocs=n, stations=[(262., 233., -141.)],
                     sources=[sem_io.point_source((163., 181., -77.), (1., 0., 0.), func="ricker", tau=0.1,
                                                  freq=8.)])
        runner = Runner(cfg)
        d = str(tmp_path / f"n{n}")
        runner.prepare(d)
        sem_io.write_input(d, cfg, cfg.sources)
        runner.run(d)
        tr.append(sem_io.read_traces(d, 1))
    assert os.path.exists(tmp_path / "mesh" / "sem" / "mesh4spec.0000.h5")
    assert os.path.exists(tmp_path / "mesh_n2" / "sem" / "mesh4spec.0001.h5")
    assert np.max(np.abs(tr[0] - tr[1])) <= 1e-10 * np.max(np.abs(tr[1]))


@pytest.mark.parametrize("param", ["lame", "vel", "kg"])
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
    def __init__(self):
        self.calls = []

    def run(self, cfg, runner, jobs):
        self.calls.append([(j.index, j.grad) for j in jobs])
        out = []
        for j in jobs:
            _, mu, _ = model.read_model(j.fwd)
            r = dict(u=np.full((len(cfg.stations), cfg.nt, 3), mu.sum() * 1e-12 + j.index))
            if j.grad:
                r.update(gl=np.ones(cfg.grid.shape), gm=np.ones(cfg.grid.shape))
            out.append(r)
        return out


def test_evaluate_cache(tmp_path):
    """evaluate resubmits only the sources whose inputs have changed."""
    from semfwi.problem import Problem
    cfg = Config(workdir=str(tmp_path), sim_time=0.01, stations=[(1., 2., 0.)],
                 sources=[sem_io.point_source((1., 2., -3.), (0., 0., 1.), func="ricker", tau=0.1, freq=8.)] * 2)
    one = np.ones(cfg.grid.shape)
    be = _FakeBackend()
    dobs = np.zeros((2, 1, cfg.nt, 3))
    pb = Problem(cfg, _FakeRunner(), dobs, cfg.lam0 * one, cfg.mu0 * one, cfg.rho0 * one, pml=None, backend=be)
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
    local = LocalBackend().run(cfg, runner, [Job(0, d, str(tmp_path / "local" / "adj"), dobs, True)])[0]
    for k in ("u", "gl", "gm"):
        assert np.array_equal(remote[k], local[k]), k
    assert np.abs(local["gm"]).max() > 0


def test_evaluate_cache_unreadable(tmp_path):
    """A truncated result.npz (interrupted while writing) is recomputed, not an error."""
    from semfwi.problem import Problem
    cfg = Config(workdir=str(tmp_path), sim_time=0.01, stations=[(1., 2., 0.)],
                 sources=[sem_io.point_source((1., 2., -3.), (0., 0., 1.), func="ricker", tau=0.1, freq=8.)])
    one = np.ones(cfg.grid.shape)
    be = _FakeBackend()
    pb = Problem(cfg, _FakeRunner(), np.zeros((1, 1, cfg.nt, 3)), cfg.lam0 * one, cfg.mu0 * one, cfg.rho0 * one,
                 pml=None, backend=be)
    x = np.zeros(pb.size)
    J, _, _ = pb.evaluate(x, "t")
    fn = tmp_path / "t" / "s0" / "result.npz"
    fn.write_bytes(fn.read_bytes()[:100])
    assert pb.evaluate(x, "t")[0] == J and len(be.calls) == 2
    assert not list((tmp_path / "t" / "s0").glob("*.tmp.npz"))


def test_input_files_cited(tmp_path):
    """Source files referenced by input.spec are inputs, whatever their name."""
    cfg = Config(workdir=str(tmp_path), stations=[(1., 2., 0.)])
    src = sem_io.point_source((1., 2., -3.), (0., 0., 1.), func="file", time_file='"ricker.txt"')
    sem_io.write_input(str(tmp_path), cfg, [src])
    for fn in ("ricker.txt", "a0_0.txt", "material.input"):
        (tmp_path / fn).write_text("x")
    assert Runner.input_files(str(tmp_path)) == ["input.spec", "material.input", "ricker.txt", "stations.txt"]


def test_run_source_errors(tmp_path, monkeypatch):
    """Forward failure: SimulationError; failure after the forward: TaskError; the strains are
    deleted in both cases."""
    from semfwi import task
    from semfwi.runner import SimulationError
    cfg = Config(workdir=str(tmp_path), sim_time=0.01, stations=[(1., 2., 0.)])
    fwd, adj = tmp_path / "fwd", tmp_path / "adj"

    class R:
        def __init__(self, fail):
            self.fail = fail

        def run(self, d):
            os.makedirs(f"{d}/strain", exist_ok=True)
            if self.fail == d:
                raise SimulationError(d)

    fwd.mkdir()
    with pytest.raises(SimulationError):
        task.run_source(cfg, R(str(fwd)), str(fwd), str(adj), None, True)
    assert not (fwd / "strain").exists()
    monkeypatch.setattr(sem_io, "read_traces", lambda d, n: np.zeros((n, cfg.nt, 3)))
    with pytest.raises(task.TaskError):
        task.run_source(cfg, R(str(adj)), str(fwd), str(adj), np.zeros((1, cfg.nt, 3)), True)
    assert not (fwd / "strain").exists() and not (adj / "strain").exists()


def test_kernels_in_sem_match_python(tmp_path):
    """Kernels accumulated by SEM3D during the adjoint simulation (strain_mode 1 and 2) equal those
    computed in Python from the forward and adjoint strains, to rounding."""
    from semfwi.backend import Job, LocalBackend
    out = {}
    for mode in (0, 1):
        cfg = Config(workdir=str(tmp_path / f"m{mode}"), sim_time=0.1, stations=[(262., 233., 0.), (121., 297., 0.)],
                     sources=[sem_io.point_source((163., 181., -77.), (1., 0., 0.), func="ricker", tau=0.05,
                                                  freq=10.)])
        runner = Runner(cfg)
        d = str(tmp_path / f"m{mode}" / "fwd")
        runner.prepare(d)
        one = np.ones(cfg.grid.shape)
        model.write_model(d, cfg.grid, cfg.lam0 * one, cfg.mu0 * (one + 0.05), cfg.rho0 * one)
        sem_io.write_input(d, cfg, cfg.sources, save_strain=True, strain_mode=mode)
        dobs = np.random.default_rng(1).standard_normal((2, cfg.nt, 3)) * 1e-9
        out[mode] = LocalBackend().run(cfg, runner, [Job(0, d, str(tmp_path / f"m{mode}" / "adj"), dobs, True)])[0]
    assert np.array_equal(out[0]["u"], out[1]["u"])
    for k in ("gl", "gm"):
        assert np.allclose(out[1][k], out[0][k], rtol=1e-12, atol=1e-12 * np.abs(out[0][k]).max()), k
    assert np.abs(out[1]["gm"]).max() > 0


def test_node_volume():
    """Assembled GLL weights of quadratic elements: they sum to the volume, and the node in the
    middle of an element carries 8 times the volume of an inner element corner."""
    g = Grid(xmin=(-20., -20., -45.), h=0.625, n=(65, 65, 73))
    w = model.node_volume(g, 1.25, (-20., -20., -45.))
    assert np.isclose(w.sum(), 40 * 40 * 45)
    assert np.isclose(w[31, 31, 31] / w[30, 30, 30], 8.)
    assert np.isclose(w[0, 0, 0], (1.25 / 6) ** 3)
    # a kernel constant in space, integrated on the nodes, becomes constant again once divided
    assert np.allclose(w / w, 1.)
    assert model.node_volume(g, 2.5, (-20., -20., -45.)) is None      # grid coarser than the GLL nodes
    assert model.node_volume(g, 1.25, (-20.3, -20., -45.)) is None    # grid shifted from the elements


def test_node_scaling():
    """Inverse of the node volume, the nodes on the faces of the grid scaled as interior nodes of
    the same kind: a corner of the grid gets the scaling of an inner element corner, not 8 times it."""
    g = Grid(xmin=(-20., -20., -45.), h=0.625, n=(65, 65, 73))
    mask = np.ones(g.shape, bool)
    d = model.node_scaling(g, 1.25, (-20., -20., -45.), mask).reshape(g.shape)
    assert np.isclose(d[30, 30, 30] / d[31, 31, 31], 8.)       # element corner / element middle
    assert np.isclose(d[0, 0, 0], d[30, 30, 30])               # grid corner = inner element corner
    assert np.isclose(d[0, 31, 31], d[30, 31, 31])             # face node = inner node of the same kind
    assert np.isclose(np.mean(1 / d), 1.)                       # mean volume / volume
    assert model.node_scaling(g, 2.5, (-20., -20., -45.), mask) is None


def test_bound():
    """Problem.bound: mu >= 0.2 mu_ref, then lambda + 2 mu / 3 >= 0.2 (lambda_ref + 2 mu_ref / 3); the
    other nodes and a model within the bounds are left unchanged."""
    from semfwi.problem import Problem
    cfg = Config(workdir="/nonexistent")
    one = np.ones(cfg.grid.shape)
    pb = Problem(cfg, None, None, cfg.lam0 * one, cfg.mu0 * one, cfg.rho0 * one)
    x = 0.1 * RNG.standard_normal(pb.size)
    assert pb.bound(x) == (x, 0)
    n = pb.n
    x[n] = -0.95             # mu of node 0 at 5 % of mu_ref
    x[1] = -3.               # lambda + 2 mu / 3 of node 1 negative
    xb, moved = pb.bound(x)
    assert moved == 2
    lam, mu, _ = pb.to_model(xb)
    m = pb.mask
    k, k_ref = (lam + 2 * mu / 3)[m], (pb.lam_ref + 2 * pb.mu_ref / 3)[m]
    assert np.isclose(mu[m][0], 0.2 * pb.mu_ref[m][0]) and np.all(mu[m] >= 0.2 * pb.mu_ref[m] * (1 - 1e-12))
    assert np.isclose(k[1], 0.2 * k_ref[1]) and np.all(k >= 0.2 * k_ref * (1 - 1e-12))
    assert np.array_equal(xb[2:n], x[2:n]) and np.array_equal(xb[n + 2:], x[n + 2:])


def test_bound_kg():
    """Problem.bound in (K, G): the same feasible set as in (lambda, mu) (G >= 0.2 G_ref, K >= 0.2 K_ref), a node
    out of bounds in one block moved in both blocks only where needed; a model within the bounds is left unchanged."""
    from semfwi.problem import Problem
    cfg = Config(workdir="/nonexistent")
    one = np.ones(cfg.grid.shape)
    pb = Problem(cfg, None, None, cfg.lam0 * one, cfg.mu0 * one, cfg.rho0 * one, param="kg")
    x = 0.1 * RNG.standard_normal(pb.size)
    assert pb.bound(x) == (x, 0)
    n = pb.n
    x[n] = -0.95             # G of node 0 at 5 % of G_ref
    x[1] = -3.               # K of node 1 negative
    xb, moved = pb.bound(x)
    assert moved == 2
    lam, mu, _ = pb.to_model(xb)
    m = pb.mask
    k = (lam + 2 * mu / 3)[m]
    assert np.isclose(mu[m][0], 0.2 * pb.mu_ref[m][0]) and np.isclose(k[1], 0.2 * pb.k_ref[m][1])
    assert np.all(mu[m] >= 0.2 * pb.mu_ref[m] * (1 - 1e-12)) and np.all(k >= 0.2 * pb.k_ref[m] * (1 - 1e-12))
    assert xb[0] == x[0] and xb[n + 1] == x[n + 1]  # the other block of each moved node is unchanged
    assert np.array_equal(xb[2:n], x[2:n]) and np.array_equal(xb[n + 2:], x[n + 2:])


def test_poisson():
    """nu = 0.25 for lambda = mu, 0.5 in the incompressible limit, and the K, G form (3K - 2G) / (2 (3K + G))."""
    from semfwi.problem import poisson
    assert np.isclose(poisson(80e6, 80e6), 0.25)
    assert np.isclose(poisson(1e15, 1e6), 0.5)
    lam, mu = 4.56e9, 84e6
    K = lam + 2 * mu / 3
    assert np.isclose(poisson(lam, mu), (3 * K - 2 * mu) / (2 * (3 * K + mu)))


def test_line_search_diverged_trial(tmp_path):
    """A trial whose J is not finite (simulation diverged without failing) halves the step, as a failed
    simulation does: J = |x - 1|^2 / 2, nan beyond |x| = 2, first step to x = 5 (step0)."""
    from types import SimpleNamespace

    from semfwi import inversion

    class Quadratic:
        size, n, active, scaling, nsim = 4, 4, ("lambda",), None, 0
        cfg = SimpleNamespace(workdir=str(tmp_path))

        def bound(self, x):
            return x, 0

        def evaluate(self, x, tag):
            J = 0.5 * np.sum((x - 1) ** 2) if np.max(np.abs(x)) <= 2 else np.nan
            return J, None, x - 1

    lines = []
    _, h = inversion.invert(str(tmp_path / "run"), [("p20", 2)], lambda si, name: Quadratic(), step0=5.,
                            log=lines.append)
    assert len(h) == 3 and h[1]["J"] < h[0]["J"]
    assert sum("failed (J = nan)" in s for s in lines) == 2  # x = 5 and 2.5, then 1.25 accepted


def test_restart_removes_stale_trials(tmp_path):
    """A restart removes the trial directories of the interrupted iteration (redone from trial 0),
    and only those."""
    from types import SimpleNamespace

    from semfwi import inversion

    class Quadratic:
        size, n, active, scaling, nsim = 4, 4, ("lambda",), None, 0
        cfg = SimpleNamespace(workdir=str(tmp_path))

        def bound(self, x):
            return x, 0

        def evaluate(self, x, tag):
            return 0.5 * np.sum((x - 1) ** 2), None, x - 1

    run = lambda n: inversion.invert(str(tmp_path / "run"), [("p20", n)],  # noqa: E731
                                     lambda si, name: Quadratic(), log=lambda s: None)
    run(1)
    for d in ("it001_t0", "it002_t2", "it002_t7", "it002_start", "it003_t0"):
        (tmp_path / d).mkdir()
    run(2)
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("it")) == \
        ["it001_t0", "it002_start", "it003_t0"]


def test_first_stage_failure_next_stage(tmp_path):
    """A line-search failure at the first stage (J flat: no trial passes Armijo) goes on to the next
    stage, as at the other stages, with the unused iterations; a restart does not go back to it."""
    from types import SimpleNamespace

    from semfwi import inversion

    class Flat:
        size, n, active, scaling, nsim = 50, 50, ("lambda",), None, 0
        cfg = SimpleNamespace(workdir=str(tmp_path))

        def bound(self, x):
            return x, 0

        def evaluate(self, x, tag):
            return 1., None, np.ones_like(x)

    class Quadratic(Flat):  # curvatures 1 to 100: not solved in 5 iterations
        c = np.logspace(0, 2, 50)

        def evaluate(self, x, tag):
            return 0.5 * np.sum(self.c * (x - 1) ** 2), None, self.c * (x - 1)

    lines = []
    run = lambda: inversion.invert(str(tmp_path / "run"), [("p20", 3), ("p30", 2)],  # noqa: E731
                                   lambda si, name: Flat() if si == 0 else Quadratic(), log=lines.append)
    _, h = run()
    # the 3 iterations of p20 are carried over to p30: 5 iterations, plus its starting point
    assert any("line search failed at iteration 1: next stage, 3 iterations carried over" in s for s in lines)
    assert [e["stage"] for e in h] == [0] + [1] * 6 and h[-1]["J"] < h[1]["J"]
    # a restart does not go back to p20 (ended by the failure): nothing left to do
    lines.clear()
    _, h2 = run()
    assert len(h2) == len(h) and not any("failed" in s for s in lines)


def test_depth_preconditioner_keeps_scaling():
    """"depth" multiplies the node scaling d0 by a factor per depth: within a horizontal plane,
    d / d0 is constant; "block" keeps d0 up to a factor per block."""
    from types import SimpleNamespace

    from semfwi import inversion

    shape = (4, 4, 6)
    mask = np.ones(shape, bool)
    n = mask.sum()
    pb = SimpleNamespace(n=n, mask=mask, grid_field=lambda v: v.reshape(shape))
    rng = np.random.default_rng(0)
    d0 = np.tile(rng.uniform(1, 8, n), 2)
    g = rng.standard_normal(2 * n) * np.tile(np.linspace(1, 10, shape[2]), 2 * shape[0] * shape[1])
    for kind in ("block", "depth"):
        d = inversion.preconditioner(kind, pb, g, d0)
        r = (d / d0).reshape(2, *shape)
        assert np.allclose(r, r[:, :1, :1, :]), kind  # one factor per block and depth
        if kind == "block":
            assert np.allclose(r, r[:, :1, :1, :1])


def test_anisotropic_grid():
    """Grid with one spacing per axis: axes, trilinear interpolation exact on linear fields, interior mask
    one step per axis away from the PMLs, and the configuration round trip."""
    from semfwi import config
    g = Grid(xmin=(-100., -100., -20.), h=(50., 50., 5.), n=(5, 5, 5))
    assert g.hs == (50., 50., 5.) and not g.isotropic and g.xmax == (100., 100., 0.)
    assert np.allclose(g.axis(2), [-20., -15., -10., -5., 0.])
    pts = np.array(g.xmin) + RNG.random((200, 3)) * (np.array(g.xmax) - np.array(g.xmin))
    idx, w = model.trilinear(g, pts)
    X, Y, Z = g.mesh()
    f = 1. + 2 * X - 3 * Y + 0.5 * Z
    assert np.allclose(model.interpolate(g, f, idx, w), 1. + pts @ [2., -3., 0.5])
    m = model.interior_mask(g, 150., 12., -75.)  # domain [-75, 75]^2 x [-12, 0], inside the grid
    assert X[m].min() == X[m].max() == 0. and Z[m].min() == -5. and Z[m].max() == 0.
    assert model.node_volume(g, 100., (-100., -100., -20.)) is None
    cfg = Config(workdir="/a", grid=g, layers=[(20., 4), (80., 2)], layer_props=[None, (4e9, 2e9, 2000.)])
    back = config.from_dict(json.loads(json.dumps(config.to_dict(cfg))), "/b")
    assert back.grid.hs == g.hs and back.grid.axis(2).tolist() == g.axis(2).tolist() and back.file_layers == [0]


def test_tv_gradient_anisotropic():
    """Total variation on a grid with one spacing per axis: the gradient matches a centered finite difference."""
    from semfwi.inversion import TotalVariation
    from semfwi.problem import Problem
    g = Grid(xmin=(0., 0., -40.), h=(50., 50., 5.), n=(7, 7, 9))
    cfg = Config(workdir="/nonexistent", L=300., depth=40., grid=g)
    one = np.ones(g.shape)
    pb = Problem(cfg, None, None, 80e6 * one, 80e6 * one, 2000. * one, param="kg")
    tv = TotalVariation(pb)
    x = 0.1 * RNG.standard_normal(pb.size)
    dx = RNG.standard_normal(pb.size)
    h = 1e-6
    fd = (tv.value(x + h * dx) - tv.value(x - h * dx)) / (2 * h)
    assert np.isclose(tv.grad(x) @ dx, fd, rtol=1e-5)


def test_layer_index():
    cfg = Config(workdir="/a", depth=100., layers=[(17.3, 1), (72.5, 1), (10.2, 1)])
    z = np.array([0., -10., -17.3, -17.31, -89.8, -95., -100., -150.])
    assert cfg.layer_index(z).tolist() == [cfg.layer_of_depth(v) for v in z] == [0, 0, 0, 1, 1, 2, 2, 2]


def test_constant_layers_inputs(tmp_path):
    """layer_props: the constant layers get their properties in mater.in and no entry in material.spec,
    the others read the model files."""
    cfg = Config(workdir="/a", layers=[(20., 1), (80., 2), (300., 2)],
                 layer_props=[None, None, (3.6e9, 1.8e9, 2600.)])
    sem_io.write_mesher_inputs(tmp_path, cfg)
    lines = (tmp_path / "mater.in").read_text().splitlines()
    assert lines[0] == "3"
    vp, vs, rho = (float(v) for v in lines[3].split()[1:4])
    assert np.isclose(vp, np.sqrt((3.6e9 + 3.6e9) / 2600.)) and np.isclose(vs, np.sqrt(1.8e9 / 2600.)) and rho == 2600.
    assert lines[1] == lines[2] and np.isclose(float(lines[1].split()[2]), np.sqrt(cfg.mu0 / cfg.rho0))
    one = np.ones(cfg.grid.shape)
    model.write_model(tmp_path, cfg.grid, cfg.lam0 * one, cfg.mu0 * one, cfg.rho0 * one,
                      nmat=3, file_layers=cfg.file_layers)
    spec = (tmp_path / "material.spec").read_text()
    assert "material 0 {" in spec and "material 1 {" in spec and "material 2" not in spec


def test_file_source(tmp_path):
    """FileSource: time function written next to input.spec, moment tensor in SEM3D's order
    (Mxx, Myy, Mzz, Mxy, Mxz, Myz), amplitude; a force keeps type = impulse and its direction."""
    cfg = Config(workdir=str(tmp_path), stations=[(0., 0., 0.)])
    t = np.arange(5) * 1e-3
    f = np.array([0., 1., 3., 2., 2.])
    m = (0., 0., 0., 1., 0., 0.)
    srcs = sem_io.materialize_sources(tmp_path, cfg, [sem_io.FileSource((1., 2., -300.), t, f, moment=m, amplitude=1e18),
                                                      sem_io.FileSource((0., 0., 0.), t, f, direction=(0., 0., -1.))])
    sem_io.write_input(tmp_path, cfg, srcs)
    spec = (tmp_path / "input.spec").read_text()
    assert spec.count("func = file;") == 2 and spec.count("type = moment;") == 1 and spec.count("type = impulse;") == 1
    assert "moment = 0.0 0.0 0.0 1.0 0.0 0.0;" in spec and "amplitude = 1e+18;" in spec
    assert 'time_file = "tL0.txt";' in spec and 'time_file = "tL1.txt";' in spec and "dir = 0.0 0.0 -1.0;" in spec
    assert np.array_equal(np.loadtxt(tmp_path / "tL0.txt"), np.c_[t, f])
    assert set(Runner.input_files(str(tmp_path))) >= {"tL0.txt", "tL1.txt"}  # in the cache key
    with pytest.raises(AssertionError):
        sem_io.FileSource((0., 0., 0.), t, f)


def _center_jacobian(nodes, elems):
    """Determinant of the trilinear map at the centre of each element (lower face then upper face,
    counterclockwise seen from above): positive for a valid element."""
    v = nodes[elems]  # (nelem, 8, 3)
    a = (v[:, 1] + v[:, 2] + v[:, 5] + v[:, 6] - v[:, 0] - v[:, 3] - v[:, 4] - v[:, 7]) / 4
    b = (v[:, 2] + v[:, 3] + v[:, 6] + v[:, 7] - v[:, 0] - v[:, 1] - v[:, 4] - v[:, 5]) / 4
    c = (v[:, 4] + v[:, 5] + v[:, 6] + v[:, 7] - v[:, 0] - v[:, 1] - v[:, 2] - v[:, 3]) / 4
    return np.einsum("ij,ij->i", a, np.cross(b, c))


def test_hex8_grid_mesh(tmp_path):
    """hexmesh.grid_mesh: the box of the on-the-fly mesher with its PMLs (sides and bottom), valid
    elements, interior elements in their layer, 8 lateral PML materials per layer plus 9 at the bottom,
    each pointing to its layer; material.input lists them as the mesher does."""
    cfg = Config(workdir="/a", L=400., h_elem=50., x0=-200., npml=2, layers=[(20., 1), (80., 2), (300., 3)])
    nodes, elems, mats, table = hexmesh.grid_mesh(cfg)
    nxy, nz = 8 + 4, 1 + 2 + 3 + 2
    assert elems.shape == (nxy * nxy * nz, 8) and nodes.shape == ((nxy + 1) ** 2 * (nz + 1), 3)
    assert np.allclose(nodes.min(axis=0), (-300., -300., -600.)) and np.allclose(nodes.max(axis=0), (300., 300., 0.))
    assert np.all(_center_jacobian(nodes, elems) > 0)
    c = nodes[elems].mean(axis=1)
    inner = (np.abs(c[:, 0]) < 200.) & (np.abs(c[:, 1]) < 200.) & (c[:, 2] > -400.)
    assert np.array_equal(mats[inner], cfg.layer_index(c[inner, 2]))
    assert np.all(np.array([table[m][0] for m in mats[~inner]]) == "P")
    assert len(table) == 3 + 3 * 8 + 9
    for e in np.nonzero(~inner)[0][::7]:
        typ, layer, (xp, xw, yp, yw, zp, zw) = table[mats[e]]
        assert layer == cfg.layer_index(c[e, 2]) and (zw != 0) == (c[e, 2] < -400.)
        assert (xw != 0) == (abs(c[e, 0]) > 200.) and (yw != 0) == (abs(c[e, 1]) > 200.)
    hexmesh.write(tmp_path, cfg, nodes, elems, mats, table)
    lines = (tmp_path / "material.input").read_text().splitlines()
    assert lines[0] == "36" and lines[1].startswith("S ") and lines[4].startswith("P ")
    assert lines[37] == "# PML properties" and len(lines) == 39 + 33
    assert (tmp_path / "mesh.input").read_text() == "1\n4\n1\nmesh.h5\n"
    with h5py.File(tmp_path / "mesh.h5") as f:
        assert f["Sem3D/Hexa8"].shape == elems.shape and np.array_equal(f["Sem3D/Mat"][:], mats)


def test_hex8_topography(tmp_path):
    """The free surface follows the topography, the mesh is unchanged below topo_depth, the
    elements stay valid; a topography reaching the PMLs is refused."""
    x = np.linspace(-400., 400., 81)
    X, Y = np.meshgrid(x, x)
    r = np.hypot(X, Y)
    hill = 60. * np.cos(np.pi * np.minimum(r, 150.) / 300.) ** 2  # zero beyond 150 m
    np.savez(tmp_path / "topo.npz", x=x, y=x, z=hill)
    cfg = Config(workdir="/a", L=400., h_elem=50., x0=-200., npml=2, layers=[(400., 8)],
                 topography=str(tmp_path / "topo.npz"), topo_depth=200.)
    flat = hexmesh.grid_mesh(cfg)[0]
    nodes, elems = hexmesh.build(cfg)[:2]
    top = flat[:, 2] == 0.
    assert np.allclose(nodes[top, 2], hexmesh.surface_offset(cfg.topography, flat[top, 0], flat[top, 1]))
    assert nodes[top, 2].max() == pytest.approx(60.)
    assert np.array_equal(nodes[flat[:, 2] <= -200.], flat[flat[:, 2] <= -200.])
    assert np.all(_center_jacobian(nodes, elems) > 0)
    np.savez(tmp_path / "wide.npz", x=x, y=x, z=hill + 1.)
    with pytest.raises(AssertionError, match="PMLs"):
        hexmesh.build(dataclasses.replace(cfg, topography=str(tmp_path / "wide.npz")))


def _hill_config(tmp_path, **kw):
    x = np.linspace(-400., 400., 81)
    X, Y = np.meshgrid(x, x)
    hill = 60. * np.cos(np.pi * np.minimum(np.hypot(X, Y), 150.) / 300.) ** 2  # zero beyond 150 m
    np.savez(tmp_path / "topo.npz", x=x, y=x, z=hill)
    return Config(workdir="/a", L=400., h_elem=50., x0=-200., npml=2, layers=[(100., 4), (300., 4)],
                  topography=str(tmp_path / "topo.npz"), topo_depth=200., **kw)


def test_flat_z(tmp_path):
    """flat_z undoes follow_surface: the moved mesh nodes come back to the flat ones; identity without
    topography; positive above the free surface."""
    cfg = _hill_config(tmp_path)
    flat = hexmesh.grid_mesh(cfg)[0]
    moved = hexmesh.build(cfg)[0]
    assert np.allclose(hexmesh.flat_z(cfg, *moved.T), flat[:, 2])
    assert hexmesh.flat_z(cfg, 0., 0., 61.) > 0 and hexmesh.flat_z(cfg, 0., 0., 59.) < 0
    plain = dataclasses.replace(cfg, topography=None)
    assert np.array_equal(hexmesh.flat_z(plain, *moved.T), moved[:, 2])
    assert np.allclose(hexmesh.on_surface(cfg, [(0., 0., -10.), (300., 0., 0.)]), [(0., 0., 50.), (300., 0., 0.)])


def test_topography_mask_and_layers(tmp_path):
    """Below a topography: the grid nodes in the air are neither inverted nor counted in the layers;
    those just above the surface whose support reaches below it are inverted; a node at a given
    depth below the hill is in the layer of that depth in the flat mesh."""
    cfg = _hill_config(tmp_path, grid=Grid(xmin=(-190., -190., -390.), h=10., n=(39, 39, 47)))  # up to z = 70
    surf = lambda x, y: hexmesh.surface(cfg, x, y)
    m = model.interior_mask(cfg.grid, cfg.L, cfg.depth, cfg.x0, surf)
    i0 = 19  # column x = y = 0, surface at 60 m
    z = cfg.grid.axis(2)
    assert m[i0, i0][z <= 60.].all() and not m[i0, i0][z > 60.].any()
    # far from the hill: the nodes up to z = 0 (the one above only touches the surface)
    assert m[1, 1][z <= 0.].all() and not m[1, 1][z > 0.].any()
    flat_m = model.interior_mask(cfg.grid, cfg.L, cfg.depth, cfg.x0)
    assert np.array_equal(m[1, 1][z < 0.], flat_m[1, 1][z < 0.])
    layers = model.node_layers(cfg)
    assert np.all(layers[i0, i0][z > 60.] == -1) and np.all(layers[1, 1][z > 0.] == -1)
    zf = hexmesh.flat_z(cfg, 0., 0., z)
    assert np.array_equal(layers[i0, i0][z <= 60.], cfg.layer_index(zf[z <= 60.]))
    g = model.pml_layer_gradient(cfg, np.array([2., 3.]))
    assert np.isclose(g.sum(), 5.) and np.all(g[layers == -1] == 0.)


def test_source_taper():
    """Gradient taper: 0 at a point source, 1 beyond the radius, increasing in between; positions of
    a FileSource and of an input.spec section; none for a source without a point."""
    g = Grid(xmin=(-100., -100., -100.), h=10., n=(21, 21, 11))
    srcs = [sem_io.FileSource((0., 0., 0.), [0., 1.], [0., 1.], direction=(0., 0., -1.)),
            sem_io.point_source((50., 0., -100.), (0., 0., 1.)), {"type": "extended"}]
    pos = [p for p in map(model.source_position, srcs) if p is not None]
    assert pos == [(0., 0., 0.), (50., 0., -100.)]
    w = model.source_taper(g, pos, 40.)
    X, Y, Z = g.mesh()
    d0 = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    d1 = np.sqrt((X - 50.) ** 2 + Y ** 2 + (Z + 100.) ** 2)
    assert w[10, 10, 10] == 0. and w[15, 10, 0] == 0.
    assert np.all(w[(d0 >= 40.) & (d1 >= 40.)] == 1.) and np.all(w[(d0 < 40.) | (d1 < 40.)] < 1.)
    line = w[10:15, 10, 10]  # x = 0 .. 40 at the first source
    assert np.all(np.diff(line) > 0) and np.isclose(line[2], 0.5)
