"""Inversion problem: J(x) and grad J(x) over all sources.

Unknowns on the interior grid nodes (model.interior_mask), as relative perturbations,
for the active parameters (active, a subset of ("a", "b")):
  param = "lame" : a = lambda / lambda_ref - 1, b = mu / mu_ref - 1;
  param = "vel"  : a = Vp / Vp_ref - 1,         b = Vs / Vs_ref - 1,
                   with mu = rho Vs^2, lambda = rho Vp^2 - 2 rho Vs^2;
  param = "kg"   : a = K / K_ref - 1,           b = G / G_ref - 1,
                   bulk and shear moduli, with lambda = K - 2 G / 3, mu = G;
  param = "vs"   : b = Vs / Vs_ref - 1 only, Vp / Vs kept at its reference value at each node:
                   lambda and mu both scale by (1 + b)^2, Poisson's ratio stays that of the reference
                   (Vs alone with a fixed Vp / Vs, as in near-surface FWI, e.g. Seylabi et al. 2020).
An inactive parameter keeps the value from fixed (the reference model by default). rho and the
nodes close to the PMLs stay fixed. SEM3D always reads lambda and mu; for velocities, the gradient
follows from the chain rule:
  dJ/dVp = 2 rho Vp dJ/dlambda,   dJ/dVs = 2 rho Vs (dJ/dmu - 2 dJ/dlambda),
  dJ/dK = dJ/dlambda,              dJ/dG = dJ/dmu - 2/3 dJ/dlambda,
  dJ/db = 2 (1 + b) (lambda_ref dJ/dlambda + mu_ref dJ/dmu)   ("vs").

zstep: the descent directions of each block in the span of a coarse vertical basis P (model.vertical_basis,
unknowns every zstep levels of the grid, linear in z between them), by the L-BFGS preconditioner S P P^T S
(S the square root of the diagonal scaling; inversion.precond_operator): L-BFGS on the coarse unknowns,
from a start in that span, up to the bounds and the size of the first step. For model files finer than the
GLL spacing, whose levels between two GLL planes get no gradient.

pml: "follow" (default: each PML takes the mean of its mesh layer, so a perturbation anywhere in a
layer changes its PMLs; the gradient includes this dependence, from the PML kernels), "copy" (SEM3D extends the current model
into the PMLs point by point, from the inner edge of each PML, as in Fathi et al.; the gradient
includes the PML kernels, but each edge node drives a whole PML column and gets a much larger
gradient than the interior, which stalls the inversion), None (mesher properties) or a fixed
(lam, mu, rho). The PML damping uses cfg.pml_damping_vp when set (independent of the model).

InterfaceProblem: the unknowns are the depth of one interface on a coarse map instead (its docstring).

observed computes the data of a true model, model_errors the errors of an inversion to it.
Run directories are prepared here; simulations and kernels are computed by the
backend (backend.LocalBackend by default, or armonik_backend.ArmoniKBackend), one source per job.
"""
import fcntl
import hashlib
import json
import os
import zipfile

import numpy as np

from . import config, hexmesh, misfit, model, sem_io
from .backend import Job, LocalBackend



def write_model(d, cfg, lam, mu, rho, pml):
    """Model files of a run directory, the PMLs as for Problem(pml=...)."""
    model.write_model(d, cfg.grid, lam, mu, rho, nmat=len(cfg.layer_list), file_layers=cfg.file_layers)
    if pml == "copy" and cfg.layer_props is not None:
        # the PMLs of the layers of constant properties keep them (follow), the others extend the files
        model.write_pml_materials(d, cfg, lam, mu, rho)
        model.write_pml_copy(d, cfg.file_layers)
    elif pml == "copy":
        model.write_pml_copy(d)
    elif pml == "follow":
        model.write_pml_materials(d, cfg, lam, mu, rho)
    elif pml is not None:
        model.write_pml_materials(d, cfg, *pml)


def observed(cfg, runner, ddir, lam, mu, rho, pml=None):
    """Data of the model (lam, mu, rho), (nsrc, nsta, nt, 3): one simulation per source in ddir/sim,
    sim1, ..., the PMLs as for Problem(pml=...). SEM3D reruns only if the inputs changed (Runner.run),
    and a file lock lets the inversions of the same case, started together, share the data."""
    os.makedirs(ddir, exist_ok=True)
    with open(f"{ddir}/.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        dirs = []
        for i, src in enumerate(cfg.sources):
            d = f"{ddir}/sim{i or ''}"
            runner.prepare(d)
            write_model(d, cfg, lam, mu, rho, pml)
            sem_io.write_input(d, cfg, sem_io.materialize_sources(d, cfg, [src]))
            dirs.append(d)
        runner.run_many(dirs)
        return np.stack([sem_io.read_traces(d, len(cfg.stations)) for d in dirs])


def model_errors(pb, x, lam_t, mu_t):
    """Relative errors to the true model on the inverted nodes, in (lambda, mu), (Vp, Vs) and K:
    ||m - m_true|| / ||m_ref - m_true|| (1: no better than the reference model of pb)."""
    lam, mu, rho = pb.to_model(x)
    m = pb.mask
    vp, vs = np.sqrt((lam + 2 * mu) / rho), np.sqrt(mu / rho)
    vp_t, vs_t = np.sqrt((lam_t + 2 * mu_t) / rho), np.sqrt(mu_t / rho)
    out = {}
    for name, a, t, ref in (("lambda", lam, lam_t, pb.lam_ref), ("mu", mu, mu_t, pb.mu_ref),
                            ("vp", vp, vp_t, pb.vp_ref), ("vs", vs, vs_t, pb.vs_ref),
                            ("k", lam + 2 * mu / 3, lam_t + 2 * mu_t / 3, pb.k_ref)):
        out["err_" + name] = float(np.linalg.norm((a - t)[m]) / np.linalg.norm((t - ref)[m]))
    return out


def poisson(lam, mu):
    """Poisson's ratio nu = lambda / (2 (lambda + mu)): between -1 and 1/2 for a physical solid,
    0.25 for lambda = mu, close to 1/2 for a saturated soil."""
    return lam / (2 * (lam + mu))


class Problem:
    def __init__(self, cfg, runner, dobs, lam_ref, mu_ref, rho_ref, param="lame", active=("a", "b"),
                 fixed=None, pml="follow", backend=None, zstep=1):
        assert param in ("lame", "vel", "kg", "vs"), param
        assert param != "vs" or tuple(active) == ("b",), active
        self.cfg, self.runner, self.dobs, self.param = cfg, runner, dobs, param
        self.backend = backend or LocalBackend()
        # gradient taper around the point sources (cfg.source_taper): a fixed factor, so a preconditioner
        pos = [p for p in map(model.source_position, cfg.sources) if p is not None]
        self.taper = model.source_taper(cfg.grid, pos, cfg.source_taper) if cfg.source_taper and pos else None
        self.active = tuple(active)
        self.lam_ref, self.mu_ref, self.rho_ref = lam_ref, mu_ref, rho_ref
        self.vp_ref = np.sqrt((lam_ref + 2 * mu_ref) / rho_ref)
        self.vs_ref = np.sqrt(mu_ref / rho_ref)
        self.k_ref = lam_ref + 2 * mu_ref / 3
        self.fixed = fixed if fixed is not None else (lam_ref, mu_ref)
        self.pml = pml
        if cfg.layer_props is not None:
            # SEM3D clips the model files to the grid: it must reach the bottom of the layers that read them
            bottom = -sum(cfg.layer_list[i][0] for i in range(max(cfg.file_layers) + 1))
            assert cfg.grid.xmin[2] <= bottom + 1e-6, (cfg.grid.xmin[2], bottom)
        surface = (lambda x, y: hexmesh.surface(cfg, x, y)) if cfg.topography else None
        self.mask = model.interior_mask(cfg.grid, cfg.L, cfg.depth, cfg.x0, surface)
        self.n = int(self.mask.sum())
        # scaling of the descent directions (model.node_scaling), per unknown; None if the grid is not on
        # the GLL nodes, which a topography moves
        d = None if cfg.topography else model.node_scaling(cfg.grid, cfg.h_elem, (cfg.x0, cfg.x0, -cfg.depth), self.mask)
        self.scaling = None if d is None else np.tile(d, len(self.active))
        self.basis = model.vertical_basis(cfg.grid, self.mask, zstep) if zstep > 1 else None
        self.nsim = 0

    @property
    def size(self):
        return len(self.active) * self.n

    def split(self, x):
        """Blocks (a, b) of x; None for an inactive parameter."""
        out, i = {}, 0
        for p in ("a", "b"):
            if p in self.active:
                out[p] = x[i * self.n:(i + 1) * self.n]
                i += 1
            else:
                out[p] = None
        return out["a"], out["b"]

    def to_model(self, x):
        rho = self.rho_ref
        xa, xb = self.split(x)
        if self.param == "lame":
            lam, mu = self.fixed[0].copy(), self.fixed[1].copy()
            if xa is not None:
                lam[self.mask] = self.lam_ref[self.mask] * (1 + xa)
            if xb is not None:
                mu[self.mask] = self.mu_ref[self.mask] * (1 + xb)
        elif self.param == "kg":
            k, mu = self.fixed[0] + 2 * self.fixed[1] / 3, self.fixed[1].copy()
            if xa is not None:
                k[self.mask] = self.k_ref[self.mask] * (1 + xa)
            if xb is not None:
                mu[self.mask] = self.mu_ref[self.mask] * (1 + xb)
            lam = k - 2 * mu / 3
        elif self.param == "vs":
            lam, mu = self.fixed[0].copy(), self.fixed[1].copy()
            s = (1 + xb) ** 2
            lam[self.mask] = self.lam_ref[self.mask] * s
            mu[self.mask] = self.mu_ref[self.mask] * s
        else:
            vp = np.sqrt((self.fixed[0] + 2 * self.fixed[1]) / rho)
            vs = np.sqrt(self.fixed[1] / rho)
            if xa is not None:
                vp[self.mask] = self.vp_ref[self.mask] * (1 + xa)
            if xb is not None:
                vs[self.mask] = self.vs_ref[self.mask] * (1 + xb)
            mu = rho * vs ** 2
            lam = rho * vp ** 2 - 2 * mu
        return lam, mu, rho

    def to_x(self, lam, mu):
        if self.param == "lame":
            a, b, a0, b0 = lam, mu, self.lam_ref, self.mu_ref
        elif self.param == "kg":
            a, b, a0, b0 = lam + 2 * mu / 3, mu, self.k_ref, self.mu_ref
        else:
            a, b = np.sqrt((lam + 2 * mu) / self.rho_ref), np.sqrt(mu / self.rho_ref)
            a0, b0 = self.vp_ref, self.vs_ref
        blocks = {"a": a[self.mask] / a0[self.mask] - 1, "b": b[self.mask] / b0[self.mask] - 1}
        return np.concatenate([blocks[p] for p in self.active])

    def bound(self, x, frac=0.2):
        """x with mu >= frac mu_ref and lambda + 2 mu / 3 >= frac (lambda_ref + 2 mu_ref / 3) on every
        inverted node (projection: mu first, then lambda), and the number of nodes moved. A safety net
        for the line-search trials, not a regularization: the drivers log the nodes it moves. param =
        "lame", "kg" or "vs" (x unchanged otherwise); in "kg", the same bounds on G and K; in "vs", both scale by
        (1 + b)^2, hence 1 + b >= sqrt(frac)."""
        if self.param == "vs":
            lo = np.sqrt(frac) - 1
            moved = x < lo
            return (np.maximum(x, lo), int(moved.sum())) if moved.any() else (x, 0)
        if self.param == "kg":
            lo = frac - 1  # G >= frac G_ref, K >= frac K_ref: x >= frac - 1 on each block
            moved = np.zeros(self.n, bool)
            for i in range(len(self.active)):
                moved |= x[i * self.n:(i + 1) * self.n] < lo
            if not moved.any():
                return x, 0
            x = x.copy()  # only the moved nodes are rewritten, the others keep x to the bit
            for i in range(len(self.active)):
                sl = x[i * self.n:(i + 1) * self.n]
                sl[moved] = np.maximum(sl[moved], lo)
            return x, int(moved.sum())
        if self.param != "lame":
            return x, 0
        lam, mu, _ = self.to_model(x)
        m = self.mask
        mu_b = np.maximum(mu[m], frac * self.mu_ref[m]) if "b" in self.active else mu[m]
        k_min = frac * (self.lam_ref[m] + 2 * self.mu_ref[m] / 3)
        lam_b = np.maximum(lam[m], k_min - 2 * mu_b / 3) if "a" in self.active else lam[m]
        moved = (mu_b != mu[m]) | (lam_b != lam[m])
        if not moved.any():
            return x, 0
        x = x.copy()  # only the moved nodes are rewritten, the others keep x to the bit
        for i, q in enumerate(self.active):
            new = lam_b / self.lam_ref[m] - 1 if q == "a" else mu_b / self.mu_ref[m] - 1
            sl = x[i * self.n:(i + 1) * self.n]
            sl[moved] = new[moved]
        return x, int(moved.sum())

    def chain(self, x, gl, gm):
        """Gradient with respect to x from dJ/dlambda and dJ/dmu on the grid."""
        if self.param == "lame":
            ga, gb = gl * self.lam_ref, gm * self.mu_ref
        elif self.param == "kg":
            ga, gb = gl * self.k_ref, (gm - 2 * gl / 3) * self.mu_ref
        elif self.param == "vs":
            mu = self.to_model(x)[1]
            ga, gb = None, 2 * np.sqrt(mu / self.mu_ref) * (gl * self.lam_ref + gm * self.mu_ref)
        else:
            lam, mu, rho = self.to_model(x)
            vp, vs = np.sqrt((lam + 2 * mu) / rho), np.sqrt(mu / rho)
            ga = 2 * rho * vp * gl * self.vp_ref
            gb = 2 * rho * vs * (gm - 2 * gl) * self.vs_ref
        blocks = {"a": ga, "b": gb}
        return np.concatenate([blocks[p][self.mask] for p in self.active])

    def grid_field(self, v):
        out = np.zeros(self.cfg.grid.shape)
        out[self.mask] = v
        return out

    def _dir(self, tag, i, kind):
        return os.path.join(self.cfg.workdir, tag, f"s{i}", kind)

    def _write_model(self, d, lam, mu, rho):
        write_model(d, self.cfg, lam, mu, rho, self.pml)

    def evaluate(self, x, tag, grad=True):
        """J(x), traces u (nsrc, nsta, nt, 3) and grad J(x) (None if grad is false). The simulations
        of source i are in workdir/tag/s{i}/fwd and adj; its result is kept there (result.npz)
        so that a new evaluation with the same inputs does not rerun them."""
        lam, mu, rho = self.to_model(x)
        model.check_bounds(lam, mu, rho)
        results = [None] * len(self.cfg.sources)
        jobs, keys = [], []
        for i, src in enumerate(self.cfg.sources):
            d = self._dir(tag, i, "fwd")
            self.runner.prepare(d)
            self._write_model(d, lam, mu, rho)
            sem_io.write_input(d, self.cfg, sem_io.materialize_sources(d, self.cfg, [src]), save_strain=grad,
                              strain_mode=1 if self.cfg.kernels_in_sem else 0, strain_pml=self.pml in ("copy", "follow"))
            key = self._key(d, i, grad)
            results[i] = self._cached(tag, i, key)
            if results[i] is None:
                jobs.append(Job(i, d, self._dir(tag, i, "adj"), self.dobs[i], grad))
                keys.append(key)
        for job, key, r in zip(jobs, keys, self.backend.run(self.cfg, self.runner, jobs) if jobs else []):
            fn = self._result_fn(tag, job.index)
            np.savez(fn + ".tmp.npz", key=key, **r)  # an interruption while writing leaves no truncated cache
            os.replace(fn + ".tmp.npz", fn)
            results[job.index] = r
        self.nsim += len(jobs) * (2 if grad else 1)
        u = np.stack([r["u"] for r in results])
        J = misfit.cost(u, self.dobs, self.cfg.dt, self.cfg.weights)
        if not grad:
            return J, u, None
        gl = sum(r["gl"] for r in results)
        gm = sum(r["gm"] for r in results)
        if "hu" in results[0]:  # illuminations of this evaluation, summed over the sources
            self.illum = (sum(r["hu"] for r in results), sum(r["hw"] for r in results))
        if self.pml == "follow" and "pml_layer" in results[0]:
            # the PML points were projected on the edge nodes (copy): moved to the layer means
            gl = gl - sum(r["pml_gl"] for r in results)
            gm = gm - sum(r["pml_gm"] for r in results)
            layer = sum(r["pml_layer"] for r in results)
            gl = gl + model.pml_layer_gradient(self.cfg, layer[0])
            gm = gm + model.pml_layer_gradient(self.cfg, layer[1])
        if self.taper is not None:
            gl, gm = gl * self.taper, gm * self.taper
        return J, u, self.chain(x, gl, gm)

    def _result_fn(self, tag, i):
        return os.path.join(self.cfg.workdir, tag, f"s{i}", "result.npz")

    def _key(self, dfwd, i, grad):
        """Fingerprint of the inputs of source i: forward directory, plus data and adjoint sources if grad."""
        h = hashlib.sha1(self.runner.input_hash(dfwd).encode())
        if grad:
            h.update(np.ascontiguousarray(self.dobs[i]).tobytes())
            d = config.to_dict(self.cfg)
            h.update(json.dumps([d["adjoint"], d["receiver_grid"]]).encode())
            if d["weights"] is not None:  # keeps the keys of the results computed without weights
                h.update(json.dumps(d["weights"]).encode())
        return h.hexdigest()

    def _cached(self, tag, i, key):
        """Result kept for key, else None (file missing, from other inputs, or unreadable)."""
        try:
            with np.load(self._result_fn(tag, i)) as st:
                return {k: st[k] for k in st.files if k != "key"} if str(st["key"]) == key else None
        except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile):
            return None


class InterfaceProblem(Problem):
    """Unknowns: the depth below the surface of one interface, h = h0 + scale x at the nodes of a regular map
    (axes = (x, y), h0[iy, ix]), bilinear between them and clamped to its edges. On the inverted nodes, above it the
    model is up = (lambda, mu), below it down (grid fields: e.g. a sediment profile and the rock), blended by
      s = 1 / (1 + exp(-(depth - h) / width)),
    so that J is differentiable in h, the nodes within a few widths of the interface carrying its gradient:
      dJ/dh = - sum over the column of (dJ/dlambda (lambda_down - lambda_up) + dJ/dmu (mu_down - mu_up)) s (1 - s) / width,
    spread back on the map nodes by the transpose of the bilinear interpolation. A few smooth unknowns move the whole
    interface, which unknowns on every node can only move by changing the velocities of many nodes by large factors
    (the shape of the top of the bedrock inverted from waveforms: Iwaki & Iwata 2011, with B-splines; the blend is
    ours). The other nodes keep fixed = (lambda, mu); the reference model (errors, model_errors) is the blend at h0.
    Bounds: 0 <= h <= hmax. depth: grid field, below the surface.
    profile = (f, p0, pscale): the properties above the interface are unknowns too, through k parameters p = p0 +
    pscale x[n:] (after the n map nodes), f(p) -> (lambda_up, mu_up, dlambda_up/dp, dmu_up/dp), grid fields, the
    derivatives (k, ...); then up is not used, and
      dJ/dp = sum of (dJ/dlambda dlambda_up/dp + dJ/dmu dmu_up/dp) (1 - s), without bounds."""

    def __init__(self, cfg, runner, dobs, axes, h0, up, down, depth, rho_ref, fixed, width, scale=100., hmax=np.inf,
                 pml="follow", backend=None, profile=None):
        super().__init__(cfg, runner, dobs, fixed[0], fixed[1], rho_ref, active=("b",), fixed=fixed, pml=pml,
                         backend=backend)
        self.param, self.active = "interface", ("h",)
        self.h0, self.scale, self.width, self.hmax = np.asarray(h0, float), scale, width, hmax
        ax, ay = (np.asarray(a, float) for a in axes)
        assert self.h0.shape == (len(ay), len(ax)), (self.h0.shape, len(ay), len(ax))
        X, Y, _ = cfg.grid.mesh()
        fx = np.clip((X[self.mask] - ax[0]) / (ax[1] - ax[0]), 0, len(ax) - 1 - 1e-9)
        fy = np.clip((Y[self.mask] - ay[0]) / (ay[1] - ay[0]), 0, len(ay) - 1 - 1e-9)
        i, j = fx.astype(int), fy.astype(int)
        u, v = fx - i, fy - j
        nx = len(ax)
        self.idx = np.stack([j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1], axis=1)
        self.w = np.stack([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v], axis=1)
        self.depth = depth[self.mask]
        self.down = [d[self.mask] for d in down]
        self.profile, self.k = None, 0
        if profile is None:
            self.up = [a[self.mask] for a in up]
        else:
            f, p0, ps = profile
            self.profile, self.p0, self.pscale = f, np.asarray(p0, float), np.asarray(ps, float)
            self.k = len(self.p0)
        self.n, self.scaling, self.basis = self.h0.size, None, None
        lam, mu, _ = self.to_model(np.zeros(self.size))
        self.lam_ref, self.mu_ref = lam, mu
        self.vp_ref = np.sqrt((lam + 2 * mu) / rho_ref)
        self.vs_ref = np.sqrt(mu / rho_ref)
        self.k_ref = lam + 2 * mu / 3

    @property
    def size(self):
        return self.n + self.k

    def interface(self, x):
        """h on the map nodes (ny, nx)."""
        return self.h0 + self.scale * x[:self.n].reshape(self.h0.shape)

    def params(self, x):
        """p of the profile above the interface."""
        return self.p0 + self.pscale * x[self.n:]

    def _up(self, x):
        """(lambda, mu) above the interface on the inverted nodes, and their derivatives in p (None without profile)."""
        if self.profile is None:
            return self.up, None
        lam, mu, dlam, dmu = self.profile(self.params(x))
        m = self.mask
        return [lam[m], mu[m]], [dlam[:, m], dmu[:, m]]

    def _blend(self, x):
        h = np.sum(self.w * self.interface(x).ravel()[self.idx], axis=1)
        t = np.tanh((self.depth - h) / (2 * self.width))
        return 0.5 * (1 + t), 0.25 * (1 - t ** 2)  # s and s (1 - s)

    def to_model(self, x):
        s, _ = self._blend(x)
        up, _ = self._up(x)
        lam, mu = self.fixed[0].copy(), self.fixed[1].copy()
        lam[self.mask] = up[0] + (self.down[0] - up[0]) * s
        mu[self.mask] = up[1] + (self.down[1] - up[1]) * s
        return lam, mu, self.rho_ref

    def to_x(self, lam, mu):
        raise NotImplementedError("the interface is not a function of (lambda, mu)")

    def bound(self, x, frac=None):
        """x with 0 <= h <= hmax, and the number of map nodes moved (p is not bounded)."""
        h = self.interface(x).ravel()
        hb = np.clip(h, 0., self.hmax)
        moved = hb != h
        if not moved.any():
            return x, 0
        x = x.copy()  # only the moved nodes are rewritten, the others keep x to the bit
        i = np.flatnonzero(moved)
        x[i] = (hb[i] - self.h0.ravel()[i]) / self.scale
        return x, int(moved.sum())

    def fit_interface(self, h):
        """x whose interface is closest to the depth h given on the inverted nodes, in the least-squares sense over
        those nodes (normal equations of the bilinear map), clipped to 0 <= h <= hmax; p at p0."""
        A, b = np.zeros((self.n, self.n)), np.zeros(self.n)
        for i in range(4):
            for j in range(4):
                np.add.at(A, (self.idx[:, i], self.idx[:, j]), self.w[:, i] * self.w[:, j])
            np.add.at(b, self.idx[:, i], self.w[:, i] * h)
        hn = np.clip(np.linalg.lstsq(A, b, rcond=None)[0], 0., self.hmax)
        return np.concatenate([(hn - self.h0.ravel()) / self.scale, np.zeros(self.k)])

    def chain(self, x, gl, gm):
        s, ds = self._blend(x)
        up, dup = self._up(x)
        gl, gm = gl[self.mask], gm[self.mask]
        g = -(gl * (self.down[0] - up[0]) + gm * (self.down[1] - up[1])) * ds / self.width
        gh = self.scale * np.bincount(self.idx.ravel(), (self.w * g[:, None]).ravel(), minlength=self.n)
        if dup is None:
            return gh
        return np.concatenate([gh, self.pscale * ((dup[0] * gl + dup[1] * gm) @ (1 - s))])

    def grid_field(self, v):
        return v[:self.n].reshape(self.h0.shape)
