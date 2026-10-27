"""Inversion problem: J(x) and grad J(x) over all sources.

Unknowns on the interior grid nodes (model.interior_mask), as relative perturbations,
for the active parameters (active, a subset of ("a", "b")):
  param = "lame" : a = lambda / lambda_ref - 1, b = mu / mu_ref - 1;
  param = "vel"  : a = Vp / Vp_ref - 1,         b = Vs / Vs_ref - 1,
                   with mu = rho Vs^2, lambda = rho Vp^2 - 2 rho Vs^2.
An inactive parameter keeps the value from fixed (the reference model by default). rho and the
nodes close to the PMLs stay fixed. SEM3D always reads lambda and mu; for velocities, the gradient
follows from the chain rule:
  dJ/dVp = 2 rho Vp dJ/dlambda,   dJ/dVs = 2 rho Vs (dJ/dmu - 2 dJ/dlambda).

pml: None (mesher properties), "follow" (the current model is extended into the PMLs, layer
by layer, as in Fathi et al.) or a fixed (lam, mu, rho). The gradient ignores the dependence of the
PMLs on the model (it is restricted to the interior domain, as in the paper).
"""
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import gradient, misfit, model, sem_io


def _kernels(args):
    return gradient.kernels(*args)


class Problem:
    def __init__(self, cfg, runner, dobs, lam_ref, mu_ref, rho_ref, param="lame", active=("a", "b"),
                 fixed=None, pml=None):
        assert param in ("lame", "vel"), param
        self.cfg, self.runner, self.dobs, self.param = cfg, runner, dobs, param
        self.active = tuple(active)
        self.lam_ref, self.mu_ref, self.rho_ref = lam_ref, mu_ref, rho_ref
        self.vp_ref = np.sqrt((lam_ref + 2 * mu_ref) / rho_ref)
        self.vs_ref = np.sqrt(mu_ref / rho_ref)
        self.fixed = fixed if fixed is not None else (lam_ref, mu_ref)
        self.pml = pml
        self.mask = model.interior_mask(cfg.grid, cfg.L, cfg.depth, cfg.x0)
        self.n = int(self.mask.sum())
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
        else:
            a, b = np.sqrt((lam + 2 * mu) / self.rho_ref), np.sqrt(mu / self.rho_ref)
            a0, b0 = self.vp_ref, self.vs_ref
        blocks = {"a": a[self.mask] / a0[self.mask] - 1, "b": b[self.mask] / b0[self.mask] - 1}
        return np.concatenate([blocks[p] for p in self.active])

    def chain(self, x, gl, gm):
        """Gradient with respect to x from dJ/dlambda and dJ/dmu on the grid."""
        if self.param == "lame":
            ga, gb = gl * self.lam_ref, gm * self.mu_ref
        else:
            lam, mu, rho = self.to_model(x)
            vp, vs = np.sqrt((lam + 2 * mu) / rho), np.sqrt(mu / rho)
            ga = 2 * rho * vp * gl * self.vp_ref
            gb = 2 * rho * vs * (gm - 2 * gl) * self.vs_ref
        blocks = {"a": ga[self.mask], "b": gb[self.mask]}
        return np.concatenate([blocks[p] for p in self.active])

    def grid_field(self, v):
        out = np.zeros(self.cfg.grid.shape)
        out[self.mask] = v
        return out

    def _dir(self, tag, i, kind):
        return os.path.join(self.cfg.workdir, tag, f"s{i}", kind)

    def _write_model(self, d, lam, mu, rho):
        model.write_model(d, self.cfg.grid, lam, mu, rho, nmat=len(self.cfg.layer_list))
        if self.pml == "follow":
            model.write_pml_materials(d, self.cfg, lam, mu, rho)
        elif self.pml is not None:
            model.write_pml_materials(d, self.cfg, *self.pml)

    def forward(self, x, tag, save_strain=True):
        """J(x); the forward simulations of all sources are in workdir/tag/s*/fwd."""
        lam, mu, rho = self.to_model(x)
        model.check_bounds(lam, mu, rho)
        dirs = []
        for i, src in enumerate(self.cfg.sources):
            d = self._dir(tag, i, "fwd")
            self.runner.prepare(d)
            self._write_model(d, lam, mu, rho)
            sem_io.write_input(d, self.cfg, sem_io.materialize_sources(d, self.cfg, [src]),
                               save_strain=save_strain)
            dirs.append(d)
        self.runner.run_many(dirs)
        self.nsim += len(dirs)
        u = np.stack([sem_io.read_traces(d, len(self.cfg.stations)) for d in dirs])
        return misfit.cost(u, self.dobs, self.cfg.dt), u

    def gradient(self, x, tag, u):
        """grad J(x) after forward(x, tag) with the strains. The strains are then deleted;
        dJ/dlambda and dJ/dmu are kept in workdir/tag/grad.npz."""
        fn = os.path.join(self.cfg.workdir, tag, "grad.npz")
        if os.path.exists(fn):
            st = np.load(fn)
            return self.chain(x, st["gl"], st["gm"])
        lam, mu, rho = self.to_model(x)
        jobs = []
        dirs = []
        for i in range(len(self.cfg.sources)):
            d = self._dir(tag, i, "adj")
            self.runner.prepare(d)
            self._write_model(d, lam, mu, rho)
            tau, hist = misfit.adjoint_history(misfit.residual(u[i], self.dobs[i]), self.cfg.dt)
            srcs = sem_io.write_adjoint_sources(d, self.cfg, tau, hist)
            sem_io.write_input(d, self.cfg, srcs, save_strain=True)
            dirs.append(d)
            for r in range(self.cfg.nprocs):
                jobs.append((self.cfg.grid, self._dir(tag, i, "fwd"), d, self.cfg.dt, 3e7, [r]))
        self.runner.run_many(dirs)
        self.nsim += len(dirs)
        gl = np.zeros(self.cfg.grid.shape)
        gm = np.zeros(self.cfg.grid.shape)
        with ProcessPoolExecutor(max_workers=max(1, min(4, len(jobs)))) as ex:
            for kl, km in ex.map(_kernels, jobs):
                gl += kl
                gm += km
        np.savez(fn, gl=gl, gm=gm)
        self.drop_strain(tag)
        return self.chain(x, gl, gm)

    def drop_strain(self, tag):
        """Strains take several GB per simulation."""
        root = os.path.join(self.cfg.workdir, tag)
        for i in range(len(self.cfg.sources)):
            for kind in ("fwd", "adj"):
                shutil.rmtree(os.path.join(root, f"s{i}", kind, "strain"), ignore_errors=True)
