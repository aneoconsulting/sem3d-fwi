"""Inversion loop shared by the drivers: L-BFGS with Armijo line search, over frequency stages.

    invert(work, stages, make_problem, ...)

stages: [(name, iterations)] or [(name, iterations, blocks)], one Problem per stage from
make_problem(stage index, name) (its data, its pulse, its run directory); the model x goes from one
stage to the next. blocks: the active blocks updated at that stage (all by default), the others frozen
(their gradient set to zero), for a successive inversion of the parameters (Kubina et al. 2018: mu,
then lambda). At each iteration:
- regularization (Tikhonov, TotalVariation or none), weighted by R = wp ||g_mis|| / ||g_reg||
  (Fathi et al. 2015, section 3.3.1), wp going linearly from wp[0] to wp[1] over the total budget;
- direction: L-BFGS, scaled by the node scaling of the problem (Problem.scaling) and, optionally,
  by a preconditioner computed at the start of each stage ("block", "depth", "illum");
- bias of the lambda direction towards the mu direction over the first `bias` iterations
  (section 3.3.3, eq. 25), both parameters inverted only;
- line search: Armijo, backtracking by quadratic interpolation, the trials brought back to the bounds
  of the problem (Problem.bound), the step halved after a failed simulation, a non-physical model or a
  non-finite J; on a failure, a second attempt by steepest descent, L-BFGS memory
  cleared. A failure ends the stage (J no longer decreases, e.g. at the noise level), the unused
  iterations carried over;
- auto_stage (w, tol): next stage as soon as J stalls (stalled), the unused iterations carried over.
Each line-search trial computes J and the gradient (speculative gradient): per-source evaluations are
independent and complete, which allows a remote backend (ArmoniK).

State in work: history.json (one entry per iteration), x<it>.npy, state.npz (x, J, gradient and
L-BFGS memory of the last iteration: a restart resumes there without re-simulating), precond_<stage>.npy.
"""
import json
import os
import shutil
import time

import numpy as np

from .model import UnphysicalModel
from .optim import LBFGS
from .runner import SimulationError


class Tikhonov:
    """J_reg(x) = 1/2 sum_blocks sum_axes sum (difference between neighbouring nodes)^2 on the grid."""

    def __init__(self, pb):
        self.pb = pb

    def _blocks(self, x):
        n = self.pb.n
        return [self.pb.grid_field(x[i * n:(i + 1) * n]) for i in range(len(x) // n)]

    def value(self, x):
        return 0.5 * sum(np.sum(np.diff(v, axis=a) ** 2) for v in self._blocks(x) for a in range(3))

    def grad(self, x):
        out = []
        for v in self._blocks(x):
            g = np.zeros_like(v)
            for a in range(3):
                d = np.diff(v, axis=a)
                sl_lo = [slice(None)] * 3
                sl_hi = [slice(None)] * 3
                sl_lo[a] = slice(None, -1)
                sl_hi[a] = slice(1, None)
                g[tuple(sl_lo)] -= d
                g[tuple(sl_hi)] += d
            out.append(g[self.pb.mask])
        return np.concatenate(out)


class TotalVariation(Tikhonov):
    """J_reg(x) = 1/2 sum_blocks sum_nodes w sqrt(|grad m|^2 + eps) on the grid (eq. 9), grad m in MPa/m
    by forward differences (zero at the last node of each axis), m = m_ref (1 + x) with a
    homogeneous reference (as the 80 MPa initial model), param = "lame" (m = lambda, mu) or "kg" (m = K, G).
    w: the z spacing after the node (before it for the last node) over the mean spacing, 1 on a regular
    grid, so that the sum stays a quadrature of the integral on an explicit z axis."""

    def __init__(self, pb, eps=0.01):
        assert pb.param in ("lame", "kg"), pb.param
        super().__init__(pb)
        self.eps = eps
        refs = {"a": pb.lam_ref if pb.param == "lame" else pb.k_ref, "b": pb.mu_ref}
        grid = pb.cfg.grid
        # per block and per axis: the differences in MPa/m (spacing per axis, per interval along z)
        inv = [1. / grid.steps(c) for c in range(3)]
        self.scale = [[1e-6 * float(np.mean(refs[p][pb.mask])) * i for i in inv] for p in pb.active]
        dz = grid.steps(2)
        self.w = np.concatenate([dz, dz[-1:]]) / dz.mean()

    @staticmethod
    def _diffs(v, s):
        out = []
        for a in range(3):
            d = np.zeros_like(v)
            sl = [slice(None)] * 3
            sl[a] = slice(None, -1)
            shape = [1, 1, 1]
            shape[a] = -1
            d[tuple(sl)] = s[a].reshape(shape) * np.diff(v, axis=a)
            out.append(d)
        return out

    def value(self, x):
        return 0.5 * sum(np.sum(self.w * np.sqrt(sum(d ** 2 for d in self._diffs(v, s)) + self.eps))
                         for v, s in zip(self._blocks(x), self.scale))

    def grad(self, x):
        out = []
        for v, s in zip(self._blocks(x), self.scale):
            d = self._diffs(v, s)
            D = np.sqrt(sum(di ** 2 for di in d) + self.eps)
            g = np.zeros_like(v)
            for a in range(3):
                shape = [1, 1, 1]
                shape[a] = -1
                sa = np.concatenate([s[a], [0.]]).reshape(shape)  # d[a] is zero at the last node
                q = 0.5 * self.w * sa * d[a] / D  # derivative with respect to v[i + e_a] - v[i]
                sl_lo = [slice(None)] * 3
                sl_hi = [slice(None)] * 3
                sl_lo[a] = slice(None, -1)
                sl_hi[a] = slice(1, None)
                g[tuple(sl_lo)] -= q[tuple(sl_lo)]
                g[tuple(sl_hi)] += q[tuple(sl_lo)]
            out.append(g[self.pb.mask])
        return np.concatenate(out)


class NoRegularization:
    def value(self, x):
        return 0.

    def grad(self, x):
        return np.zeros_like(x)


def regularization(kind, pb, eps=0.01):
    """"tikhonov", "tv" (TotalVariation, eps in (MPa/m)^2) or None."""
    if kind is None:
        return NoRegularization()
    return {"tikhonov": lambda: Tikhonov(pb), "tv": lambda: TotalVariation(pb, eps)}[kind]()


def preconditioner(kind, pb, g, d0=None, illum_eps=0.01):
    """Diagonal scaling of the L-BFGS directions, computed from the first gradient g of a frequency stage.
    "block": the lambda and mu blocks of the scaled gradient get the same norm (the data are much more
    sensitive to mu); "depth": in addition, each block is divided by the horizontal mean of |g| at each
    depth (smoothed over 5 nodes, ratio clipped at 100), the surface data being less sensitive to the
    deep nodes; "illum": each node divided by the geometric mean of the forward and adjoint
    illuminations of the evaluation of g (Problem.illum, H_u = int eps(u):eps(u) dt and H_w; diagonal
    pseudo-Hessian, Shin et al. 2001; a normalization by the forward and adjoint fields, Kubina et al.
    2014), plus illum_eps times its maximum on the inverted nodes, instead of d0 (projected like the
    kernels, the illumination carries the node volume as the gradient does). d0: scaling applied first
    (Problem.scaling), returned as is for "none". A frozen block (gradient zero) is not normalized."""
    if kind == "none":
        return d0
    d = np.ones_like(g) if d0 is None or kind == "illum" else d0.copy()
    n, m = pb.n, pb.mask
    if kind == "illum":
        H = np.sqrt(pb.illum[0] * pb.illum[1])[m]
        h = 1. / (H + illum_eps * H.max())
    for i in range(len(g) // n):
        sl = slice(i * n, (i + 1) * n)
        if kind == "depth":
            a = np.abs(pb.grid_field(g[sl]))
            prof = np.array([a[:, :, k][m[:, :, k]].mean() if m[:, :, k].any() else 0. for k in range(a.shape[2])])
            prof = np.convolve(prof, np.ones(5) / 5, mode="same")
            w = np.broadcast_to(1. / np.maximum(prof, prof.max() / 100), a.shape)
            d[sl] *= w[m]
        elif kind == "illum":
            d[sl] *= h
        nrm = np.linalg.norm(d[sl] * g[sl])
        if nrm > 0:
            d[sl] /= nrm
    return d


def stalled(J, w, tol):
    """True when the best of the last w values of J is not tol (relative) below the best before them."""
    return len(J) > 2 * w and min(J[-w:]) > (1 - tol) * min(J[:-w])


def _clean(workdir, it):
    """Traces and adjoint runs of the evaluations of iteration it (summarised in history and state)."""
    for d in os.listdir(workdir):
        if d.startswith(f"it{it:03d}") and os.path.isdir(os.path.join(workdir, d)):
            for s in os.listdir(os.path.join(workdir, d)):
                shutil.rmtree(os.path.join(workdir, d, s, "fwd", "traces"), ignore_errors=True)
                shutil.rmtree(os.path.join(workdir, d, s, "adj"), ignore_errors=True)


def _clean_stale(workdir, it):
    """Trial directories of iteration it left by an interrupted run: its trials are redone and numbered
    from 0 again, and a trial directory numbered after the current one could be taken for a newer one."""
    if os.path.isdir(workdir):
        for d in os.listdir(workdir):
            if d.startswith(f"it{it:03d}_t"):
                shutil.rmtree(os.path.join(workdir, d), ignore_errors=True)


def invert(work, stages, make_problem, x0=None, reg=None, eps=0.01, wp=(0.5, 0.3), bias=0, step0=0.05,
           m=15, c1=1e-4, precond="none", auto_stage=None, callback=None, log=print):
    """Runs (or resumes) the inversion in work. x0: starting x, or pb -> x, zeros by default;
    step0: maximum relative change of the model at the first step of a stage (no L-BFGS memory);
    callback(pb, x): extra entries of each history line (errors to the true model)."""
    os.makedirs(work, exist_ok=True)
    total = sum(s[1] for s in stages)
    hist_fn, state_fn = f"{work}/history.json", f"{work}/state.npz"
    history = json.load(open(hist_fn)) if os.path.exists(hist_fn) else []
    x, saved = None, {}
    if os.path.exists(state_fn):
        saved = dict(np.load(state_fn))
        x = saved["x"]
    it_global = history[-1]["it"] if history else 0

    auto = auto_stage and (int(auto_stage[0]), auto_stage[1])
    carry = 0  # iterations left unused by the stages ended early (auto_stage, line-search failure)
    for si, (pn, niter, *blocks) in enumerate(stages):
        last = si == len(stages) - 1
        niter += carry
        Js = [h["J"] for h in history if h["stage"] == si]
        done_in_stage = len(Js) - 1 if Js else 0
        # done, or ended early: stalled, or a later stage already started (line-search failure)
        later = any(h["stage"] > si for h in history)
        if done_in_stage >= niter or later or (auto and not last and stalled(Js, *auto)):
            carry = max(0, niter - done_in_stage)
            continue
        carry = 0
        pb = make_problem(si, pn)
        pb.nsim = history[-1]["nsim"] if history else 0  # cumulative count, including before a restart
        R_fn = regularization(reg, pb, eps)
        if x is None:
            x = (x0(pb) if callable(x0) else x0.copy()) if x0 is not None else np.zeros(pb.size)
        # frozen blocks: gradient set to zero, so L-BFGS (memory, scaling) never moves them
        free = None
        if blocks and tuple(blocks[0]) != tuple(pb.active):
            assert set(blocks[0]) <= set(pb.active), (blocks[0], pb.active)
            free = np.concatenate([np.full(pb.n, float(p in blocks[0])) for p in pb.active])
        fr = (lambda v: v) if free is None else (lambda v: v * free)  # noqa: E731
        two = len(pb.active) == 2 and free is None
        opt = LBFGS(m=m)

        def record(it_stage, J, R, alpha, trials, dt, g):
            h = dict(it=it_global, stage=si, pulse=pn, it_stage=it_stage, J=J, Jreg=float(R_fn.value(x)),
                     R=R, alpha=alpha, trials=trials, time=dt)
            if callback:
                h.update(callback(pb, x))
            h.update(gnorm=float(np.linalg.norm(g)), nsim=pb.nsim)
            history.append(h)
            np.save(f"{work}/x{it_global:03d}.npy", x)
            # J, gradient and L-BFGS memory at model x: a restart resumes from here without re-simulating
            np.savez(state_fn + ".tmp.npz", x=x, J=J, g=g, pulse=pn, S=np.array(opt.s), Y=np.array(opt.y))
            os.replace(state_fn + ".tmp.npz", state_fn)
            json.dump(history, open(hist_fn, "w"), indent=1)
            log("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in h.items()))

        t0 = time.time()
        if done_in_stage > 0 and "g" in saved and str(saved["pulse"]) == pn:
            J, gmis = float(saved["J"]), saved["g"]
            opt.s, opt.y = list(saved["S"]), list(saved["Y"])
            log(f"  resuming at iteration {it_global} without re-simulating ({len(opt.s)} L-BFGS pairs)")
        else:
            J, _, gmis = pb.evaluate(x, f"it{it_global:03d}_start")
            gmis = fr(gmis)
            if done_in_stage == 0:
                record(0, J, 0., 0., 0, time.time() - t0, gmis)
                Js.append(J)
        saved = {}
        _clean_stale(pb.cfg.workdir, it_global + 1)
        # computed once per stage and kept, so that a restart uses the same scaling
        d_fn = f"{work}/precond_{pn}.npy"
        if precond != "none" and not os.path.exists(d_fn):
            np.save(d_fn, preconditioner(precond, pb, gmis, pb.scaling))
        d = np.load(d_fn) if precond != "none" else pb.scaling
        if d is not None:
            opt.precond = lambda v: d * v
        for k in range(done_in_stage + 1, niter + 1):
            t0 = time.time()
            it_global += 1
            w = wp[0] + (wp[1] - wp[0]) * min(1., (it_global - 1) / max(1, total - 1))
            gr = fr(R_fn.grad(x))
            R = w * np.linalg.norm(gmis) / np.linalg.norm(gr) if np.linalg.norm(gr) > 0 else 0.
            Jt0 = J + R * R_fn.value(x)
            g = gmis + R * gr
            p = opt.direction(g)
            biased = bias and two and it_global <= bias
            if biased:
                W = 1. - (it_global - 1) / bias
                pa, pbm = p[:pb.n], p[pb.n:]
                p = np.concatenate([np.linalg.norm(pa) * (W * pbm / np.linalg.norm(pbm)
                                                          + (1 - W) * pa / np.linalg.norm(pa)), pbm])
            gp = g @ p
            if gp >= 0:
                log("  not a descent direction: reset")
                opt.reset()
                p = -opt.precond(g)
                gp = g @ p
            # already steepest descent (no memory, no bias): a second attempt would change nothing
            plain = not opt.s and not biased
            accepted, trial0 = False, 0
            while True:
                alpha = 1. if opt.s else step0 / np.max(np.abs(p))
                for trial in range(trial0, trial0 + 6):
                    tag = f"it{it_global:03d}_t{trial}"
                    xt, moved = pb.bound(x + alpha * p)
                    if moved:
                        log(f"  trial {trial}: {moved} nodes brought back to the bounds")
                    try:
                        Jm, _, gmis_t = pb.evaluate(xt, tag)
                        gmis_t = fr(gmis_t)
                    except (SimulationError, UnphysicalModel) as e:  # step too large; other failures propagate
                        log(f"  trial {trial} alpha={alpha:.3e} failed ({e})")
                        alpha *= 0.5
                        continue
                    if not np.isfinite(Jm):  # the simulation diverged without failing: step too large
                        log(f"  trial {trial} alpha={alpha:.3e} failed (J = {Jm})")
                        alpha *= 0.5
                        continue
                    Jt = Jm + R * R_fn.value(xt)
                    log(f"  trial {trial} alpha={alpha:.3e} J={Jt:.6e} (Armijo {Jt0 + c1 * alpha * gp:.6e})")
                    if Jt <= Jt0 + c1 * alpha * gp:
                        accepted = True
                        break
                    # minimum of the parabola through Jt0, J'(0) = gp and Jt, clamped to [0.1 alpha, 0.5 alpha]
                    a_q = -gp * alpha ** 2 / (2 * (Jt - Jt0 - gp * alpha))
                    alpha = min(max(a_q, 0.1 * alpha), 0.5 * alpha)
                if accepted or trial0 > 0 or plain:
                    break
                # before giving up on this stage: clear L-BFGS memory, steepest descent (no bias)
                log("  line search failed: L-BFGS memory cleared, retrying with steepest descent")
                opt.reset()
                p = -opt.precond(g)
                gp = g @ p
                trial0 = trial + 1
            if not accepted:
                carry = niter - (k - 1)
                log(f"line search failed at iteration {it_global}"
                    + ("" if last else f": next stage, {carry} iterations carried over"))
                break
            opt.update(xt - x, (gmis_t + R * fr(R_fn.grad(xt))) - g)
            x, J, gmis = xt, Jm, gmis_t
            record(k, J, R, alpha, trial + 1, time.time() - t0, gmis)
            Js.append(J)
            # the simulation directories are no longer needed (traces and gradients are summarised)
            _clean(pb.cfg.workdir, it_global - 1)
            _clean(pb.cfg.workdir, it_global - 2)
            if auto and not last and stalled(Js, *auto):
                log(f"[{pn}] J stalled over {auto[0]} iterations: next stage, {niter - k} iterations carried over")
                carry = niter - k
                break
    return x, history
