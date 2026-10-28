"""L-BFGS with Armijo line search (backtracking by quadratic interpolation).

Each line-search trial computes J and the gradient (speculative gradient): the accepted trial,
usually the first one, costs two simulations per source, like a gradient computation;
a rejected trial costs two instead of one. Per-source evaluations are thus independent and
complete, which allows handing them to a remote backend (ArmoniK) without keeping strains there.
The state is saved at each iteration (workdir/state.npz and history.json) so it can be resumed.
"""
import json
import os
import time

import numpy as np

from .model import UnphysicalModel
from .runner import SimulationError


class LBFGS:
    def __init__(self, m=5, precond=None):
        self.m = m
        self.precond = precond or (lambda v: v)
        self.s, self.y = [], []

    def direction(self, g):
        q = g.copy()
        alphas = []
        for s, y in reversed(list(zip(self.s, self.y))):
            a = s @ q / (y @ s)
            alphas.append(a)
            q -= a * y
        r = self.precond(q)
        if self.s:
            s, y = self.s[-1], self.y[-1]
            r *= (s @ y) / (y @ self.precond(y))
        for (s, y), a in zip(zip(self.s, self.y), reversed(alphas)):
            b = y @ r / (y @ s)
            r += (a - b) * s
        return -r

    def update(self, s, y):
        if s @ y <= 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
            return False
        self.s.append(s)
        self.y.append(y)
        if len(self.s) > self.m:
            self.s.pop(0)
            self.y.pop(0)
        return True

    def reset(self):
        self.s, self.y = [], []


def minimize(problem, x0, niter, workdir, step0=0.02, c1=1e-4, max_trials=6, m=5, precond=None,
             callback=None, log=print):
    """Minimizes J; step0: maximum relative change of the model at the first step."""
    state_fn = os.path.join(workdir, "state.npz")
    hist_fn = os.path.join(workdir, "history.json")
    opt = LBFGS(m, precond)
    if os.path.exists(state_fn):
        st = np.load(state_fn)
        x, J, g, it0 = st["x"], float(st["J"]), st["g"], int(st["it"]) + 1
        opt.s, opt.y = list(st["s"]), list(st["y"])
        history = json.load(open(hist_fn))
        log(f"resuming at iteration {it0}, J = {J:.6e}")
    else:
        t0 = time.time()
        x = x0.copy()
        J, _, g = problem.evaluate(x, "it000")
        it0 = 1
        history = [dict(it=0, J=J, alpha=0., trials=0, time=time.time() - t0)]
        if callback:
            history[-1].update(callback(0, x, g))
        _save(state_fn, hist_fn, x, J, g, 0, opt, history)
        log(_line(history[-1]))

    for it in range(it0, niter + 1):
        t0 = time.time()
        p = opt.direction(g)
        gp = g @ p
        if gp >= 0:
            log("not a descent direction: resetting L-BFGS")
            opt.reset()
            p, gp = -g, -(g @ g)
        alpha = 1. if opt.s else step0 / np.max(np.abs(p))
        accepted = False
        for trial in range(max_trials):
            tag = f"it{it:03d}_t{trial}"
            xt = x + alpha * p
            try:
                Jt, _, gt = problem.evaluate(xt, tag)
            except (SimulationError, UnphysicalModel) as e:  # step too large; other failures propagate
                log(f"  trial {trial} alpha = {alpha:.3e}: failed ({e})")
                alpha *= 0.5
                continue
            log(f"  trial {trial} alpha = {alpha:.3e} J = {Jt:.6e} (Armijo: {J + c1 * alpha * gp:.6e})")
            if Jt <= J + c1 * alpha * gp:
                accepted = True
                break
            # minimum of the parabola through J, J'(0) = gp and Jt, clamped to [0.1 alpha, 0.5 alpha]
            a_q = -gp * alpha ** 2 / (2 * (Jt - J - gp * alpha))
            alpha = min(max(a_q, 0.1 * alpha), 0.5 * alpha)
        if not accepted:
            log(f"iteration {it}: line search failed, stopping")
            break
        opt.update(xt - x, gt - g)
        x, J, g = xt, Jt, gt
        history.append(dict(it=it, J=J, alpha=alpha, trials=trial + 1, time=time.time() - t0))
        if callback:
            history[-1].update(callback(it, x, g))
        _save(state_fn, hist_fn, x, J, g, it, opt, history)
        log(_line(history[-1]))
    return x, history


def _line(h):
    return "  ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in h.items())


def _save(state_fn, hist_fn, x, J, g, it, opt, history):
    np.savez(state_fn, x=x, J=J, g=g, it=it, s=np.array(opt.s).reshape(-1, len(x)),
             y=np.array(opt.y).reshape(-1, len(x)))
    with open(hist_fn, "w") as f:
        json.dump(history, f, indent=1)
