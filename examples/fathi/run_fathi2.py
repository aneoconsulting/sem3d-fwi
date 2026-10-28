"""Test cases 2 and 3 of Fathi et al. (2015, sections 4.2 and 4.3): inversion of a smooth profile,
and of a layered profile (--case 3).

    python run_fathi2.py mu      [--stages p20:20 p30:10]   # 4.2.1, mu only, lambda at the true profile
    python run_fathi2.py lambda  [--stages p20:25]          # 4.2.1, lambda only, mu at the true profile
    python run_fathi2.py both    [--stages p20:25] [--bias 15]   # 4.2.2, simultaneous, lambda bias
    python run_fathi2.py both --case 3 [--stages p20:310 p30:550 p40:252] [--bias 15]   # 4.3

As in the paper: homogeneous 80 MPa initial model; data computed on the inversion mesh;
regularization with R = rho_k ||g_mis|| / ||g_reg|| (section 3.3.1), rho_k decreasing from 0.5 to
0.3: Tikhonov for case 2, Total Variation with eps = 0.01 for case 3 (eq. 9); frequency continuation
(3.3.2), with T = 0.45 s for p20 and 0.4 s for p30 and p40 in case 3 (0.5 s in case 2); bias of the
lambda direction towards the mu direction (3.3.3, eq. 25), the weight W decreasing from 1 to 0 over
--bias iterations; L-BFGS with 15 vectors. Main deviation: the number of iterations (compute budget;
the paper reaches 1112 iterations in case 3).
"""
import argparse
import json
import os
import time

import numpy as np

from fathi import homogeneous, make_config, true_model
from semfwi import model, sem_io
from semfwi.config import RUNS_DIR
from semfwi.optim import LBFGS
from semfwi.problem import Problem
from semfwi.runner import Runner, SimulationError

SIM_TIME = {2: {}, 3: {"p20": 0.45, "p30": 0.4, "p40": 0.4}}  # per pulse; 0.5 s if absent


def root(case):
    return os.path.join(RUNS_DIR, "fathi", f"case{case}")


def config(workdir, case, pulse_name):
    return make_config(workdir, case, pulse_name, sim_time=SIM_TIME[case].get(pulse_name))


def observed(pulse_name, case=2):
    cfg = config(f"{root(case)}/data_{pulse_name}", case, pulse_name)
    runner = Runner(cfg)
    lam, mu, rho = true_model(cfg.grid, case)
    d = f"{cfg.workdir}/sim"
    if os.path.exists(f"{d}/.done") and os.path.isdir(f"{d}/traces"):
        # already computed (shared by the inversions running in parallel): read only
        return sem_io.read_traces(d, len(cfg.stations))[None]
    runner.prepare(d)
    model.write_model(d, cfg.grid, lam, mu, rho, nmat=len(cfg.layer_list))
    model.write_pml_materials(d, cfg, lam, mu, rho)
    sem_io.write_input(d, cfg, sem_io.materialize_sources(d, cfg, cfg.sources))
    runner.run(d)
    return sem_io.read_traces(d, len(cfg.stations))[None]


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
    """J_reg(x) = 1/2 sum_blocks sum_nodes sqrt(|grad m|^2 + eps) on the grid (eq. 9), grad m in MPa/m
    by forward differences (zero at the last node of each axis), m = m_ref (1 + x) with a
    homogeneous reference (as the 80 MPa initial model), param = "lame"."""

    def __init__(self, pb, eps=0.01):
        assert pb.param == "lame", pb.param
        super().__init__(pb)
        self.eps = eps
        refs = {"a": pb.lam_ref, "b": pb.mu_ref}
        self.scale = [1e-6 * float(np.mean(refs[p][pb.mask])) / pb.cfg.grid.h for p in pb.active]

    @staticmethod
    def _diffs(v, s):
        out = []
        for a in range(3):
            d = np.zeros_like(v)
            sl = [slice(None)] * 3
            sl[a] = slice(None, -1)
            d[tuple(sl)] = s * np.diff(v, axis=a)
            out.append(d)
        return out

    def value(self, x):
        return 0.5 * sum(np.sum(np.sqrt(sum(d ** 2 for d in self._diffs(v, s)) + self.eps))
                         for v, s in zip(self._blocks(x), self.scale))

    def grad(self, x):
        out = []
        for v, s in zip(self._blocks(x), self.scale):
            d = self._diffs(v, s)
            D = np.sqrt(sum(di ** 2 for di in d) + self.eps)
            g = np.zeros_like(v)
            for a in range(3):
                q = 0.5 * s * d[a] / D  # derivative with respect to v[i + e_a] - v[i]
                sl_lo = [slice(None)] * 3
                sl_hi = [slice(None)] * 3
                sl_lo[a] = slice(None, -1)
                sl_hi[a] = slice(1, None)
                g[tuple(sl_lo)] -= q[tuple(sl_lo)]
                g[tuple(sl_hi)] += q[tuple(sl_lo)]
            out.append(g[self.pb.mask])
        return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("mu", "lambda", "both"))
    ap.add_argument("--case", type=int, choices=(2, 3), default=2)
    ap.add_argument("--reg", choices=("tikhonov", "tv"), default=None, help="default: tikhonov (2), tv (3)")
    ap.add_argument("--eps", type=float, default=0.01, help="TV parameter, (MPa/m)^2")
    ap.add_argument("--stages", nargs="+", default=None, help="pulse:iterations")
    ap.add_argument("--bias", type=int, default=0)
    ap.add_argument("--step0", type=float, default=0.05)
    ap.add_argument("--wp", nargs=2, type=float, default=(0.5, 0.3))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    stages = [(s.split(":")[0], int(s.split(":")[1])) for s in (args.stages or ["p20:20"])]
    name = args.tag or args.which + ("_bias" if args.bias else "")
    work = f"{root(args.case)}/{name}"
    reg_kind = args.reg or ("tv" if args.case == 3 else "tikhonov")
    os.makedirs(work, exist_ok=True)
    active = {"mu": ("b",), "lambda": ("a",), "both": ("a", "b")}[args.which]
    total = sum(n for _, n in stages)

    hist_fn, state_fn = f"{work}/history.json", f"{work}/state.npz"
    history = json.load(open(hist_fn)) if os.path.exists(hist_fn) else []
    x, saved = None, {}
    if os.path.exists(state_fn):
        saved = dict(np.load(state_fn))
        x = saved["x"]
    it_global = history[-1]["it"] if history else 0

    for si, (pn, niter) in enumerate(stages):
        done_in_stage = sum(1 for h in history if h["stage"] == si and h["it_stage"] > 0)
        if done_in_stage >= niter:
            continue
        cfg = config(f"{work}/{pn}", args.case, pn)
        runner = Runner(cfg)
        lam_t, mu_t, rho = true_model(cfg.grid, args.case)
        lam0, mu0, _ = homogeneous(cfg.grid)
        t0 = time.time()
        dobs = observed(pn, args.case)
        print(f"[{pn}] data {time.time() - t0:.0f} s", flush=True)
        pb = Problem(cfg, runner, dobs, lam0, mu0, rho, active=active, fixed=(lam_t, mu_t), pml="follow")
        pb.nsim = history[-1]["nsim"] if history else 0  # cumulative count, including before a restart
        reg = TotalVariation(pb, args.eps) if reg_kind == "tv" else Tikhonov(pb)
        x_true = pb.to_x(lam_t, mu_t)
        if x is None:
            x = np.zeros(pb.size)

        opt = LBFGS(m=15)

        def record(it_stage, J, R, alpha, trials, dt, g):
            lam, mu, _ = pb.to_model(x)
            m = pb.mask
            el = float(np.linalg.norm((lam - lam_t)[m]) / np.linalg.norm((lam0 - lam_t)[m]))
            em = float(np.linalg.norm((mu - mu_t)[m]) / np.linalg.norm((mu0 - mu_t)[m]))
            h = dict(it=it_global, stage=si, pulse=pn, it_stage=it_stage, J=J, Jreg=float(reg.value(x)),
                     R=R, alpha=alpha, trials=trials, time=dt, err_lambda=el, err_mu=em,
                     gnorm=float(np.linalg.norm(g)), nsim=pb.nsim)
            history.append(h)
            np.save(f"{work}/x{it_global:03d}.npy", x)
            # J, gradient and L-BFGS memory at model x: a restart resumes from here without re-simulating
            np.savez(state_fn + ".tmp.npz", x=x, J=J, g=g, pulse=pn, S=np.array(opt.s), Y=np.array(opt.y))
            os.replace(state_fn + ".tmp.npz", state_fn)
            json.dump(history, open(hist_fn, "w"), indent=1)
            print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in h.items()),
                  flush=True)

        t0 = time.time()
        if done_in_stage > 0 and "g" in saved and str(saved["pulse"]) == pn:
            J, gmis = float(saved["J"]), saved["g"]
            opt.s, opt.y = list(saved["S"]), list(saved["Y"])
            print(f"  resuming at iteration {it_global} without re-simulating ({len(opt.s)} L-BFGS pairs)", flush=True)
        else:
            tag = f"it{it_global:03d}_start"
            J, _, gmis = pb.evaluate(x, tag)
            if done_in_stage == 0:
                record(0, J, 0., 0., 0, time.time() - t0, gmis)
        saved = {}
        for k in range(done_in_stage + 1, niter + 1):
            t0 = time.time()
            it_global += 1
            wp = args.wp[0] + (args.wp[1] - args.wp[0]) * min(1., (it_global - 1) / max(1, total - 1))
            gr = reg.grad(x)
            R = wp * np.linalg.norm(gmis) / np.linalg.norm(gr) if np.linalg.norm(gr) > 0 else 0.
            Jt0 = J + R * reg.value(x)
            g = gmis + R * gr
            p = opt.direction(g)
            if args.bias and len(active) == 2 and it_global <= args.bias:
                W = 1. - (it_global - 1) / args.bias
                pa, pbm = p[:pb.n], p[pb.n:]
                p = np.concatenate([np.linalg.norm(pa) * (W * pbm / np.linalg.norm(pbm)
                                                          + (1 - W) * pa / np.linalg.norm(pa)), pbm])
            gp = g @ p
            if gp >= 0:
                print("  not a descent direction: reset", flush=True)
                opt.reset()
                p, gp = -g, -(g @ g)
            # already steepest descent (no memory, no bias): a second attempt would change nothing
            plain = not opt.s and not (args.bias and len(active) == 2 and it_global <= args.bias)
            accepted, trial0 = False, 0
            while True:
                alpha = 1. if opt.s else args.step0 / np.max(np.abs(p))
                for trial in range(trial0, trial0 + 6):
                    tag = f"it{it_global:03d}_t{trial}"
                    xt = x + alpha * p
                    try:
                        Jm, _, gmis_t = pb.evaluate(xt, tag)
                    except (SimulationError, model.UnphysicalModel) as e:
                        print(f"  trial {trial} alpha={alpha:.3e} failed ({e})", flush=True)
                        alpha *= 0.5
                        continue
                    Jt = Jm + R * reg.value(xt)
                    print(f"  trial {trial} alpha={alpha:.3e} J={Jt:.6e} (Armijo {Jt0 + 1e-4 * alpha * gp:.6e})",
                          flush=True)
                    if Jt <= Jt0 + 1e-4 * alpha * gp:
                        accepted = True
                        break
                    a_q = -gp * alpha ** 2 / (2 * (Jt - Jt0 - gp * alpha))
                    alpha = min(max(a_q, 0.1 * alpha), 0.5 * alpha)
                if accepted or trial0 > 0 or plain:
                    break
                # before giving up on this frequency: clear L-BFGS memory, steepest descent (no bias)
                print("  line search failed: L-BFGS memory cleared, retrying with steepest descent", flush=True)
                opt.reset()
                p, gp = -g, -(g @ g)
                trial0 = trial + 1
            if not accepted:
                print(f"line search failed at iteration {it_global}", flush=True)
                break
            opt.update(xt - x, (gmis_t + R * reg.grad(xt)) - g)
            x, J, gmis = xt, Jm, gmis_t
            record(k, J, R, alpha, trial + 1, time.time() - t0, gmis)
            # the simulation directories are no longer needed (traces and gradients are summarised)
            for old in os.listdir(cfg.workdir):
                if old.startswith(f"it{it_global - 1:03d}") or old.startswith(f"it{it_global - 2:03d}"):
                    os.system(f"rm -rf {cfg.workdir}/{old}/s0/fwd/traces {cfg.workdir}/{old}/s0/adj")


if __name__ == "__main__":
    main()
