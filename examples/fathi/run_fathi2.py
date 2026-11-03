"""Test cases 2 to 4 of Fathi et al. (2015, sections 4.2 to 4.4): inversion of a smooth profile,
of a layered profile (--case 3) and of a layered profile with an ellipsoidal inclusion (--case 4).

    python run_fathi2.py mu      [--stages p20:20 p30:10]   # 4.2.1, mu only, lambda at the true profile
    python run_fathi2.py lambda  [--stages p20:25]          # 4.2.1, lambda only, mu at the true profile
    python run_fathi2.py both    [--stages p20:25] [--bias 15]   # 4.2.2, simultaneous, lambda bias
    python run_fathi2.py both --case 3 [--stages p20:310 p30:550 p40:252] [--bias 15]   # 4.3
    python run_fathi2.py both --case 4 [--stages p20:410 p30:320 p40:430] [--bias 50] [--noise 0.05]   # 4.4
    python run_fathi2.py both --case 3 --stages p20:310 p30:550 p40:252 --auto-stage 30 0.05
        # same total budget, next frequency as soon as J stalls (30 iterations without a 5 % gain)
    python run_fathi2.py both --case 3 ... --init grad:80:125           # another starting model
    python run_fathi2.py both --case 3 ... --station-step 2.5 --loads z x   # another acquisition

As in the paper: homogeneous 80 MPa initial model; data computed on the inversion mesh;
regularization with R = rho_k ||g_mis|| / ||g_reg|| (section 3.3.1), rho_k decreasing from 0.5 to
0.3: Tikhonov for case 2, Total Variation with eps = 0.01 for cases 3 and 4 (eq. 9); frequency
continuation (3.3.2), with T = 0.45 s for p20 and 0.4 s for p30 and p40 in cases 3 and 4 (0.5 s in
case 2); bias of the lambda direction towards the mu direction (3.3.3, eq. 25), the weight W
decreasing from 1 to 0 over --bias iterations (50 in case 4: kink of figure 26); L-BFGS with 15
vectors. --noise adds Gaussian noise to the data (case 4, figures 27 to 30): standard deviation
noise x max |d| of each trace (station, component), an interpretation, the paper does not define
the level; fixed seed. Main deviation: the number of iterations (compute budget; the paper reaches
1112 iterations in case 3, 1160 in case 4).

--precond (not in the paper, default none) scales the L-BFGS directions (see semfwi.inversion.preconditioner).
Other cases on the same geometry (examples/nappe) reuse invert() with their own Case.
"""
import argparse
import dataclasses
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from fathi import PULSES, homogeneous, layered, make_config, true_model
from semfwi import inversion, problem
from semfwi.config import RUNS_DIR
from semfwi.runner import Runner

SIM_TIME = {2: {}, 3: {"p20": 0.45, "p30": 0.4, "p40": 0.4}}  # per pulse; 0.5 s if absent
SIM_TIME[4] = SIM_TIME[3]


@dataclass
class Case:
    """What differs between the test cases: where the runs go, the true model (grid -> lambda, mu,
    rho), the simulated time per pulse (0.5 s if absent), the geometry of make_config, the
    default regularization, the reference model (homogeneous 80 MPa, as in the paper: x = 0, TV and
    the errors are relative to it), the starting model if different (grid -> lambda, mu) and the
    acquisition (make_config arguments station_step, station_extent, loads)."""
    root: str
    true_model: Callable
    sim_time: dict
    geometry: int = 2
    reg: str = "tv"
    initial: Callable = homogeneous  # reference model, grid -> lambda, mu, rho
    start: Callable = None
    acq: dict = field(default_factory=dict)


def fathi_case(n):
    return Case(os.path.join(RUNS_DIR, "fathi", f"case{n}"), lambda grid: true_model(grid, n), SIM_TIME[n],
                n, "tikhonov" if n == 2 else "tv")


def config(workdir, case, pulse_name):
    case = fathi_case(case) if isinstance(case, int) else case
    return make_config(workdir, case.geometry, pulse_name, sim_time=case.sim_time.get(pulse_name), **case.acq)


def observed(pulse_name, case):
    """Data of the true model, (nsrc, nsta, nt, 3), PMLs extending it as in the paper: computed once
    in <case.root>/data_<pulse> and shared by the inversions of the same case and acquisition."""
    cfg = config(f"{case.root}/data_{pulse_name}", case, pulse_name)
    return problem.observed(cfg, Runner(cfg), cfg.workdir, *case.true_model(cfg.grid), pml="copy")


def smooth(f, sigma, h):
    """Gaussian smoothing of a grid field, standard deviation sigma in m, mirrored edges."""
    if sigma <= 0:
        return f.copy()
    s = sigma / h
    k = np.exp(-0.5 * (np.arange(-int(4 * s), int(4 * s) + 1) / s) ** 2)
    k /= k.sum()
    for a in range(3):
        pad = [(0, 0)] * 3
        pad[a] = (len(k) // 2, len(k) // 2)
        f = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), a, np.pad(f, pad, mode="symmetric"))
    return f


def starting_model(spec, case):
    """--init: hom:V (homogeneous V MPa), grad:A:B (A MPa at the surface to B MPa at the bottom,
    linear in depth), smooth:S (true model smoothed, S m), layers:S (layers of case 3 without the
    inclusion, smoothed, S m; 0: exact). lambda = mu, as in all the Fathi models."""
    kind, *v = spec.split(":")
    v = [float(a) for a in v]

    def start(grid):
        X, Y, Z = grid.mesh()
        if kind == "hom":
            m = np.full(grid.shape, v[0] * 1e6)
        elif kind == "grad":
            m = 1e6 * (v[0] + (v[1] - v[0]) * (-Z) / (-grid.xmin[2]))
        elif kind == "smooth":
            m = smooth(case.true_model(grid)[0], v[0], grid.h)
        elif kind == "layers":
            m = smooth(layered(Z), v[0], grid.h)
        else:
            raise ValueError(spec)
        return m, m.copy()
    return start


def add_noise(d, level, seed=0):
    """d + N(0, (level max |d_trace|)^2), traces along axis -2 (time); d: (nsrc, nsta, nt, 3)."""
    if not level:
        return d
    rng = np.random.default_rng(seed)
    return d + level * np.max(np.abs(d), axis=-2, keepdims=True) * rng.standard_normal(d.shape)


def parser():
    """Arguments shared by the cases on this geometry (run_fathi2 adds --case)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("mu", "lambda", "both"))
    ap.add_argument("--reg", choices=("tikhonov", "tv"), default=None, help="default: tikhonov (2), tv (others)")
    ap.add_argument("--noise", type=float, default=0., help="Gaussian noise on the data, relative (0.05: 5 %%)")
    ap.add_argument("--eps", type=float, default=0.01, help="TV parameter, (MPa/m)^2")
    ap.add_argument("--stages", nargs="+", default=None, help="pulse:iterations")
    ap.add_argument("--bias", type=int, default=0)
    ap.add_argument("--step0", type=float, default=0.05)
    ap.add_argument("--wp", nargs=2, type=float, default=(0.5, 0.3))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--auto-stage", nargs=2, type=float, default=None, metavar=("W", "TOL"),
                    help="move to the next frequency when the best J of the last W iterations is not TOL "
                         "(relative) below the best J before them; the unused iterations go to the next stage, "
                         "the last stage runs to the end of the total budget")
    ap.add_argument("--precond", choices=("none", "block", "depth"), default="none")
    ap.add_argument("--init", default=None, help="starting model (hom:V, grad:A:B, smooth:S, layers:S; "
                    "see starting_model); default: the reference, homogeneous 80 MPa")
    ap.add_argument("--station-step", type=float, default=None, help="m, default 0.625")
    ap.add_argument("--station-extent", type=float, default=None, help="half side of the station grid, m, default 17.5")
    ap.add_argument("--loads", nargs="+", default=None, choices=("z", "x", "y", "zs"),
                    help="one shot per load (fathi.LOADS), default z")
    ap.add_argument("--data-only", action="store_true", help="compute the data of every stage and stop")
    ap.add_argument("--pml", choices=("copy", "follow", "true"), default="follow",
                    help="PML properties: the mean of each mesh layer (follow), the model extended point by "
                         "point by SEM3D (copy, stalls: see problem.py), or frozen at the true model (true)")
    return ap


def main():
    ap = parser()
    ap.add_argument("--case", type=int, choices=(2, 3, 4), default=2)
    args = ap.parse_args()
    invert(args, with_options(fathi_case(args.case), args))


def with_options(case, args):
    """Case with the starting model and acquisition of the options. A non-default acquisition has its
    own directory (its data differ): <root>/acq_s<step>_e<extent>_<loads>."""
    acq = {k: v for k, v in (("station_step", args.station_step), ("station_extent", args.station_extent),
                             ("loads", tuple(args.loads) if args.loads else None)) if v is not None}
    case = dataclasses.replace(case, acq=acq, start=starting_model(args.init, case) if args.init else None)
    if acq:
        name = (f"acq_s{acq.get('station_step', 0.625):g}_e{acq.get('station_extent', 17.5):g}_"
                + "-".join(acq.get("loads", ("z",))))
        case = dataclasses.replace(case, root=f"{case.root}/{name}")
    return case


def invert(args, case):
    """Inversion of case with the options of parser() (semfwi.inversion.invert), in <case.root>/<tag>."""
    stages = [(s.split(":")[0], int(s.split(":")[1])) for s in (args.stages or ["p20:20"])]
    name = args.tag or (args.which + ("_bias" if args.bias else "")
                        + (f"_noise{round(100 * args.noise)}" if args.noise else "")
                        + (f"_{args.precond}" if args.precond != "none" else ""))
    work = f"{case.root}/{name}"
    active = {"mu": ("b",), "lambda": ("a",), "both": ("a", "b")}[args.which]
    if args.data_only:
        for pn, _ in stages:
            observed(pn, case)
        return
    truth = {}

    def make_problem(si, pn):
        cfg = config(f"{work}/{pn}", case, pn)
        lam_t, mu_t, rho = case.true_model(cfg.grid)
        lam0, mu0, _ = case.initial(cfg.grid)
        truth.update(lam_t=lam_t, mu_t=mu_t)
        t0 = time.time()
        dobs = add_noise(observed(pn, case), args.noise, seed=list(PULSES).index(pn))
        print(f"[{pn}] data {time.time() - t0:.0f} s", flush=True)
        return problem.Problem(cfg, Runner(cfg), dobs, lam0, mu0, rho, active=active, fixed=(lam_t, mu_t),
                               pml=(lam_t, mu_t, rho) if args.pml == "true" else args.pml)

    def errors(pb, x):  # in lambda and mu only (1: no better than the reference)
        e = problem.model_errors(pb, x, truth["lam_t"], truth["mu_t"])
        return dict(err_lambda=e["err_lambda"], err_mu=e["err_mu"])

    x0 = (lambda pb: pb.to_x(*case.start(pb.cfg.grid))) if case.start else None
    inversion.invert(work, stages, make_problem, x0=x0, reg=args.reg or case.reg, eps=args.eps, wp=args.wp,
                     bias=args.bias, step0=args.step0, precond=args.precond, auto_stage=args.auto_stage,
                     callback=errors, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
