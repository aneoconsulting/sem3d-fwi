"""Synthetic inversion on the small cube (levels 2, 5 and 6 of the plan):
two Gaussian anomalies, one on lambda and one on mu, at different locations;
4 buried point forces, 25 surface stations; homogeneous initial model.

    python run_synth.py [--param vel|lame] [--niter 15] [--taylor] [--workdir ...]
By default, workdir = $SEMFWI_RUNS/fwi_synth (lame) or fwi_synth_vel (vel).
"""
import argparse
import os

import numpy as np

from semfwi import model, optim, sem_io
from semfwi.config import RUNS_DIR, Config
from semfwi.problem import Problem
from semfwi.runner import Runner


def make_config(workdir):
    ricker = dict(func="ricker", tau=0.2, freq=5.)
    sources = [
        sem_io.point_source((100., 100., -300.), (0., 0., 1.), **ricker),
        sem_io.point_source((300., 110., -260.), (1., 0., 0.), **ricker),
        sem_io.point_source((110., 290., -220.), (0., 1., 0.), **ricker),
        sem_io.point_source((290., 300., -310.), (1., 1., 1.), **ricker),
    ]
    xs = np.linspace(50., 350., 5)
    stations = [(x, y, 0.) for x in xs for y in xs]
    return Config(workdir=workdir, stations=stations, sources=sources)


def true_model(cfg):
    X, Y, Z = cfg.grid.mesh()
    blob_l = np.exp(-((X - 150) ** 2 + (Y - 250) ** 2 + (Z + 150) ** 2) / 60. ** 2)
    blob_m = np.exp(-((X - 250) ** 2 + (Y - 160) ** 2 + (Z + 190) ** 2) / 60. ** 2)
    one = np.ones(cfg.grid.shape)
    return cfg.lam0 * (1 - 0.15 * blob_l), cfg.mu0 * (1 - 0.10 * blob_m), cfg.rho0 * one


def default_workdir(param):
    return os.path.join(RUNS_DIR, "fwi_synth" + ("_vel" if param == "vel" else ""))


def model_errors(pb, x, lam_t, mu_t):
    """Relative errors to the true model on the inverted nodes, in (lambda, mu) and in (Vp, Vs)."""
    lam, mu, rho = pb.to_model(x)
    m = pb.mask
    vp, vs = np.sqrt((lam + 2 * mu) / rho), np.sqrt(mu / rho)
    vp_t, vs_t = np.sqrt((lam_t + 2 * mu_t) / rho), np.sqrt(mu_t / rho)
    out = {}
    for name, a, t, ref in (("lambda", lam, lam_t, pb.lam_ref), ("mu", mu, mu_t, pb.mu_ref),
                            ("vp", vp, vp_t, pb.vp_ref), ("vs", vs, vs_t, pb.vs_ref)):
        out["err_" + name] = float(np.linalg.norm((a - t)[m]) / np.linalg.norm((t - ref)[m]))
    return out


def observed(cfg, runner, lam, mu, rho):
    dirs = []
    for i, src in enumerate(cfg.sources):
        d = os.path.join(cfg.workdir, "obs", f"s{i}")
        runner.prepare(d)
        model.write_model(d, cfg.grid, lam, mu, rho)
        sem_io.write_input(d, cfg, [src])
        dirs.append(d)
    runner.run_many(dirs)
    return np.stack([sem_io.read_traces(d, len(cfg.stations)) for d in dirs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", choices=("lame", "vel"), default="vel")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--niter", type=int, default=15)
    ap.add_argument("--step0", type=float, default=0.05)
    ap.add_argument("--taylor", action="store_true", help="Taylor test of the multi-source gradient, then stop")
    args = ap.parse_args()

    cfg = make_config(args.workdir or default_workdir(args.param))
    os.makedirs(cfg.workdir, exist_ok=True)
    runner = Runner(cfg)
    lam_t, mu_t, rho_t = true_model(cfg)
    dobs = observed(cfg, runner, lam_t, mu_t, rho_t)
    one = np.ones(cfg.grid.shape)
    pb = Problem(cfg, runner, dobs, cfg.lam0 * one, cfg.mu0 * one, cfg.rho0 * one, param=args.param)
    x_true = pb.to_x(lam_t, mu_t)
    n = pb.n
    print(f"parameters {args.param}, {len(cfg.sources)} sources, {len(cfg.stations)} stations, {n} inverted nodes x 2 parameters, "
          f"{cfg.nt} steps of {cfg.dt} s")

    if args.taylor:
        x0 = np.zeros(pb.size)
        J0, u0 = pb.forward(x0, "taylor/x0")
        g0 = pb.gradient(x0, "taylor/x0", u0)
        dx = x_true
        dJ = g0 @ dx
        print(f"J0 = {J0:.6e}, dJ.dx = {dJ:.6e}")
        prev = None
        for h in (0.5, 0.25, 0.125, 0.0625):
            Jh, _ = pb.forward(h * dx, f"taylor/h{h:g}", save_strain=False)
            r1, r2 = abs(Jh - J0), abs(Jh - J0 - h * dJ)
            rate = "" if prev is None else f"  order {np.log2(prev / r2):.2f}"
            print(f"  h = {h:<7g} |J-J0| = {r1:.3e}  |J-J0-h dJ| = {r2:.3e}{rate}")
            prev = r2
        return

    def callback(it, x, g):
        np.save(os.path.join(cfg.workdir, f"x{it:03d}.npy"), x)
        out = model_errors(pb, x, lam_t, mu_t)
        out.update(gnorm=float(np.linalg.norm(g)), nsim=pb.nsim)
        return out

    optim.minimize(pb, np.zeros(pb.size), args.niter, cfg.workdir, step0=args.step0, callback=callback,
                   log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
