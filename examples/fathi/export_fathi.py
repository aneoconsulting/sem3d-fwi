"""Gather the results of the Fathi test cases into one JSON file:
table 2 (case 1), histories and profiles (cases 2 and 3), pulses and spectra, traces.

    python export_fathi.py [--out $SEMFWI_RUNS/fathi/summary.json]
"""
import argparse
import glob
import json
import os

import numpy as np

from fathi import PULSES, homogeneous, layered, make_config, profile, pulse, true_model
from run_fathi2 import config
from semfwi import sem_io
from semfwi.config import RUNS_DIR
from semfwi.problem import Problem

ROOT = os.path.join(RUNS_DIR, "fathi")
PROFILE_XY = [(0., 0.), (10., 10.), (20., 20.)]
NAMES = {2: ("mu", "lambda", "both", "both_bias"), 3: ("both_bias", "both_bias_test")}


def pulses():
    t = np.arange(0, 0.4, 1e-3)
    out = {}
    for pn in PULSES:
        f = pulse(pn)(t)
        tt = np.arange(0, 2.0, 1e-4)
        s = np.abs(np.fft.rfft(pulse(pn)(tt))) * 1e-4
        fr = np.fft.rfftfreq(len(tt), 1e-4)
        keep = fr <= 50
        m, sg = PULSES[pn]
        literal = np.exp(-((t - m) / sg) ** 2)
        out[pn] = dict(t=t.round(4).tolist(), f=f.round(5).tolist(), literal=literal.round(5).tolist(),
                       freq=fr[keep].round(3).tolist()[::5], spec=s[keep].round(6).tolist()[::5])
    return out


def case1():
    fn = f"{ROOT}/case1/table2.json"
    out = dict(rows=json.load(open(fn)) if os.path.exists(fn) else [])
    gfile = f"{ROOT}/case1/p20/g0.npy"
    if os.path.exists(gfile):
        cfg = make_config(f"{ROOT}/case1/p20", 1, "p20")
        lam0, mu0, rho = homogeneous(cfg.grid)
        pb = Problem(cfg, None, None, lam0, mu0, rho)
        g = np.load(gfile)
        gl, gm = pb.grid_field(g[:pb.n]), pb.grid_field(g[pb.n:])
        j = int(np.argmin(np.abs(cfg.grid.axis(1) - 0.)))
        out["kernel_slice"] = dict(x=cfg.grid.axis(0).tolist(), z=cfg.grid.axis(2).tolist(),
                                   lam=gl[:, j, :].T.tolist(), mu=gm[:, j, :].T.tolist())
    return out


def experiment(name, case=2):
    work = f"{ROOT}/case{case}/{name}"
    hf = f"{work}/history.json"
    if not os.path.exists(hf):
        return None
    hist = json.load(open(hf))
    cfg = make_config(work, case)
    lam_t, mu_t, rho = true_model(cfg.grid, case)
    lam0, mu0, _ = homogeneous(cfg.grid)
    active = {"mu": ("b",), "lambda": ("a",)}.get(name.split("_")[0], ("a", "b"))
    pb = Problem(cfg, None, None, lam0, mu0, rho, active=active, fixed=(lam_t, mu_t))
    last = hist[-1]["it"]
    x = np.load(f"{work}/x{last:03d}.npy")
    lam, mu, _ = pb.to_model(x)
    X, Y, Z = cfg.grid.mesh()
    z = cfg.grid.axis(2)
    prof = {}
    for (px, py) in PROFILE_XY:
        i = int(np.argmin(np.abs(cfg.grid.axis(0) - px)))
        j = int(np.argmin(np.abs(cfg.grid.axis(1) - py)))
        prof[f"{px:g},{py:g}"] = dict(lam=(lam[i, j, :] / 1e6).round(3).tolist(),
                                      mu=(mu[i, j, :] / 1e6).round(3).tolist())
    j0 = int(np.argmin(np.abs(cfg.grid.axis(1) - 0.)))
    sl = dict(x=cfg.grid.axis(0).tolist(), z=z.tolist(),
              lam=(lam[:, j0, :].T / 1e6).round(2).tolist(), mu=(mu[:, j0, :].T / 1e6).round(2).tolist())
    # observed, initial and final traces at a few stations (vertical component)
    waves = {}
    first = [d for d in sorted(glob.glob(f"{work}/initial_fwd/s0/fwd")) if os.path.isdir(f"{d}/traces")]
    finals = sorted(glob.glob(f"{work}/*/it{last:03d}_t*/s0/fwd")) or sorted(glob.glob(f"{work}/*/it{last:03d}_start/s0/fwd"))
    obs = sorted(glob.glob(f"{work}/data_*/sim")) or sorted(glob.glob(f"{ROOT}/case{case}/*/data_p20/sim"))
    try:
        nsta = len(cfg.stations)
        sel = [nsta // 2, 0, nsta // 2 + 20]
        for lab, dirs in (("initial", first), ("final", finals)):
            # initial traces are computed with the p20 pulse: irrelevant once the pulse has changed
            if lab == "initial" and hist[-1]["pulse"] != "p20":
                continue
            if dirs and os.path.isdir(f"{dirs[-1]}/traces"):
                u = sem_io.read_traces(dirs[-1], nsta)
                waves[lab] = u[sel, :, 2].round(12).tolist()
        pdir = f"{work}/{hist[-1]['pulse']}/data_{hist[-1]['pulse']}/sim"
        dd = glob.glob(f"{ROOT}/case{case}/data_{hist[-1]['pulse']}/sim")
        if dd:
            waves["observed"] = sem_io.read_traces(dd[0], nsta)[sel, :, 2].round(12).tolist()
        waves["stations"] = [cfg.stations[s] for s in sel]
        waves["dt"] = cfg.dt
    except Exception as e:  # the traces are only a complement
        waves["error"] = str(e)
    return dict(history=hist, z=z.tolist(), profiles=prof,
                target={"mu": ((layered if case == 3 else profile)(z) / 1e6).round(3).tolist()}, slice=sl, waves=waves)


def initial_forward(name, case=2):
    """Traces at the initial model (the iteration 0 directories are cleaned up during the inversion)."""
    from semfwi.runner import Runner
    work = f"{ROOT}/case{case}/{name}"
    if os.path.isdir(f"{work}/initial_fwd/s0/fwd/traces") or not os.path.isdir(work):
        return
    pn = json.load(open(f"{work}/history.json"))[-1]["pulse"]   # same pulse as the final traces
    cfg = config(work, case, pn)
    lam_t, mu_t, rho = true_model(cfg.grid, case)
    lam0, mu0, _ = homogeneous(cfg.grid)
    active = {"mu": ("b",), "lambda": ("a",)}.get(name.split("_")[0], ("a", "b"))
    runner = Runner(cfg)
    runner.mesh_dir = f"{work}/{pn}/mesh"
    pb = Problem(cfg, runner, None, lam0, mu0, rho, active=active, fixed=(lam_t, mu_t), pml="follow")
    d = f"{work}/initial_fwd/s0/fwd"
    runner.prepare(d)
    lam, mu, _ = pb.to_model(np.zeros(pb.size))
    pb._write_model(d, lam, mu, rho)
    sem_io.write_input(d, cfg, sem_io.materialize_sources(d, cfg, cfg.sources))
    runner.run(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/summary.json")
    ap.add_argument("--initial", action="store_true", help="compute the traces for the initial model")
    args = ap.parse_args()
    if args.initial:
        for case, names in NAMES.items():
            for name in names:
                initial_forward(name, case)
    out = dict(pulses=pulses(), case1=case1(), case2={}, case3={})
    for case, names in NAMES.items():
        for name in names:
            e = experiment(name, case)
            if e:
                out[f"case{case}"][name] = e
    json.dump(out, open(args.out, "w"))
    print(args.out, os.path.getsize(args.out) // 1024, "KB")


if __name__ == "__main__":
    main()
