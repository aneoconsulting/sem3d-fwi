"""Test case 1 of Fathi et al. (2015, section 4.1): gradient check, equivalent of their table 2.

Data: true profile on a fine mesh (0.5 m elements), brought back to the coarse time grid.
Gradient evaluated at the initial model of the inversions (homogeneous, 80 MPa), PML fixed to that model.
Directional derivatives: perturbation of one node of the parameter grid (x relative to the
reference model), compared with forward finite differences (eq. 26 of the paper) for
h = 1e-3, 1e-4, 1e-5.

    python run_fathi1.py [--pulses p20 p40]
"""
import argparse
import json
import os
import time

import numpy as np

from fathi import homogeneous, make_config, true_model
from semfwi import model, sem_io
from semfwi.config import RUNS_DIR
from semfwi.problem import Problem
from semfwi.runner import Runner

ROOT = os.path.join(RUNS_DIR, "fathi", "case1")
CASES = {"p20": [((1., 1., 0.), "a"), ((1., 1., 0.), "b"), ((1., 1., -40.), "a"), ((1., 1., -40.), "b")],
         "p40": [((1., 1., 0.), "a"), ((1., 1., 0.), "b")]}


def observed(pulse_name):
    """Traces of the true profile on the fine mesh, sampled at the coarse times (j + 1) dt."""
    cfg = make_config(f"{ROOT}/data_{pulse_name}", 1, pulse_name, fine=True)
    runner = Runner(cfg)
    lam, mu, rho = true_model(cfg.grid)
    d = f"{cfg.workdir}/sim"
    runner.prepare(d)
    model.write_model(d, cfg.grid, lam, mu, rho, nmat=len(cfg.layer_list))
    model.write_pml_materials(d, cfg, lam, mu, rho)
    sem_io.write_input(d, cfg, sem_io.materialize_sources(d, cfg, cfg.sources))
    runner.run(d)
    u = sem_io.read_traces(d, len(cfg.stations))
    return u[:, 1::2, :][None]   # fine step k = 2 j + 1  <->  coarse step j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulses", nargs="+", default=["p20", "p40"])
    args = ap.parse_args()
    results = []
    out_fn = f"{ROOT}/table2.json"
    if os.path.exists(out_fn):
        results = json.load(open(out_fn))
    for pn in args.pulses:
        t0 = time.time()
        dobs = observed(pn)
        print(f"{pn}: fine data {time.time() - t0:.0f} s, max |d| = {np.abs(dobs).max():.3e}", flush=True)
        cfg = make_config(f"{ROOT}/{pn}", 1, pn)
        runner = Runner(cfg)
        assert dobs.shape[2] == cfg.nt
        lam0, mu0, rho = homogeneous(cfg.grid)
        pb = Problem(cfg, runner, dobs, lam0, mu0, rho, pml=(lam0, mu0, rho))
        x0 = np.zeros(pb.size)
        t0 = time.time()
        J0, _, g = pb.evaluate(x0, "x0")
        print(f"{pn}: J0 = {J0:.6e}, gradient in {time.time() - t0:.0f} s", flush=True)
        np.save(f"{cfg.workdir}/g0.npy", g)
        X, Y, Z = cfg.grid.mesh()
        for node, par in CASES[pn]:
            idx = np.nonzero((np.abs(X - node[0]) < 1e-9) & (np.abs(Y - node[1]) < 1e-9)
                             & (np.abs(Z - node[2]) < 1e-9))
            k = int(np.ravel_multi_index(idx, cfg.grid.shape)[0])
            pos = int(np.nonzero(np.flatnonzero(pb.mask.ravel()) == k)[0][0])
            j = pos + (0 if par == "a" else pb.n)
            row = dict(pulse=pn, node=node, param="lambda" if par == "a" else "mu", dco=float(g[j]), dfdh={})
            for h in (1e-3, 1e-4, 1e-5):
                x = x0.copy()
                x[j] = h
                Jh, _, _ = pb.evaluate(x, f"fd_{par}_{node[2]:g}_{h:g}", grad=False)
                row["dfdh"][f"{h:g}"] = (Jh - J0) / h
            print(f"  {pn} {node} {row['param']:6s} dco = {row['dco']:+.6e}; dfdh: "
                  + "  ".join(f"h={h}: {v:+.6e}" for h, v in row["dfdh"].items()), flush=True)
            results = [r for r in results if not (r["pulse"] == pn and tuple(r["node"]) == node
                                                  and r["param"] == row["param"])] + [row]
            json.dump(results, open(out_fn, "w"), indent=1)


if __name__ == "__main__":
    main()
