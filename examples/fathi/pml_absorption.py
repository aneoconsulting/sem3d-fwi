"""Absorption of the PMLs in the geometry of cases 2 to 4: the data of case 3 (true layered model,
load and stations of the paper, p20, T = 0.45 s) simulated in several domains, all with the
same elements (1.25 m, ngll = 3), the same time step and the same stations:

    prod   : 40 x 40 x 45 m, PML of 5 elements (6.25 m), the inversion setting;
    pml10  : same domain, PML of 10 elements (12.5 m);
    big    : 60 x 60 x 65 m (10 m more on each side, 20 m deeper), PML of 5 elements.

Boundary reflections are the only difference between these simulations at the stations (the model
is layered, so it is the same in the common part). The script writes, in RUNS_DIR/fathi/pml_test,
the traces of each variant and pml_test.json: relative L2 differences between variants and their
misfit 1/2 ||u_a - u_b||^2 compared with J0, the misfit of the homogeneous initial model of case 3.

    python pml_absorption.py [prod pml10 big]
"""
import json
import os
import sys

import numpy as np

from fathi import make_config, true_model
from semfwi import misfit, model, sem_io
from semfwi.config import RUNS_DIR, Grid
from semfwi.runner import Runner

VARIANTS = {"prod": (20., 45., 5), "pml10": (20., 45., 10), "big": (30., 65., 5)}  # half width, depth, PML elements
J0 = 3.713222421399098e-08  # case 3, homogeneous 80 MPa model, p20 (iteration 0 of the inversions)


def config(workdir, half, depth, npml):
    cfg = make_config(workdir, 3, "p20", sim_time=0.45)
    gh, h = cfg.grid.h, cfg.h_elem
    n, nz = int(round(2 * half / gh)) + 1, int(round(depth / gh)) + 1
    cfg.grid = Grid(xmin=(-half, -half, -depth), h=gh, n=(n, n, nz))
    cfg.L, cfg.x0, cfg.depth, cfg.npml = 2 * half, -half, depth, npml
    cfg.layers = [(h, 1)] * int(round(depth / h))
    return cfg


def simulate(name):
    cfg = config(os.path.join(RUNS_DIR, "fathi", "pml_test", name), *VARIANTS[name])
    runner = Runner(cfg)
    d = f"{cfg.workdir}/sim"
    runner.prepare(d)
    lam, mu, rho = true_model(cfg.grid, 3)
    model.write_model(d, cfg.grid, lam, mu, rho, nmat=len(cfg.layer_list))
    model.write_pml_materials(d, cfg, lam, mu, rho)
    sem_io.write_input(d, cfg, sem_io.materialize_sources(d, cfg, cfg.sources))
    runner.run(d)
    u = sem_io.read_traces(d, len(cfg.stations))
    np.save(f"{cfg.workdir}/traces.npy", u)
    return cfg, u


def main():
    names = sys.argv[1:] or list(VARIANTS)
    out, dt = {}, None
    for n in names:
        cfg, out[n] = simulate(n)
        dt = cfg.dt
        print(f"{n}: {out[n].shape}", flush=True)
    res = {}
    for a in names:
        for b in names:
            if a < b:
                d = out[a] - out[b]
                res[f"{a}-{b}"] = dict(rel_l2=float(np.linalg.norm(d) / np.linalg.norm(out[b])),
                                        J_over_J0=float(misfit.cost(out[a][None], out[b][None], dt) / J0))
                print(a, b, res[f"{a}-{b}"], flush=True)
    with open(os.path.join(RUNS_DIR, "fathi", "pml_test", "pml_test.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
