"""Summary of the synthetic inversion: decrease of J, errors to the true model, recovered anomalies,
horizontal and vertical slices through the centre of each anomaly (PNG figure).

    python analyse_synth.py [--param vel|lame] [--workdir ...] [--it N]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from semfwi.problem import Problem
from semfwi.runner import Runner
from run_synth import default_workdir, make_config, model_errors, true_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", choices=("lame", "vel"), default="vel")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--it", type=int, default=None)
    args = ap.parse_args()
    cfg = make_config(args.workdir or default_workdir(args.param))
    args.workdir = cfg.workdir
    lam_t, mu_t, rho_t = true_model(cfg)
    one = np.ones(cfg.grid.shape)
    pb = Problem(cfg, Runner(cfg), None, cfg.lam0 * one, cfg.mu0 * one, cfg.rho0 * one, param=args.param)
    x_true = pb.to_x(lam_t, mu_t)
    n = pb.n

    hist = json.load(open(os.path.join(args.workdir, "history.json")))
    J0 = hist[0]["J"]
    print(" it   J/J0       err_lambda  err_mu  err_vp  err_vs  trials  simulations")
    for h in hist:
        e = model_errors(pb, np.load(os.path.join(args.workdir, f"x{h['it']:03d}.npy")), lam_t, mu_t)
        print(f"{h['it']:3d}  {h['J'] / J0:.3e}  {e['err_lambda']:.3f}       {e['err_mu']:.3f}   "
              f"{e['err_vp']:.3f}   {e['err_vs']:.3f}   {h['trials']}       {h['nsim']}")

    files = sorted(glob.glob(os.path.join(args.workdir, "x[0-9][0-9][0-9].npy")))
    it = args.it if args.it is not None else int(os.path.basename(files[-1])[1:4])
    x = np.load(os.path.join(args.workdir, f"x{it:03d}.npy"))
    dl, dm = pb.grid_field(x[:n]), pb.grid_field(x[n:])
    tl, tm = pb.grid_field(x_true[:n]), pb.grid_field(x_true[n:])
    X, Y, Z = cfg.grid.mesh()
    pnames = ("lambda", "mu") if args.param == "lame" else ("Vp", "Vs")
    print(f"\niteration {it}")
    for name, rec, true in (zip(pnames, (dl, dm), (tl, tm))):
        it_min = np.unravel_index(np.argmin(true), true.shape)
        ir_min = np.unravel_index(np.argmin(rec), rec.shape)
        corr = np.corrcoef(rec[pb.mask], true[pb.mask])[0, 1]
        print(f"  {name:6s} true: min {true.min():+.3f} at ({X[it_min]:.0f}, {Y[it_min]:.0f}, {Z[it_min]:.0f})"
              f"; recovered: min {rec.min():+.3f} at ({X[ir_min]:.0f}, {Y[ir_min]:.0f}, {Z[ir_min]:.0f}),"
              f" max {rec.max():+.3f}; correlation {corr:.3f}")

    # Slices through the centre of each true anomaly
    fig, axes = plt.subplots(4, 2, figsize=(9, 15), constrained_layout=True)
    ext_xy = [cfg.grid.xmin[0], cfg.grid.xmax[0], cfg.grid.xmin[1], cfg.grid.xmax[1]]
    ext_xz = [cfg.grid.xmin[0], cfg.grid.xmax[0], cfg.grid.xmin[2], cfg.grid.xmax[2]]
    row = 0
    for name, rec, true in (zip(pnames, (dl, dm), (tl, tm))):
        c = np.unravel_index(np.argmin(true), true.shape)
        vmax = np.abs(true).max()
        for col, (fld, title) in enumerate(((true, "true"), (rec, f"iteration {it}"))):
            ax = axes[row, col]
            im = ax.imshow(fld[:, :, c[2]].T, origin="lower", extent=ext_xy, cmap="RdBu", vmin=-vmax, vmax=vmax)
            ax.set_title(f"d{name}/{name}0, {title}, z = {Z[c]:.0f} m")
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            ax = axes[row + 1, col]
            ax.imshow(fld[:, c[1], :].T, origin="lower", extent=ext_xz, cmap="RdBu", vmin=-vmax, vmax=vmax)
            ax.set_title(f"d{name}/{name}0, {title}, y = {Y[c]:.0f} m")
            ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        fig.colorbar(im, ax=axes[row:row + 2, :], shrink=0.6)
        row += 2
    out = os.path.join(args.workdir, f"coupes_it{it:03d}.png")
    fig.savefig(out, dpi=110)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
