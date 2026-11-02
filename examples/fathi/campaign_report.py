"""Summary of a campaign of inversions (one file per node, examples/fathi/jeanzay/campaigns/*.txt), finished
or not: what each run has reached, and figures to compare them.

    python campaign_report.py jeanzay/campaigns/*.txt [--out <SEMFWI_RUNS>/report]

For each campaign file: a table (stdout and <out>/<name>.txt) with, per run, the last iteration and
its frequency, J / J at the start, the errors on lambda and mu (relative to the 80 MPa reference,
1 = no better than the reference) and the number of simulations; and <out>/<name>.png: errors and J
along the iterations, errors against the number of simulations (the cost, which differs when a run
has several shots), and the last models: mean over |x|, |y| <= 10 m against depth, true model in black.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from run_fathi2 import config, fathi_case, parser, with_options  # noqa: E402
from semfwi import model  # noqa: E402
from semfwi.config import RUNS_DIR  # noqa: E402


def runs_of(fn):
    """(tag, case with the options of the line, args) for each line of a campaign file."""
    ap = parser()
    ap.add_argument("--case", type=int, default=2)
    out = []
    for line in open(fn):
        if line.strip() and not line.startswith("#"):
            tag, *argv = line.split()
            args = ap.parse_args(["both", *argv, "--tag", tag])
            out.append((tag, with_options(fathi_case(args.case), args), args))
    return out


def last_model(case, work, it):
    """lambda, mu of iteration it on the grid, from x (reference 80 MPa, both parameters inverted)."""
    cfg = config(work, case, "p20")
    lam, mu, _ = case.true_model(cfg.grid)
    ref, _, _ = case.initial(cfg.grid)
    mask = model.interior_mask(cfg.grid, cfg.L, cfg.depth, cfg.x0)
    x = np.load(f"{work}/x{it:03d}.npy")
    n = int(mask.sum())
    lam, mu = lam.copy(), mu.copy()
    lam[mask] = ref[mask] * (1 + x[:n])
    mu[mask] = ref[mask] * (1 + x[n:])
    return cfg.grid, lam, mu


def profile(grid, f):
    X, Y, Z = grid.mesh()
    c = (np.abs(X[:, :, 0]) <= 10) & (np.abs(Y[:, :, 0]) <= 10)
    return Z[0, 0, :], np.array([f[:, :, k][c].mean() for k in range(f.shape[2])]) / 1e6


def report(fn, out):
    name = os.path.splitext(os.path.basename(fn))[0]
    rows, curves = [], []
    for tag, case, _ in runs_of(fn):
        work = f"{case.root}/{tag}"
        if not os.path.exists(f"{work}/history.json"):
            rows.append(f"{tag:16s} not started")
            continue
        h = json.load(open(f"{work}/history.json"))
        it = np.array([e["it"] for e in h])
        J = np.array([e["J"] for e in h])
        el = np.array([e["err_lambda"] for e in h])
        em = np.array([e["err_mu"] for e in h])
        ns = np.array([e["nsim"] for e in h])
        J0 = {}
        for e in h:  # J at the start of each frequency (J is not comparable across frequencies)
            J0.setdefault(e["stage"], e["J"])
        e = h[-1]
        rows.append(f"{tag:16s} {e['it']:5d} {e['pulse']:>5s} {e['J'] / J0[e['stage']]:9.3f} "
                    f"{el[0]:7.3f} {el[-1]:7.3f} {em[0]:7.3f} {em[-1]:7.3f} {ns[-1]:6d} "
                    f"{np.median([x['time'] for x in h[-20:]]):6.0f}")
        curves.append((tag, case, work, it, J, el, em, ns, e["it"]))
    head = (f"{'run':16s} {'it':>5s} {'freq':>5s} {'J/J_stage':>9s} {'eλ 0':>7s} {'eλ':>7s} {'eμ 0':>7s} "
            f"{'eμ':>7s} {'sims':>6s} {'s/it':>6s}")
    text = "\n".join([f"== {name}", head] + rows)
    print(text + "\n")
    open(f"{out}/{name}.txt", "w").write(text + "\n")
    if not curves:
        return
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    for tag, case, work, it, J, el, em, ns, last in curves:
        l, = ax[0, 0].plot(it, em, lw=1, label=tag)
        ax[0, 1].plot(it, el, lw=1, color=l.get_color())
        ax[0, 2].semilogy(it, J, lw=1, color=l.get_color())
        ax[1, 0].plot(ns, em, lw=1, color=l.get_color())
        grid, lam, mu = last_model(case, work, last)
        z, p = profile(grid, mu)
        ax[1, 1].plot(p, z, lw=1, color=l.get_color())
        z, p = profile(grid, lam)
        ax[1, 2].plot(p, z, lw=1, color=l.get_color())
    lt, mt, _ = curves[0][1].true_model(grid)
    for a, f in ((ax[1, 1], mt), (ax[1, 2], lt)):
        z, p = profile(grid, f)
        a.plot(p, z, "k", lw=2, label="true")
        a.set_xlabel("MPa (mean over |x|, |y| <= 10 m)")
        a.set_ylabel("z (m)")
    titles = ("error on mu", "error on lambda", "J (misfit, per frequency)", "error on mu vs simulations",
              "mu, last iteration", "lambda, last iteration")
    for a, t in zip(ax.flat, titles):
        a.set_title(t)
        a.grid(alpha=0.3)
    for a in (ax[0, 0], ax[0, 1], ax[0, 2]):
        a.set_xlabel("iteration")
    ax[1, 0].set_xlabel("simulations (forward and adjoint, all shots)")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(f"{out}/{name}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=os.path.join(RUNS_DIR, "report"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for fn in args.files:
        report(fn, args.out)
    print(f"figures and tables in {args.out}")


if __name__ == "__main__":
    main()
