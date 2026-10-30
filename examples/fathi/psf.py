"""Point-spread functions of Fathi's acquisition (semfwi.hessian) at the true model: what the load and
the 3-component surface receivers resolve, and how much of a lambda perturbation leaks into mu and
the other way round. One Hessian-vector product (one gradient) per bump, plus the gradient at the
true model.

    python psf.py --case 3 [--pulse p20] [--depths 3 8 15 25] [--width 1]

Results in $SEMFWI_RUNS/fathi/case<n>/psf_<pulse>/psf.npz (v and H v per bump) and printed:
  same  : ||H v|| in the block of the bump (sensitivity there),
  cross : ||H v|| in the other block / same (lambda-mu crosstalk),
  sym   : |v_i . H v_j - v_j . H v_i| / |v_i . H v_j| between the lambda and mu bumps at the same
          depth (finite-difference check, H is symmetric),
  zhalf : depth extent where |H v| in the bump's block exceeds half its maximum, on the vertical
          through the bump (vertical resolution).
"""
import argparse

import numpy as np

from run_fathi2 import config, fathi_case, observed
from semfwi.hessian import Hessian, bump
from semfwi.problem import Problem
from semfwi.runner import Runner


def run(args, case):
    work = f"{case.root}/psf_{args.pulse}"
    cfg = config(work, case, args.pulse)
    lam_t, mu_t, rho = case.true_model(cfg.grid)
    pb = Problem(cfg, Runner(cfg), observed(args.pulse, case), lam_t, mu_t, rho, active=("a", "b"),
                 fixed=(lam_t, mu_t), pml="follow")
    x = np.zeros(pb.size)  # relative to the true model
    H = Hessian(pb, x, "psf", dx=args.dx)
    print(f"[{args.pulse}] |g(x_true)| = {np.linalg.norm(H.gradient()):.3e}", flush=True)
    names = {"a": "lambda", "b": "mu"}
    grid = cfg.grid
    z = grid.axis(2)
    i0, j0 = (int(np.argmin(np.abs(grid.axis(c)))) for c in range(2))
    out = {}
    for depth in args.depths:
        for blk in ("a", "b"):
            v = bump(pb, blk, (0., 0., -depth), args.width)
            hv = H(v)
            out[f"{names[blk]}_{depth:g}_v"], out[f"{names[blk]}_{depth:g}_hv"] = v, hv
            same, other = pb.split(hv) if blk == "a" else pb.split(hv)[::-1]
            col = np.abs(pb.grid_field(same)[i0, j0])
            zs = z[col >= 0.5 * col.max()]
            print(f"  {names[blk]:6s} z={-depth:6.1f}  same={np.linalg.norm(same):.3e}  "
                  f"cross={np.linalg.norm(other) / np.linalg.norm(same):.2f}  "
                  f"zhalf=[{zs.min():.1f}, {zs.max():.1f}]", flush=True)
            np.savez(f"{work}/psf.npz", **out)
        va, ha = out[f"lambda_{depth:g}_v"], out[f"lambda_{depth:g}_hv"]
        vb, hb = out[f"mu_{depth:g}_v"], out[f"mu_{depth:g}_hv"]
        print(f"  sym z={-depth:.1f}: {abs(va @ hb - vb @ ha) / abs(va @ hb):.2e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, choices=(2, 3, 4), default=3)
    ap.add_argument("--pulse", default="p20")
    ap.add_argument("--depths", type=float, nargs="+", default=[3., 8., 15., 25.])
    ap.add_argument("--width", type=float, default=1., help="standard deviation of the bumps (m)")
    ap.add_argument("--dx", type=float, default=1e-3, help="largest relative model change per product")
    args = ap.parse_args()
    run(args, fathi_case(args.case))
