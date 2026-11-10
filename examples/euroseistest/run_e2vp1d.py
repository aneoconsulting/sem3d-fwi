"""1-D case of E2VP (Chaljub et al. 2015, 1D-sharp), reduced to a 4 x 4 x 4 km box (e2vp.config_1d):

    python run_e2vp1d.py site       # site response: sediments / outcrop, against Thomson-Haskell
    python run_e2vp1d.py gradcheck  # adjoint gradient against finite differences, (K, G)
    python run_e2vp1d.py invert [--iters 20] [--param kg] [--taper R]   # the three sedimentary layers

site: two forward simulations with the double couple only and the exact layered model (constant
properties per mesh layer), one with the sediments and one with the bedrock up to the surface; the ratio
of their mean horizontal amplitude spectra over the stations is the site response, compared with the
vertically incident SH transfer function of the layers (fundamental frequency about 0.67 Hz in the
paper). gradcheck and invert: the sediments on the parameter grid, data of the true model computed on
the same grid (inverse crime), double couple and surface force, start from homogeneous sediments
(the properties of L2). Runs in $SEMFWI_RUNS/e2vp1d.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2vp  # noqa: E402
from semfwi import inversion, problem  # noqa: E402
from semfwi.config import RUNS_DIR  # noqa: E402
from semfwi.runner import Runner  # noqa: E402

ROOT = os.path.join(RUNS_DIR, "e2vp1d")


def site(args):
    out = {}
    for name, outcrop in (("sediments", False), ("outcrop", True)):
        cfg = e2vp.config_1d(f"{ROOT}/site_{name}", fmax=args.fmax, sim_time=args.sim_time, exact=True,
                             outcrop=outcrop, sources=("dc",))
        one = np.ones(cfg.grid.shape)
        u = problem.observed(cfg, Runner(cfg), cfg.workdir, cfg.lam0 * one, cfg.mu0 * one, cfg.rho0 * one,
                             pml="follow")[0]
        out[name] = np.abs(np.fft.rfft(u[:, :, :2], axis=1))  # (nsta, nf, 2): horizontal components
    fr = np.fft.rfftfreq(u.shape[1], cfg.dt)
    ratio = out["sediments"].mean(axis=(0, 2)) / out["outcrop"].mean(axis=(0, 2))
    th = e2vp.sh_transfer(fr)
    band = (fr > 0.2) & (fr <= args.fmax)
    f0_sim, f0_th = fr[band][np.argmax(ratio[band])], fr[band][np.argmax(th[band])]
    print(f"fundamental frequency: simulation {f0_sim:.3f} Hz, Thomson-Haskell {f0_th:.3f} Hz "
          f"(resolution {fr[1]:.3f} Hz); peak ratio {ratio[band].max():.2f}, theory {th[band].max():.2f}")
    for f in np.arange(0.25, args.fmax + 1e-9, 0.125):
        i = np.argmin(np.abs(fr - f))
        print(f"  {fr[i]:.3f} Hz  simulation {ratio[i]:6.2f}  theory {th[i]:6.2f}")
    json.dump(dict(f=fr[band].tolist(), ratio=ratio[band].tolist(), theory=th[band].tolist(),
                   f0_sim=float(f0_sim), f0_th=float(f0_th)), open(f"{ROOT}/site.json", "w"))


def make_problem(args, work):
    cfg = e2vp.config_1d(work, fmax=args.fmax, sim_time=args.sim_time)
    cfg.source_taper = args.taper
    data = e2vp.config_1d(f"{ROOT}/data_f{args.fmax:g}", fmax=args.fmax, sim_time=args.sim_time)
    lam_t, mu_t, rho = e2vp.true_model_1d(cfg.grid)
    dobs = problem.observed(data, Runner(data), data.workdir, lam_t, mu_t, rho, pml="follow")
    lam0, mu0, _ = e2vp.initial_1d(cfg.grid)
    pb = problem.Problem(cfg, Runner(cfg), dobs, lam0, mu0, rho, param=args.param, pml="follow")
    return pb, (lam_t, mu_t)


def gradcheck(args):
    """Directional derivative along a smooth bump of K (block a) then of G (block b), centred at
    (0, 0, -50 m) in L2: g . dx against (J(x + h dx) - J(x)) / h."""
    pb, _ = make_problem(args, f"{ROOT}/gradcheck")
    x = np.zeros(pb.size)
    J0, _, g = pb.evaluate(x, "x0")
    X, Y, Z = (c[pb.mask] for c in pb.cfg.grid.mesh())
    bump = np.exp(-(X ** 2 + Y ** 2) / (2 * 400. ** 2) - (Z + 50.) ** 2 / (2 * 15. ** 2))
    rows = []
    for i, blk in enumerate(pb.active):
        dx = np.zeros(pb.size)
        dx[i * pb.n:(i + 1) * pb.n] = bump
        dco = g @ dx
        for h in args.h:
            Jh = pb.evaluate(x + h * dx, f"{blk}_h{h:g}", grad=False)[0]
            dfdh = (Jh - J0) / h
            rows.append(dict(block=blk, h=h, dco=float(dco), dfdh=float(dfdh), rel=float(abs(dfdh - dco) / abs(dco))))
            print(f"{blk} h={h:g}: adjoint {dco:.6e}  finite difference {dfdh:.6e}  relative gap {rows[-1]['rel']:.2e}", flush=True)
    json.dump(dict(J0=float(J0), rows=rows), open(f"{ROOT}/gradcheck.json", "w"), indent=1)


def invert(args):
    work = f"{ROOT}/invert_{args.param}" + (f"_taper{args.taper:g}" if args.taper else "")
    truth = {}

    def mk(si, pn):
        pb, t = make_problem(args, f"{work}/{pn}")
        truth.update(lam_t=t[0], mu_t=t[1])
        return pb

    def errors(pb, x):  # in lambda, mu and K, and Poisson's ratio range on the inverted nodes
        e = problem.model_errors(pb, x, truth["lam_t"], truth["mu_t"])
        lam, mu, _ = pb.to_model(x)
        nu = problem.poisson(lam, mu)[pb.mask]
        return dict(err_lambda=e["err_lambda"], err_mu=e["err_mu"], err_k=e["err_k"],
                    nu_min=float(nu.min()), nu_max=float(nu.max()))

    inversion.invert(work, [(f"f{args.fmax:g}", args.iters)], mk, reg="tv", callback=errors,
                     log=lambda s: print(s, flush=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("site", "gradcheck", "invert"))
    ap.add_argument("--fmax", type=float, default=1.)
    ap.add_argument("--sim-time", type=float, default=None, help="s, default 16 (site) or 8; the source peaks at 3 / fmax s")
    ap.add_argument("--param", choices=("lame", "kg"), default="kg")
    ap.add_argument("--h", type=float, nargs="+", default=[1e-2, 1e-3])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--taper", type=float, default=0., help="m, radius of the gradient taper around the sources (invert)")
    args = ap.parse_args()
    args.sim_time = args.sim_time or (16. if args.what == "site" else 8.)
    os.makedirs(ROOT, exist_ok=True)
    {"site": site, "gradcheck": gradcheck, "invert": invert}[args.what](args)


if __name__ == "__main__":
    main()
