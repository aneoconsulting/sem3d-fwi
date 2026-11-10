"""3-D case of the Mygdonian basin (E2VP2-MODEL archive, e2vp2.py), a 6 x 6 km box around the epicentre:

    python run_e2vp3d.py prepare                  # crop the archive's grids into $SEMFWI_RUNS/e2vp3d/basin.npz
    python run_e2vp3d.py forward [--fmax 0.5] [--flat]   # true model, the archive's earthquake
    python run_e2vp3d.py forward --rock           # reference: the sediments replaced by rock
    python run_e2vp3d.py site [--fmax 0.5]        # basin response from forward, against the 1-D column
    python run_e2vp3d.py gradcheck [--fmax 0.5]   # adjoint gradient against finite differences, (K, G)
    python run_e2vp3d.py invert [--iters 10] [--nsources 3]   # the first 500 m, from a 1-D start

forward: one simulation, traces and timing saved in forward_f<fmax>[_t<time>][_flat][_rock].npz. site: ratio of the
mean horizontal amplitude spectra of the basin to those of the rock reference at each station, and the
vertically incident SH transfer function of the epicentre column (the gradient layers in 2 m slices).
gradcheck and invert: the first 500 m on the parameter grid, data of the true model on the same grid
(inverse crime); the start is a 1-D profile that does not use the true model (see start_model).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2vp  # noqa: E402
import e2vp2  # noqa: E402
from semfwi import inversion, problem  # noqa: E402
from semfwi.config import RUNS_DIR  # noqa: E402
from semfwi.runner import Runner  # noqa: E402

ROOT = os.path.join(RUNS_DIR, "e2vp3d")
NPZ = os.path.join(ROOT, "basin.npz")


def true_model(cfg, basin, rock=False):
    """lambda, mu, rho of the archive's model on the parameter grid; rock: the sediments replaced by the
    first crustal layer (reference without the basin, same geometry)."""
    vp, vs, rho = basin.properties(*cfg.grid.mesh())
    if rock:
        sed, r = vs < 2000., basin.layers[2]
        vp, vs, rho = (np.where(sed, r[k][0], a) for a, k in ((vp, "vp"), (vs, "vs"), (rho, "rho")))
    return e2vp.lame(vp, vs, rho)


def start_model(cfg, basin):
    """1-D start: in the first 500 m below the meshed surface, Vs from 250 m/s at the surface to 1200 m/s
    at 500 m, Vp = 1600 m/s + 1.5 Vs, the density of the true model (not inverted); a generic sediment
    profile, the same everywhere (neither the basin's thickness nor its gradients)."""
    X, Y, Z = cfg.grid.mesh()
    depth = np.clip(basin.mesh_surface(X, Y) - Z, 0., 500.)
    vs = 250. + (1200. - 250.) * depth / 500.
    return e2vp.lame(1600. + 1.5 * vs, vs, basin.properties(X, Y, Z)[2])


def base_tag(args):
    return f"f{args.fmax:g}" + (f"_t{args.time:g}" if args.time else "")


def forward(args):
    tag = base_tag(args) + ("_flat" if args.flat else "") + ("_rock" if args.rock else "")
    cfg, basin = e2vp2.config_3d(f"{ROOT}/forward_{tag}", NPZ, fmax=args.fmax, sim_time=args.time, relief=not args.flat)
    lam, mu, rho = true_model(cfg, basin, rock=args.rock)
    t = time.time()
    u = problem.observed(cfg, Runner(cfg), cfg.workdir, lam, mu, rho, pml="follow")[0]
    el = time.time() - t
    print(f"{tag}: {el:.0f} s, dt {cfg.dt:.3e}, {cfg.nt} steps, max |u| {np.abs(u).max():.3e}, finite {np.isfinite(u).all()}")
    np.savez(f"{ROOT}/forward_{tag}.npz", u=u, stations=np.array(cfg.stations), dt=cfg.dt, elapsed=el)


def column(basin, x=0., y=0., slab=2.):
    """Layers (th, vp, vs, rho) of the column at (x, y) down to the bedrock, in slab m slices, and the
    bedrock (vp, vs, rho)."""
    s = basin.mesh_surface(x, y)
    d = np.arange(slab / 2, 1000., slab)
    vp, vs, rho = basin.properties(np.full_like(d, x), np.full_like(d, y), s - d)
    sed = vs < 2000.
    n = np.argmin(sed)
    return [(slab, vp[i], vs[i], rho[i]) for i in range(n)], (vp[n], vs[n], rho[n])


def site(args):
    """Ratio of the mean horizontal amplitude spectra, basin over rock reference (forward --rock: same
    source, path and surface), at each surface station; at the epicentre, against the vertically incident
    SH transfer function of its column. Spectra zero-padded 4 times (interpolated, not resolved), up to
    0.8 fmax: above, the rock reference falls below a tenth of its maximum and the ratio is noise."""
    out = {}
    for flat in (False, True):
        tag = base_tag(args) + ("_flat" if flat else "")
        if not all(os.path.exists(f"{ROOT}/forward_{t}.npz") for t in (tag, tag + "_rock")):
            continue
        r, r0 = np.load(f"{ROOT}/forward_{tag}.npz"), np.load(f"{ROOT}/forward_{tag}_rock.npz")
        st, dt = r["stations"], float(r["dt"])
        n = 4 * r["u"].shape[1]
        spec, spec0 = (np.abs(np.fft.rfft(a["u"][:, :, :2], n=n, axis=1)).mean(axis=2) for a in (r, r0))
        fr = np.fft.rfftfreq(n, dt)
        band = (fr > 0.1) & (fr <= 0.8 * args.fmax)
        ratio = spec[:, band] / spec0[:, band]
        surface = np.arange(len(st)) < len(st) - 3
        epi = int(np.argmin(np.hypot(st[:, 0], st[:, 1]) + 1e9 * ~surface))
        basin = e2vp2.Basin(NPZ, relief=not flat)
        sed = basin._bilinear(basin.surface, st[:, 0], st[:, 1]) - basin._bilinear(basin.bedrock, st[:, 0], st[:, 1])
        out[tag] = dict(f=fr[band].tolist(), epicentre=ratio[epi].tolist(), stations=st.tolist(),
                        sediments=sed.tolist(), peak=ratio.max(axis=1).tolist(),
                        f_peak=fr[band][np.argmax(ratio, axis=1)].tolist(), resolution=float(1 / (r["u"].shape[1] * dt)))
        on, off = surface & (sed > 50.), surface & (sed < 1.)
        print(f"{tag}: epicentre ratio peak {ratio[epi].max():.2f} at {fr[band][np.argmax(ratio[epi])]:.3f} Hz "
              f"(resolution {out[tag]['resolution']:.3f} Hz); mean peak over {on.sum()} stations on > 50 m of "
              f"sediments {ratio[on].max(axis=1).mean():.2f}, over {off.sum()} on rock {ratio[off].max(axis=1).mean():.2f}")
    layers, bed = column(e2vp2.Basin(NPZ))
    fr = np.linspace(0.05, max(1., args.fmax), 400)
    th = e2vp.sh_transfer(fr, layers, bed)
    print(f"1-D column at the epicentre: {sum(l[0] for l in layers):.0f} m of sediments, SH fundamental "
          f"{fr[np.argmax(th)]:.3f} Hz, peak {th.max():.2f}; within {args.fmax:g} Hz: max {th[fr <= args.fmax].max():.2f}")
    out["theory"] = dict(f=fr.tolist(), ratio=th.tolist())
    json.dump(out, open(f"{ROOT}/site_{base_tag(args)}.json", "w"))


def make_problem(args, work):
    cfg, basin = e2vp2.config_3d(work, NPZ, fmax=args.fmax, nsources=args.nsources)
    data, _ = e2vp2.config_3d(f"{ROOT}/data_f{args.fmax:g}_s{args.nsources}", NPZ, fmax=args.fmax, nsources=args.nsources)
    lam_t, mu_t, rho = true_model(cfg, basin)
    dobs = problem.observed(data, Runner(data), data.workdir, lam_t, mu_t, rho, pml="follow")
    lam0, mu0, _ = start_model(cfg, basin)
    pb = problem.Problem(cfg, Runner(cfg), dobs, lam0, mu0, rho, param=args.param, pml="follow")
    return pb, (lam_t, mu_t), basin


def gradcheck(args):
    """Directional derivative along a smooth bump of K (block a) then G (block b), 60 m below the surface
    at (500, -500) m: g . dx against (J(x + h dx) - J(x)) / h."""
    pb, _, basin = make_problem(args, f"{ROOT}/gradcheck_f{args.fmax:g}")
    x = np.zeros(pb.size)
    J0, _, g = pb.evaluate(x, "x0")
    X, Y, Z = (c[pb.mask] for c in pb.cfg.grid.mesh())
    depth = basin.mesh_surface(X, Y) - Z
    bump = np.exp(-((X - 500.) ** 2 + (Y + 500.) ** 2) / (2 * 600. ** 2) - (depth - 60.) ** 2 / (2 * 40. ** 2))
    rows = []
    for i, blk in enumerate(pb.active):
        dx = np.zeros(pb.size)
        dx[i * pb.n:(i + 1) * pb.n] = bump
        dco = g @ dx
        for h in args.h:
            Jh = pb.evaluate(x + h * dx, f"{blk}_{h:g}", grad=False)[0]
            dfdh = (Jh - J0) / h
            rows.append(dict(block=blk, h=h, dco=float(dco), dfdh=float(dfdh), rel=float(abs(dfdh - dco) / abs(dco))))
            print(f"{blk} h={h:g}: adjoint {dco:.6e}  finite difference {dfdh:.6e}  relative gap {rows[-1]['rel']:.2e}", flush=True)
    json.dump(dict(J0=float(J0), rows=rows), open(f"{ROOT}/gradcheck_f{args.fmax:g}.json", "w"), indent=1)


def invert(args):
    work = f"{ROOT}/invert_f{args.fmax:g}_s{args.nsources}_{args.param}"
    truth = {}

    def mk(si, pn):
        pb, t, basin = make_problem(args, f"{work}/{pn}")
        X, Y, Z = pb.cfg.grid.mesh()
        sed = (basin.properties(X, Y, Z)[1] < 2000.)[pb.mask]  # nodes in the sediments of the true model
        truth.update(lam_t=t[0], mu_t=t[1], sed=sed)
        return pb

    def errors(pb, x):  # in lambda, mu and K on all inverted nodes, and in Vs on the sediments only
        e = problem.model_errors(pb, x, truth["lam_t"], truth["mu_t"])
        lam, mu, rho = pb.to_model(x)
        m, s = pb.mask, truth["sed"]
        vs, vs_t, vs_0 = (np.sqrt(a[m] / rho[m])[s] for a in (mu, truth["mu_t"], pb.mu_ref))
        nu = problem.poisson(lam, mu)[m]
        return dict(err_lambda=e["err_lambda"], err_mu=e["err_mu"], err_k=e["err_k"],
                    err_vs_sed=float(np.linalg.norm(vs - vs_t) / np.linalg.norm(vs_0 - vs_t)),
                    nu_min=float(nu.min()), nu_max=float(nu.max()))

    inversion.invert(work, [(f"f{args.fmax:g}", args.iters)], mk, reg="tv", callback=errors,
                     log=lambda s: print(s, flush=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("prepare", "forward", "site", "gradcheck", "invert"))
    ap.add_argument("--fmax", type=float, default=0.5)
    ap.add_argument("--flat", action="store_true", help="forward: flat free surface")
    ap.add_argument("--rock", action="store_true", help="forward: sediments replaced by rock")
    ap.add_argument("--time", type=float, default=None, help="forward, site: simulated time (s), default 3 / fmax + 6")
    ap.add_argument("--param", choices=("lame", "kg"), default="kg")
    ap.add_argument("--h", type=float, nargs="+", default=[1e-2, 1e-3])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--nsources", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    if args.what == "prepare":
        e2vp2.prepare(NPZ)
    else:
        {"forward": forward, "site": site, "gradcheck": gradcheck, "invert": invert}[args.what](args)


if __name__ == "__main__":
    main()
