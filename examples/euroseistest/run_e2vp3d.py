"""3-D case of the Mygdonian basin (E2VP2-MODEL archive, e2vp2.py), a 6 x 6 km box around the epicentre:

    python run_e2vp3d.py prepare                  # crop the archive's grids into $SEMFWI_RUNS/e2vp3d/basin.npz
    python run_e2vp3d.py forward [--fmax 0.5] [--flat]   # true model, the archive's earthquake
    python run_e2vp3d.py forward --rock           # reference: the sediments replaced by rock
    python run_e2vp3d.py forward --pml copy --npml 4   # PMLs extending the model files, or thicker (_pmlcopy, _npml4)
    python run_e2vp3d.py site [--fmax 0.5]        # basin response from forward, against the 1-D column
    python run_e2vp3d.py bedrock --fmax 1 --time 24   # sediment thickness from H/V and the vertical array of that forward
    python run_e2vp3d.py gradcheck [--fmax 0.5]   # adjoint gradient against finite differences, (K, G)
    python run_e2vp3d.py invert [--iters 10] [--nsources 3] [--bedrock f1_t24] [--param vs|interface|interface-v0g|interface-pw]   # the first 500 m, from a 1-D start
    python run_e2vp3d.py psf --bedrock f1_t24     # point-spread functions of K and G at 100 and 300 m depth
    python run_e2vp3d.py psf --bedrock f1_t24 --at 500 -500   # ... under the station (500, -500)
    python run_e2vp3d.py bedmap --bedrock f1_t24 --from invert_f0.5_s3_interface_z4_bedrock-f1_t24/x010.npy --tag f1_t24_if05
    python run_e2vp3d.py bedmap --bedrock f1_t24 --mean --tag f1_t24_mean   # bedrock_<tag>.npz, a start for --bedrock <tag>
    python run_e2vp3d.py bedmap --bedrock f1_t24 --truth --fmax 0.75 --tag f1_t24_truth   # the best map of the true bedrock
    python run_e2vp3d.py jmap --fmax 0.75 --bedrock f1_t24_truth   # J of that map
    python run_e2vp3d.py bedmap --bedrock f1_t24 --truth --knots --fmax 0.75 --tag f1_t24_truthpw   # and the best knots
    python run_e2vp3d.py jswap --fmax 0.5 --swap vp   # J of the true model with one fixed part of --param interface

forward: one simulation, traces and timing saved in forward_f<fmax>[_t<time>][_flat][_rock].npz. site: ratio of the
mean horizontal amplitude spectra of the basin to those of the rock reference at each station, and the
vertically incident SH transfer function of the epicentre column (the gradient layers in 2 m slices).
gradcheck and invert: the first 500 m on the parameter grid, data of the true model on the same grid
(inverse crime); the start is a 1-D profile that does not use the true model, or with --bedrock sediments
down to the thickness map of bedrock, estimated from the data (see start_model and bedrock). The descent
directions vary linearly in z between every ZSTEP-th level of the grid (_z4 in the work directory).
--param: kg (K and G, default), vs (Vs alone, Vp / Vs of the start kept at each node, problem.Problem) or interface
(with --bedrock: the depth of the bedrock on the 11 x 11 map of the start, sediments and rock of the start above and
below it, problem.InterfaceProblem), or interface-v0g (the same, and v0 and g of the sediment profile above it), or
interface-pw (the same, and the Vs of the sediments at the depths ZK, linear in between: a profile richer than one line).
psf: Hessian-vector products at the start for Gaussian bumps of K and G (standard deviation 100 m) at 100 and 300 m
below the surface, where the true sediments are thickest or under the station --at (semfwi.hessian; point-spread
functions as in Fichtner & Trampert 2011): the spread of the response is the
resolution there, its part in the other parameter the K-G crosstalk; psf_<tag>.npz and .json.
bedmap: a new bedrock map from that of --bedrock, its depth replaced by the map of an interface inversion (--from, an
iterate x###.npy of its work directory, relative to $SEMFWI_RUNS/e2vp3d), or by its mean (--mean: a flat bedrock, to see
what the inversion owes to the H/V map); the sediment profile kept, or that of the iterate of an interface-v0g or
interface-pw inversion. --bedrock <tag> then starts any inversion from it: from the bedrock map of 0.5 Hz to 1 Hz, or
the velocities (--param vs) after the interface. bedmap --truth: the map closest to the true bedrock in the
least-squares sense over the inverted nodes of the grid of --fmax (the transition
of --param interface applied to it), a diagnostic that judges the parametrisation and never starts an inversion; with
--knots, also the Vs of interface-pw at the knots closest to the true sediments (least squares over their nodes).
jmap: J of the map of --bedrock, sediments and rock as in --param interface; on a map of bedmap --truth, whether the data prefer
the true bedrock to the maps the inversion finds (the J of these is in the history of the inversions).
jswap: J of the true model (0 on the data, an inverse crime) with one fixed part of --param interface swapped in, the
rule of Vp, the linear sediment profile or the transition: which one keeps the true bedrock above the J of the
inverted maps.
--nprocs: MPI ranks of every simulation (default 4); other than 4, it is part of the names of forward and bedrock
files (forward_f2_t24_n160.npz), and of the inversion's data and work directories.
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

# directions on every ZSTEP-th level of the grid (h / 10 vertically): h / 2.5, its horizontal step, wider than the
# largest gap between GLL planes (0.33 h), so that no level is left without gradient (Problem zstep)
ZSTEP = 4
HSCALE = 100.  # m of bedrock depth per unit of x (InterfaceProblem scale)
PSCALE = (100., 0.5)  # interface-v0g: m/s of v0 and 1/s of g per unit of x (100 m/s of Vs at 200 m)
ZK = (0., 50., 100., 200., 300., 500.)  # interface-pw: depths of the knots of the sediment Vs, linear in between
PWSCALE = 100.  # interface-pw: m/s of Vs at a knot per unit of x
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


def start_model(cfg, basin, bedrock=None):
    """1-D start: in the first 500 m below the meshed surface, Vs from 250 m/s at the surface to 1200 m/s
    at 500 m, Vp = 1600 m/s + 1.5 Vs, the density of the true model (not inverted); a generic sediment
    profile, the same everywhere (neither the basin's thickness nor its gradients).
    bedrock (npz of the bedrock command): sediments of Vs = v0 + g depth down to its thickness map, Vp =
    1600 m/s + 1.5 Vs, then the first crustal layer (the one meshed below the parameter grid)."""
    X, Y, Z = cfg.grid.mesh()
    depth = np.clip(basin.mesh_surface(X, Y) - Z, 0., 500.)
    rho = basin.properties(X, Y, Z)[2]
    if bedrock is None:
        vs = 250. + (1200. - 250.) * depth / 500.
        return e2vp.lame(1600. + 1.5 * vs, vs, rho)
    b = np.load(bedrock)
    sed = depth < regular_bilinear(b["x"], b["y"], b["h"], X, Y)
    (ls, ms), (lr, mr), _ = layers(cfg, basin, b)
    return np.where(sed, ls, lr), np.where(sed, ms, mr), rho


def knots(b):
    """Vs of the sediments of the bedrock map b at the knots ZK: its own (vsk, from an interface-pw inversion), or those
    of v0 + g depth."""
    if "vsk" in b:
        assert np.allclose(b["zk"], ZK), (b["zk"], ZK)
        return np.asarray(b["vsk"], float)
    return float(b["v0"]) + float(b["g"]) * np.array(ZK)


def profile_basis(depth, param):
    """phi (k, ...) with the sediment Vs = sum_k p_k phi_k(depth): (1, depth) for interface-v0g, the hat functions of
    the knots ZK for interface-pw (Vs piecewise linear in depth, p its values at the knots)."""
    if param == "interface-v0g":
        return np.stack([np.ones_like(depth), depth])
    return np.stack([np.interp(depth, ZK, e) for e in np.eye(len(ZK))])


def layers(cfg, basin, b):
    """(lambda, mu) of the sediments of the bedrock map b (Vs = v0 + g depth, or piecewise linear through knots(b) for
    a map of an interface-pw inversion; Vp = 1600 m/s + 1.5 Vs) and of the first crustal layer, at every node of the
    grid, and the depth below the meshed surface, clipped to [0, 500] m."""
    X, Y, Z = cfg.grid.mesh()
    depth = np.clip(basin.mesh_surface(X, Y) - Z, 0., 500.)
    rho = basin.properties(X, Y, Z)[2]
    vs = np.interp(depth, ZK, knots(b)) if "vsk" in b else b["v0"] + b["g"] * depth
    rock = basin.layers[2]
    return (e2vp.lame(1600. + 1.5 * vs, vs, rho)[:2], e2vp.lame(np.full_like(vs, rock["vp"][0]), np.full_like(vs, rock["vs"][0]), rho)[:2],
            depth)


def sediment_profile(cfg, basin, param):
    """f(p) of InterfaceProblem: (lambda, mu) of the sediments Vs = sum_k p_k phi_k(depth) (profile_basis: v0 + g depth,
    or piecewise linear through the knots ZK), Vp = 1600 m/s + 1.5 Vs, on the grid, and their derivatives in p."""
    X, Y, Z = cfg.grid.mesh()
    depth = np.clip(basin.mesh_surface(X, Y) - Z, 0., 500.)
    rho = basin.properties(X, Y, Z)[2]
    dvs = profile_basis(depth, param)

    def f(p):
        vs = np.tensordot(p, dvs, 1)
        vp = 1600. + 1.5 * vs
        return rho * (vp ** 2 - 2 * vs ** 2), rho * vs ** 2, rho * (3 * vp - 4 * vs) * dvs, 2 * rho * vs * dvs
    return f


def regular_bilinear(x, y, a, X, Y):
    """a[iy, ix] given on the regular axes x, y, at (X, Y), clamped to the edges."""
    fx = np.clip((X - x[0]) / (x[1] - x[0]), 0, len(x) - 1 - 1e-9)
    fy = np.clip((Y - y[0]) / (y[1] - y[0]), 0, len(y) - 1 - 1e-9)
    i, j = fx.astype(int), fy.astype(int)
    u, v = fx - i, fy - j
    return (1 - u) * (1 - v) * a[j, i] + u * (1 - v) * a[j, i + 1] + (1 - u) * v * a[j + 1, i] + u * v * a[j + 1, i + 1]


def konno_ohmachi(spec, fr, fc, b=40.):
    """spec (..., len(fr)) smoothed at the frequencies fc (Konno and Ohmachi 1998, bandwidth b)."""
    x = b * np.log10(fr[None, 1:] / fc[:, None])
    w = (np.sin(x) / np.where(x == 0, 1., x)) ** 4
    w[x == 0] = 1.
    return spec[..., 1:] @ w.T / w.sum(axis=1)


def sh_downhole(f, v0, g, h, zr, bed, rho=2000., q=30., slab=2.):
    """|surface / depth zr| of vertically incident SH waves (propagator of the stress-free surface down to
    zr) in sediments of Vs = v0 + g z and density rho down to h, then the bedrock bed = (vs, rho); a
    quality factor q in every layer keeps the ratio finite at its peaks."""
    w = 2 * np.pi * np.asarray(f)
    p11, p21 = np.ones_like(w, complex), np.zeros_like(w, complex)
    z = 0.
    while z < zr - 1e-9:
        d = min(slab, zr - z)
        vs, r = (v0 + g * (z + d / 2), rho) if z + d / 2 < h else bed
        c = vs * (1 + 0.5j / q)
        k, mu = w / c, r * c ** 2
        p11, p21 = np.cos(k * d) * p11 + np.sin(k * d) / (mu * k) * p21, -mu * k * np.sin(k * d) * p11 + np.cos(k * d) * p21
        z += d
    return 1. / np.abs(p11)


def bedrock(args):
    """Sediment thickness map from the observed data alone (forward with the same --fmax, --time and --nprocs:
    the archive's earthquake), as a site study would get it, written to bedrock_<tag>.npz for start_model:
    - at each surface station, the H/V ratio (Konno-Ohmachi smoothed horizontal over vertical spectrum,
      0.1 Hz to fmax) and its peak f0;
    - at the epicentre, the profile v0 + g z down to h fitted (grid search, least squares on the log) to the
      ratios surface / 90 m and surface / 990 m of the vertical array (0.15 to 0.9 fmax; 1-D SH, bedrock = the first crustal
      layer; v0 >= 100 m/s): it fixes the S travel time through the sediments tt, not the shape;
    - at each station, tt = c / f0 with c = f0 tt at the epicentre, and the depth where this profile
      reaches tt; where the H/V peak is below AMP the sediments are too thin for f0 to fall within fmax
      (thickness H_THIN, assumed).
    The true bedrock only scores the map (printed)."""
    AMP, H_THIN = 5., 50.
    tag = base_tag(args)
    r = np.load(f"{ROOT}/forward_{tag}.npz")
    u, st, dt = r["u"], r["stations"], float(r["dt"])
    basin = e2vp2.Basin(NPZ)
    n = 4 * u.shape[1]
    S = np.abs(np.fft.rfft(u, n=n, axis=1))
    fr = np.fft.rfftfreq(n, dt)
    H = np.sqrt((S[..., 0] ** 2 + S[..., 1] ** 2) / 2)
    fh = fr[(fr > 0.1) & (fr <= args.fmax)]
    surf = np.arange(len(st)) < len(st) - 3
    hv = konno_ohmachi(H[surf], fr, fh) / konno_ohmachi(S[surf, :, 2], fr, fh)
    f0, amp = fh[np.argmax(hv, axis=1)], hv.max(axis=1)
    epi = int(np.argmin(np.hypot(st[surf, 0], st[surf, 1])))
    fb = fr[(fr > 0.15) & (fr < 0.9 * args.fmax)][::4]
    s0 = float(basin.mesh_surface(0., 0.))
    Hs = konno_ohmachi(H, fr, fb)
    obs = [(s0 - st[k, 2], np.log(Hs[epi] / Hs[k])) for k in (len(st) - 2, len(st) - 1)]  # 90 and 990 m
    bed = (basin.layers[2]["vs"][0], basin.layers[2]["rho"][0])
    fits = [(sum(((np.log(sh_downhole(fb, v0, g, h, zr, bed)) - o) ** 2).sum() for zr, o in obs), v0, g, h)
            for v0 in np.arange(100., 401., 20.) for g in np.arange(0., 8.01, 0.25) for h in np.arange(60., 501., 10.)
            if v0 + g * h < bed[0]]
    m, v0, g, he = min(fits)
    tt = (lambda h: np.log1p(g * h / v0) / g) if g else (lambda h: h / v0)
    c = f0[epi] * tt(he)
    h_st = np.where(amp > AMP, v0 * np.expm1(g * c / f0) / g if g else v0 * c / f0, H_THIN)
    x, y = np.unique(st[surf, 0]), np.unique(st[surf, 1])
    hm = np.full((len(y), len(x)), np.nan)
    hm[np.searchsorted(y, st[surf, 1]), np.searchsorted(x, st[surf, 0])] = h_st
    assert not np.isnan(hm).any(), "surface stations not on a regular grid"
    true = basin._bilinear(basin.surface, st[surf, 0], st[surf, 1]) - basin._bilinear(basin.bedrock, st[surf, 0], st[surf, 1])
    a = np.arange(x[0], x[-1] + 1e-9, 100.)
    A, B = np.meshgrid(a, a)
    err = regular_bilinear(x, y, hm, A, B) - (basin._bilinear(basin.surface, A, B) - basin._bilinear(basin.bedrock, A, B))
    print(f"profile at the epicentre: v0 {v0:.0f} m/s, g {g:.2f} /s, bedrock at {he:.0f} m (misfit {m:.2f}), "
          f"tt {tt(he):.3f} s; c = f0 tt = {c:.3f}; {(amp > AMP).sum()} of {surf.sum()} stations with an H/V peak above {AMP:g}")
    print(f"against the true bedrock: at the stations |error| median {np.median(np.abs(h_st - true)):.0f} m, "
          f"max {np.abs(h_st - true).max():.0f} m; map every 100 m rms {np.sqrt((err ** 2).mean()):.0f} m "
          f"(true thickness rms {np.sqrt((true ** 2).mean()):.0f} m)")
    np.savez(f"{ROOT}/bedrock_{tag}.npz", x=x, y=y, h=hm, v0=v0, g=g, f0=f0, amp=amp, stations=st[surf])


def base_tag(args):
    return f"f{args.fmax:g}" + (f"_t{args.time:g}" if args.time else "") + ("" if args.nprocs == 4 else f"_n{args.nprocs}")


def forward(args):
    tag = (base_tag(args) + ("_flat" if args.flat else "") + ("_rock" if args.rock else "")
           + ("" if args.pml == "follow" else f"_pml{args.pml}") + ("" if args.npml == 2 else f"_npml{args.npml}"))
    cfg, basin = e2vp2.config_3d(f"{ROOT}/forward_{tag}", NPZ, fmax=args.fmax, sim_time=args.time, relief=not args.flat,
                                 nprocs=args.nprocs, npml=args.npml)
    lam, mu, rho = true_model(cfg, basin, rock=args.rock)
    t = time.time()
    u = problem.observed(cfg, Runner(cfg), cfg.workdir, lam, mu, rho, pml=args.pml)[0]
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


def observed(args, work):
    """cfg and basin of a problem in work, the data of the true model on its grid, and that model (lambda, mu, rho)."""
    n = "" if args.nprocs == 4 else f"_n{args.nprocs}"
    cfg, basin = e2vp2.config_3d(work, NPZ, fmax=args.fmax, nsources=args.nsources, nprocs=args.nprocs)
    data, _ = e2vp2.config_3d(f"{ROOT}/data_f{args.fmax:g}_s{args.nsources}{n}", NPZ, fmax=args.fmax, nsources=args.nsources,
                              nprocs=args.nprocs)
    lam_t, mu_t, rho = true_model(cfg, basin)
    return cfg, basin, problem.observed(data, Runner(data), data.workdir, lam_t, mu_t, rho, pml="follow"), (lam_t, mu_t, rho)


def make_problem(args, work):
    cfg, basin, dobs, (lam_t, mu_t, rho) = observed(args, work)
    lam0, mu0, _ = start_model(cfg, basin, f"{ROOT}/bedrock_{args.bedrock}.npz" if args.bedrock else None)
    if args.param.startswith("interface"):
        assert args.bedrock, "--param interface moves the bedrock map of --bedrock"
        b = np.load(f"{ROOT}/bedrock_{args.bedrock}.npz")
        up, down, depth = layers(cfg, basin, b)
        prof = None
        if args.param == "interface-v0g":
            assert "vsk" not in b, "interface-v0g starts from a linear profile"
            prof = (sediment_profile(cfg, basin, args.param), (float(b["v0"]), float(b["g"])), PSCALE)
        elif args.param == "interface-pw":
            prof = (sediment_profile(cfg, basin, args.param), knots(b), (PWSCALE,) * len(ZK))
        pb = problem.InterfaceProblem(cfg, Runner(cfg), dobs, (b["x"], b["y"]), b["h"], up, down, depth, rho, (lam0, mu0),
                                      width=cfg.grid.hs[2] / 2, scale=HSCALE, hmax=500., pml="follow", profile=prof)
        return pb, (lam_t, mu_t), basin
    pb = problem.Problem(cfg, Runner(cfg), dobs, lam0, mu0, rho, param=args.param, pml="follow", zstep=ZSTEP,
                         active=("b",) if args.param == "vs" else ("a", "b"))
    return pb, (lam_t, mu_t), basin


def gradcheck(args):
    """Directional derivative along a smooth bump of K (block a) then G (block b), 60 m below the surface
    at (500, -500) m: g . dx against (J(x + h dx) - J(x)) / h, and the centered difference for each h given with -h. --param interface: a bump of the bedrock depth
    on its map, centred at the same point (h = 1e-2: 1 m deeper at its centre); interface-v0g: then v0 alone (h = 1e-2:
    1 m/s) and g alone (5e-3 /s); interface-pw: then the Vs of each knot alone (h = 1e-2: 1 m/s)."""
    tag = f"f{args.fmax:g}" + ("" if args.param == "kg" else f"_{args.param}") + (f"_bedrock-{args.bedrock}" if args.bedrock else "")
    pb, _, basin = make_problem(args, f"{ROOT}/gradcheck_{tag}")
    x = np.zeros(pb.size)
    J0, _, g = pb.evaluate(x, "x0")
    dirs = []
    if args.param.startswith("interface"):
        b = np.load(f"{ROOT}/bedrock_{args.bedrock}.npz")
        X, Y = (c.ravel() for c in np.meshgrid(b["x"], b["y"]))  # map nodes, h[iy, ix]
        bump = np.exp(-((X - 500.) ** 2 + (Y + 500.) ** 2) / (2 * 600. ** 2))
        dirs.append(("h", np.r_[bump, np.zeros(pb.k)]))
        names = ("v0", "g") if args.param == "interface-v0g" else tuple(f"vs{z:g}" for z in ZK)
        dirs += [(k, np.eye(pb.size)[pb.n + i]) for i, k in enumerate(names[:pb.k])]  # InterfaceProblem.params
    else:
        X, Y, Z = (c[pb.mask] for c in pb.cfg.grid.mesh())
        depth = basin.mesh_surface(X, Y) - Z
        bump = np.exp(-((X - 500.) ** 2 + (Y + 500.) ** 2) / (2 * 600. ** 2) - (depth - 60.) ** 2 / (2 * 40. ** 2))
        for i, blk in enumerate(pb.active):
            dx = np.zeros(pb.size)
            dx[i * pb.n:(i + 1) * pb.n] = bump
            dirs.append((blk, dx))
    rows = []
    for blk, dx in dirs:
        dco = g @ dx
        for h in args.h:
            Jh = pb.evaluate(x + h * dx, f"{blk}_{h:g}", grad=False)[0]
            dfdh = (Jh - J0) / h
            rows.append(dict(block=blk, h=h, dco=float(dco), dfdh=float(dfdh), rel=float(abs(dfdh - dco) / abs(dco))))
            print(f"{blk} h={h:g}: adjoint {dco:.6e}  finite difference {dfdh:.6e}  relative gap {rows[-1]['rel']:.2e}", flush=True)
    # +h and -h: the centered difference, free of the first-order term of the curvature of J (at 1 Hz, the forward
    # difference of the interface converges only for h < 1e-4)
    fd = {(r["block"], r["h"]): r for r in rows}
    central = []
    for (blk, h), r in fd.items():
        if h > 0 and (blk, -h) in fd:
            c = 0.5 * (r["dfdh"] + fd[(blk, -h)]["dfdh"])
            central.append(dict(block=blk, h=h, dco=r["dco"], dfdh=c, rel=abs(c - r["dco"]) / abs(r["dco"])))
            print(f"{blk} +-{h:g}: adjoint {r['dco']:.6e}  centered difference {c:.6e}  relative gap {central[-1]['rel']:.2e}")
    json.dump(dict(J0=float(J0), rows=rows, central=central), open(f"{ROOT}/gradcheck_{tag}.json", "w"), indent=1)


def psf(args):
    """Point-spread functions at the start model (module docstring): for each bump, the response in each block as a
    grid field, and its centre of mass, its standard deviations horizontally and vertically (weights |H v|), and its
    norm per block (crosstalk)."""
    from semfwi import hessian
    assert not args.param.startswith("interface"), "psf: bumps on the grid nodes"
    tag = f"f{args.fmax:g}_{args.param}" + (f"_bedrock-{args.bedrock}" if args.bedrock else "") \
        + (f"_at{args.at[0]:g}_{args.at[1]:g}" if args.at else "")
    pb, _, basin = make_problem(args, f"{ROOT}/psf_{tag}")
    st = np.array(pb.cfg.stations)
    sed = basin._bilinear(basin.surface, st[:, 0], st[:, 1]) - basin._bilinear(basin.bedrock, st[:, 0], st[:, 1])
    x0, y0 = args.at or st[int(np.argmax(sed))][:2]  # by default the station on the thickest sediments of the true model
    H = hessian.Hessian(pb, np.zeros(pb.size), "psf", dx=1e-2)
    X, Y, Z = pb.cfg.grid.mesh()
    depth = (basin.mesh_surface(X, Y) - Z)[pb.mask]
    X, Y = X[pb.mask], Y[pb.mask]
    out, rows = {}, []
    for blk in pb.active:
        for d in (100., 300.):
            z = float(basin.mesh_surface(x0, y0)) - d
            v = hessian.bump(pb, blk, (x0, y0, z), width=100.)
            hv = H(v)
            row = dict(block=blk, depth=d, x=float(x0), y=float(y0),
                       sediments=float(basin._bilinear(basin.surface, x0, y0) - basin._bilinear(basin.bedrock, x0, y0)))
            for i, b in enumerate(pb.active):
                h = hv[i * pb.n:(i + 1) * pb.n]
                w = np.abs(h) / np.abs(h).sum()
                cx, cy, cd = (float(w @ c) for c in (X, Y, depth))
                row[b] = dict(norm=float(np.linalg.norm(h)), centre=(cx, cy, cd),
                              sigma_h=float(np.sqrt(w @ ((X - cx) ** 2 + (Y - cy) ** 2) / 2)),
                              sigma_z=float(np.sqrt(w @ (depth - cd) ** 2)), peak=float(np.abs(h).max()))
                out[f"{blk}{d:g}_{b}"] = pb.grid_field(h)
            rows.append(row)
            print(json.dumps(row), flush=True)
    np.savez(f"{ROOT}/psf_{tag}.npz", **out)
    json.dump(rows, open(f"{ROOT}/psf_{tag}.json", "w"), indent=1)


def invert(args):
    work = (f"{ROOT}/invert_f{args.fmax:g}_s{args.nsources}_{args.param}_z{ZSTEP}" + (f"_bedrock-{args.bedrock}" if args.bedrock else "")
            + ("" if args.nprocs == 4 else f"_n{args.nprocs}"))
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

    # interface: a few smooth unknowns (the bedrock map), no regularization
    inversion.invert(work, [(f"f{args.fmax:g}", args.iters)], mk, reg=None if args.param.startswith("interface") else "tv", callback=errors,
                     log=lambda s: print(s, flush=True))


def bedmap(args):
    """bedrock_<--tag>.npz: the map of --bedrock with its depth h replaced (module docstring)."""
    b = dict(np.load(f"{ROOT}/bedrock_{args.bedrock}.npz"))
    if args.mean:
        h = np.full_like(b["h"], b["h"].mean())
    elif args.truth:
        cfg, basin = e2vp2.config_3d(f"{ROOT}/bedmap_truth_f{args.fmax:g}", NPZ, fmax=args.fmax, nsources=args.nsources)
        X, Y, Z = cfg.grid.mesh()
        lam0, mu0, rho = start_model(cfg, basin, f"{ROOT}/bedrock_{args.bedrock}.npz")
        up, down, depth = layers(cfg, basin, b)
        pb = problem.InterfaceProblem(cfg, None, None, (b["x"], b["y"]), b["h"], up, down, depth, rho, (lam0, mu0),
                                      width=cfg.grid.hs[2] / 2, scale=HSCALE, hmax=500.)
        thick = np.clip(basin._bilinear(basin.surface, X, Y) - basin._bilinear(basin.bedrock, X, Y), 0., None)
        h = pb.interface(pb.fit_interface(thick[pb.mask]))
        if args.knots:  # the profile of interface-pw closest to the true sediments
            vs_t = basin.properties(X, Y, Z)[1]
            sed = (vs_t < 2000.) & pb.mask & (basin.mesh_surface(X, Y) >= Z)
            b["zk"] = np.array(ZK)
            b["vsk"] = np.linalg.lstsq(profile_basis(depth[sed], "interface-pw").T, vs_t[sed], rcond=None)[0]
    else:
        assert f"bedrock-{args.bedrock}" in args.src, f"{args.src} is not an inversion from bedrock_{args.bedrock}"
        x = np.load(f"{ROOT}/{args.src}")
        h = b["h"] + HSCALE * x[:b["h"].size].reshape(b["h"].shape)  # InterfaceProblem.interface
        if x.size == b["h"].size + 2:  # interface-v0g: InterfaceProblem.params
            b["v0"], b["g"] = b["v0"] + PSCALE[0] * x[-2], b["g"] + PSCALE[1] * x[-1]
        elif x.size == b["h"].size + len(ZK):  # interface-pw
            b["zk"], b["vsk"] = np.array(ZK), knots(b) + PWSCALE * x[-len(ZK):]
    prof = (f"Vs {np.round(b['vsk']).astype(int).tolist()} m/s at {list(ZK)} m" if "vsk" in b
            else f"v0 {float(b['v0']):.0f} m/s, g {float(b['g']):.3f} /s")
    print(f"bedrock_{args.tag}: depth {h.min():.0f} to {h.max():.0f} m, mean {h.mean():.0f} m, {prof}"
          f" (bedrock_{args.bedrock}: {b['h'].min():.0f} to {b['h'].max():.0f}, mean {b['h'].mean():.0f})")
    np.savez(f"{ROOT}/bedrock_{args.tag}.npz", **{**b, "h": h})


def jmap(args):
    """J of the bedrock map of --bedrock, sediments and rock as at the start of --param interface (x = 0): one forward
    per source, no gradient; jmap_<tag>.json."""
    args.param = "interface"
    tag = f"f{args.fmax:g}_bedrock-{args.bedrock}" + ("" if args.nprocs == 4 else f"_n{args.nprocs}")
    pb, _, _ = make_problem(args, f"{ROOT}/jmap_{tag}")
    J = pb.evaluate(np.zeros(pb.size), "x0", grad=False)[0]
    print(f"jmap_{tag}: J = {J:.6e}")
    with open(f"{ROOT}/jmap_{tag}.json", "w") as f:
        json.dump(dict(J=float(J), fmax=args.fmax, bedrock=args.bedrock, nsources=args.nsources), f, indent=1)


def jswap(args):
    """J of the true model with one fixed part of --param interface swapped in (a diagnostic, like bedmap --truth), the
    density true: vp, Vp = 1600 m/s + 1.5 Vs in the sediments; profile, the Vs and the Vp of the sediments each replaced by
    its best line in depth (least squares over the sediment nodes below the surface); transition, the sharp bedrock
    replaced by the logistic transition of InterfaceProblem (width hz / 2) at the true depth, between the sediments
    continued below it by their base values and the first crustal layer. One forward per source; jswap_<tag>.json."""
    tag = f"f{args.fmax:g}_{args.swap}" + ("" if args.nprocs == 4 else f"_n{args.nprocs}")
    cfg, basin, dobs, (_, _, rho) = observed(args, f"{ROOT}/jswap_{tag}")
    X, Y, Z = cfg.grid.mesh()
    vp, vs, _ = basin.properties(X, Y, Z)
    vp0, vs0 = vp.copy(), vs.copy()
    sed = vs < 2000.
    depth = basin.mesh_surface(X, Y) - Z
    if args.swap == "vp":
        vp = np.where(sed, 1600. + 1.5 * vs, vp)
    elif args.swap == "profile":
        fit = sed & (depth >= 0.)
        d = np.maximum(depth[sed], 0.)
        for a in (vp, vs):
            a[sed] = np.polyval(np.polyfit(depth[fit], a[fit], 1), d)
    else:
        thick = basin._bilinear(basin.surface, X, Y) - basin._bilinear(basin.bedrock, X, Y)
        s = 0.5 * (1 + np.tanh((np.clip(depth, 0., 500.) - thick) / cfg.grid.hs[2]))  # InterfaceProblem._blend
        base, rock = basin.layers[1], basin.layers[2]
        for a, k in ((vp, "vp"), (vs, "vs")):
            up = np.where(sed, a, base[k][1])
            a[...] = up + (rock[k][0] - up) * s
    lam, mu, _ = e2vp.lame(vp, vs, rho)
    pb = problem.Problem(cfg, Runner(cfg), dobs, lam, mu, rho, param="lame", pml="follow")
    J = pb.evaluate(np.zeros(pb.size), "x0", grad=False)[0]
    m = pb.mask & (depth >= 0.)
    rms = {k: float(np.sqrt(np.mean((a - a0)[m] ** 2))) for k, a, a0 in (("vp", vp, vp0), ("vs", vs, vs0))}
    print(f"jswap_{tag}: J = {J:.6e}, change rms over the inverted nodes Vp {rms['vp']:.0f} m/s, Vs {rms['vs']:.0f} m/s")
    with open(f"{ROOT}/jswap_{tag}.json", "w") as f:
        json.dump(dict(J=float(J), fmax=args.fmax, swap=args.swap, nsources=args.nsources, rms=rms), f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("prepare", "forward", "site", "bedrock", "gradcheck", "invert", "psf", "bedmap", "jmap", "jswap"))
    ap.add_argument("--fmax", type=float, default=0.5)
    ap.add_argument("--flat", action="store_true", help="forward: flat free surface")
    ap.add_argument("--rock", action="store_true", help="forward: sediments replaced by rock")
    ap.add_argument("--time", type=float, default=None, help="forward, site: simulated time (s), default 3 / fmax + 6")
    ap.add_argument("--param", choices=("lame", "kg", "vs", "interface", "interface-v0g", "interface-pw"), default="kg")
    ap.add_argument("--h", type=float, nargs="+", default=[1e-2, 1e-3])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--nsources", type=int, default=3)
    ap.add_argument("--bedrock", metavar="TAG", help="invert, gradcheck: start from bedrock_TAG.npz (bedrock command)")
    ap.add_argument("--from", dest="src", metavar="X_NPY", help="bedmap: iterate of an interface inversion")
    ap.add_argument("--mean", action="store_true", help="bedmap: the depth of the map replaced by its mean")
    ap.add_argument("--truth", action="store_true",
                    help="bedmap: the depth of the map fitted to the true bedrock (diagnostic, never a start)")
    ap.add_argument("--knots", action="store_true",
                    help="bedmap --truth: also the Vs at the knots ZK fitted to the true sediments (diagnostic)")
    ap.add_argument("--tag", help="bedmap: written to bedrock_TAG.npz")
    ap.add_argument("--swap", choices=("vp", "profile", "transition"), help="jswap: the part of --param interface swapped in")
    ap.add_argument("--at", type=float, nargs=2, metavar=("X", "Y"), help="psf: bumps under this surface station")
    ap.add_argument("--nprocs", type=int, default=4, help="MPI ranks of each simulation")
    ap.add_argument("--pml", choices=("follow", "copy"), default="follow",
                    help="forward: PMLs of the sediment layers, mean of their layer or extension of the model files")
    ap.add_argument("--npml", type=int, default=2, help="forward: layers of PML elements")
    args = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    if args.what == "prepare":
        e2vp2.prepare(NPZ)
    else:
        {"forward": forward, "site": site, "bedrock": bedrock, "gradcheck": gradcheck, "invert": invert, "psf": psf, "bedmap": bedmap, "jmap": jmap, "jswap": jswap}[args.what](args)


if __name__ == "__main__":
    main()
