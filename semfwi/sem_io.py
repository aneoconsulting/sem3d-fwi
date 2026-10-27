"""SEM3D inputs and outputs: input.spec, stations, adjoint sources, traces, log, strains."""
import os
import h5py
import numpy as np

def write_mesher_inputs(d, cfg):
    """Layered cube [x0, x0 + L]^2 x [-depth, 0], free surface on top, cfg.npml layers
    of PML elements at the bottom and on the sides."""
    layers = cfg.layer_list
    lines = [cfg.x0, cfg.x0 + cfg.L, cfg.h_elem, cfg.x0, cfg.x0 + cfg.L, cfg.h_elem, 0., len(layers)]
    lines += [f"{th!r} {n}" for th, n in layers]
    lines += [cfg.npml, "0 1", 1]
    with open(f"{d}/mat.dat", "w") as f:
        f.write("".join(f"{x}\n" for x in lines))
    # Reference properties: the solid reads material.spec, the PMLs are rewritten by
    # model.write_pml_materials if the profile must be extended
    vp = np.sqrt((cfg.lam0 + 2 * cfg.mu0) / cfg.rho0)
    vs = np.sqrt(cfg.mu0 / cfg.rho0)
    with open(f"{d}/mater.in", "w") as f:
        f.write(f"{len(layers)}\n" + f"S {vp} {vs} {cfg.rho0} 0.0 0.0\n" * len(layers))
    with open(f"{d}/mesh.input", "w") as f:
        f.write(f"{cfg.nprocs}\n1\n")


def _section(s):
    """source section (default) or extended_source if s["_section"] says so."""
    s = dict(s)
    name = s.pop("_section", "source")
    return name + " {\n" + "".join(f"    {k} = {v};\n" for k, v in s.items()) + "};\n"


def write_input(d, cfg, sources, save_strain=False):
    with open(f"{d}/stations.txt", "w") as f:
        for s in cfg.stations:
            f.write("%r %r %r\n" % tuple(float(x) for x in s))
    src = "".join(_section(s) for s in sources)
    with open(f"{d}/input.spec", "w") as f:
        f.write(f"""run_name = "fwi";
sim_time = {cfg.sim_time!r};
mesh_file = "mesh4spec";
mat_file = "material.input";
dim = 3;
mpml_atn_param = 0.002;
ngll = {cfg.ngll};
snapshots {{ save_snap = false; }};
save_traces = true;
traces_format = hdf5;
prorep = false;
{src}
time_scheme {{
    accel_scheme = false;
    veloc_scheme = true;
    alpha = 0.5;
    beta = 0.5;
    gamma = 1;
    courant = {cfg.courant!r};
    dt = {cfg.dt!r};
}};
amortissement {{ nsolids = 0; }};
capteurs "UU" {{
    type = points;
    file = "stations.txt";
    period = 1;
}};
out_variables {{
    enP = 0; enS = 0; evol = 0; pre = 0;
    dis = 1; vel = 0; acc = 0;
    edev = 0; sdev = 0;
}};
{f"save_strain = true; strain_period = {cfg.strain_period};" if save_strain else ""}
""")


def point_source(coords, direction, **kw):
    s = dict(coords="%r %r %r" % tuple(float(x) for x in coords), type="impulse",
             dir="%r %r %r" % tuple(float(x) for x in direction))
    s.update(kw)
    return s


def write_extended_force(d, tag, X, Y, Z, direction, tau, hist):
    """Extended source of point forces (is_force = 1): X, Y, Z of shape (Ns, Nd), Ns, Nd >= 2,
    hist (Ns, Nd, nt) force in N at times tau. SEM3D does not normalize Dir and truncates file
    names to 30 characters."""
    kine, slip = f"k{tag}.h5", f"s{tag}.h5"
    assert len(kine) <= 30 and X.shape[0] >= 2 and X.shape[1] >= 2, (kine, X.shape)
    ns, nd = X.shape
    nt = hist.shape[-1]
    with h5py.File(f"{d}/{kine}", "w") as f:
        f.attrs["Ns"] = np.int32(ns)
        f.attrs["Nd"] = np.int32(nd)
        f.attrs["Nt"] = np.int32(nt)
        f.attrs["dt"] = float(tau[1] - tau[0])
        f.attrs["Dir"] = np.asarray(direction, dtype="f8")
        for name, a in (("x", X), ("y", Y), ("z", Z)):
            f[name] = np.ascontiguousarray(a, dtype="f8")
    with h5py.File(f"{d}/{slip}", "w") as f:
        f["moment"] = np.ascontiguousarray(hist, dtype="f8")
        f["time"] = np.asarray(tau, dtype="f8")
    return dict(_section="extended_source", kine_file=f'"{kine}"', slip_file=f'"{slip}"', is_force=1)


class SurfaceLoad:
    """Uniform traction on a rectangle of the free surface, discretized into nodal forces:
    F_i = t(t) * w_i, where w_i is the integral of the basis function of GLL node i over the
    loaded area (GLL weights x 2D Jacobian, assembled). Elements of side h aligned with the area."""

    def __init__(self, xa, xb, h, ngll, amplitude, pulse, direction=(0., 0., -1.)):
        from numpy.polynomial import legendre
        # GLL points and weights on [-1, 1]
        n = ngll - 1
        inner = legendre.Legendre.basis(n).deriv().roots() if n > 1 else np.array([])
        xi = np.concatenate([[-1.], np.sort(inner), [1.]])
        pn = legendre.Legendre.basis(n)(xi)
        wg = 2. / (n * (n + 1) * pn ** 2)
        ne = int(round((xb - xa) / h))
        x = np.unique(np.round(np.concatenate([xa + h * e + 0.5 * h * (xi + 1) for e in range(ne)]), 9))
        w = np.zeros(len(x))
        for e in range(ne):
            for j in range(ngll):
                k = np.argmin(np.abs(x - (xa + h * e + 0.5 * h * (xi[j] + 1))))
                w[k] += wg[j] * 0.5 * h
        self.x, self.w1 = x, w
        self.amplitude, self.pulse, self.direction = amplitude, pulse, direction

    def total_force(self):
        return self.amplitude * self.w1.sum() ** 2

    def write(self, d, cfg, tag="L"):
        tau = cfg.dt * (np.arange(cfg.nt) + 0.5)
        X, Y = np.meshgrid(self.x, self.x, indexing="ij")
        W = np.outer(self.w1, self.w1) * self.amplitude
        f = self.pulse(tau)
        f[-1] = 0.  # SEM3D holds the last value beyond the last sample
        return [write_extended_force(d, tag, X, Y, np.zeros_like(X), self.direction, tau,
                                     W[:, :, None] * f[None, None, :])]


def materialize_sources(d, cfg, sources):
    """input.spec sections for a list of sources: dictionaries (point sources or ready-made
    sections) or objects with a write(d, cfg) method."""
    out = []
    for i, s in enumerate(sources):
        out += s.write(d, cfg, tag=f"L{i}") if hasattr(s, "write") else [s]
    return out


def write_adjoint_sources(d, cfg, tau, hist):
    """hist: (nsta, nt, 3) adjoint histories (already time-reversed).
    adjoint = "points": one point source per station and per component, func = file, zero
    components being omitted; adjoint = "extended": one extended source per component,
    on the station grid cfg.receiver_grid (stations ordered with x slow, y fast)."""
    if cfg.adjoint == "extended":
        xs, ys, z = cfg.receiver_grid
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        h = hist.reshape(len(xs), len(ys), hist.shape[1], 3).copy()
        h[:, :, -1, :] = 0.
        return [write_extended_force(d, f"A{c}", X, Y, np.full_like(X, z), np.eye(3)[c], tau, h[..., c])
                for c in range(3) if np.any(h[..., c])]
    sources = []
    for r, st in enumerate(cfg.stations):
        for c in range(3):
            if not np.any(hist[r, :, c]):
                continue
            fn = f"a{r}_{c}.txt"
            np.savetxt(f"{d}/{fn}", np.c_[tau, hist[r, :, c]])
            sources.append(point_source(st, np.eye(3)[c], func="file", time_file=f'"{fn}"'))
    return sources


def read_log(d):
    dt = n = None
    for line in open(f"{d}/sem3d.log"):
        if "Time step size" in line:
            dt = float(line.split(":")[1])
        if "Number of time steps" in line:
            n = int(line.split(":")[1])
    return dt, n


def read_traces(d, nsta):
    """Displacements at the stations, (nsta, nt, 3); sample k is the field at time (k+1) dt."""
    out = {}
    for fn in sorted(os.listdir(f"{d}/traces")):
        if not fn.endswith(".h5"):
            continue
        with h5py.File(f"{d}/traces/{fn}", "r") as f:
            var = [(v.decode() if isinstance(v, bytes) else str(v)).split()[0] for v in f["Variables"][...]]
            assert var[:4] == ["Time", "Displ", "Displ", "Displ"], var
            for k in f.keys():
                if k.startswith("UU_") and not k.endswith("_pos"):
                    out[int(k[3:])] = f[k][:, 1:4]
    assert sorted(out) == list(range(nsta)), sorted(out)
    return np.stack([out[i] for i in range(nsta)])


def read_strain(d):
    """Strains per rank: list of (coords (npts,3), weight (npts), eps (nrec,npts,6) memmap,
    period). Record j holds the field at time (j+1) * period * dt."""
    out = []
    for fn in sorted(os.listdir(f"{d}/strain")):
        if not fn.startswith("strain_geom."):
            continue
        rank = fn.split(".")[1]
        with h5py.File(f"{d}/strain/{fn}", "r") as f:
            npts = int(f.attrs["npts"])
            period = int(f.attrs["period"]) if "period" in f.attrs else 1
            if npts == 0:
                continue
            coords, weight = f["coords"][...], f["weight"][...]
        eps = np.memmap(f"{d}/strain/strain.{rank}.bin", dtype="<f8", mode="r").reshape(-1, npts, 6)
        out.append((coords, weight, eps, period))
    return out
