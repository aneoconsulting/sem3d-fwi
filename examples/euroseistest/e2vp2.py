"""EUROSEISTEST, 3-D model of the Mygdonian basin (E2VP2-MODEL archive): two sedimentary layers with linear
gradients between the free surface, an intermediate interface and the bedrock (10 m grids, UTM 34N), then
flat crustal layers; a normal-fault earthquake below the centre of the basin. Nothing of the archive is
copied here (its licence is unknown): its directory is read (velocity_profile.txt, source_info.txt and the
three interface grids) and the part used is cropped into a local npz file by prepare().

Axes: x east, y north, z up, origin at the epicentre; z = 0 is the elevation Z0 of the box (its lowest
surface point), and the mesh follows the surface elevation minus Z0 (hexmesh), brought back to 0 over the
margin before the PMLs (the PMLs must stay flat). The sediments keep their thicknesses below this surface.

Assumption: in a "linear gradient" layer, the properties vary linearly with the relative depth between
its top and bottom interfaces (velocity_profile.txt does not say whether with depth or relative depth).
"""
import os
import re

import numpy as np

from semfwi import hexmesh, sem_io
from semfwi.config import Config, Grid

import e2vp

DATA = os.path.expanduser(os.environ.get("E2VP2_DIR", "~/data/euroseistest/e2vp2_model/E2VP2-MODEL"))


def read_profile(path=f"{DATA}/velocity_profile.txt"):
    """Layers of velocity_profile.txt: list of dict(top, bottom, vs, vp, rho, gradient), top and bottom
    the interface (a grid file name or a z-plane elevation), each property a (top, bottom) pair."""
    layers = []
    for block in re.split(r"\n\s*layer \d+\s*\n", "\n" + open(path).read())[1:]:
        kv = dict((" ".join(k.split()), v.strip()) for k, v in (l.split("=", 1) for l in block.splitlines() if "=" in l))
        lay = dict(top=kv["interface top"], bottom=kv["interface bottom"],
                   gradient=kv["structure type"] == "linear gradient")
        for p in ("Vs", "Vp", "rho"):
            lay[p.lower()] = (float(kv[f"{p} top"]), float(kv[f"{p} bottom"]))
        layers.append(lay)
    return layers


def read_source(path=f"{DATA}/source_info.txt"):
    """source_info.txt: dict(x, y, z, mw, strike, dip, rake), UTM and elevation in m."""
    kv = dict(re.match(r"\s*(\w+).*?=\s*(\S+)", l).groups() for l in open(path) if "=" in l)
    return dict(x=float(kv["x"]), y=float(kv["y"]), z=float(kv["z"]), mw=float(kv["mw"]),
                strike=float(kv["strike"]), dip=float(kv["dip"]), rake=float(kv["rake"]))


def prepare(out, half=3000., margin=1000.):
    """Crops the surface, intermediate and bedrock grids to the box of half-width half + margin around
    the epicentre and writes out (npz): x, y (local axes), surface, mp, bedrock (elevations; outside the
    basin mp = bedrock = surface)."""
    import h5py
    from PIL import Image  # the bedrock grid is a TIFF despite its name
    Image.MAX_IMAGE_PIXELS = None
    src = read_source()
    w = half + margin
    with h5py.File(f"{DATA}/DEM_surface_10m.grd") as f:
        gx, gy = f["x"][:], f["y"][:]  # centres of the cells, regular
        ix = slice(int(round((src["x"] - w - gx[0]) / (gx[1] - gx[0]))), int(round((src["x"] + w - gx[0]) / (gx[1] - gx[0]))) + 1)
        iy = slice(int(round((src["y"] - w - gy[0]) / (gy[1] - gy[0]))), int(round((src["y"] + w - gy[0]) / (gy[1] - gy[0]))) + 1)
        x, y, s = f["x"][ix] - src["x"], f["y"][iy] - src["y"], f["z"][iy, ix]
    with h5py.File(f"{DATA}/DEM_MP_10m.grd") as f:
        mp = f["z"][iy, ix]
    b = np.array(Image.open(f"{DATA}/DEM_bedrock_10m.grd"), dtype=np.float32)[iy, ix]
    mp, b = np.where(np.isnan(mp), s, mp), np.where(np.isnan(b), s, b)
    assert np.all(s >= mp) and np.all(mp >= b), "interfaces out of order"
    np.savez(out, x=x, y=y, surface=s, mp=mp, bedrock=b)


class Basin:
    """The cropped interfaces (prepare) and the layers of velocity_profile.txt."""

    def __init__(self, npz, half=3000., taper=800., relief=True):
        t = np.load(npz)
        self.x, self.y = t["x"], t["y"]
        self.surface, self.mp, self.bedrock = t["surface"], t["mp"], t["bedrock"]
        self.half, self.taper, self.relief = half, taper, relief
        inside = (np.abs(self.x)[None, :] <= half) & (np.abs(self.y)[:, None] <= half)
        self.z0 = float(self.surface[inside].min())
        self.layers = read_profile()

    def _bilinear(self, a, x, y):
        fx = np.clip((x - self.x[0]) / (self.x[1] - self.x[0]), 0, len(self.x) - 1 - 1e-9)
        fy = np.clip((y - self.y[0]) / (self.y[1] - self.y[0]), 0, len(self.y) - 1 - 1e-9)
        i, j = fx.astype(int), fy.astype(int)
        u, v = fx - i, fy - j
        return (1 - u) * (1 - v) * a[j, i] + u * (1 - v) * a[j, i + 1] + (1 - u) * v * a[j + 1, i] + u * v * a[j + 1, i + 1]

    def weight(self, x, y):
        """1 inside the box minus the taper band, down to 0 at its edge (cos^2)."""
        d = np.maximum(np.abs(x), np.abs(y)) - (self.half - self.taper)
        return np.where(d <= 0, 1., np.cos(0.5 * np.pi * np.clip(d / self.taper, 0, 1)) ** 2)

    def mesh_surface(self, x, y):
        """Elevation of the meshed free surface minus z0 (0 without relief)."""
        if not self.relief:
            return np.zeros(np.broadcast(x, y).shape)
        return (self._bilinear(self.surface, x, y) - self.z0) * self.weight(x, y)

    def write_topography(self, path, step=20.):
        a = np.arange(-self.half, self.half + 1e-9, step)
        X, Y = np.meshgrid(a, a)
        np.savez(path, x=a, y=a, z=self.mesh_surface(X, Y))

    def properties(self, x, y, z):
        """(Vp, Vs, rho) at mesh coordinates (x, y, z): the sediments by depth below the meshed surface
        with their thicknesses at (x, y), the crust by elevation; points above the surface get the
        surface values (they only enter the interpolation near it)."""
        x, y, z = np.broadcast_arrays(*(np.asarray(a, float) for a in (x, y, z)))
        s = self._bilinear(self.surface, x, y)
        t1 = s - self._bilinear(self.mp, x, y)
        t2 = s - self._bilinear(self.bedrock, x, y)
        sm = self.mesh_surface(x, y)
        depth = np.maximum(sm - z, 0.)
        elev = np.minimum(z, sm) + self.z0
        out = [np.full(x.shape, np.nan) for _ in range(3)]
        sed1, sed2 = depth < t1, (depth >= t1) & (depth < t2)
        for sel, lay, frac in ((sed1, self.layers[0], depth / np.maximum(t1, 1e-9)),
                               (sed2, self.layers[1], (depth - t1) / np.maximum(t2 - t1, 1e-9))):
            for o, p in zip(out, ("vp", "vs", "rho")):
                a, b = lay[p]
                o[sel] = (a + (b - a) * frac)[sel]
        rock = ~(sed1 | sed2)
        for lay in self.layers[2:]:
            top = 1e9 if not lay["top"].startswith("z-plane") else float(lay["top"].split()[1])
            bottom = float(lay["bottom"].split()[1])
            sel = rock & (elev <= top) & (elev > bottom)
            for o, p in zip(out, ("vp", "vs", "rho")):
                o[sel] = lay[p][0]
        assert not np.isnan(out[0]).any(), "points below the model"
        return tuple(out)


# Synthetic earthquakes added for the inversion tests (not in the archive): (x, y, elevation, strike), the
# archive's mechanism otherwise
EXTRA_SOURCES = [(-1500., 1200., -3000., 200.), (1600., -1000., -2500., 320.), (1200., 1500., -4000., 80.)]


def config_3d(workdir, npz, fmax=0.5, half=3000., depth=6000., sim_time=None, station_step=500.,
              nsources=1, nprocs=4, grid_h=None, relief=True):
    """Box of 2 half around the epicentre, from the meshed surface down to the elevation z0 - depth.
    Elements of h = 125 m / fmax horizontally (about one S wavelength at the surface, at the slowest Vs of the sediments);
    vertically 4 elements of h over the sediments (500 m below the flat surface, the model files), then
    constant crustal layers (the first two down to their bottoms in velocity_profile.txt, the third to the bottom of the box) in elements of about 4 h. The model files cover the
    first 500 m on a grid of grid_h (default h / 2.5) horizontally and h / 10 vertically, up to the highest
    point of the meshed surface. dt for a Courant number of 0.5 at the largest Vp of each layer.
    relief=False: flat free surface at z = 0 (same sediments below it).
    nsources: the archive's earthquake, then nsources - 1 of EXTRA_SOURCES.
    Stations every station_step on the surface, plus the vertical array of the archive (0, 10, 100, 1000 m
    below the epicentre). Source: the archive's earthquake (position, mechanism, Mw), with the analytic I2
    time function of e2vp (its stf.asc has an unknown unit), which peaks at 3 / fmax s: sim_time defaults
    to 3 / fmax + 6 s (S waves at the surface about 2 s after the origin, then the basin response)."""
    basin = Basin(npz, half, relief=relief)
    topo = os.path.join(workdir, "topography.npz")
    os.makedirs(workdir, exist_ok=True)
    basin.write_topography(topo)
    h = 125. / fmax
    n_top = 4
    sed = n_top * h
    crust = [l for l in basin.layers[2:]]
    zb = [float(l["bottom"].split()[1]) - basin.z0 for l in crust[:2]] + [-depth]  # bottoms (flat mesh z)
    layers, props, top = [(h, 1)] * n_top, [None] * n_top, -sed
    for lay, bottom in zip(crust, zb):
        th = top - bottom
        layers.append((th, max(1, int(np.ceil(th / (4 * h))))))
        props.append(e2vp.lame(lay["vp"][0], lay["vs"][0], lay["rho"][0]))
        top = bottom
    gh = grid_h or h / 2.5
    hz = h / 10
    zmax = float(np.load(topo)["z"].max())
    nz_below, nz_above = int(round(sed / hz)), int(np.ceil(zmax / hz)) + 1
    grid = Grid(xmin=(-half, -half, -sed), h=(gh, gh, hz),
                n=(int(round(2 * half / gh)) + 1,) * 2 + (nz_below + nz_above + 1,))
    # Courant: smallest GLL spacing (ngll = 5) 0.1727 h_vertical, largest Vp of the layer
    vmax = [max(l["vp"]) for l in basin.layers[:2]] + [l["vp"][0] for l in crust]
    dts = [0.5 * 0.1727 * min(h, layers[0][0]) / max(vmax[2:3] + vmax[:2])]  # top layers: sediments or rock
    for (th, n), v in zip(layers[n_top:], vmax[2:]):
        dts.append(0.5 * 0.1727 * min(h, th / n) / v)
    dt = min(dts)
    sim_time = sim_time or 3. / fmax + 6.
    period = max(1, int(round(0.02 / dt)))  # strains every ~0.02 s
    nt = int(round(sim_time / dt / period)) * period
    src = read_source()
    m0 = 10 ** (1.5 * src["mw"] + 9.1)
    t, f = e2vp.time_function("I2", fmax)
    srcs = [sem_io.FileSource((x, y, src["z"] - basin.z0 if z is None else z - basin.z0), t, f,
                              moment=e2vp.moment_tensor(strike, src["dip"], src["rake"], m0))
            for x, y, z, strike in [(0., 0., None, src["strike"])] + EXTRA_SOURCES[:nsources - 1]]
    xs = np.arange(-half + station_step, half - station_step + 1e-9, station_step)
    pts = [(a, b, 0.) for a in xs for b in xs] + [(0., 0., -d) for d in (10., 100., 1000.)]
    cfg = Config(workdir=workdir, L=2 * half, x0=-half, depth=depth, h_elem=h, ngll=5, layers=layers,
                 layer_props=props, npml=2, sim_time=nt * dt, dt=dt, courant=1.0, grid=grid,
                 stations=[], sources=srcs, nprocs=nprocs, workers=1, strain_period=period,
                 adjoint="points", mesh_hex8=True, topography=topo if relief else None, topo_depth=1500.)
    cfg.stations = hexmesh.on_surface(cfg, pts)
    vp, vs, rho = basin.properties(*cfg.grid.mesh())
    cfg.lam0, cfg.mu0, cfg.rho0 = (float(np.median(a)) for a in e2vp.lame(vp, vs, rho))
    return cfg, basin
