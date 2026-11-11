"""EUROSEISTEST Verification and Validation Project (E2VP), Mygdonian basin (Greece): models, sources
and configurations. Values from Chaljub et al. (2015), 3-D numerical simulations of earthquake ground
motion in sedimentary basins: testing accuracy through stringent models, GJI 201, 90-111
(doi:10.1093/gji/ggu472):
  - table 1 (model with three homogeneous sedimentary layers, "sharp") and table 2 (piecewise linear,
    "smooth"), thicknesses at the central station TST, bedrock H below; elastic here (the paper uses
    Q_S = V_S / 10, no bulk attenuation);
  - section 3.1, 1-D cases: double couple at 3 km depth, vertical strike-slip (dip 90, rake 0),
    strike 22.5 degrees, M0 = 1e18 N.m, time function I2; vertical force at the free surface above it,
    amplitude 5e11 N, time function I1, the derivative of I2 (I1 is a filtered Dirac, flat from 0.3 to
    3 Hz, nothing above 4 Hz).
Nothing is read from the benchmark files: the time functions are built here (time_function), the same
below fmax as the benchmark's up to a scale factor.

Axes: x east, y north, z up, origin at the epicentre on the free surface.
"""
import numpy as np

from semfwi import model, sem_io
from semfwi.config import Config, Grid

# (thickness m, Vp m/s, Vs m/s, rho kg/m3) from the free surface down; then the bedrock (Vp, Vs, rho)
SHARP = [(17.3, 1500., 200., 2100.), (72.5, 1800., 350., 2100.), (115.6, 2500., 650., 2200.)]
BEDROCK = (4500., 2600., 2600.)
SOURCE_DEPTH = 3000.
M0 = 1e18
FORCE = 5e11
STRIKE, DIP, RAKE = 22.5, 90., 0.


def lame(vp, vs, rho):
    """(lambda, mu, rho) from (Vp, Vs, rho)."""
    return rho * (vp ** 2 - 2 * vs ** 2), rho * vs ** 2, rho


def sediment_bottom(layers=SHARP):
    return sum(l[0] for l in layers)


def moment_tensor(strike, dip, rake, m0):
    """Double couple in SEM3D's order (Mxx, Myy, Mzz, Mxy, Mxz, Myz), x east, y north, z up, from the
    Aki and Richards formulas (box 4.4: x north, y east, z down)."""
    s, d, r = np.radians([strike, dip, rake])
    mnn = -m0 * (np.sin(d) * np.cos(r) * np.sin(2 * s) + np.sin(2 * d) * np.sin(r) * np.sin(s) ** 2)
    mne = m0 * (np.sin(d) * np.cos(r) * np.cos(2 * s) + 0.5 * np.sin(2 * d) * np.sin(r) * np.sin(2 * s))
    mnd = -m0 * (np.cos(d) * np.cos(r) * np.cos(s) + np.cos(2 * d) * np.sin(r) * np.sin(s))
    mee = m0 * (np.sin(d) * np.cos(r) * np.sin(2 * s) - np.sin(2 * d) * np.sin(r) * np.cos(s) ** 2)
    med = -m0 * (np.cos(d) * np.cos(r) * np.sin(s) - np.cos(2 * d) * np.sin(r) * np.cos(s))
    mdd = m0 * np.sin(2 * d) * np.sin(r)
    # east = y_AR, north = x_AR, up = -z_AR
    return (mee, mnn, mdd, mne, -med, -mnd)


def time_function(name, fmax, dt=1e-4):
    """I1 (force) or I2 (double couple), (t, f): I1 is a Dirac low-passed by a cos^2 taper from 0.5 fmax to
    fmax (zero phase), centred at t0 = 3 / fmax s so that its spread before the peak stays after t = 0,
    scaled to a unit integral over its duration t0 + 6 / fmax, and I2 its integral, from 0 to 1. The
    benchmark's I1 is flat up to 3 Hz: below fmax <= 2 Hz the two have the same spectrum, up to the scale
    (its integral is 0.0113 instead of 1) and a delay. SEM3D holds the last value beyond the duration (I1
    back to 0, I2 at 1)."""
    t0 = 3. / fmax
    n = int(round((t0 + 6. / fmax) / dt))
    fr = np.fft.rfftfreq(2 * n, dt)
    taper = np.where(fr <= 0.5 * fmax, 1., np.where(fr >= fmax, 0., np.cos(np.pi * (fr / fmax - 0.5)) ** 2))
    t = dt * np.arange(n)
    f = np.fft.irfft(taper * np.exp(-2j * np.pi * fr * t0), 2 * n)[:n]
    f /= np.sum(0.5 * (f[1:] + f[:-1])) * dt
    if name == "I2":
        f = np.concatenate([[0.], np.cumsum(0.5 * (f[1:] + f[:-1]) * dt)])
    return t, f


def sh_transfer(freqs, layers=SHARP, bedrock=BEDROCK):
    """|surface / outcrop| for vertically incident SH waves through the elastic layers (Thomson-Haskell:
    up- and downgoing amplitudes equal at the free surface, continuity of displacement and traction at
    each interface; outcrop = twice the incident amplitude)."""
    w = 2 * np.pi * np.asarray(freqs, float)
    a, b = np.ones_like(w, complex), np.ones_like(w, complex)
    vs = [l[2] for l in layers] + [bedrock[1]]
    rho = [l[3] for l in layers] + [bedrock[2]]
    for m, (h, _, _, _) in enumerate(layers):
        e = np.exp(1j * w / vs[m] * h)
        alpha = rho[m] * vs[m] / (rho[m + 1] * vs[m + 1])
        a, b = 0.5 * a * (1 + alpha) * e + 0.5 * b * (1 - alpha) / e, 0.5 * a * (1 - alpha) * e + 0.5 * b * (1 + alpha) / e
    return 1 / np.abs(a)


def config_1d(workdir, fmax=1., half=2000., depth=4000., sim_time=8., exact=False, outcrop=False,
              sources=("dc", "force"), station_step=400., station_extent=1600., nprocs=4):
    """1-D case of Chaljub et al. (1D-sharp), reduced to a box of 2 half x 2 half x depth m for this machine.
    Spectral elements of degree 4 (ngll = 5) of h = 200 m / fmax horizontally (one element per S
    wavelength in L1); one element per sedimentary layer, so that the mesh follows the interfaces, and
    elements of about 500 m in the bedrock; 2 PML elements. dt = 6.25e-4 s: Courant number 0.31 in L1,
    whose 17.3 m elements set the step. Stations on the free surface every station_step m up to
    station_extent m from the epicentre.
    exact: every layer has constant properties (material.input), the true model exactly, no parameter
    grid needed (forward checks). Otherwise the sediments read the model files on a parameter grid of
    100 m horizontally whose z nodes are the 13 GLL depths of the sedimentary elements (inversion), the
    only depths where SEM3D samples the model, and only the bedrock is constant.
    outcrop: the bedrock up to the surface (the reference of the site response), exact, a single element
    in place of the three sedimentary layers (the bedrock in 17.3 m elements would need dt / 3)."""
    h, ngll = 200. / fmax, 5
    nbed = int(np.ceil((depth - sediment_bottom()) / 500.))
    layers = [(l[0], 1) for l in SHARP] + [(depth - sediment_bottom(), nbed)]
    rock = lame(*BEDROCK)
    if outcrop:
        exact, layers = True, [(sediment_bottom(), 1), layers[-1]]
    if exact:
        props = [rock if outcrop else lame(*l[1:]) for l in SHARP][:len(layers) - 1] + [rock]
        grid = Grid(xmin=(-half, -half, -sediment_bottom()), h=(2 * half, 2 * half, sediment_bottom()), n=(2, 2, 2))
    else:
        props = [None] * len(SHARP) + [rock]
        z = model.gll_axis(-np.cumsum([0.] + [l[0] for l in SHARP])[::-1], ngll)
        grid = Grid(xmin=(-half, -half), h=(100., 100.), n=(int(2 * half / 100) + 1,) * 2, z=z)
    xs = np.arange(-station_extent, station_extent + 1e-9, station_step)
    stations = [(x, y, 0.) for x in xs for y in xs]
    srcs = []
    if "dc" in sources:
        t, f = time_function("I2", fmax)
        srcs.append(sem_io.FileSource((0., 0., -SOURCE_DEPTH), t, f, moment=moment_tensor(STRIKE, DIP, RAKE, M0)))
    if "force" in sources:
        t, f = time_function("I1", fmax)
        srcs.append(sem_io.FileSource((0., 0., 0.), t, f, direction=(0., 0., -1.), amplitude=FORCE))
    dt = 6.25e-4
    period = 64  # strains every 0.04 s, 12 samples per period at 2 Hz
    nt = int(round(sim_time / dt / period)) * period
    lam0, mu0, rho0 = lame(*SHARP[1][1:])  # reference: L2, homogeneous sediments
    return Config(workdir=workdir, L=2 * half, x0=-half, depth=depth, h_elem=h, ngll=ngll, layers=layers,
                  layer_props=props, npml=2, sim_time=nt * dt, dt=dt, courant=1.0, grid=grid,
                  lam0=lam0, mu0=mu0, rho0=rho0, stations=stations, sources=srcs, nprocs=nprocs, workers=1,
                  strain_period=period, adjoint="points")


def true_model_1d(grid, layers=SHARP):
    """(lambda, mu, rho) of the sharp model at the grid nodes (the sediments; the bedrock below)."""
    Z = grid.mesh()[2]
    out = [np.zeros(grid.shape) for _ in range(3)]
    top = 0.
    for th, vp, vs, rho in layers:
        sel = (Z <= top + 1e-9) & (Z > top - th - 1e-9)
        for o, v in zip(out, lame(vp, vs, rho)):
            o[sel] = v
        top -= th
    for o, v in zip(out, lame(*BEDROCK)):
        o[Z <= top - 1e-9] = v
    return tuple(out)


def initial_1d(grid):
    """Homogeneous sediments with the properties of L2, the reference model of config_1d; rho is the true one."""
    lam, mu, _ = lame(*SHARP[1][1:])
    one = np.ones(grid.shape)
    return lam * one, mu * one, true_model_1d(grid)[2]
