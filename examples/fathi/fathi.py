"""Test cases of Fathi, Kallivokas and Poursartip (2015), section 4: smooth profile
lambda(z) = mu(z) = 80 + 0.45 |z| + 35 exp(-(|z| - 22.5)^2 / 150) MPa (cases 1 and 2, sections 4.1
and 4.2), layered profile lambda(z) = mu(z) = 80, 101.25, 125 MPa above -12 m, above -27 m and below
(case 3, section 4.3, eq. 30), the same with an ellipsoidal inclusion of 156.8 MPa
((x - 7.5) / 7.5)^2 + (y / 5)^2 + ((z + 12) / 5.5)^2 <= 1 (case 4, section 4.4, eq. 31),
rho = 2000 kg/m3, uniform vertical load of 1 kPa on the surface,
Gaussian pulses of table 1 f(t) = exp(-(t - mu)^2 / sigma) (sigma is a variance: see the report,
typo in the paper).
"""
import numpy as np

from semfwi import sem_io
from semfwi.config import Config, Grid

RHO = 2000.
PULSES = {"p20": (0.11, 0.0014), "p30": (0.08, 0.0007), "p40": (0.06, 0.0004)}


def profile(z):
    """lambda = mu in Pa at depth z <= 0."""
    a = np.abs(z)
    return 1e6 * (80. + 0.45 * a + 35. * np.exp(-(a - 22.5) ** 2 / 150.))


def layered(z):
    """lambda = mu in Pa at depth z <= 0 (case 3)."""
    return 1e6 * np.where(z >= -12., 80., np.where(z >= -27., 101.25, 125.))


def pulse(name):
    m, s = PULSES[name]
    return lambda t: np.exp(-(t - m) ** 2 / s)


def inclusion(x, y, z):
    """Inside the ellipsoidal inclusion of case 4."""
    return ((x - 7.5) / 7.5) ** 2 + (y / 5.) ** 2 + ((z + 12.) / 5.5) ** 2 <= 1.


def true_model(grid, case=2):
    X, Y, Z = grid.mesh()
    p = layered(Z) if case in (3, 4) else profile(Z)
    if case == 4:
        p[inclusion(X, Y, Z)] = 156.8e6
    return p.copy(), p.copy(), np.full(grid.shape, RHO)


def homogeneous(grid, value=80e6):
    one = np.ones(grid.shape)
    return value * one, value * one, RHO * one


def make_config(workdir, case, pulse_name="p20", fine=False, nprocs=4, sim_time=None):
    """case 1: 24 x 24 x 45 m, 1 m elements (0.5 m for the data), 5 m PML.
    cases 2 to 4: 40 x 40 x 45 m, 1.25 m elements, 6.25 m PML. Quadratic elements (ngll = 3),
    T = 0.5 s unless sim_time is given. Deviation from the paper: SEM3D's Newmark scheme diverges
    at Fathi's time steps (Runge-Kutta 4: 9e-4 and 1e-3 s); we use 4.5e-4 s (case 1, 2.25e-4 s for
    the fine data) and 6.25e-4 s (cases 2 to 4), i.e. a Courant number of about 0.4."""
    if case == 1:
        half, depth, zone = 12., 45., 11.
        h = 0.5 if fine else 1.
        dt = 2.25e-4 if fine else 4.5e-4
        sim_time = 1112 * 4.5e-4        # 1112 coarse steps, 2224 fine steps
        gh = 0.25 if fine else 0.5      # parameter grid: quadratic nodes
        period = 4
    else:
        half, depth, zone = 20., 45., 17.5
        h, dt, gh, period = 1.25, 6.25e-4, 0.625, 8
        sim_time = sim_time or 0.5
    n = int(round(2 * half / gh)) + 1
    nz = int(round(depth / gh)) + 1
    grid = Grid(xmin=(-half, -half, -depth), h=gh, n=(n, n, nz))
    nl = int(round(depth / h))
    xs = np.arange(-zone, zone + 1e-9, 0.5 * h if not fine else 0.5 * 2 * h)  # coarse mesh nodes
    if case != 1:
        xs = np.arange(-zone, zone + 1e-9, 0.625)
    stations = [(x, y, 0.) for x in xs for y in xs]
    load = sem_io.SurfaceLoad(-zone, zone, h, 3, 1000., pulse(pulse_name))
    return Config(workdir=workdir, L=2 * half, x0=-half, depth=depth, h_elem=h, ngll=3,
                  layers=[(h, 1)] * nl, npml=int(round(5. / h)) if case == 1 else 5,
                  sim_time=sim_time, dt=dt, courant=1.0, grid=grid, lam0=80e6, mu0=80e6, rho0=RHO,
                  stations=stations, sources=[load], nprocs=nprocs, workers=1, strain_period=period,
                  adjoint="extended", receiver_grid=(xs, xs, 0.))
