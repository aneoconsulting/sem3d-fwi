"""Water-table case (ours, not in Fathi et al. 2015): the three layers of Fathi's case 3 for mu
(80, 101.25, 125 MPa above -12 m, above -27 m and below), lambda = mu above a water table at -10 m and
5 mu below it (Poisson's ratio 0.25 above, 0.42 below, a nearly saturated soil), rho = 2000 kg/m3.
The interfaces of lambda (-10, -12, -27 m) no longer coincide with those of mu (-12, -27 m), which
the bias of the lambda direction towards the mu direction cannot represent. Same geometry, load,
receivers and pulses as Fathi's cases 2 to 4.

lambda is 5 mu over the whole depth below -10 m: starting from the homogeneous 80 MPa model, the
inversion must first rebuild this background before the interface at -10 m matters. Two variants
separate the two questions: lens_model, a saturated lens (lambda = 3 mu between -16 and -10 m only),
and smoothed_start, the true model smoothed along z as the initial model.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fathi"))
from fathi import RHO, layered  # noqa: E402

WATER_TABLE = -10.
LAMBDA_FACTOR = 5.


def true_model(grid):
    Z = grid.mesh()[2]
    mu = layered(Z)
    lam = mu * np.where(Z < WATER_TABLE, LAMBDA_FACTOR, 1.)
    return lam, mu.copy(), np.full(grid.shape, RHO)


LENS = (-16., -10.)
LENS_FACTOR = 3.


def lens_model(grid):
    """mu of case 3, lambda = 3 mu in the saturated lens -16 m <= z < -10 m and lambda = mu elsewhere."""
    Z = grid.mesh()[2]
    mu = layered(Z)
    lam = mu * np.where((Z >= LENS[0]) & (Z < LENS[1]), LENS_FACTOR, 1.)
    return lam, mu.copy(), np.full(grid.shape, RHO)


def smoothed_start(grid, sigma=5.):
    """True model of the water-table case smoothed along z by a Gaussian of standard deviation sigma (m),
    edges extended: the background of lambda is right, its interface at -10 m and those of mu are not."""
    lam, mu, rho = true_model(grid)
    h = grid.h
    k = np.arange(-int(3 * sigma / h), int(3 * sigma / h) + 1) * h
    w = np.exp(-k ** 2 / (2 * sigma ** 2))
    w /= w.sum()
    n = len(k) // 2

    def smooth(v):
        p = np.concatenate([np.repeat(v[:, :, :1], n, axis=2), v, np.repeat(v[:, :, -1:], n, axis=2)], axis=2)
        return sum(wi * p[:, :, i:i + v.shape[2]] for i, wi in enumerate(w))

    return smooth(lam), smooth(mu), rho
