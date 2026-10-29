"""Water-table case (ours, not in Fathi et al. 2015): the three layers of Fathi's case 3 for mu
(80, 101.25, 125 MPa above -12 m, above -27 m and below), lambda = mu above a water table at -10 m and
5 mu below it (Poisson's ratio 0.25 above, 0.42 below, a nearly saturated soil), rho = 2000 kg/m3.
The interfaces of lambda (-10, -12, -27 m) no longer coincide with those of mu (-12, -27 m), which
the bias of the lambda direction towards the mu direction cannot represent. Same geometry, load,
receivers and pulses as Fathi's cases 2 to 4.
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
