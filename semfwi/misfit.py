"""L2 cost function and adjoint sources.

Conventions (plan-inversion-sem3d.md, section 1.3):
  - trace k <-> field at time (k+1) dt;
  - J = 1/2 sum_r sum_k |u_k - d_k|^2 dt (rectangle rule);
  - adjoint history at times tau_k = (k + 1/2) dt, value res[N-1-k]: SEM3D evaluates the
    sources at t + dt/2 and therefore lands exactly on the file's samples.
"""
import numpy as np


def cost(u, dobs, dt):
    return 0.5 * np.sum((u - dobs) ** 2) * dt


def residual(u, dobs):
    return u - dobs


def adjoint_history(res, dt):
    """res: (nsta, nt, 3). Returns tau (nt) and the time-reversed history (nsta, nt, 3)."""
    nt = res.shape[1]
    tau = dt * (np.arange(nt) + 0.5)
    return tau, res[:, ::-1, :].copy()
