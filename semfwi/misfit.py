"""L2 cost function and adjoint sources.

Conventions (plan-inversion-sem3d.md, section 1.3):
  - trace k <-> field at time (k+1) dt;
  - J = 1/2 sum_r sum_k |u_k - d_k|^2 dt (rectangle rule);
  - optional weights w (nsta, 3), one per station and component (0 removes a trace):
    J = 1/2 sum_r sum_k sum_c w_rc (u_k - d_k)_c^2 dt, and the residual is w (u - d);
  - adjoint history at times tau_k = (k + 1/2) dt, value res[N-1-k]: SEM3D evaluates the
    sources at t + dt/2 and therefore lands exactly on the file's samples.
"""
import numpy as np


def cost(u, dobs, dt, w=None):
    w = 1. if w is None else np.asarray(w)[..., None, :]
    return 0.5 * np.sum(w * (u - dobs) ** 2) * dt


def residual(u, dobs, w=None):
    return u - dobs if w is None else np.asarray(w)[:, None, :] * (u - dobs)


def adjoint_history(res, dt):
    """res: (nsta, nt, 3). Returns tau (nt) and the time-reversed history (nsta, nt, 3)."""
    nt = res.shape[1]
    tau = dt * (np.arange(nt) + 0.5)
    return tau, res[:, ::-1, :].copy()
