"""Sensitivity kernels from forward and adjoint strains (save_strain output, M2).

  K_lambda(x) = - int div u(t) div w(T - t) dt
  K_mu(x)     = - int 2 eps(u)(t) : eps(w)(T - t) dt
  dJ/dp_n     = sum_g omega_g |J_g| K(x_g) P_{g,n}   (P: trilinear interpolation)

Forward record k is correlated with adjoint record N-2-k (section 1.3 of the plan).
With output every p steps (strain_period, N a multiple of p), the M = N/p records are
correlated the same way (r <-> M-2-r) and the time integral uses the step p dt.
"""
import numpy as np

from . import model, sem_io


def kernels(grid, dfwd, dadj, dt, mem=2e8, ranks=None):
    """dJ/dlambda and dJ/dmu on the grid (nx, ny, nz) for one pair of simulations
    (contribution of only the rank files whose index is in ranks, if given)."""
    gl = np.zeros(grid.shape)
    gm = np.zeros(grid.shape)
    pairs = list(zip(sem_io.read_strain(dfwd), sem_io.read_strain(dadj)))
    if ranks is not None:
        pairs = [p for i, p in enumerate(pairs) if i in ranks]
    for (cf, wf, ef, pf), (ca, wa, ea, pa) in pairs:
        assert np.array_equal(cf, ca), "forward and adjoint meshes differ"
        assert ef.shape == ea.shape and pf == pa, (ef.shape, ea.shape, pf, pa)
        dtp = dt * pf
        n = ef.shape[0]
        chunk = max(1, int(mem / (len(wf) * 48)))  # reads about mem bytes per array
        Kl = np.zeros(len(wf))
        Km = np.zeros(len(wf))
        for k0 in range(0, n, chunk):
            k = np.arange(k0, min(n, k0 + chunk))
            kk = n - 2 - k
            ok = kk >= 0
            k, kk = k[ok], kk[ok]
            eu = np.asarray(ef[k])
            ew = np.asarray(ea[kk])
            Kl -= dtp * np.sum(eu[..., :3].sum(-1) * ew[..., :3].sum(-1), axis=0)
            # eps:eps = diagonal terms + 2 * off-diagonal terms
            Km -= dtp * 2 * np.sum(np.sum(eu[..., :3] * ew[..., :3], -1)
                                  + 2 * np.sum(eu[..., 3:] * ew[..., 3:], -1), axis=0)
        idx, w = model.trilinear(grid, cf)
        gl += model.project(grid, wf * Kl, idx, w)
        gm += model.project(grid, wf * Km, idx, w)
    return gl, gm
