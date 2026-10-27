"""End-to-end evaluation of one source: forward simulation, residual, adjoint simulation, kernels.

This is the unit of work of the backends (local or ArmoniK). Only the traces and the kernels
projected on the parameter grid come out: the strains (several GB per simulation) stay in the
run directories and are deleted at the end.
"""
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import gradient, misfit, sem_io

# Model files the adjoint simulation takes over from the forward simulation
MODEL_FILES = ("Lambda.h5", "Mu.h5", "Rho.h5", "material.spec", "material.input")


def _kernels(args):
    return gradient.kernels(*args)


def prepare_adjoint(dadj, dfwd):
    """Adjoint directory: same mesh and same model (PMLs included) as the forward simulation."""
    os.makedirs(dadj, exist_ok=True)
    if not os.path.lexists(f"{dadj}/sem"):
        os.symlink(os.path.realpath(f"{dfwd}/sem"), f"{dadj}/sem")
    for fn in MODEL_FILES:
        if os.path.exists(f"{dfwd}/{fn}"):
            shutil.copy(f"{dfwd}/{fn}", dadj)


def run_source(cfg, runner, dfwd, dadj, dobs, grad):
    """dfwd: ready forward directory (mesh, model, input.spec with save_strain if grad).
    Return dict(u = traces (nsta, nt, 3)) and, if grad, gl and gm (dJ/dlambda, dJ/dmu on the grid)."""
    runner.run(dfwd)
    u = sem_io.read_traces(dfwd, len(cfg.stations))
    if not grad:
        return dict(u=u)
    prepare_adjoint(dadj, dfwd)
    tau, hist = misfit.adjoint_history(misfit.residual(u, dobs), cfg.dt)
    srcs = sem_io.write_adjoint_sources(dadj, cfg, tau, hist)
    sem_io.write_input(dadj, cfg, srcs, save_strain=True)
    runner.run(dadj)
    jobs = [(cfg.grid, dfwd, dadj, cfg.dt, 3e7, [r]) for r in range(cfg.nprocs)]
    gl = np.zeros(cfg.grid.shape)
    gm = np.zeros(cfg.grid.shape)
    with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as ex:
        for kl, km in ex.map(_kernels, jobs):
            gl += kl
            gm += km
    for d in (dfwd, dadj):
        shutil.rmtree(f"{d}/strain", ignore_errors=True)
    return dict(u=u, gl=gl, gm=gm)
