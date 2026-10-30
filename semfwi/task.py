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


class TaskError(RuntimeError):
    """Evaluation failure not caused by the forward simulation (adjoint, kernels, configuration,
    infrastructure): a smaller step in the line search would not help."""


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
    Return dict(u = traces (nsta, nt, 3)) and, if grad, gl and gm (dJ/dlambda, dJ/dmu on the grid).
    A forward simulation failure propagates as is (SimulationError), any later failure as
    TaskError; the strains are deleted in all cases."""
    try:
        runner.run(dfwd)
        u = sem_io.read_traces(dfwd, len(cfg.stations))
        if not grad:
            return dict(u=u)
        try:
            gl, gm = _adjoint(cfg, runner, dfwd, dadj, dobs, u)
        except Exception as e:
            raise TaskError(f"after the forward simulation of {dfwd}: {type(e).__name__}: {e}") from e
        return dict(u=u, gl=gl, gm=gm)
    finally:
        for d in (dfwd, dadj):
            shutil.rmtree(f"{d}/strain", ignore_errors=True)


def _adjoint(cfg, runner, dfwd, dadj, dobs, u):
    """Adjoint simulation and kernels projected on the grid: (gl, gm)."""
    prepare_adjoint(dadj, dfwd)
    tau, hist = misfit.adjoint_history(misfit.residual(u, dobs, cfg.weights), cfg.dt)
    srcs = sem_io.write_adjoint_sources(dadj, cfg, tau, hist)
    if any(fn.startswith("disp.") for fn in os.listdir(f"{dfwd}/strain")):
        # the forward simulation wrote displacements (strain_mode = 1): SEM3D computes the kernels
        # the adjoint traces are not used: not recorded
        sem_io.write_input(dadj, cfg, srcs, save_strain=True, strain_mode=2, traces=False)
        link = f"{dadj}/fwd_strain"
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.realpath(f"{dfwd}/strain"), link)
        runner.run(dadj)
        return gradient.kernels_from_sem(cfg.grid, dadj)
    sem_io.write_input(dadj, cfg, srcs, save_strain=True)
    runner.run(dadj)
    jobs = [(cfg.grid, dfwd, dadj, cfg.dt, 3e7, [r]) for r in range(cfg.nprocs)]
    gl = np.zeros(cfg.grid.shape)
    gm = np.zeros(cfg.grid.shape)
    with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as ex:
        for kl, km in ex.map(_kernels, jobs):
            gl += kl
            gm += km
    return gl, gm
