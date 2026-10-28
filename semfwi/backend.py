"""Execution of the per-source evaluations (task.run_source).

A backend has a method run(cfg, runner, jobs) that returns, in job order, the run_source
dictionaries; cfg and runner are those of the Problem submitting the jobs. LocalBackend runs
them on the local machine; armonik_backend.ArmoniKBackend submits them to ArmoniK, one task per job.
A forward simulation failure raises runner.SimulationError, any other failure task.TaskError.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import task


@dataclass
class Job:
    index: int          # source index
    fwd: str            # ready forward directory
    adj: str            # adjoint directory (created by run_source)
    dobs: np.ndarray    # observed data of the source, (nsta, nt, 3)
    grad: bool


class LocalBackend:
    """cfg.workers sources in parallel."""

    def run(self, cfg, runner, jobs):
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            return list(ex.map(lambda j: task.run_source(cfg, runner, j.fwd, j.adj, j.dobs, j.grad), jobs))
