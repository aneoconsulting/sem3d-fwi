"""Execution of the per-source evaluations (task.run_source).

A backend has a method run(jobs) that returns, in job order, the run_source dictionaries.
LocalBackend runs them on the local machine; armonik_backend.ArmoniKBackend submits them to
ArmoniK, one task per job.
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

    def __init__(self, cfg, runner):
        self.cfg, self.runner = cfg, runner

    def _run(self, job):
        return task.run_source(self.cfg, self.runner, job.fwd, job.adj, job.dobs, job.grad)

    def run(self, jobs):
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            return list(ex.map(self._run, jobs))
