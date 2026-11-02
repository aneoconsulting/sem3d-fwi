"""Cost of one inversion iteration of case 3 as a function of the number of MPI ranks: before
running the inversions on a cluster, we measure rather than assume linear scaling.

    python scaling.py --nprocs 1 2 4 8 10 20 40 [--pulse p20] [--reps 1] [--tag strong]
    python scaling.py --data-only            # observed data, before several copies run at once
    python scaling.py --report               # speed-up, efficiency, core-hours per iteration

One evaluation = what an iteration costs when the first trial step is accepted: forward simulation
(displacements saved), adjoint simulation (SEM3D accumulates the kernels), kernels projected on the
grid. Each (ranks, repetition) runs in its own directory, mesh included, so that nothing comes from
a cache. One JSON line per evaluation in <SEMFWI_RUNS>/fathi/case3/scaling/<tag>.jsonl: wall times
of the mesher, of each SEM3D run and of the rest (Python), and J (must not depend on the ranks).
The launcher is SEMFWI_MPIRUN (default "mpirun --oversubscribe -n {n}"; on Slurm "srun -n {n}").
"""
import argparse
import json
import os
import socket
import time

import numpy as np

from fathi import make_config
from run_fathi2 import SIM_TIME, fathi_case, observed
from semfwi.problem import Problem
from semfwi.runner import Runner


class TimedRunner(Runner):
    """Runner that records the wall time of the mesher and of each SEM3D run."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.times = {"mesh": 0., "sem3d": []}

    def mesh(self):
        t0 = time.time()
        super().mesh()
        self.times["mesh"] += time.time() - t0

    def run(self, d):
        t0 = time.time()
        r = super().run(d)
        self.times["sem3d"].append(time.time() - t0)
        return r


def report(root):
    """Table per (tag family, ranks): median wall time per iteration, speed-up and efficiency
    relative to 1 rank alone, core-hours per iteration and per inversion of 1112 iterations."""
    rows = {}
    for fn in sorted(os.listdir(root)):
        if fn.endswith(".jsonl"):
            for line in open(f"{root}/{fn}"):
                r = json.loads(line)
                rows.setdefault((r["tag"].split("_c")[0], r["nprocs"]), []).append(r)
    base = np.median([r["total"] for r in rows.get(("strong", 1), [{"total": np.nan}])])
    print(f"{'run':8} {'ranks':>5} {'runs':>4} {'wall s':>7} {'sem3d s':>7} {'speed-up':>8} {'effic.':>6} "
          f"{'core-h/it':>9} {'core-h/inv':>10}")
    for (tag, n), rs in sorted(rows.items()):
        t = float(np.median([r["total"] for r in rs]))
        s = float(np.median([sum(r["sem3d"]) for r in rs]))
        print(f"{tag:8} {n:5d} {len(rs):4d} {t:7.1f} {s:7.1f} {base / t:8.2f} {base / (n * t):6.2f} "
              f"{n * t / 3600:9.4f} {1112 * n * t / 3600:10.1f}")
    js = [r["J"] for rs in rows.values() for r in rs]
    if js:
        print(f"J: relative spread {(max(js) - min(js)) / max(js):.1e} (must be ~0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprocs", type=int, nargs="+", default=[4])
    ap.add_argument("--pulse", default="p20")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--tag", default="strong")
    ap.add_argument("--data-only", action="store_true")
    ap.add_argument("--report", action="store_true", help="table of the measurements already made")
    args = ap.parse_args()
    case = fathi_case(3)
    if args.report:
        return report(f"{case.root}/scaling")
    dobs = observed(args.pulse, case)
    if args.data_only:
        return
    root = f"{case.root}/scaling"
    os.makedirs(root, exist_ok=True)
    for n in args.nprocs:
        for rep in range(args.reps):
            cfg = make_config(f"{root}/{args.tag}_{args.pulse}_n{n}_r{rep}", case.geometry, args.pulse,
                              nprocs=n, sim_time=SIM_TIME[3][args.pulse])
            runner = TimedRunner(cfg)
            lam_t, mu_t, rho = case.true_model(cfg.grid)
            lam0, mu0, _ = case.initial(cfg.grid)
            pb = Problem(cfg, runner, dobs, lam0, mu0, rho, active=("a", "b"), fixed=(lam_t, mu_t),
                         pml="follow")
            t0 = time.time()
            J, _, g = pb.evaluate(np.zeros(pb.size), "it000")
            total = time.time() - t0
            t = runner.times
            r = dict(host=socket.gethostname(), tag=args.tag, pulse=args.pulse, nprocs=n, rep=rep,
                     mesh=t["mesh"], sem3d=t["sem3d"], python=total - t["mesh"] - sum(t["sem3d"]),
                     total=total, core_hours=n * total / 3600, J=float(J), gnorm=float(np.linalg.norm(g)))
            print(json.dumps(r), flush=True)
            with open(f"{root}/{args.tag}.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
