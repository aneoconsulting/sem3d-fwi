# sem3d-fwi on Jean Zay

The CPU allocation (100 000 h) ends on **1 December 2026**. Before running inversions there, we measure what one
costs, without assuming linear scaling: the scalability test below takes about one hour of wall time and
should use under 100 core-hours. Everything marked CHANGE_ME, and the Jean Zay details flagged
"to check", must be confirmed on site.

## 1. Build SEM3D (branch `adjoint`)

The inversion needs the `adjoint` branch of the fork (sensitivity kernels computed in SEM3D, PML kernels):
`https://github.com/aneojgurhem/SEM`, branch `adjoint`, commit `e3c8898a` or later. Do not use the upstream
SEM3D: it lacks these kernels.

```bash
source examples/fathi/jeanzay/env.sh          # after fixing the module names
git clone -b adjoint https://github.com/aneojgurhem/SEM $WORK/SEM   # or scp it if outgoing git is blocked
cmake -S $WORK/SEM -B $WORK/build/sem3d-adjoint -DCMAKE_BUILD_TYPE=RelWithDebInfo -DOPT_MPI=ON \
      -DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_C_COMPILER=mpicc -DCMAKE_INSTALL_PREFIX=$WORK/install/sem3d-adjoint
cmake --build $WORK/build/sem3d-adjoint -j 8 && cmake --install $WORK/build/sem3d-adjoint
```

Options as on the reference machine (AWS Graviton, gfortran, OpenMPI, HDF5 1.14): RelWithDebInfo, MPI on,
everything else off (no CPML, no OpenACC, double precision). HDF5 must be built with the same MPI.

## 2. Python

```bash
python -m venv $WORK/venvs/semfwi && source $WORK/venvs/semfwi/bin/activate
pip install numpy h5py pytest
cd $SEMFWI_SRC && SEMFWI_MPIRUN="srun -n {n}" pytest -q tests   # 24 tests; run inside a short allocation
```

`SEMFWI_MPIRUN` sets how SEM3D is launched (`{n}` = ranks). The default, `mpirun --oversubscribe -n {n}`,
is for a workstation; under Slurm use `srun`.

## 3. Scalability test

From the root of the repository:

```bash
sbatch examples/fathi/jeanzay/scaling_strong.slurm   # one run at a time: 1, 2, 4, 8, 10, 20, 40 ranks, twice each
sbatch examples/fathi/jeanzay/scaling_pack.slurm     # 40/n runs of n ranks at once on the node
python examples/fathi/scaling.py --report            # table, once both jobs are done
```

What one measurement is: one inversion iteration of Fathi case 3 (40 x 40 x 45 m, 1.25 m elements,
ngll = 3, 5 PML elements, 3249 stations, 0.45 s): forward simulation, adjoint simulation, kernels.
An iteration whose first trial step is accepted costs exactly this; in our case-3 runs the average
is closer to 1.5 evaluations per iteration.

The two jobs answer two different questions:

- `strong`: how fast one inversion goes with more ranks. The efficiency is expected to drop well
  before 40 ranks (small mesh, MPI exchanges, strains written to disk).
- `pack`: what a full node delivers when it runs independent inversions. The inversions of a campaign
  are independent, so the number of ranks per run should be the one with the best throughput per
  node, not the fastest single run. Running 40/n copies at once exposes the shared memory bandwidth
  and the shared file system (each run writes several GB of displacements per iteration to $SCRATCH).

Read in the table: `core-h/it` per number of ranks, for `strong` and `pack`. The campaign budget is
`core-h/it` x iterations x runs, with the `pack` value at the chosen number of ranks. `J` must be
identical for every measurement (it does not depend on the ranks); otherwise stop and report.

Reference point (AWS Graviton, 4 cores, under load from another run, so pessimistic):
1 rank 174 s, 2 ranks 127 s per evaluation; with 4 ranks, unloaded, about 40 s.

## To check on Jean Zay

- account (`--account=...@cpu`), partition `cpu_p1`, QoS `qos_cpu-dev` (time limit and node count
  allowed), and how exclusive jobs are charged (all 40 cores of the node, presumably).
- module names in `env.sh` (compiler, MPI, parallel HDF5, CMake, Python >= 3.11).
- whether git can reach GitHub from the login nodes; otherwise copy the repositories with scp.
- `srun --exact` sharing a node between concurrent job steps (scaling_pack).
