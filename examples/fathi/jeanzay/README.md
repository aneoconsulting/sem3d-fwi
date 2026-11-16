# sem3d-fwi on Jean Zay

The CPU allocation (100 000 h) ends on **1 December 2026**. Before running inversions there, we measure what one
costs, without assuming linear scaling: the scalability test below takes about one hour of wall time and
costs at most 80 core-hours (2 jobs, 1 h limit, 40 cores), if an exclusive node is charged 40 cores. Everything marked CHANGE_ME, and the Jean Zay details flagged
"to check", must be confirmed on site.

## 1. Build SEM3D (branch `adjoint`)

The inversion needs the `adjoint` branch of the fork (sensitivity kernels computed in SEM3D, PML kernels):
`https://github.com/aneojgurhem/SEM`, branch `adjoint`, commit `e3c8898a` or later. Do not use the upstream
SEM3D: it lacks these kernels.

```bash
source examples/fathi/jeanzay/env.sh
git clone -b adjoint https://github.com/aneojgurhem/SEM $WORK/SEM   # or copy it with scp/rsync
cmake -S $WORK/SEM -B $WORK/build/sem3d-adjoint -DCMAKE_BUILD_TYPE=Release -DOPT_VEC=8 -DOPT_MIN_GLL_OPT=3 \
      -DCMAKE_Fortran_FLAGS=-march=cascadelake -DCMAKE_C_FLAGS=-march=cascadelake -DOPT_MPI=ON \
      -DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_C_COMPILER=mpicc -DCMAKE_INSTALL_PREFIX=$WORK/install/sem3d-adjoint
cmake --build $WORK/build/sem3d-adjoint -j 4 && cmake --install $WORK/build/sem3d-adjoint
```

Options as on the reference machine (see the main README and `docker/Dockerfile`): Release,
`OPT_VEC=8`, `OPT_MIN_GLL_OPT=3` (kernels specialized for ngll = 3; with the default 4, case 3 runs the
generic kernels), the native architecture (`-mcpu=native` on the Graviton, `-march=cascadelake` on the
Jean Zay CPU nodes, Xeon Gold 6248), MPI on, everything else off (no CPML, no OpenACC, double
precision). HDF5 must be built with the same MPI. `-O3 -march=cascadelake` alone (generic kernels)
gives nothing over RelWithDebInfo: 557 s against 542 s for one case-3 iteration on 1 rank.

Building on the login node takes about 6 minutes with `-j 4` and is allowed; the `compil` partition can
queue for hours.

This ngll = 3 build is the default binary (`SEM3D_BIN` in `env.sh`). One case-3 iteration, alone on a
node: 245 s on 1 rank, 35 s on 20, 30 s on 40, against 542, 56 and 44 s with RelWithDebInfo; J identical
to the bit. The option that selects the ngll = 3 kernels is `-DOPT_MIN_GLL_OPT=3`.

An install shared by the members of a project goes in `$WORK_SHARED` (group-writable), built the same
way on the login node; `env.sh` then points `SEMFWI_SRC`, `SEM3D_BIN` and the venv there:

```bash
D=$WORK_SHARED            # or a subdirectory of it
module purge && module load gcc/12.2.0 openmpi/4.1.5 hdf5/1.12.0-mpi cmake python/3.11.5
cmake -S $D/SEM -B $D/build/sem3d-adjoint-ngll3 \
      -DCMAKE_BUILD_TYPE=Release -DOPT_VEC=8 -DOPT_MIN_GLL_OPT=3 \
      -DCMAKE_Fortran_FLAGS=-march=cascadelake -DCMAKE_C_FLAGS=-march=cascadelake -DOPT_MPI=ON \
      -DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_C_COMPILER=mpicc \
      -DCMAKE_INSTALL_PREFIX=$D/install/sem3d-adjoint-ngll3
cmake --build $D/build/sem3d-adjoint-ngll3 -j 4 && cmake --install $D/build/sem3d-adjoint-ngll3
```

## 2. Python

```bash
python -m venv $WORK/venvs/semfwi && source $WORK/venvs/semfwi/bin/activate
pip install numpy h5py pytest matplotlib   # matplotlib: campaign_report.py
cd $SEMFWI_SRC && SEMFWI_MPIRUN="srun -n {n}" pytest -q tests --basetemp=$SCRATCH/semfwi-pytest
# 24 tests; run inside a short allocation. --basetemp: /tmp of the compute nodes is too small
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
An iteration whose first trial step is accepted costs exactly this. Measured in our case-3 runs: 1.3 to
1.6 evaluations per iteration on average while the inversion progresses, 2 to 4 when it stalls.

The two jobs answer two different questions:

- `strong`: how fast one inversion goes with more ranks. The efficiency is expected to drop well
  before 40 ranks (small mesh, MPI exchanges, strains written to disk).
- `pack`: what a full node delivers when it runs independent inversions. The inversions of a campaign
  are independent, so the number of ranks per run should be the one with the best throughput per
  node, not the fastest single run. Running 40/n copies at once exposes the shared memory bandwidth
  and the shared file system (each evaluation writes about 650 MB of displacements to $SCRATCH, then deletes them).

Read in the table: `core-h/it` per number of ranks, for `strong` and `pack`. The campaign budget is
`core-h/it` x iterations x runs, with the `pack` value at the chosen number of ranks. `J` must be
identical for every measurement (it does not depend on the ranks); otherwise stop and report.

Reference point, measured on our machine (AWS Graviton, 4 cores, nothing else running), one evaluation:

| ranks | wall time | speed-up | efficiency | memory of SEM3D (all ranks) |
|---|---|---|---|---|
| 1 | 117 s | 1.00 | 1.00 | 1.0 GB |
| 2 | 68 s | 1.73 | 0.86 | 1.1 GB |
| 4 | 47 s | 2.49 | 0.62 | 1.1 GB |

Scaling is already sub-linear at 4 ranks. The mesher and the Python part take about 3 s in total; the
rest is the two SEM3D runs. 40 runs of 1 rank on a node need about 40 GB of memory.

## 4. Campaigns

One job = one node = 10 inversions of 4 ranks run at once (the scalability test: 4 ranks per run,
10 runs per node, is a good compromise between throughput and the duration of one inversion), with
a 20 h limit. The job is not exclusive: `--ntasks` on the `sbatch` line reserves fewer cores (long runs in
`qos_cpu-t4`, up to 100 h), and `--nprocs` in a line gives that run other ranks (a run resumed with
other ranks gets its own mesh; same J). A campaign file has one inversion per line: its tag, then the arguments of
`run_fathi2.py both`. The runs use `--auto-stage 30 0.05`: next frequency as soon as J gains less than 5 % over
30 iterations, the unused iterations carried over (same total budget as the paper's stages). Use the SEM3D binary of section 1 (optimized for ngll = 3: 2.2 times faster, same J to the bit).

```bash
sbatch examples/fathi/jeanzay/campaign.slurm examples/fathi/jeanzay/campaigns/case3_start.txt   # starting models
sbatch examples/fathi/jeanzay/campaign.slurm examples/fathi/jeanzay/campaigns/case4.txt         # case 4
sbatch examples/fathi/jeanzay/campaign.slurm examples/fathi/jeanzay/campaigns/case3_acq.txt     # acquisition
cd examples/fathi && python campaign_report.py jeanzay/campaigns/*.txt   # at any time, finished or not
```

The report writes, in `$SEMFWI_RUNS/report`, one table (`.txt`) and one figure (`.png`) per campaign
file. A run stopped by the time limit resumes from its last iteration when the same file is submitted
again. Logs: `$SEMFWI_RUNS/logs/<tag>.log`. The job deletes old iteration directories as it goes
(about 200 MB each; the start of each frequency and the last 5 iterations are kept).

## 5. GPU (H100)

SEM3D built with OpenACC, branch `adjoint` at `2221ee68` or later (strain output and kernels on the
GPU, sources and stations without per-step transfers, a race in the solid PMLs fixed). On the login
node, about 10 minutes:

```bash
module purge
for m in arch/h100 nvidia-compilers/25.11 cuda/12.8.0 openmpi/4.1.8-cuda hdf5/1.14.5-mpi-cuda cmake; do
  module load $m   # one at a time: on one line, a failed prerequisite unloads them all
done
cmake -S $WORK/SEM -B $WORK/build/sem3d-acc -DCMAKE_BUILD_TYPE=Release -DOPT_OPENACC=ON -DOPT_MPI=ON \
      -DOPT_VEC=32 -DOPT_MIN_GLL_OPT=3 -DCMAKE_Fortran_FLAGS=-gpu=cc90 \
      -DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
      -DCMAKE_INSTALL_PREFIX=$WORK/install/sem3d-acc
cmake --build $WORK/build/sem3d-acc -j 6 && cmake --install $WORK/build/sem3d-acc
```

`OPT_VEC` > 1 is required by the OpenACC kernels (with 1 they do not compile). The Python of the CPU
nodes is not on the H100 nodes: a venv of its own, from the H100 Python
(`module purge; module load arch/h100 python/3.11.5; python -m venv $WORK/venvs/semfwi-h100`, then
`module purge` and `pip install numpy h5py pytest matplotlib`). `env_gpu.sh` loads the modules above
and this venv. SEM3D puts rank r of a node on its GPU r: a run of n ranks needs n GPUs.

Validation against the CPU build (`export SBATCH_ACCOUNT=<project>@h100`, 2 GPUs, a few minutes):

```bash
SEM3D_BIN=$WORK/install/sem3d-acc/bin sbatch examples/fathi/jeanzay/gpu_check.slurm <label>
```

It runs the tests, then one evaluation of case 3 with `scaling.py`, and compares its traces and
gradient with those of a CPU run of `scaling.py` (default `ngll3_pack_c10_p20_n1_r0`, from section 3).
Measured (November 2026), one evaluation of case 3 on 1 H100: 34 s (forward 16 s, adjoint 13 s, of
which about 5 to 8 s of initialization and 3 s of I/O on the CPU per run), 0.0094 GPU-hour; J
identical to the CPU in all its digits, traces within 3e-15, gradient within 4e-13 (PML kernels
included). For reference, 472 s on 1 CPU core, about 30 s on 40. Two tests fail on GPU:
`test_kernels_in_sem_match_python` and `test_worker_compute_matches_local` require two runs to be
identical to the bit, and the GPU assembly sums with atomics in an order that changes from one run
to the next (differences of 1e-15).

## Checked on Jean Zay (November 2026)

- the account is not in the repository: `export SBATCH_ACCOUNT=<project>@cpu` before `sbatch`. Partition `cpu_p1`. CPU QoS: `qos_cpu-dev` (2 h, 10 jobs submitted), `qos_cpu-t3`
  (20 h), `qos_cpu-t4` (4 days, 1280 cores). An exclusive node is charged 40 cores (`billing=40`).
- modules: `gcc/12.2.0 openmpi/4.1.5 hdf5/1.12.0-mpi cmake python/3.11.5` (in `env.sh`). The only
  HDF5 1.14 (`hdf5/1.14.5-mpi`) requires Intel MPI.
- git reaches GitHub over https from the login nodes.
- `srun --exact` shares a node between concurrent job steps (scaling_pack ran as expected).
- the CPU nodes have 2 x 20 cores with SMT (80 logical CPUs); the scripts use one rank per physical core.
- H100: `--constraint=h100`, account `<project>@h100`, QoS `qos_gpu_h100-dev`, `-t3`, `-t4`; 4 H100
  (80 GB) and 96 cores per node (`gpu_p6`), 24 cores per GPU.
