# sem3d-fwi

Adjoint-state full-waveform inversion (FWI) on top of the
[SEM3D](https://github.com/sem3d/SEM) solver. SEM3D runs the forward and adjoint
simulations; `semfwi` writes their inputs, computes the misfit function, the sensitivity
kernels and the gradient, and drives the optimization (L-BFGS, Armijo line search).

## Requirements

- Python ≥ 3.11 with numpy and h5py (matplotlib for the examples, pytest for the
  tests);
- SEM3D (`openacc-stable` branch) with:
  - #6: receiver location;
  - #7: imposed time step, `time_scheme { dt = ...; }` (merged);
  - #9: strain output, `save_strain` and `strain_period`;
  - kernels computed during the adjoint simulation (`strain_mode`, used by default) and
    faster ngll = 3 simulations (branch `adjoint` of `~/SEM`, not proposed upstream yet);
  - #8 (sources on a fluid-solid interface) is only needed for those cases;
- OpenMPI and HDF5 (the ones SEM3D was compiled with).

## Installation

```
pip install -e '.[examples,test]'
```

Environment variables:

| Variable      | Role                                                     | Default         |
|---------------|----------------------------------------------------------|-----------------|
| `SEM3D_BIN`   | directory of `sem3d.exe` and `mesher`, prepended to `PATH` | **required**  |
| `MPI_BIN`     | directory of `mpirun`, prepended to `PATH`               | `mpirun` from `PATH` |
| `MPI_LIB`     | MPI libraries, prepended to `LD_LIBRARY_PATH`            | —               |
| `HDF5_LIB`    | HDF5 libraries, prepended to `LD_LIBRARY_PATH`           | —               |
| `SEMFWI_RUNS` | root of the examples' run directories                    | `~/SEM/exp/runs` |
| `SEMFWI_KERNELS_IN_SEM` | `0`: kernels computed in Python from the strain files (SEM3D without `strain_mode`) | `1` |

`SEM3D_BIN` has no default: a `sem3d.exe` found in `PATH` might not have the strain
output. Without it, the simulations stop with a message and the test that runs SEM3D is
skipped. Example:

```
export SEM3D_BIN=~/install/sem3d-adjoint/bin
export MPI_BIN=/usr/lib64/openmpi/bin MPI_LIB=/usr/lib64/openmpi/lib
export HDF5_LIB=~/install/hdf5-1.14/lib
```

SEM3D build options for ngll = 3 (a factor 1.5 on the simulations, same results):
`-DCMAKE_BUILD_TYPE=Release -DOPT_VEC=8 -DOPT_MIN_GLL_OPT=3`, and `-mcpu=native` in the
Fortran flags on the machine that runs it. With the kernels in SEM3D, one evaluation of
case 3 of Fathi (J and gradient, 4 cores of a Graviton) takes 46 s instead of 155 s.

## `semfwi` package

| Module     | Role |
|------------|------|
| `config`   | case parameters: mesh, time, parameter grid, stations, sources |
| `model`    | parameter grid, Lambda/Mu/Rho files, trilinear interpolation |
| `sem_io`   | writing SEM3D inputs, reading traces, the log and strains |
| `runner`   | single mesh, launching simulations, cache, parallel execution |
| `misfit`   | residuals, misfit function (optional weights per station and component, `Config.weights`), adjoint source time histories |
| `gradient` | K_lambda and K_mu kernels and projection onto the parameter grid |
| `task`     | one source end to end: forward, residual, adjoint, kernels |
| `backend`  | source execution: `LocalBackend` (local machine) |
| `problem`  | J(x) and grad J(x) over all sources (`Problem.evaluate`) |
| `optim`    | L-BFGS with Armijo line search, checkpoint and restart |
| `hessian`  | Hessian-vector products from two gradients, point-spread functions, low-rank Hessian |
| `armonik_*`| ArmoniK backend (client), worker and the contract between them |

For each source, a gradient evaluation runs a forward simulation (traces and strains),
then an adjoint simulation whose sources are the time-reversed residuals; the kernels are
the correlation of the strains of the two simulations. Each source is an independent unit
of work (`task.run_source`) from which only the traces and the kernels on the grid come
out; the strains stay in its run directory. `Problem.evaluate(x, tag, grad=True)` keeps
the result of each source (`tag/s*/result.npz`) and only resubmits the sources whose
inputs have changed. Each line-search trial also computes the gradient (speculative
gradient): the accepted trial costs two simulations per source, a rejected trial two
instead of one.

## Examples

- `examples/synth`: small cube, two Gaussian anomalies (lambda and mu), 4 point
  forces, 25 stations.
  ```
  python examples/synth/run_synth.py [--param vel|lame] [--niter 15] [--taylor]
  python examples/synth/analyse_synth.py
  ```
- `examples/fathi`: test cases of Fathi, Kallivokas and Poursartip (2015,
  arXiv:1504.08340), sections 4.1 to 4.4.
  ```
  python examples/fathi/run_fathi1.py [--pulses p20 p40]       # gradient check
  python examples/fathi/run_fathi2.py mu|lambda|both [--stages p20:20 p30:10] [--bias 15]
  python examples/fathi/run_fathi2.py both --case 3 [--stages p20:310 p30:550 p40:252] [--bias 15]
  python examples/fathi/run_fathi2.py both --case 4 [--stages p20:410 p30:320 p40:430] [--bias 50] [--noise 0.05]
  python examples/fathi/export_fathi.py                        # results -> summary.json
  python examples/fathi/psf.py --case 3 [--depths 3 8 15 25]   # point-spread functions, true model
  ```
  `--precond block|depth` (not in the paper) scales the L-BFGS directions: same norm for the
  lambda and mu blocks, and with `depth` a compensation of the sensitivity loss with depth.
- `examples/nappe`: our case, not in the paper. Fathi's geometry and three layers, with a water
  table at -10 m below which lambda is 5 times mu: the interfaces of lambda and mu differ.
  `--variant lens`: lambda = 3 mu only in a saturated lens between -16 and -10 m;
  `--variant smooth`: the water table, starting from the true model smoothed along z.
  ```
  python examples/nappe/run_nappe.py both [--stages p20:20] [--bias 10] [--precond depth]
  python examples/nappe/run_nappe.py both --variant lens|smooth [--tag name] ...
  python examples/nappe/psf_nappe.py [--depths 3 8 12 15 25]
  ```

## ArmoniK

With `ArmoniKBackend`, each source becomes an
[ArmoniK](https://github.com/aneoconsulting/ArmoniK.Core) task run by the
`semfwi.armonik_worker` worker. The client prepares the run directories as it does locally
(mesh included, so it needs SEM3D); it sends the mesh once per session, then, for each task,
the input files of the forward directory and the data. In return, it receives the traces
and the kernels. The contract is described in `semfwi/armonik_io.py`.

Worker image (Amazon Linux 2023, parallel HDF5 1.14, OpenMPI, SEM3D, semfwi):

```
pip install -e '.[armonik]'           # client
docker/build.sh [--test]              # semfwi_worker:0.1.0; --test runs pytest in the image
```

SEM3D is compiled there from a copy (`git archive`) of `SEM_SRC` (default `~/SEM`) at
revision `SEM_REF` (default `adjoint`).

Local deployment with ArmoniK.Core (4 agent + worker pairs, one `TestPartition0` partition):

```
cd ../ArmoniK.Core
WORKER_IMAGE=semfwi_worker:0.1.0 just tag=0.0.0.0-local replicas=4 partitions=1 \
  grafana=false prometheus=false seq=false deploy
```

Do not use `just build-deploy`: it would rebuild `WORKER_IMAGE` with the Dockerfile of
ArmoniK's test worker.

Usage:

```
export ARMONIK_ENDPOINT=http://localhost:5001 ARMONIK_PARTITION=TestPartition0
python examples/synth/run_synth.py --backend armonik [--taylor | --niter 2]
```

In a script: `Problem(..., backend=ArmoniKBackend())`, then `backend.close()`, which closes
the session. A failure of the forward simulation in a task is raised as `SimulationError`,
with the end of `sem3d.log`; any other failure (adjoint, kernels, timeout) as `TaskError`,
which the line search does not catch.

## Tests

```
pytest -q tests -m "not sem3d"   # without SEM3D
pytest -q tests                  # with SEM3D if SEM3D_BIN is set (about 15 s)
```
