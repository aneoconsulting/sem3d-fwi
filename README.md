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

`SEM3D_BIN` has no default: a `sem3d.exe` found in `PATH` might not have the strain
output. Without it, the simulations stop with a message and the test that runs SEM3D is
skipped. Example:

```
export SEM3D_BIN=~/install/sem3d-adjoint/bin
export MPI_BIN=/usr/lib64/openmpi/bin MPI_LIB=/usr/lib64/openmpi/lib
export HDF5_LIB=~/install/hdf5-1.14/lib
```

## `semfwi` package

| Module     | Role |
|------------|------|
| `config`   | case parameters: mesh, time, parameter grid, stations, sources |
| `model`    | parameter grid, Lambda/Mu/Rho files, trilinear interpolation |
| `sem_io`   | writing SEM3D inputs, reading traces, the log and strains |
| `runner`   | single mesh, launching simulations, cache, parallel execution |
| `misfit`   | residuals, misfit function, adjoint source time histories |
| `gradient` | K_lambda and K_mu kernels and projection onto the parameter grid |
| `problem`  | J(x) and grad J(x) over all sources |
| `optim`    | L-BFGS with Armijo line search, checkpoint and restart |

For each source, a gradient evaluation runs a forward simulation (traces and strains),
then an adjoint simulation whose sources are the time-reversed residuals; the kernels are
the correlation of the strains of the two simulations.

## Examples

- `examples/synth`: small cube, two Gaussian anomalies (lambda and mu), 4 point
  forces, 25 stations.
  ```
  python examples/synth/run_synth.py [--param vel|lame] [--niter 15] [--taylor]
  python examples/synth/analyse_synth.py
  ```
- `examples/fathi`: test cases of Fathi, Kallivokas and Poursartip (2015,
  arXiv:1504.08340), sections 4.1 and 4.2.
  ```
  python examples/fathi/run_fathi1.py [--pulses p20 p40]       # gradient check
  python examples/fathi/run_fathi2.py mu|lambda|both [--stages p20:20 p30:10] [--bias 15]
  python examples/fathi/export_fathi.py                        # results -> summary.json
  ```

## Tests

```
pytest -q tests -m "not sem3d"   # without SEM3D
pytest -q tests                  # with SEM3D if SEM3D_BIN is set (about 15 s)
```
