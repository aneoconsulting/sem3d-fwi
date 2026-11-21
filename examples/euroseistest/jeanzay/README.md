# EUROSEISTEST 3-D on Jean Zay

The CPU part of the setup of `examples/fathi/jeanzay/README.md` (modules, `env.sh`, venv, account) applies;
this page only adds what the 3-D case of the Mygdonian basin needs.

## Data

The E2VP2-MODEL archive (licence unknown: in `$WORK`, never in `$WORK_SHARED` nor in the repository):
`velocity_profile.txt`, `source_info.txt`, `DEM_surface_10m.grd`, `DEM_MP_10m.grd`, `DEM_bedrock_10m.grd`.

```bash
export E2VP2_DIR=$WORK/data/E2VP2-MODEL
pip install pillow                       # the bedrock .grd is a TIFF (prepare)
cd examples/euroseistest && python run_e2vp3d.py prepare   # basin.npz
```

## SEM3D for ngll = 5

The case uses ngll = 5; the default binary of the Fathi page is built for ngll = 3 (`-DOPT_MIN_GLL_OPT=3`).
Same source (branch `adjoint`, commit `32daa083` or later: the source element no longer depends on the MPI
partition) and options, without that one:

```bash
cmake -S $WORK/SEM -B $WORK/build/sem3d-adjoint-ngll5 -DCMAKE_BUILD_TYPE=Release -DOPT_VEC=8 \
      -DCMAKE_Fortran_FLAGS=-march=cascadelake -DCMAKE_C_FLAGS=-march=cascadelake -DOPT_MPI=ON \
      -DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_C_COMPILER=mpicc -DCMAKE_INSTALL_PREFIX=$WORK/install/sem3d-adjoint-ngll5
cmake --build $WORK/build/sem3d-adjoint-ngll5 -j 4 && cmake --install $WORK/build/sem3d-adjoint-ngll5
```

`forward.slurm` uses it (`E2VP_SEM3D_BIN` to point elsewhere).

## Forward simulations and scaling

```bash
sbatch --nodes=1 examples/euroseistest/jeanzay/forward.slurm 2
sbatch --nodes=2 examples/euroseistest/jeanzay/forward.slurm 2
sbatch --nodes=4 examples/euroseistest/jeanzay/forward.slurm 2
sacct -j <job> --format=JobID,Elapsed,MaxRSS,NNodes   # wall time and memory of each
```

The archive's earthquake, 24 s, at 2 Hz: elements of 62.5 m, about 8 times the 3.2 M GLL points at 1 Hz and twice
its time steps, so about 16 times its cost (1 Hz on 4 Graviton cores: 7857 s, 8.7 core-h). The three jobs give the
strong scaling; the traces of any of them feed the bedrock map: `run_e2vp3d.py bedrock --fmax 2 --time 24 --nprocs
<ranks>`. The sources lie on faces or edges of elements (the epicentre on a layer interface); from `32daa083` on,
SEM3D picks the same element for any number of ranks (traces of 1 to 4 ranks within 1e-14), before it the
traces depended on the partition (8e-6 of their maximum between 1 and 2 ranks at 0.5 Hz).
`sacct`'s MaxRSS does not measure SEM3D: put `/usr/bin/time -v` before `srun`, or use `sstat` during the run.

## Gradient checks and inversions

`invert.slurm` runs `run_e2vp3d.py gradcheck`, `psf` or `invert` with every simulation on all the allocated cores, the
sources one after the other; its arguments are those of `run_e2vp3d.py`, `--nprocs` set to the number of tasks. The
start with the bedrock map needs `bedrock_<tag>.npz` in `$SEMFWI_RUNS/e2vp3d` (`run_e2vp3d.py bedrock`, from a forward
of the same `--fmax`, `--time` and `--nprocs`):

```bash
sbatch --nodes=1 examples/euroseistest/jeanzay/forward.slurm 1               # forward_f1_t24_n40.npz
python run_e2vp3d.py bedrock --fmax 1 --time 24 --nprocs 40                   # bedrock_f1_t24_n40.npz
sbatch --nodes=2 --qos=qos_cpu-dev --time=02:00:00 examples/euroseistest/jeanzay/invert.slurm gradcheck --fmax 1 --bedrock f1_t24_n40 --param vs
sbatch --nodes=2 examples/euroseistest/jeanzay/invert.slurm psf --fmax 1 --bedrock f1_t24_n40 --at 500 -500
sbatch --nodes=2 examples/euroseistest/jeanzay/invert.slurm invert --fmax 1 --bedrock f1_t24_n40 --param vs
```

`--param vs` inverts Vs alone, Vp / Vs of the start kept: at 0.5 Hz, a bump of K or of G answers in both blocks, and
K and G together brought Poisson's ratio below 0. `psf --at 500 -500` is the point-spread function under the station
(500, -500), 2.5 km from every edge, to compare with `psf_f0.5_kg_bedrock-f1_t24_at500_-500` on the resolution.

`--param interface` inverts the depth of the bedrock on the 11 x 11 map instead, the sediments and rock of the start
above and below it. `bedmap` turns an iterate of such an inversion into a new map, the start of the next one, at a
higher frequency or in Vs alone; `bedmap --mean` makes a flat map, the same start without what the H/V map says of the
shape (on the login node, it only reads and writes small files):

```bash
sbatch --nodes=2 examples/euroseistest/jeanzay/invert.slurm invert --fmax 0.5 --bedrock f1_t24_n40 --param interface --iters 30
python run_e2vp3d.py bedmap --bedrock f1_t24_n40 --from invert_f0.5_s3_interface_z4_bedrock-f1_t24_n40_n80/x030.npy --tag if05
sbatch --nodes=2 examples/euroseistest/jeanzay/invert.slurm invert --fmax 1 --bedrock if05 --param interface --iters 30
sbatch --nodes=2 examples/euroseistest/jeanzay/invert.slurm invert --fmax 1 --bedrock if05 --param vs --iters 30
python run_e2vp3d.py bedmap --bedrock f1_t24_n40 --mean --tag mean
sbatch --nodes=2 examples/euroseistest/jeanzay/invert.slurm invert --fmax 0.5 --bedrock mean --param interface --iters 30
```

`--param interface-v0g` inverts v0 and g of the sediment profile (Vs = v0 + g depth) with the map; `bedmap --from`
carries them on. A gradient check given `+h` and `-h` also prints the centered difference: at 1 Hz, J is so curved
along the depth of the bedrock that the forward difference converges only for h < 1e-4 (gap 9.5 % at 1e-3, centered
0.9 %):

```bash
sbatch --nodes=2 --qos=qos_cpu-dev --time=01:00:00 examples/euroseistest/jeanzay/invert.slurm gradcheck --fmax 1 --bedrock f1_t24_n40 --param interface --h 1e-3 -1e-3
sbatch --nodes=2 examples/euroseistest/jeanzay/invert.slurm invert --fmax 0.5 --bedrock if05 --param interface-v0g --iters 30
```

`jmap` gives J of the map of `--bedrock` (one forward per source, a few minutes in QoS dev). `bedmap --truth` writes the
map closest to the true bedrock on the grid of `--fmax`, a diagnostic that only judges the parametrisation: if its J
is above that of the inverted maps (in the history of the inversions), the data prefer a wrong map, and the fixed
part of the model (the sediment profile, the transition) is to blame rather than the inversion:

```bash
python run_e2vp3d.py bedmap --bedrock f1_t24_n40 --truth --fmax 0.75 --tag truth075
sbatch --nodes=2 --qos=qos_cpu-dev --time=00:30:00 examples/euroseistest/jeanzay/invert.slurm jmap --fmax 0.75 --bedrock truth075
```

An inversion stopped by the time limit resumes from its last iteration when submitted again with the same arguments
and the same number of tasks (the ranks are part of the names of its data and work directories).
