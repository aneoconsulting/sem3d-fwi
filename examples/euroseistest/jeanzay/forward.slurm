#!/bin/bash
# EUROSEISTEST 3-D: forward simulation of the archive's earthquake (24 s) on every core of the allocated nodes.
# From the root of the sem3d-fwi clone of the branch, E2VP2_DIR exported, basin.npz prepared:
#   sbatch --nodes=1 examples/euroseistest/jeanzay/forward.slurm 2     # then --nodes=2, --nodes=4: strong scaling
#   sbatch --nodes=1 examples/euroseistest/jeanzay/forward.slurm 2 --pml copy   # or --npml 4: PML variants
# Traces in $SEMFWI_RUNS/e2vp3d/forward_f<fmax>_t24_n<ranks>.npz, then:
#   python run_e2vp3d.py bedrock --fmax 2 --time 24 --nprocs <ranks>
#SBATCH --job-name=e2vp3d-forward
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --ntasks-per-node=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out

FMAX=${1:-2}
shift  # other arguments of run_e2vp3d.py forward (--pml copy, --npml 4)
: "${E2VP2_DIR:?export E2VP2_DIR (the E2VP2-MODEL directory)}"
source examples/fathi/jeanzay/env.sh
export SEMFWI_SRC=$PWD PYTHONPATH=$PWD                            # this clone, not the one of env.sh
export SEM3D_BIN=${E2VP_SEM3D_BIN:-$WORK/install/sem3d-adjoint-ngll5/bin}  # ngll = 5 build
export SEMFWI_MPIRUN="srun -n {n}"
cd examples/euroseistest || exit 1
python -u run_e2vp3d.py forward --fmax "$FMAX" --time 24 --nprocs "$SLURM_NTASKS" "$@"
