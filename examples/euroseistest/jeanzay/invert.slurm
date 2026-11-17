#!/bin/bash
# EUROSEISTEST 3-D: gradcheck, psf or invert (run_e2vp3d.py) with every simulation on every core of the allocated
# nodes, one source after the other. From the root of the sem3d-fwi clone of the branch, E2VP2_DIR exported,
# basin.npz prepared, bedrock_<tag>.npz copied to $SEMFWI_RUNS/e2vp3d for --bedrock:
#   sbatch --nodes=1 --qos=qos_cpu-dev --time=02:00:00 examples/euroseistest/jeanzay/invert.slurm gradcheck --fmax 1 --bedrock f1_t24
#   sbatch --nodes=1 examples/euroseistest/jeanzay/invert.slurm invert --fmax 1 --bedrock f1_t24 --param vs
#   sbatch --nodes=4 --qos=qos_cpu-t4 --time=100:00:00 examples/euroseistest/jeanzay/invert.slurm invert --fmax 2 ...
# An inversion stopped by the time limit resumes from its last iteration when submitted again with the same arguments.
#SBATCH --job-name=e2vp3d-invert
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --ntasks-per-node=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out

: "${E2VP2_DIR:?export E2VP2_DIR (the E2VP2-MODEL directory)}"
source examples/fathi/jeanzay/env.sh
export SEMFWI_SRC=$PWD PYTHONPATH=$PWD                            # this clone, not the one of env.sh
export SEM3D_BIN=${E2VP_SEM3D_BIN:-$WORK/install/sem3d-adjoint-ngll5/bin}  # ngll = 5 build
export SEMFWI_MPIRUN="srun -n {n}"
cd examples/euroseistest || exit 1
python -u run_e2vp3d.py "$@" --nprocs "$SLURM_NTASKS"
