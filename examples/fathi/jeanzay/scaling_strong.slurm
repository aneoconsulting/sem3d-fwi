#!/bin/bash
# One inversion iteration of case 3 (forward + adjoint + kernels) on 1 to 40 ranks, one after the other,
# on a whole node: wall time and core-hours per iteration as a function of the ranks per run.
#SBATCH --job-name=fwi-strong
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

source examples/fathi/jeanzay/env.sh  # sbatch from the root of sem3d-fwi
export SEMFWI_MPIRUN="srun --exact --cpu-bind=cores -n {n}"
cd "$SEMFWI_SRC/examples/fathi" || exit 1
python -u scaling.py --data-only
python -u scaling.py --nprocs 1 2 4 8 10 20 40 --reps 2 --tag strong
