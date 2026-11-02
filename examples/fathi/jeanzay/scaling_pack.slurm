#!/bin/bash
# The inversions are independent: what counts is the throughput of a full node. For n ranks per run,
# 40/n runs at once on the node (memory bandwidth and file system shared), compared with the same n
# alone on the node (scaling_strong.slurm).
#SBATCH --job-name=fwi-pack
#SBATCH --account=CHANGE_ME@cpu
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
python -u scaling.py --data-only  # computed once, before the copies read it
for n in 1 2 4 8 10 20; do
  for i in $(seq 1 $((40 / n))); do
    python -u scaling.py --nprocs $n --tag pack_c$i &
  done
  wait
done
