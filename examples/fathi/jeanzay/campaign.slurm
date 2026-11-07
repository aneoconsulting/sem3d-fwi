#!/bin/bash
# One node, one inversion of 4 ranks per line of a campaign file (10 lines: the 40 cores), all at once.
#   sbatch examples/fathi/jeanzay/campaign.slurm examples/fathi/jeanzay/campaigns/case3_start.txt
# A run stopped by the time limit resumes where it was when the same file is submitted again.
# Not exclusive: fewer cores with --ntasks (a few long runs: sbatch --qos=qos_cpu-t4 --time=100:00:00
# --ntasks=20 ...), and other ranks per run with --nprocs in the line (their sum: --ntasks).
#SBATCH --job-name=fwi-campaign
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out

LIST=$(readlink -f "$1")
source examples/fathi/jeanzay/env.sh  # sbatch from the root of sem3d-fwi
export SEMFWI_MPIRUN="srun --exact --cpu-bind=cores -n {n}"
cd "$SEMFWI_SRC/examples/fathi" || exit 1
mkdir -p "$SEMFWI_RUNS/logs"
while read -r tag args; do
  [ -z "$tag" ] && continue
  python -u run_fathi2.py both $args --tag "$tag" >> "$SEMFWI_RUNS/logs/$tag.log" 2>&1 &
done < <(grep -v '^#' "$LIST")
# about 200 MB per iteration directory: keep the start of each stage and the last 5 iterations
while [ -n "$(jobs -r)" ]; do
  sleep 300
  find "$SEMFWI_RUNS/fathi" -mindepth 2 -maxdepth 4 -type d -name 'p[0-9]0' | while read -r st; do
    ls -d "$st"/it[0-9]* 2>/dev/null | grep -v _start | sort -V | head -n -5 | xargs -r rm -rf
  done
done
wait
