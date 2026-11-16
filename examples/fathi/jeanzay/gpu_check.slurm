#!/bin/bash
# Validation of a GPU (OpenACC) build of SEM3D on H100, against the CPU build:
#   SEM3D_BIN=<dir of the GPU sem3d.exe> sbatch examples/fathi/jeanzay/gpu_check.slurm <label> [<CPU reference>]
# 1. the semfwi tests (those that run SEM3D included: traces, kernels computed in SEM3D; one on
#    2 ranks, hence 2 GPUs: SEM3D puts rank r of a node on its GPU r);
# 2. one evaluation of case 3 (forward, adjoint, kernels, PML included: scaling.py), wall times,
#    and its J, traces and gradient compared with those of a CPU run of scaling.py (<CPU reference>:
#    its directory in fathi/case3/scaling, default ngll3_pack_c10_p20_n1_r0).
# The tests that require two runs to be identical to the bit fail on GPU: the assembly sums
# with atomics, in an order that changes from one run to the next.
# export SBATCH_ACCOUNT=<project>@h100 before sbatch.
#SBATCH --job-name=fwi-gpu-check
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

label=${1:?label}
ref=${2:-ngll3_pack_c10_p20_n1_r0}
source examples/fathi/jeanzay/env_gpu.sh  # sbatch from the root of sem3d-fwi
export SEMFWI_MPIRUN="srun -n {n}"
echo "SEM3D_BIN=$SEM3D_BIN"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd "$SEMFWI_SRC" || exit 1
pytest -q tests --basetemp="$SCRATCH/semfwi-pytest-$label"

cd examples/fathi || exit 1
NV_ACC_TIME=0 python -u scaling.py --nprocs 1 --tag "gpu_$label"
python - "$SEMFWI_RUNS/fathi/case3/scaling" "gpu_${label}_p20_n1_r0" "$ref" <<'EOF'
import sys
import numpy as np
root, gpu, cpu = sys.argv[1:]
a = np.load(f"{root}/{gpu}/it000/s0/result.npz")
b = np.load(f"{root}/{cpu}/it000/s0/result.npz")
for k in b.files:
    if k == "key":
        continue
    if k not in a.files:
        print(f"{k:10} missing on GPU")
        continue
    x, y = a[k], b[k]
    err = np.abs(x - y).max() / max(np.abs(y).max(), 1e-300)
    print(f"{k:10} {str(y.shape):18} max |gpu - cpu| / max |cpu| = {err:.2e}")
EOF
