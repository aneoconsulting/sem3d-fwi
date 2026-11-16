# Environment of sem3d-fwi on the H100 nodes of Jean Zay: SEM3D built with OpenACC (nvfortran), see
# the GPU section of README.md. Same layout as env.sh.
module purge
for m in arch/h100 nvidia-compilers/25.11 cuda/12.8.0 openmpi/4.1.8-cuda hdf5/1.14.5-mpi-cuda; do
  module load $m  # one at a time: on one line, a failed prerequisite unloads them all
done
export SEMFWI_SRC=$WORK/sem3d-fwi                         # this repository
export SEM3D_BIN=${SEM3D_BIN:-$WORK/install/sem3d-acc/bin}  # sem3d.exe and mesher, OpenACC build
export SEMFWI_RUNS=$SCRATCH/semfwi-runs                   # simulations: large files, not in $WORK
export PYTHONPATH=$SEMFWI_SRC
source $WORK/venvs/semfwi-h100/bin/activate               # numpy, h5py, pytest: own venv (the Python of the CPU nodes is not on the H100 nodes)
