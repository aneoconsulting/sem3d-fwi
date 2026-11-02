# Environment of sem3d-fwi on Jean Zay. The module names are to be checked with `module avail`
# (same MPI and HDF5 to build SEM3D and to run it).
module purge
module load gcc/12.2.0 openmpi/4.1.5 hdf5/1.12.0-mpi cmake python/3.11.5
export SEMFWI_SRC=$WORK/sem3d-fwi                 # this repository
export SEM3D_BIN=$WORK/install/sem3d-adjoint/bin  # sem3d.exe and mesher, branch adjoint of SEM
export SEMFWI_RUNS=$SCRATCH/semfwi-runs           # simulations: large files, not in $WORK
export PYTHONPATH=$SEMFWI_SRC
source $WORK/venvs/semfwi/bin/activate            # numpy, h5py
