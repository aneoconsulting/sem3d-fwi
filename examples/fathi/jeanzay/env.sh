# Environment of sem3d-fwi on Jean Zay. The module names are to be checked with `module avail`
# (same MPI and HDF5 to build SEM3D and to run it).
module purge
module load gcc openmpi hdf5 cmake python  # CHANGE_ME: names and versions not checked (module avail)
export SEMFWI_SRC=$WORK/sem3d-fwi                 # this repository
export SEM3D_BIN=$WORK/install/sem3d-adjoint/bin  # sem3d.exe and mesher, branch adjoint of SEM
export SEMFWI_RUNS=$SCRATCH/semfwi-runs           # simulations: large files, not in $WORK
export PYTHONPATH=$SEMFWI_SRC
source $WORK/venvs/semfwi/bin/activate            # numpy, h5py
