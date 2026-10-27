# semfwi ArmoniK worker image: SEM3D (copy of the sources, see build.sh) + semfwi.
# Amazon Linux 2023: same gcc and OpenMPI as the development machine.
FROM amazonlinux:2023 AS build
RUN dnf install -y gcc gcc-c++ gcc-gfortran make cmake openmpi-devel zlib-devel tar gzip && dnf clean all
ENV PATH=/opt/hdf5/bin:/usr/lib64/openmpi/bin:$PATH \
    LD_LIBRARY_PATH=/opt/hdf5/lib:/usr/lib64/openmpi/lib \
    HDF5_ROOT=/opt/hdf5
COPY docker/vendor/hdf5.tar.gz /src/
RUN tar xzf /src/hdf5.tar.gz -C /src && cd /src/hdf5-*/ \
 && CC=mpicc FC=mpif90 ./configure --prefix=/opt/hdf5 --enable-parallel --enable-fortran \
      --enable-shared --disable-tests > /dev/null \
 && make -j"$(nproc)" > /dev/null && make install > /dev/null && rm -rf /src/hdf5-*/
COPY docker/vendor/sem3d.tar /src/
RUN mkdir /src/sem && tar xf /src/sem3d.tar -C /src/sem \
 && cmake -S /src/sem -B /src/build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DOPT_MPI=ON \
      -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_Fortran_COMPILER=mpif90 \
      -DCMAKE_INSTALL_PREFIX=/opt/sem3d > /dev/null \
 && cmake --build /src/build -j"$(nproc)" > /dev/null && cmake --install /src/build > /dev/null \
 && rm -rf /src/build /src/sem

FROM amazonlinux:2023 AS base
RUN dnf install -y openmpi libgfortran zlib python3.11 python3.11-pip && dnf clean all
COPY --from=build /opt/hdf5/lib /opt/hdf5/lib
COPY --from=build /opt/sem3d/bin/sem3d.exe /opt/sem3d/bin/mesher /opt/sem3d/bin/
# HOME: OpenMPI writes to ~; single_copy_mechanism=none: no CMA (ptrace) in a container
ENV SEM3D_BIN=/opt/sem3d/bin MPI_BIN=/usr/lib64/openmpi/bin MPI_LIB=/usr/lib64/openmpi/lib HDF5_LIB=/opt/hdf5/lib \
    SEMFWI_SCRATCH=/tmp/semfwi HOME=/tmp OMPI_MCA_btl_vader_single_copy_mechanism=none \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY docker/requirements.txt .
RUN python3.11 -m pip install --only-binary=:all: -r requirements.txt
COPY pyproject.toml README.md ./
COPY semfwi ./semfwi
RUN python3.11 -m pip install --no-deps .

# semfwi tests in the image, including those that run SEM3D (build.sh --test)
FROM base AS test
RUN python3.11 -m pip install --only-binary=:all: pytest==9.1.1
COPY tests ./tests
COPY pytest.ini .
USER 1654
RUN python3.11 -m pytest -q -p no:cacheprovider tests

FROM base AS final
# Same UID as the ArmoniK polling agent, to share the unix sockets
USER 1654
ENTRYPOINT ["python3.11", "-m", "semfwi.armonik_worker"]
