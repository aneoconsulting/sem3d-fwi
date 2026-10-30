# semfwi ArmoniK worker image: SEM3D + semfwi.
# Built by docker/build.sh, which puts the sources in docker/vendor:
#   - hdf5.tar.gz: HDF5 1.14 sources;
#   - sem3d.tar  : copy of the SEM3D sources (git archive).
# Amazon Linux 2023 base: same gcc and OpenMPI as the development machine.
#
# Stages: build (compilation), base (runtime), test (pytest), final (worker image).


# ---------------------------------------------------------------------------
# build: parallel HDF5 then SEM3D, compiled with the MPI wrappers
# ---------------------------------------------------------------------------
FROM amazonlinux:2023 AS build

RUN dnf install -y \
        gcc gcc-c++ gcc-gfortran make cmake \
        openmpi-devel zlib-devel tar gzip \
    && dnf clean all

ENV PATH=/opt/hdf5/bin:/usr/lib64/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/hdf5/lib:/usr/lib64/openmpi/lib
ENV HDF5_ROOT=/opt/hdf5

# HDF5 -> /opt/hdf5
COPY docker/vendor/hdf5.tar.gz /src/
WORKDIR /src
RUN tar xzf hdf5.tar.gz && mv hdf5-*/ hdf5
WORKDIR /src/hdf5
RUN CC=mpicc FC=mpif90 ./configure \
        --prefix=/opt/hdf5 \
        --enable-parallel \
        --enable-fortran \
        --enable-shared \
        --disable-tests
RUN make -j"$(nproc)" && make install

# SEM3D -> /opt/sem3d
COPY docker/vendor/sem3d.tar /src/
WORKDIR /src/sem
RUN tar xf /src/sem3d.tar
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DOPT_VEC=8 \
        -DOPT_MIN_GLL_OPT=3 \
        -DOPT_MPI=ON \
        -DCMAKE_C_COMPILER=mpicc \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DCMAKE_Fortran_COMPILER=mpif90 \
        -DCMAKE_INSTALL_PREFIX=/opt/sem3d
RUN cmake --build build -j"$(nproc)" && cmake --install build


# ---------------------------------------------------------------------------
# base: runtime libraries, SEM3D binaries and semfwi
# ---------------------------------------------------------------------------
FROM amazonlinux:2023 AS base

RUN dnf install -y openmpi libgfortran zlib python3.11 python3.11-pip \
    && dnf clean all

COPY --from=build /opt/hdf5/lib /opt/hdf5/lib
COPY --from=build /opt/sem3d/bin/sem3d.exe /opt/sem3d/bin/mesher /opt/sem3d/bin/

# Locations read by semfwi.runner
ENV SEM3D_BIN=/opt/sem3d/bin
ENV MPI_BIN=/usr/lib64/openmpi/bin
ENV MPI_LIB=/usr/lib64/openmpi/lib
ENV HDF5_LIB=/opt/hdf5/lib
# Task working directories (see semfwi.armonik_worker)
ENV SEMFWI_SCRATCH=/tmp/semfwi
# OpenMPI writes to HOME, which must be writable
ENV HOME=/tmp
# No CMA (ptrace) copy between processes inside a container
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY docker/requirements.txt .
RUN python3.11 -m pip install --only-binary=:all: -r requirements.txt

COPY pyproject.toml README.md ./
COPY semfwi ./semfwi
RUN python3.11 -m pip install --no-deps .


# ---------------------------------------------------------------------------
# test: semfwi tests, including those that run SEM3D (build.sh --test)
# ---------------------------------------------------------------------------
FROM base AS test

RUN python3.11 -m pip install --only-binary=:all: pytest==9.1.1
COPY tests ./tests
COPY pytest.ini .

USER 1654
RUN python3.11 -m pytest -q -p no:cacheprovider tests


# ---------------------------------------------------------------------------
# final: worker image
# ---------------------------------------------------------------------------
FROM base AS final

# Same UID as the ArmoniK polling agent, to share the unix sockets
USER 1654
ENTRYPOINT ["python3.11", "-m", "semfwi.armonik_worker"]
