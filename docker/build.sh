#!/usr/bin/env bash
# Build the semfwi ArmoniK worker image.
#
#   docker/build.sh                  # semfwi_worker:0.1.0
#   docker/build.sh --test           # also run the semfwi tests in the image ("test" target)
#   docker/build.sh --tag 0.2.0
#
# SEM3D is a copy of the sources (git archive) of SEM_SRC (default ~/SEM) at revision SEM_REF
# (default perf-graviton). HDF5_TARBALL: HDF5 1.14 sources (default ~/install/src/hdf5-1.14.3.tar.gz,
# downloaded otherwise).
set -euo pipefail
cd "$(dirname "$0")/.."

tag=0.1.0
run_tests=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) tag=$2; shift ;;
    --test) run_tests=true ;;
    -h | --help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

sem_src=${SEM_SRC:-$HOME/SEM}
sem_ref=${SEM_REF:-perf-graviton}
hdf5=${HDF5_TARBALL:-$HOME/install/src/hdf5-1.14.3.tar.gz}
hdf5_url=https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.14/hdf5-1.14.3/src/hdf5-1.14.3.tar.gz

mkdir -p docker/vendor
echo "==> SEM3D: $sem_src @ $sem_ref ($(git -C "$sem_src" rev-parse --short "$sem_ref"))"
git -C "$sem_src" archive --format=tar -o "$PWD/docker/vendor/sem3d.tar" "$sem_ref"
if [[ -f "$hdf5" ]]; then
  cp "$hdf5" docker/vendor/hdf5.tar.gz
elif [[ ! -f docker/vendor/hdf5.tar.gz ]]; then
  curl -fsSL -o docker/vendor/hdf5.tar.gz "$hdf5_url"
fi

if [[ "$run_tests" == true ]]; then
  echo "==> tests"
  docker build -f docker/Dockerfile --target test .
fi
echo "==> semfwi_worker:$tag"
docker build -f docker/Dockerfile --label "sem3d.ref=$sem_ref" \
  --label "sem3d.commit=$(git -C "$sem_src" rev-parse "$sem_ref")" -t "semfwi_worker:$tag" .
