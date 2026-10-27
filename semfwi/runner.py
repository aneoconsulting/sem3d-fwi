"""Launching SEM3D simulations: a single shared mesh, a cache keyed by input hashes,
running several simulations in parallel."""
import hashlib
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

from . import sem_io

INPUT_FILES = ("input.spec", "stations.txt", "material.spec", "Lambda.h5", "Mu.h5", "Rho.h5")


class SimulationError(RuntimeError):
    pass


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mesh_dir = os.path.join(cfg.workdir, "mesh")

    def env(self):
        """Simulation environment: sem3d_bin, then MPI_BIN if set, prepended to PATH;
        HDF5_LIB and MPI_LIB, if set, prepended to LD_LIBRARY_PATH."""
        b = self.cfg.sem3d_bin
        if not b:
            raise SimulationError("SEM3D_BIN not set: directory of sem3d.exe and mesher")
        for exe in ("sem3d.exe", "mesher"):
            if not os.access(os.path.join(b, exe), os.X_OK):
                raise SimulationError(f"{exe} not found in {b} (SEM3D_BIN)")
        e = dict(os.environ)
        path = [b] + [os.path.expanduser(e[v]) for v in ("MPI_BIN",) if e.get(v)]
        e["PATH"] = ":".join(path + [e.get("PATH", "")])
        libs = [os.path.expanduser(e[v]) for v in ("HDF5_LIB", "MPI_LIB") if e.get(v)]
        e["LD_LIBRARY_PATH"] = ":".join(libs + [e.get("LD_LIBRARY_PATH", "")])
        if shutil.which("mpirun", path=e["PATH"]) is None:
            raise SimulationError("mpirun not found in PATH (set MPI_BIN)")
        return e

    def mesh(self):
        """The mesh does not depend on the model (the solid reads material.spec): it is built once."""
        d = self.mesh_dir
        if os.path.exists(f"{d}/sem/mesh4spec.0000.h5"):
            return
        os.makedirs(d, exist_ok=True)
        sem_io.write_mesher_inputs(d, self.cfg)
        sem_io.write_input(d, self.cfg, [])  # the mesher reads input.spec (ngll in particular)
        with open(f"{d}/mesh.input") as fin, open(f"{d}/mesher.log", "w") as fout:
            subprocess.run(["mesher"], stdin=fin, stdout=fout, stderr=subprocess.STDOUT, cwd=d,
                           env=self.env(), check=True)
        os.makedirs(f"{d}/sem", exist_ok=True)
        for fn in os.listdir(d):
            if fn.startswith("mesh4spec"):
                shutil.move(f"{d}/{fn}", f"{d}/sem/{fn}")

    def prepare(self, d):
        """Directory linked to the mesh; the caller then writes the model and inputs into it.
        An existing directory is kept: run() only reruns the computation if the inputs changed."""
        self.mesh()
        os.makedirs(d, exist_ok=True)
        if not os.path.lexists(f"{d}/sem"):
            os.symlink(os.path.abspath(f"{self.mesh_dir}/sem"), f"{d}/sem")
        shutil.copy(f"{self.mesh_dir}/material.input", d)

    @staticmethod
    def input_hash(d):
        h = hashlib.sha1()
        for fn in sorted(os.listdir(d)):
            if (fn in INPUT_FILES or fn == "material.input"
                    or (fn[0] in "aks" and fn.endswith((".txt", ".h5")))):
                h.update(fn.encode())
                with open(f"{d}/{fn}", "rb") as f:
                    h.update(f.read())
        return h.hexdigest()

    def is_done(self, d, h):
        try:
            with open(f"{d}/.done") as f:
                return f.read().strip() == h
        except FileNotFoundError:
            return False

    def run(self, d):
        """Runs SEM3D in d unless the same computation has already finished there (and its strains,
        if requested, have not been deleted). Checks dt and the number of steps."""
        h = self.input_hash(d)
        with open(f"{d}/input.spec") as f:
            want_strain = "save_strain = true;" in f.read()
        if not self.is_done(d, h) or (want_strain and not os.path.isdir(f"{d}/strain")):
            cmd = ["mpirun", "--oversubscribe", "-n", str(self.cfg.nprocs), "sem3d.exe"]
            with open(f"{d}/sem3d.log", "w") as fout:
                r = subprocess.run(cmd, stdout=fout, stderr=subprocess.STDOUT, cwd=d, env=self.env())
            if r.returncode != 0:
                raise SimulationError(f"SEM3D failed in {d} (code {r.returncode}), see sem3d.log")
            with open(f"{d}/.done", "w") as f:
                f.write(h)
        dt, n = sem_io.read_log(d)
        if dt != self.cfg.dt or n != self.cfg.nt:
            raise SimulationError(f"{d} : dt = {dt}, {n} steps; expected {self.cfg.dt}, {self.cfg.nt}")
        return d

    def run_many(self, dirs):
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            return list(ex.map(self.run, dirs))
