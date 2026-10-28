"""semfwi ArmoniK worker: python -m semfwi.armonik_worker

A task evaluates one source (task.run_source) in a temporary directory under SEMFWI_SCRATCH
(default /tmp/semfwi); contract in armonik_io. The mesh, identical for all tasks of a case,
is kept in SEMFWI_SCRATCH/mesh under the hash of its content.
Variables: SEM3D_BIN (required), MPI_BIN, MPI_LIB, HDF5_LIB (see runner), LOG_LEVEL.
"""
import hashlib
import json
import logging
import os
import shutil
import tempfile
from urllib.parse import urlparse

from . import armonik_io, config, task
from .runner import Runner

# absolute: the mesh symlink is created from each task's forward directory
SCRATCH = os.path.abspath(os.environ.get("SEMFWI_SCRATCH", "/tmp/semfwi"))


def mesh_dir(data):
    """Mesh directory (with a sem subdirectory), extracted once per content."""
    d = os.path.join(SCRATCH, "mesh", hashlib.sha1(data).hexdigest())
    if not os.path.isdir(d):
        os.makedirs(os.path.dirname(d), exist_ok=True)
        tmp = tempfile.mkdtemp(dir=os.path.dirname(d))
        armonik_io.unpack(data, os.path.join(tmp, "sem"))
        try:
            os.rename(tmp, d)
        except OSError:  # extracted meanwhile by another task
            shutil.rmtree(tmp)
    return d


def compute(payload, deps, workdir):
    """Run a task; deps: id -> content. Return the result .npz."""
    try:
        p = json.loads(payload)
    except ValueError as e:
        raise armonik_io.InvalidPayload(f"payload is not JSON: {e}") from e
    if p.get("version") != armonik_io.VERSION:
        raise armonik_io.InvalidPayload(f"unsupported version {p.get('version')} (expected {armonik_io.VERSION})")
    cfg = config.from_dict(p["cfg"], workdir)
    mesh = mesh_dir(deps[p["mesh"]])
    dfwd = os.path.join(workdir, "fwd")
    armonik_io.unpack(deps[p["inputs"]], dfwd)
    os.symlink(os.path.join(mesh, "sem"), os.path.join(dfwd, "sem"))
    dobs = armonik_io.decode_array(deps[p["dobs"]]) if p["grad"] else None
    r = task.run_source(cfg, Runner(cfg), dfwd, os.path.join(workdir, "adj"), dobs, p["grad"])
    return armonik_io.encode_result(r)


def _log_tail(workdir, n=20):
    out = []
    for kind in ("fwd", "adj"):
        fn = os.path.join(workdir, kind, "sem3d.log")
        if os.path.exists(fn):
            with open(fn, errors="replace") as f:
                out.append(f"--- {kind}/sem3d.log\n" + "".join(f.readlines()[-n:]))
    return "\n".join(out)


def process(task_handler):
    from armonik.common import Output
    logger = logging.getLogger("semfwi")
    if len(task_handler.expected_results) != 1:
        return Output(f"one output expected, {len(task_handler.expected_results)} received")
    os.makedirs(SCRATCH, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as w:
        try:
            out = compute(task_handler.payload, task_handler.data_dependencies, w)
        except Exception as e:
            logger.warning(f"task {task_handler.task_id} failed: {e}")
            return Output(f"{type(e).__name__}: {e}\n{_log_tail(w)}".strip())
    task_handler.send_results({task_handler.expected_results[0]: out})
    return Output()


def _endpoints(channel, default_socket):
    """(listen address, client URI) of a ComputePlane channel: ArmoniK.Core gives a socket path
    in unixdomainsocket mode and an http://host:port URI in tcp mode."""
    socket_type = os.getenv(f"ComputePlane__{channel}__SocketType", "unixdomainsocket").lower()
    address = os.getenv(f"ComputePlane__{channel}__Address", default_socket)
    if socket_type == "unixdomainsocket":
        path = urlparse(address).path if "://" in address else address
        return f"unix://{path}", f"unix://{path}"
    uri = address if "://" in address else f"http://{address}"
    return f"0.0.0.0:{urlparse(uri).port}", uri


def main():
    from concurrent import futures

    import grpc
    from armonik.common.channel import create_channel
    from armonik.protogen.worker.worker_service_pb2_grpc import add_WorkerServicer_to_server
    from armonik.worker import ClefLogger
    from armonik.worker.worker import ArmoniKWorker
    ClefLogger.setup_logging(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
    logger = ClefLogger.getLogger("semfwi")
    worker_bind, _ = _endpoints("WorkerChannel", "/cache/armonik_worker.sock")
    _, agent_uri = _endpoints("AgentChannel", "/cache/armonik_agent.sock")
    logger.info(f"semfwi worker on {worker_bind}, agent {agent_uri}, SEM3D_BIN={os.getenv('SEM3D_BIN')}")
    with create_channel(agent_uri, options=(("grpc.default_authority", "localhost"),)) as channel:
        worker = ArmoniKWorker(channel, process, logger=logger)
        # ArmoniKWorker.start serves calls on a single thread: during a task (minutes of SEM3D),
        # the polling agent's health checks would wait and the agent would shut down
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        add_WorkerServicer_to_server(worker, server)
        server.add_insecure_port(worker_bind)
        server.start()
        server.wait_for_termination()


if __name__ == "__main__":
    main()
