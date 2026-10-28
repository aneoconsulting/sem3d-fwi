"""ArmoniK backend: one task per source, executed by the semfwi worker (armonik_worker).

The client prepares the forward directories as in local mode (mesh, model, input.spec); only
their input files, the mesh (once per session) and the data are sent to ArmoniK, and only the
results of task.run_source come back. Contract in armonik_io.

    backend = ArmoniKBackend()             # ARMONIK_ENDPOINT, default http://localhost:5001
    pb = Problem(cfg, runner, dobs, ..., backend=backend)
    ...
    backend.close()                        # or: with ArmoniKBackend() as backend
"""
import datetime
import json
import os
import time

from armonik.client import ArmoniKResults, ArmoniKSessions, ArmoniKTasks
from armonik.common import ResultStatus, TaskDefinition, TaskOptions
from armonik.common.channel import create_channel

from . import armonik_io, config
from .runner import SimulationError
from .task import TaskError

# Final statuses of an output
DONE = (ResultStatus.COMPLETED, ResultStatus.ABORTED, ResultStatus.DELETED, ResultStatus.NOTFOUND)


class ArmoniKBackend:
    def __init__(self, endpoint=None, partition=None, max_duration=3600., max_retries=1, wait_timeout=None,
                 poll=5.):
        """wait_timeout: maximum wait for the outputs of a run; by default the time to run its tasks
        one by one, retries included (max_duration x (max_retries + 1) x number of tasks)."""
        self.max_duration, self.max_retries = max_duration, max_retries
        self.wait_timeout, self.poll = wait_timeout, poll
        self.channel = create_channel(endpoint or os.environ.get("ARMONIK_ENDPOINT", "http://localhost:5001"))
        self.sessions, self.tasks = ArmoniKSessions(self.channel), ArmoniKTasks(self.channel)
        self.results = ArmoniKResults(self.channel)
        partition = partition or os.environ.get("ARMONIK_PARTITION") or None
        options = TaskOptions(max_duration=datetime.timedelta(seconds=max_duration), priority=1,
                              max_retries=max_retries, partition_id=partition)
        self.session_id = self.sessions.create_session(options, partition_ids=[partition] if partition else None)
        self._mesh = {}  # real path of the sem directory -> result id

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.channel is not None:
            self.sessions.close_session(self.session_id)
            self.channel.close()
            self.channel = None

    def _upload(self, data):
        """Create results and upload their content in chunks; return name -> id."""
        meta = self.results.create_results_metadata(list(data), self.session_id)
        for name, b in data.items():
            self.results.upload_result_data(meta[name].result_id, self.session_id, b)
        return {name: meta[name].result_id for name in data}

    def _mesh_id(self, dfwd):
        sem = os.path.realpath(os.path.join(dfwd, "sem"))
        if sem not in self._mesh:
            self._mesh[sem] = self._upload({"mesh": armonik_io.pack_tree(sem)})["mesh"]
        return self._mesh[sem]

    def _wait(self, output_ids):
        """Wait until every output has a final status (DONE); TaskError beyond wait_timeout
        (partition without a worker, stuck task...). Uses polling: the SDK event stream has no
        timeout and ignores deleted outputs."""
        timeout = self.wait_timeout
        if timeout is None:
            timeout = self.max_duration * (self.max_retries + 1) * len(output_ids)
        deadline = time.monotonic() + timeout
        pending = list(output_ids)
        while True:
            pending = [r for r in pending if self.results.get_result(r).status not in DONE]
            if not pending:
                return
            if time.monotonic() > deadline:
                states = []
                for r in pending:
                    t = self.tasks.get_task(self.results.get_result(r).owner_task_id)
                    states.append(f"task {t.id}: {t.status.name}")
                raise TaskError(f"{len(pending)} output(s) not finished after {timeout:.0f} s "
                                f"(session {self.session_id}) :\n" + "\n".join(states))
            time.sleep(self.poll)

    def run(self, cfg, runner, jobs):
        if not jobs:
            return []
        cfg_dict = config.to_dict(cfg)
        data, deps = {}, []
        for k, j in enumerate(jobs):
            data[f"inputs-{k}"] = armonik_io.pack(j.fwd, runner.input_files(j.fwd))
            if j.grad:
                data[f"dobs-{k}"] = armonik_io.encode_array(j.dobs)
        ids = self._upload(data)
        payloads = {}
        for k, j in enumerate(jobs):
            d = dict(mesh=self._mesh_id(j.fwd), inputs=ids[f"inputs-{k}"], dobs=ids.get(f"dobs-{k}"))
            deps.append([v for v in d.values() if v])
            payloads[f"payload-{k}"] = json.dumps(dict(version=armonik_io.VERSION, cfg=cfg_dict,
                                                       grad=j.grad, **d)).encode()
        pids = self._upload(payloads)
        names = [f"output-{k}" for k in range(len(jobs))]
        meta = self.results.create_results_metadata(names, self.session_id)
        out_ids = [meta[n].result_id for n in names]
        self.tasks.submit_tasks(self.session_id, [
            TaskDefinition(payload_id=pids[f"payload-{k}"], expected_output_ids=[out_ids[k]], data_dependencies=deps[k])
            for k in range(len(jobs))])
        self._wait(out_ids)
        out, errors = [], []
        for j, rid in zip(jobs, out_ids):
            r = self.results.get_result(rid)
            if r.status == ResultStatus.COMPLETED:
                out.append(armonik_io.decode_result(self.results.download_result_data(rid, self.session_id)))
                continue
            t = self.tasks.get_task(r.owner_task_id)
            msg = ((t.output.error if t.output else None) or t.status_message
                   or f"output {ResultStatus.name_from_value(r.status)}")
            errors.append((j.index, t.id, msg))
        if errors:
            text = "\n".join(f"source {i} (task {tid}): {msg}" for i, tid, msg in errors)
            # the worker prefixes the error with its type: only a forward simulation failure can come
            # from the model, everything else (adjoint, timeout, infrastructure) is a TaskError
            forward = all(msg.startswith(f"{SimulationError.__name__}:") for _, _, msg in errors)
            raise (SimulationError if forward else TaskError)(text)
        return out
