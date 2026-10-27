"""ArmoniK backend: one task per source, executed by the semfwi worker (armonik_worker).

The client prepares the forward directories as in local mode (mesh, model, input.spec); only
their input files, the mesh (once per session) and the data are sent to ArmoniK, and only the
results of task.run_source come back. Contract in armonik_io.

    backend = ArmoniKBackend(cfg)          # ARMONIK_ENDPOINT, default http://localhost:5001
    pb = Problem(cfg, runner, dobs, ..., backend=backend)
    ...
    backend.close()                        # or: with ArmoniKBackend(cfg) as backend
"""
import datetime
import json
import os

from armonik.client import ArmoniKEvents, ArmoniKResults, ArmoniKSessions, ArmoniKTasks
from armonik.common import ResultStatus, TaskDefinition, TaskOptions
from armonik.common.channel import create_channel

from . import armonik_io, config
from .runner import Runner, SimulationError


class ArmoniKBackend:
    def __init__(self, cfg, endpoint=None, partition=None, max_duration=3600., max_retries=1):
        self.cfg = cfg
        self.cfg_dict = config.to_dict(cfg)
        self.channel = create_channel(endpoint or os.environ.get("ARMONIK_ENDPOINT", "http://localhost:5001"))
        self.sessions, self.tasks = ArmoniKSessions(self.channel), ArmoniKTasks(self.channel)
        self.results, self.events = ArmoniKResults(self.channel), ArmoniKEvents(self.channel)
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
        """Wait until each output is completed or aborted (the SDK stops at the first aborted one)."""
        pending = set(output_ids)
        while pending:
            try:
                self.events.wait_for_result_availability(list(pending), self.session_id)
                return
            except RuntimeError:
                pending = {r for r in pending
                           if self.results.get_result(r).status not in (ResultStatus.COMPLETED, ResultStatus.ABORTED)}

    def run(self, jobs):
        if not jobs:
            return []
        data, deps = {}, []
        for k, j in enumerate(jobs):
            data[f"inputs-{k}"] = armonik_io.pack(j.fwd, Runner.input_files(j.fwd))
            if j.grad:
                data[f"dobs-{k}"] = armonik_io.encode_array(j.dobs)
        ids = self._upload(data)
        payloads = {}
        for k, j in enumerate(jobs):
            d = dict(mesh=self._mesh_id(j.fwd), inputs=ids[f"inputs-{k}"], dobs=ids.get(f"dobs-{k}"))
            deps.append([v for v in d.values() if v])
            payloads[f"payload-{k}"] = json.dumps(dict(version=armonik_io.VERSION, cfg=self.cfg_dict,
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
            else:
                t = self.tasks.get_task(r.owner_task_id)
                errors.append(f"source {j.index} (task {t.id}): "
                              + ((t.output.error if t.output else None) or t.status_message or "failed"))
        if errors:
            raise SimulationError("\n".join(errors))
        return out
