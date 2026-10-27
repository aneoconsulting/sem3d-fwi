"""Contract between the ArmoniK backend (client) and the worker: one task per source.

Payload (JSON):
  {"version": 1, "cfg": config.to_dict(cfg), "grad": bool,
   "mesh": id, "inputs": id, "dobs": id or null}
Data dependencies (ArmoniK result ids referenced in the payload):
  mesh   : tar of the mesher's sem directory (mesh4spec.*), shared by all tasks;
  inputs : tar of the input files of the forward directory (Runner.input_files);
  dobs   : observed data of the source, .npy (nsta, nt, 3), if grad.
Output: .npz from task.run_source (u, plus gl, gm if grad).

This module does not depend on the ArmoniK SDK.
"""
import io
import os
import tarfile

import numpy as np

VERSION = 1


class InvalidPayload(ValueError):
    pass


def pack(d, files):
    """Uncompressed tar of the given files of directory d, under their own names."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for fn in files:
            tar.add(os.path.join(d, fn), arcname=fn)
    return buf.getvalue()


def pack_tree(d):
    return pack(d, sorted(os.listdir(d)))


def unpack(data, d):
    os.makedirs(d, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
        tar.extractall(d, filter="data")


def encode_array(a):
    buf = io.BytesIO()
    np.save(buf, np.asarray(a), allow_pickle=False)
    return buf.getvalue()


def decode_array(data):
    return np.load(io.BytesIO(data), allow_pickle=False)


def encode_result(r):
    buf = io.BytesIO()
    np.savez(buf, **r)
    return buf.getvalue()


def decode_result(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as st:
        return {k: st[k] for k in st.files}
