"""Canonical, path-independent model identity.

Experiment 1 discovered that a checkpoint's raw SHA-256 is *filename*-
sensitive: ``torch.save`` embeds a zip archive name derived from the output
filename, so the same weights written to ``policy_ppo.pt`` and
``ppo_repro.pt`` hash differently. That hash still has a job — it identifies
the exact artifact bytes — but it cannot answer "is this the same model?".

``parameter_sha256`` answers that. It hashes only the model's parameter
content: every state-dict entry in sorted key order, each contributing its
key name, dtype, shape, and raw CPU-contiguous tensor bytes, with explicit
separators so no two different structures can serialize to the same stream.
It depends on nothing else — not the filename, the archive metadata, the
filesystem path, or the timestamp.

    from ml.model_fingerprint import parameter_sha256, checkpoint_fingerprint
    parameter_sha256(net.state_dict())
    checkpoint_fingerprint("ml/policy_ppo.pt")   # both hashes for a file
"""

import hashlib
from typing import Dict, Mapping

_SEP = b"\x00"


def parameter_sha256(state: Mapping) -> str:
    """Deterministic hash of a state dict's parameter content alone."""
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        tensor = state[key]
        h.update(key.encode("utf-8"))
        h.update(_SEP)
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(_SEP)
        h.update(repr(tuple(tensor.shape)).encode("utf-8"))
        h.update(_SEP)
        # .detach().cpu().contiguous() so device, autograd state and stride
        # layout can never change the bytes for identical values.
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        h.update(_SEP)
    return h.hexdigest()


def file_sha256(path: str) -> str:
    """SHA-256 of the artifact bytes (filename-sensitive, by nature)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def checkpoint_parameter_sha256(path: str) -> str:
    """parameter_sha256 of a saved PolicyNet checkpoint's weights."""
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = blob["state"] if isinstance(blob, dict) and "state" in blob else blob
    return parameter_sha256(state)


def checkpoint_fingerprint(path: str) -> Dict[str, str]:
    """Both identities for a checkpoint file: artifact bytes and parameters."""
    return {"checkpoint_sha256": file_sha256(path),
            "parameter_sha256": checkpoint_parameter_sha256(path)}
