# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Block-boundary checkpointing for resumable quantization.

AutoRound tunes transformer blocks sequentially: block ``N``'s tuning input is
the quantized output of block ``N-1`` chained through all preceding blocks (when
``enable_quanted_input`` is set, which is the default).  The only state needed to
resume tuning at block ``N`` is therefore the *activations entering block N* —
the floating-point reference input (``input_ids``), the quantized companion
input (``q_input``), and the static per-block side inputs (``input_others``).

This module persists exactly those tensors after each block completes, so a run
that is killed mid-way (OOM, crash, preemption) can be restarted and skip
straight to the first un-tuned block.  It deliberately does *not* re-load the
quantized weights of completed blocks into the tuning loop — once their output
activations are captured, those blocks are done as far as the loop is concerned.

Two environment variables drive it:

* ``AR_RESUME_DIR`` — directory to read/write block checkpoints.  When set,
  checkpointing is enabled; when a fresh process finds existing checkpoints in
  it, the quantization loop resumes from the next block.
* ``AR_STOP_AFTER_BLOCK`` — block index after which to raise
  :class:`StopAfterBlock`.  This exists to create a reproducible "interrupted"
  state for tests (and is reused by the resume feature's clean-stop path).
"""
from __future__ import annotations

import os
import random
import re
from typing import Any, Optional

import torch

from auto_round.logger import logger

_CKPT_RE = re.compile(r"^block_(\d{6})\.pt$")


def _capture_rng_state() -> dict:
    """Snapshot CPU/CUDA/NumPy/Python RNG so a resume re-enters the loop with
    the exact randomness state a clean run would have at the same seam."""
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[dict]) -> None:
    if not state:
        return
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class StopAfterBlock(Exception):
    """Raised to stop the quantization loop cleanly after a given block index.

    Carries the index of the block after which the stop fired so callers (and
    tests) can assert where the interruption happened.
    """

    def __init__(self, block_index: int):
        self.block_index = block_index
        super().__init__(f"stopped after block {block_index} (AR_STOP_AFTER_BLOCK)")


def _ckpt_name(block_index: int) -> str:
    return f"block_{block_index:06d}.pt"


def _to_cpu(obj: Any) -> Any:
    """Recursively move tensors in lists/tuples/dicts to CPU for serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    if isinstance(obj, list):
        return [_to_cpu(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    return obj


class BlockCheckpointer:
    """Persist and restore block-boundary activations for resumable tuning.

    A checkpoint file ``block_<index>.pt`` holds the activations *entering the
    next* block — i.e. after block ``index`` finishes, we save the inputs that
    block ``index + 1`` will consume.  ``latest_completed_index()`` therefore
    reports the highest fully-tuned block, and the loop resumes at
    ``index + step``.
    """

    def __init__(
        self,
        resume_dir: Optional[str],
        stop_after_block: Optional[int] = None,
    ):
        self.resume_dir = resume_dir
        self.stop_after_block = stop_after_block
        self.active = resume_dir is not None
        if self.active:
            os.makedirs(self.resume_dir, exist_ok=True)

    # ── Construction from environment ──────────────────────────────────────
    @classmethod
    def from_env(cls) -> "BlockCheckpointer":
        """Build a checkpointer from ``AR_RESUME_DIR`` / ``AR_STOP_AFTER_BLOCK``.

        Returns an inactive checkpointer (a no-op) when ``AR_RESUME_DIR`` is
        unset, so the default code path is entirely unaffected.
        """
        resume_dir = os.environ.get("AR_RESUME_DIR") or None
        stop_raw = os.environ.get("AR_STOP_AFTER_BLOCK")
        stop_after = None
        if stop_raw is not None and stop_raw != "":
            try:
                stop_after = int(stop_raw)
            except ValueError:
                logger.warning(
                    f"AR_STOP_AFTER_BLOCK={stop_raw!r} is not an integer; ignoring."
                )
        # A stop request alone (without a resume dir) is meaningless — the stop
        # path is only useful when checkpoints are being written.  Honour it
        # anyway by treating the stop as active so tests can force an interrupt,
        # but checkpoint persistence requires a dir.
        return cls(resume_dir=resume_dir, stop_after_block=stop_after)

    # ── Write side ─────────────────────────────────────────────────────────
    def save(
        self, block_index: int, *, input_ids, q_input, input_others, shard_state=None
    ) -> None:
        """Persist the activations entering block ``block_index + 1``.

        ``shard_state`` is the optional ShardWriter snapshot (from
        ``ShardWriter.export_state()``) needed to reassemble the final model on
        resume when immediate-saving is active.
        """
        if not self.active:
            return
        payload = {
            "block_index": block_index,
            "input_ids": _to_cpu(input_ids),
            "q_input": _to_cpu(q_input),
            "input_others": _to_cpu(input_others),
            "rng_state": _capture_rng_state(),
            "shard_state": shard_state,
        }
        # Write to a temp file then rename so a kill mid-write never leaves a
        # half-written checkpoint that resume would later trust.
        path = os.path.join(self.resume_dir, _ckpt_name(block_index))
        tmp_path = path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        logger.info(f"[resume] checkpointed block {block_index} -> {path}")

    def should_stop(self, block_index: int) -> bool:
        return self.stop_after_block is not None and block_index >= self.stop_after_block

    # ── Read side ──────────────────────────────────────────────────────────
    def latest_completed_index(self) -> Optional[int]:
        """Highest contiguous block index with a checkpoint on disk, or None.

        Requires contiguity from 0: a gap (e.g. a missing earlier checkpoint)
        stops the scan, since resuming past a gap would feed a later block the
        wrong activations.
        """
        if not self.active or not os.path.isdir(self.resume_dir):
            return None
        present = set()
        for fname in os.listdir(self.resume_dir):
            m = _CKPT_RE.match(fname)
            if m:
                present.add(int(m.group(1)))
        if not present:
            return None
        latest = -1
        i = 0
        while i in present:
            latest = i
            i += 1
        return latest if latest >= 0 else None

    def load(self, block_index: int) -> dict:
        """Load the activations entering block ``block_index + 1``."""
        path = os.path.join(self.resume_dir, _ckpt_name(block_index))
        # weights_only=False: payload contains lists/dicts of tensors, not a
        # bare state_dict.  Source is our own trusted scratch dir.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(f"[resume] loaded checkpoint for block {block_index} <- {path}")
        return payload

    @staticmethod
    def restore_rng(payload: dict) -> None:
        """Restore RNG state captured in a checkpoint payload (no-op if absent)."""
        _restore_rng_state(payload.get("rng_state"))
