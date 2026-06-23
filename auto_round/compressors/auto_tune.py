# Copyright (c) 2026 Intel Corporation
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
"""Auto-tuner for spark-auto-round quantization.

Adjusts user settings down a relaxation priority ladder until estimated peak
memory fits within available GPU memory. Runs pre-flight (before quantization)
and on resume (with OOM awareness).

Usage:
    from auto_round.compressors.auto_tune import auto_tune

    adjusted, steps = auto_tune(
        user_settings={"batch_size": 8, "seqlen": 2048, "nsamples": 512, "adam": True},
        model_config=config,
        available_memory=128 * 1024**3,  # 128 GB
        memory_utilization=0.75,
    )
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from auto_round.compressors.memory_estimator import estimate_peak_memory_per_block

# ---------------------------------------------------------------------------
# Relaxation priority ladder
# ---------------------------------------------------------------------------

# Each entry: (setting_key, step_values, impact_description)
# Values are listed from most aggressive (best quality) → most conservative.
# The tuner walks this list in order, advancing to the next level per iteration.
_RELAXATION_LADDER: List[Dict[str, Any]] = [
    {
        "key": "batch_size",
        "levels": [8, 4, 2, 1],
        "impact": "noisier gradients",
    },
    {
        "key": "seqlen",
        "levels": [2048, 1024, 512, 256],
        "impact": "truncated context",
    },
    {
        "key": "nsamples",
        "levels": [512, 256, 128],
        "impact": "less calibration coverage",
    },
    {
        "key": "adam",
        "levels": [True, False],  # enabled → disabled
        "impact": "SignSGD vs Adam optimizer",
    },
]

# Settings that the auto-tuner must never touch
_NEVER_AUTO_TUNE = {"iters", "group_size"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def auto_tune(
    user_settings: Dict[str, Any],
    model_config: "AutoConfig",  # noqa: F821
    available_memory: int,
    memory_utilization: float = 0.75,
    resume_context: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Adjust user settings to fit within available memory budget.

    Parameters
    ----------
    user_settings : dict
        Must contain: batch_size, seqlen, nsamples, adam (bool).
        May contain: group_size, iters (ignored by tuner).
    model_config : AutoConfig
        HuggingFace model config (for extracting hidden dimensions).
    available_memory : int
        GPU memory available in bytes (from torch.cuda.mem_get_info()).
    memory_utilization : float
        Fraction of available_memory to use as budget (default 0.75).
    resume_context : dict or None
        From checkpoint progress.json, if resuming:
            exit_reason : str | None     ("oom", "interrupted", None)
            oom_count : int              (number of prior OOMs, default 0)

    Returns
    -------
    adjusted_settings : dict
        Copy of user_settings with some values relaxed.
    steps : list[dict]
        Each dict: {"setting": str, "old": val, "new": val, "impact": str}.
        Empty if no changes were needed.
    """
    # -- Deep copy so we don't mutate caller's dict -------------------------
    adjusted = copy.deepcopy(user_settings)

    # -- Determine starting position in the relaxation ladder ----------------
    _, skip_count = _resolve_resume_offset(resume_context)

    # -- Compute budget ------------------------------------------------------
    budget_bytes = int(available_memory * memory_utilization)

    # -- Copy the ladder to avoid mutating the module-level list -------------
    ladder: List[Dict[str, Any]] = copy.deepcopy(_RELAXATION_LADDER)

    # -- Walk the relaxation ladder -----------------------------------------
    steps: List[Dict[str, Any]] = []
    ladder_idx = 0
    setting_idx = 0  # within the current ladder entry's levels

    # Fast-skip to resume offset (skip_n levels for OOM recover)
    for _ in range(skip_count):
        if ladder_idx >= len(ladder):
            break
        entry = ladder[ladder_idx]
        key = entry["key"]
        levels = entry["levels"]

        # Record the skip
        old_val = adjusted.get(key)
        # Advance: the first level is the one that OOM'd; skip it
        if len(levels) > 1:
            new_val = levels[1]
            levels.pop(0)  # discard the failed level
            # Update adjusted setting to the next level
            adjusted[key] = new_val
        else:
            # This ladder entry is exhausted; move to next
            ladder_idx += 1
            setting_idx = 0
            continue

        steps.append({
            "setting": key,
            "old": old_val,
            "new": new_val,
            "impact": entry["impact"],
            "skipped": True,  # marker for display
        })

    # Now the main relaxation loop
    while True:
        peak_gb, _ = estimate_peak_memory_per_block(model_config, adjusted)
        peak_bytes = int(peak_gb * (1024 ** 3))

        if peak_bytes <= budget_bytes:
            break  # fits within budget

        # Find the next setting to relax
        while ladder_idx < len(ladder):
            entry = ladder[ladder_idx]
            key = entry["key"]
            levels = entry["levels"]
            current_val = adjusted.get(key)

            # Find current value in levels list
            if current_val in levels:
                idx = levels.index(current_val)
                if idx < len(levels) - 1:
                    # Relax to next level
                    new_val = levels[idx + 1]
                    adjusted[key] = new_val
                    steps.append({
                        "setting": key,
                        "old": current_val,
                        "new": new_val,
                        "impact": entry["impact"],
                        "skipped": False,
                    })
                    setting_idx = idx + 1
                    break  # re-check peak with new setting
                else:
                    # Already at minimum for this setting — move to next ladder entry
                    ladder_idx += 1
                    setting_idx = 0
                    continue
            else:
                # Current value not in standard levels — set to the
                # lowest level (most conservative)
                adjusted[key] = levels[-1]
                steps.append({
                    "setting": key,
                    "old": current_val,
                    "new": levels[-1],
                    "impact": entry["impact"],
                    "skipped": False,
                })
                break
        else:
            # All settings at minimum — can't relax further
            break

    return adjusted, steps


def format_preflight_message(
    user_settings: Dict[str, Any],
    adjusted_settings: Dict[str, Any],
    steps: List[Dict[str, Any]],
    peak_gb: float,
    budget_gb: float,
    memory_utilization: float,
) -> str:
    """Return a plain-text string for the pre-flight log message.

    v1: No rich/tabulate dependency. Plain lines.
    """
    if not steps:
        if peak_gb <= budget_gb:
            return (
                f"Memory OK: est. peak {peak_gb:.1f} GB / {budget_gb:.1f} GB "
                f"({memory_utilization*100:.0f}%) — proceeding with user settings."
            )
        else:
            return (
                f"Memory budget exceeded (peak {peak_gb:.1f} GB / budget "
                f"{budget_gb:.1f} GB, {memory_utilization*100:.0f}%). "
                f"No further relaxations available — ⚠️ still exceeds budget."
            )

    skipped_steps = [s for s in steps if s.get("skipped")]
    active_steps = [s for s in steps if not s.get("skipped")]

    lines = ["Memory budget exceeded. Adjusting settings:"]
    for s in active_steps:
        lines.append(f"  {s['setting']:<12} {s['old']} → {s['new']:<8} ({s['impact']})")
    if skipped_steps:
        lines.append(
            "  (additionally, skipped {} OOM'd setting(s))".format(len(skipped_steps))
        )
    lines.append(
        f"Estimated peak: {peak_gb:.1f} GB / {budget_gb:.1f} GB "
        f"({memory_utilization*100:.0f}%) "
        + ("✓" if peak_gb <= budget_gb else "⚠️ still exceeds budget")
    )
    return "\n".join(lines)


def format_resume_message(
    completed: int,
    total: int,
    exit_reason: Optional[str],
    adjusted_settings: Dict[str, Any],
    steps: List[Dict[str, Any]],
    peak_gb: float,
    budget_gb: float,
    memory_utilization: float,
    oom_count: int = 0,
) -> str:
    """Return a plain-text string for resume log message."""
    lines = [
        f"Resuming from block {completed}/{total}.",
    ]
    if exit_reason == "oom":
        lines.append(f"Previous run OOM'd on block {completed}.")
        if oom_count > 0:
            lines.append(f"OOM count: {oom_count}. Accelerating relaxation.")
    elif exit_reason == "interrupted":
        lines.append("Previous run was interrupted (user stopped). Using fresh settings.")
    elif exit_reason is not None:
        lines.append(f"Previous exit reason: {exit_reason}.")

    active_steps = [s for s in steps if not s.get("skipped")]
    if active_steps:
        lines.append("Adjusting settings:")
        for s in active_steps:
            lines.append(
                f"  {s['setting']:<12} {s['old']} → {s['new']:<8} ({s['impact']})"
            )

    lines.append(
        f"Estimated peak: {peak_gb:.1f} GB / {budget_gb:.1f} GB "
        f"({memory_utilization*100:.0f}%)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_resume_offset(
    resume_context: Optional[Dict[str, Any]],
) -> Tuple[int, int]:
    """Determine how many ladder steps to skip based on resume context.

    Returns (start_step, skip_count).
    - interrupt: skip 0 (fresh auto-tune)
    - oom: skip (1 + floor(oom_count / 2))  — accelerates after repeated OOMs
    - None/unknown: skip 0 (fresh auto-tune with original margin)
    """
    if resume_context is None:
        return 0, 0

    exit_reason = resume_context.get("exit_reason")
    oom_count = max(0, resume_context.get("oom_count", 0))

    if exit_reason == "oom":
        # Skip one step by default, plus one extra per 2 OOMs
        skip = 1 + (oom_count // 2)
        return 0, skip

    # interrupt or unknown — fresh start
    return 0, 0