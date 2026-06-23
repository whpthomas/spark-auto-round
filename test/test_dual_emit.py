# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test that pre-flight message is dual-emitted (print + logger.info)."""

import logging
from unittest.mock import patch, MagicMock

import pytest


def _make_args(**overrides):
    """Build a minimal argparse.Namespace for tune()."""
    import argparse
    defaults = dict(
        model="test-model",
        batch_size=8, seqlen=2048, nsamples=512, iters=1000,
        group_size=128, dataset="opencode-instruct", output_dir="/tmp/test",
        seed=42, disable_torch_compile=False, memory_utilization=75,
        memory_budget=96, trust_remote_code=True, dry_run=True,
        clear_cache=False, lr=None, minmax_lr=None,
        quant_lm_head=False, ignore_layers="", layer_config=None,
        model_dtype=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _run_tune_with_patches(capsys, extra_patches=None):
    """Run tune() with all necessary mocks. Returns captured stdout."""
    from auto_round.__main__ import tune

    patches = {
        # estimate_memory_strategy is imported locally in tune() from auto_round.utils.device
        "auto_round.utils.device.estimate_memory_strategy": MagicMock(
            return_value=(False, {"strategy": "whole-model", "block_size_bytes": 1e9})
        ),
        "auto_round.__main__.AutoConfig": MagicMock(),
        # auto_tune is imported at module level in __main__.py, so patch the local name
        "auto_round.__main__.auto_tune": MagicMock(return_value=(
            {"batch_size": 4, "seqlen": 1024},
            [{"setting": "batch_size", "old": 8, "new": 4, "impact": "noisier gradients"}],
        )),
        "auto_round.compressors.memory_estimator.estimate_peak_memory_per_block": MagicMock(
            return_value=(32.0, None)
        ),
        "auto_round.AutoRound": MagicMock(),
        "auto_round.__main__.clear_memory": MagicMock(),
        "auto_round.utils.device.log_memory_analysis": MagicMock(),
    }
    if extra_patches:
        patches.update(extra_patches)

    # Configure AutoConfig mock
    patches["auto_round.__main__.AutoConfig"].from_pretrained.return_value = {
        "model_type": "qwen3"
    }
    patches["auto_round.AutoRound"].return_value.quantize_and_save.return_value = (None, [])

    # Enter all context managers
    active_patches = {}
    for key, value in patches.items():
        cm = patch(key, value)
        active_patches[key] = (cm, cm.__enter__())

    try:
        args = _make_args()
        tune(args)
        return capsys.readouterr()
    finally:
        for cm, _ in active_patches.values():
            cm.__exit__(None, None, None)


class TestDualEmit:
    """Verify pre-flight message reaches stdout via print() regardless of log level."""

    def test_preflight_message_always_printed(self, capsys):
        """Pre-flight message is printed to stdout even when logger is at WARNING level."""
        root_logger = logging.getLogger("auto_round")
        old_level = root_logger.level
        root_logger.setLevel(logging.WARNING)
        try:
            captured = _run_tune_with_patches(capsys)
            assert "batch_size" in captured.out
            assert "noisier gradients" in captured.out
        finally:
            root_logger.setLevel(old_level)

    def test_logger_info_still_called(self, capsys):
        """logger.info() is still called with the pre-flight message (log files preserved)."""
        # Patch logger at its source
        with patch("auto_round.utils.logger") as mock_logger:
            captured = _run_tune_with_patches(capsys, extra_patches={
                "auto_round.utils.logger": mock_logger,
            })
            # Find the call with the pre-flight message (contains "batch_size")
            info_calls = [call for call in mock_logger.info.call_args_list
                          if call[0] and "batch_size" in str(call[0][0])]
            assert len(info_calls) >= 1, "logger.info was never called with the pre-flight message"

    def test_memory_ok_message_printed(self, capsys):
        """Memory-OK pre-flight message (no adjustments) is also printed."""
        captured = _run_tune_with_patches(capsys, extra_patches={
            "auto_round.__main__.auto_tune": MagicMock(return_value=(
                {"batch_size": 8, "seqlen": 2048},
                [],
            )),
        })
        assert "Memory OK" in captured.out
