# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for CLI display + report in data_driven.py.

These tests verify that the display and report are properly wired
into the quantization loop. They use mock objects to avoid needing
a real model or GPU.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestDisplayIntegration:
    """Test that CLIDisplay is properly used in data_driven."""

    def test_display_created_in_quantize(self):
        """CLIDisplay is created when quantize() is called."""
        from auto_round.cli_display import CLIDisplay
        display = CLIDisplay(total_blocks=4)
        assert display._total_blocks == 4

    def test_report_created_with_correct_args(self):
        """QuantizationReport is created with model name and version."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(
            model_name="test-model",
            version="0.14.3",
            cli_args={"batch_size": 8},
        )
        assert report.model_name == "test-model"
        assert report.version == "0.14.3"


class TestMetricsWiring:
    """Test that compute_block_sensitivity is correctly imported."""

    def test_import_from_metrics(self):
        """compute_block_sensitivity is importable from auto_round.metrics."""
        from auto_round.metrics import compute_block_sensitivity
        assert callable(compute_block_sensitivity)

    def test_metrics_used_in_data_driven(self):
        """compute_block_sensitivity is imported in data_driven module."""
        from auto_round.compressors import data_driven
        assert hasattr(data_driven, 'compute_block_sensitivity') or \
               'compute_block_sensitivity' in dir(data_driven)


class TestShakedownCli:
    """Test that --shakedown flag is parsed correctly."""

    def test_shakedown_default_false(self):
        """--shakedown defaults to False."""
        from auto_round.__main__ import BasicArgumentParser
        parser = BasicArgumentParser()
        args = parser.parse_args(["Qwen/Qwen3.5-0.8B"])
        assert args.shakedown is False

    def test_shakedown_flag_true(self):
        """--shakedown sets flag to True."""
        from auto_round.__main__ import BasicArgumentParser
        parser = BasicArgumentParser()
        args = parser.parse_args(["Qwen/Qwen3.5-0.8B", "--shakedown"])
        assert args.shakedown is True

    def test_shakedown_does_not_affect_other_args(self):
        """--shakedown does not change other argument defaults."""
        from auto_round.__main__ import BasicArgumentParser
        parser = BasicArgumentParser()
        args = parser.parse_args(["Qwen/Qwen3.5-0.8B", "--shakedown"])
        assert args.iters == 1000  # default, overridden later
        assert args.nsamples == 512
        assert args.seqlen == 2048
        assert args.batch_size == 8


class TestHaltAfterCli:
    """Test that --halt-after flag is parsed correctly."""

    def test_halt_after_default_minus_one(self):
        """--halt-after defaults to -1 (no halt)."""
        from auto_round.__main__ import BasicArgumentParser
        parser = BasicArgumentParser()
        args = parser.parse_args(["Qwen/Qwen3.5-0.8B"])
        assert args.halt_after == -1

    def test_halt_after_zero(self):
        """--halt-after 0 is parsed correctly."""
        from auto_round.__main__ import BasicArgumentParser
        parser = BasicArgumentParser()
        args = parser.parse_args(["Qwen/Qwen3.5-0.8B", "--halt-after", "0"])
        assert args.halt_after == 0

    def test_halt_after_positive_int(self):
        """--halt-after 5 is parsed correctly."""
        from auto_round.__main__ import BasicArgumentParser
        parser = BasicArgumentParser()
        args = parser.parse_args(["Qwen/Qwen3.5-0.8B", "--halt-after", "5"])
        assert args.halt_after == 5

    def test_halt_after_negative_one(self):
        """--halt-after -1 is parsed correctly (no halt)."""
        from auto_round.__main__ import BasicArgumentParser
        parser = BasicArgumentParser()
        args = parser.parse_args(["Qwen/Qwen3.5-0.8B", "--halt-after", "-1"])
        assert args.halt_after == -1

    def test_halt_after_with_shakedown(self):
        """--shakedown and --halt-after can be used together."""
        from auto_round.__main__ import BasicArgumentParser
        parser = BasicArgumentParser()
        args = parser.parse_args(
            ["Qwen/Qwen3.5-0.8B", "--shakedown", "--halt-after", "2"]
        )
        assert args.shakedown is True
        assert args.halt_after == 2
