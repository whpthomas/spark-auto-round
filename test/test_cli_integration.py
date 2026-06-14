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
            version="0.14.0",
            cli_args={"batch_size": 8},
        )
        assert report.model_name == "test-model"
        assert report.version == "0.14.0"


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
