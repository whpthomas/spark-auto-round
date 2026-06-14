# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for auto_round.cli_display — self-managed progress bar and sensitivity lines."""

import io
import sys

import pytest


class TestColors:
    """Test the Colors helper class."""

    def test_green_formatting(self):
        """Green wraps text in ANSI codes when TTY enabled."""
        from auto_round.cli_display import Colors

        old = Colors.ENABLED
        try:
            Colors.ENABLED = True
            result = Colors.green("hello")
            assert result == "\033[32mhello\033[0m"
        finally:
            Colors.ENABLED = old

    def test_green_passthrough_when_disabled(self):
        """Green returns plain text when TTY disabled."""
        from auto_round.cli_display import Colors

        old = Colors.ENABLED
        try:
            Colors.ENABLED = False
            result = Colors.green("hello")
            assert result == "hello"
        finally:
            Colors.ENABLED = old

    def test_orange_formatting(self):
        """Orange wraps text in ANSI codes when TTY enabled."""
        from auto_round.cli_display import Colors

        old = Colors.ENABLED
        try:
            Colors.ENABLED = True
            result = Colors.orange("hello")
            assert result == "\033[38;5;208mhello\033[0m"
        finally:
            Colors.ENABLED = old

    def test_orange_passthrough_when_disabled(self):
        """Orange returns plain text when TTY disabled."""
        from auto_round.cli_display import Colors

        old = Colors.ENABLED
        try:
            Colors.ENABLED = False
            result = Colors.orange("hello")
            assert result == "hello"
        finally:
            Colors.ENABLED = old

    def test_bold_formatting(self):
        """Bold wraps text in ANSI codes when TTY enabled."""
        from auto_round.cli_display import Colors

        old = Colors.ENABLED
        try:
            Colors.ENABLED = True
            result = Colors.bold("hello")
            assert result == "\033[1mhello\033[0m"
        finally:
            Colors.ENABLED = old

    def test_grey_formatting(self):
        """Grey wraps text in ANSI codes when TTY enabled."""
        from auto_round.cli_display import Colors

        old = Colors.ENABLED
        try:
            Colors.ENABLED = True
            result = Colors.grey("hello")
            assert result == "\033[90mhello\033[0m"
        finally:
            Colors.ENABLED = old


class TestCLIDisplay:
    """Test the CLIDisplay class."""

    def test_init(self):
        """Display initializes with total_blocks."""
        from auto_round.cli_display import CLIDisplay

        display = CLIDisplay(total_blocks=64)
        assert display._total_blocks == 64
        assert display._blocks_done == 0

    def test_format_time(self):
        """Time formatting works correctly."""
        from auto_round.cli_display import CLIDisplay

        assert CLIDisplay._format_time(65) == "1:05"
        assert CLIDisplay._format_time(3661) == "1:01:01"
        assert CLIDisplay._format_time(30) == "0:30"
        assert CLIDisplay._format_time(0) == "0:00"

    def test_print_sensitivity_increments_counter(self):
        """Each sensitivity call increments blocks_done."""
        from auto_round.cli_display import CLIDisplay

        display = CLIDisplay(total_blocks=4)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            display.begin()
            assert display._blocks_done == 0
            display.print_sensitivity("model.layers.0", 0.9998, 52.3, 5.2e-6)
            assert display._blocks_done == 1
            display.print_sensitivity("model.layers.1", 0.9870, 50.1, 8.0e-6)
            assert display._blocks_done == 2
        finally:
            sys.stdout = old_stdout

    def test_sensitivity_line_content(self):
        """Sensitivity line contains expected content."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=4)
            display.begin()
            display.print_sensitivity("model.layers.0", 0.9998, 52.3, 5.2e-6)
            output = sys.stdout.getvalue()
            assert "🟢" in output
            assert "model.layers.0" in output
            assert "Cosine similarity 0.9998" in output
            assert "Peak Signal-to-Noise Ratio 52.3" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_sensitivity_line_warn_icon(self):
        """Below-threshold metrics show 🟠 icon."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=4)
            display.begin()
            display.print_sensitivity("model.layers.1", 0.9870, 50.1, 8.0e-6)
            output = sys.stdout.getvalue()
            assert "🟠" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_sensitivity_line_below_cosine_threshold(self):
        """Cosine sim below threshold shows 🟠."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=2)
            display.begin()
            # cos_sim below 0.99, psnr above 45
            display.print_sensitivity("model.layers.0", 0.9800, 52.0, 1.0e-5)
            output = sys.stdout.getvalue()
            assert "🟠" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_sensitivity_line_below_psnr_threshold(self):
        """PSNR below threshold shows 🟠."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=2)
            display.begin()
            # cos_sim above 0.99, psnr below 45
            display.print_sensitivity("model.layers.0", 0.9998, 42.0, 1.0e-5)
            output = sys.stdout.getvalue()
            assert "🟠" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_sensitivity_line_infinite_psnr(self):
        """Infinite PSNR shown as ∞."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=2)
            display.begin()
            display.print_sensitivity("model.layers.0", 1.0, float("inf"), 0.0)
            output = sys.stdout.getvalue()
            assert "∞" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_end_prints_summary(self):
        """end() prints completion message."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=2)
            display.begin()
            display.print_sensitivity("model.layers.0", 0.9998, 52.3, 5.2e-6)
            display.print_sensitivity("model.layers.1", 0.9990, 52.0, 1.0e-5)
            display.end(peak_ram_gb=31.25, peak_vram_gb=40.93)
            output = sys.stdout.getvalue()
            assert "Quantization complete" in output
            assert "2/2 blocks" in output
            assert "31.2 GB RAM" in output
            assert "40.9 GB VRAM" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_end_without_memory(self):
        """end() works without memory info."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=1)
            display.begin()
            display.print_sensitivity("model.layers.0", 0.9998, 52.3, 5.2e-6)
            display.end()
            output = sys.stdout.getvalue()
            # Check last line specifically (end() output), not the full output
            last_line = output.strip().split("\n")[-1]
            assert "Quantization complete" in last_line
            assert "Peak" not in last_line
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_end_with_only_ram(self):
        """end() works with only RAM info."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=1)
            display.begin()
            display.print_sensitivity("model.layers.0", 0.9998, 52.3, 5.2e-6)
            display.end(peak_ram_gb=16.0)
            output = sys.stdout.getvalue()
            assert "Quantization complete" in output
            assert "16.0 GB RAM" in output
            assert "VRAM" not in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_end_with_only_vram(self):
        """end() works with only VRAM info."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=1)
            display.begin()
            display.print_sensitivity("model.layers.0", 0.9998, 52.3, 5.2e-6)
            display.end(peak_vram_gb=40.93)
            output = sys.stdout.getvalue()
            # Check last line specifically (end() output)
            last_line = output.strip().split("\n")[-1]
            assert "Quantization complete" in last_line
            # Only VRAM, no separate RAM info
            assert " GB RAM" not in last_line
            assert "40.9 GB VRAM" in last_line
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_begin_prints_progress(self):
        """begin() prints initial progress bar."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=64)
            display.begin()
            output = sys.stdout.getvalue()
            assert "Quantizing model.layers.0" in output
            assert "0%" in output
            assert "0/64" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_full_lifecycle(self):
        """Full lifecycle: begin → sensitivity → end."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=3)
            display.begin()
            display.print_sensitivity("model.layers.0", 0.9998, 52.3, 5.2e-6)
            display.print_sensitivity("model.layers.1", 0.9870, 50.1, 8.0e-6)
            display.print_sensitivity("model.layers.2", 0.9990, 52.0, 1.0e-5)
            display.end()
            output = sys.stdout.getvalue()
            # Check all sensitivity lines present
            assert "🟢 model.layers.0" in output
            assert "🟠 model.layers.1" in output
            assert "🟢 model.layers.2" in output
            # Check completion
            assert "Quantization complete" in output
            assert "3/3 blocks" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_update_progress_is_noop(self):
        """update_progress() doesn't change state or produce output."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=2)
            display.begin()
            output_before = sys.stdout.getvalue()
            display.update_progress("anything")
            output_after = sys.stdout.getvalue()
            # No new output from update_progress
            assert output_before == output_after
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_single_block(self):
        """Works with a single block."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=1)
            display.begin()
            display.print_sensitivity("model.layers.0", 0.9998, 52.3, 5.2e-6)
            display.end()
            output = sys.stdout.getvalue()
            assert "🟢 model.layers.0" in output
            assert "1/1 blocks" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_many_blocks(self):
        """Works with many blocks (64)."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=64)
            display.begin()
            for i in range(64):
                display.print_sensitivity(
                    f"model.language_model.layers.{i}",
                    0.9998,
                    52.3,
                    5.2e-6,
                )
            display.end()
            output = sys.stdout.getvalue()
            # Check first and last layer
            assert "🟢 model.language_model.layers.0" in output
            assert "🟢 model.language_model.layers.63" in output
            assert "64/64 blocks" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_zero_psnr(self):
        """Zero PSNR formatted as '0'."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=1)
            display.begin()
            display.print_sensitivity("model.layers.0", 1.0, 0.0, 0.0)
            output = sys.stdout.getvalue()
            assert "Peak Signal-to-Noise Ratio 0.0" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled
