# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for post-hoc quality summary (Phases 2-3)."""

import io
import sys

import pytest


class TestGetQualitySummary:
    """Test QuantizationReport.get_quality_summary()."""

    def test_basic_summary(self):
        """Returns correct aggregate metrics for a typical run."""
        from auto_round.report import QuantizationReport

        report = QuantizationReport(model_name="test", version="0.14.2")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3)
        report.add_layer("model.layers.1", cosine_sim=0.9870, psnr_db=50.1)
        report.add_layer("model.layers.2", cosine_sim=0.9990, psnr_db=52.0)

        summary = report.get_quality_summary()

        assert summary["total"] == 3
        assert summary["passed"] == 2  # layers 0 and 2
        assert summary["warn"] == 1    # layer 1
        assert abs(summary["avg_cosine_sim"] - (0.9998 + 0.9870 + 0.9990) / 3) < 1e-6
        assert abs(summary["avg_psnr_db"] - (52.3 + 50.1 + 52.0) / 3) < 0.01
        assert summary["min_cosine_sim"] == 0.9870
        assert summary["min_psnr_db"] == 50.1

    def test_all_passed(self):
        """All layers above thresholds → warn=0."""
        from auto_round.report import QuantizationReport

        report = QuantizationReport(model_name="test", version="0.14.2")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3)
        report.add_layer("model.layers.1", cosine_sim=0.9999, psnr_db=55.0)

        summary = report.get_quality_summary()

        assert summary["total"] == 2
        assert summary["passed"] == 2
        assert summary["warn"] == 0

    def test_infinite_psnr_excluded(self):
        """Infinite PSNR (perfect reconstruction) excluded from average."""
        from auto_round.report import QuantizationReport

        report = QuantizationReport(model_name="test", version="0.14.2")
        report.add_layer("model.layers.0", cosine_sim=1.0, psnr_db=float("inf"))
        report.add_layer("model.layers.1", cosine_sim=0.9998, psnr_db=52.3)

        summary = report.get_quality_summary()

        assert summary["avg_psnr_db"] == 52.3  # Only the finite value
        assert summary["min_psnr_db"] == 52.3
        assert summary["avg_cosine_sim"] == (1.0 + 0.9998) / 2

    def test_empty_report(self):
        """Empty report returns defaults."""
        from auto_round.report import QuantizationReport

        report = QuantizationReport(model_name="test", version="0.14.2")
        summary = report.get_quality_summary()

        assert summary["total"] == 0
        assert summary["passed"] == 0
        assert summary["warn"] == 0
        assert summary["avg_cosine_sim"] == 1.0
        assert summary["avg_psnr_db"] == float("inf")

    def test_single_layer(self):
        """Single layer summary matches that layer's values."""
        from auto_round.report import QuantizationReport

        report = QuantizationReport(model_name="test", version="0.14.2")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3)

        summary = report.get_quality_summary()

        assert summary["avg_cosine_sim"] == 0.9998
        assert summary["avg_psnr_db"] == 52.3
        assert summary["min_cosine_sim"] == 0.9998
        assert summary["min_psnr_db"] == 52.3


class TestAutoTunerSection:
    """Test QuantizationReport auto-tuner section in saved report."""

    def test_auto_tuner_section_in_report(self, tmp_path):
        """Report with auto_tuner_steps includes adjustments section."""
        from auto_round.report import QuantizationReport

        report = QuantizationReport(model_name="test", version="0.14.2")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3)
        report.auto_tuner_steps = [
            {"setting": "batch_size", "old": 8, "new": 4,
             "impact": "noisier gradients", "skipped": False},
            {"setting": "seqlen", "old": 2048, "new": 1024,
             "impact": "truncated context", "skipped": False},
            {"setting": "nsamples", "old": 512, "new": 512,
             "impact": "unchanged", "skipped": True},
        ]

        path = report.save(str(tmp_path))
        content = path.read_text()

        assert "Auto-Tuner Adjustments:" in content
        assert "batch_size" in content
        assert "8 → 4" in content
        assert "noisier gradients" in content
        # Skipped step should NOT appear
        assert "nsamples" not in content

    def test_no_auto_tuner_section_when_none(self, tmp_path):
        """Report without auto_tuner_steps omits adjustments section."""
        from auto_round.report import QuantizationReport

        report = QuantizationReport(model_name="test", version="0.14.2")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3)

        path = report.save(str(tmp_path))
        content = path.read_text()

        assert "Auto-Tuner Adjustments:" not in content

    def test_empty_auto_tuner_steps_no_section(self, tmp_path):
        """Report with empty auto_tuner_steps omits adjustments section."""
        from auto_round.report import QuantizationReport

        report = QuantizationReport(model_name="test", version="0.14.2")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3)
        report.auto_tuner_steps = []

        path = report.save(str(tmp_path))
        content = path.read_text()

        assert "Auto-Tuner Adjustments:" not in content


class TestCLIDisplayEnd:
    """Test CLIDisplay.end() with quality_summary and tune_steps."""

    def test_end_with_quality_summary(self):
        """end() with quality_summary prints quality line."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=4)
            display.begin()
            for i in range(4):
                display.print_sensitivity(
                    f"model.layers.{i}", 0.9998, 52.3,
                    init_loss=8.0e-6, best_loss=5.2e-6,
                    best_iter=950, total_iters=1000,
                )

            quality_summary = {
                "avg_cosine_sim": 0.9998,
                "avg_psnr_db": 52.3,
                "min_cosine_sim": 0.9997,
                "min_psnr_db": 52.2,
                "total": 4,
                "passed": 4,
                "warn": 0,
            }
            display.end(
                peak_ram_gb=31.25,
                peak_vram_gb=40.93,
                quality_summary=quality_summary,
            )
            output = sys.stdout.getvalue()
            assert "Quality:" in output
            assert "avg cosine 0.9998" in output
            assert "avg PSNR 52.3 dB" in output
            assert "4/4 passed" in output
            assert "0 warnings" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_end_with_tune_steps(self):
        """end() with tune_steps prints auto-tuner adjustments."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=4)
            display.begin()
            for i in range(4):
                display.print_sensitivity(
                    f"model.layers.{i}", 0.9998, 52.3,
                    init_loss=8.0e-6, best_loss=5.2e-6,
                    best_iter=950, total_iters=1000,
                )

            tune_steps = [
                {"setting": "batch_size", "old": 8, "new": 4,
                 "impact": "noisier gradients", "skipped": False},
                {"setting": "seqlen", "old": 2048, "new": 1024,
                 "impact": "truncated context", "skipped": False},
            ]
            display.end(
                peak_ram_gb=31.25,
                peak_vram_gb=40.93,
                tune_steps=tune_steps,
            )
            output = sys.stdout.getvalue()
            assert "Auto-tuner adjustments:" in output
            assert "batch_size" in output
            assert "8 → 4" in output
            assert "noisier gradients" in output
            assert "seqlen" in output
            assert "2048 → 1024" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_end_skipped_steps_not_shown(self):
        """end() does not show skipped auto-tuner steps."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=2)
            display.begin()
            for i in range(2):
                display.print_sensitivity(
                    f"model.layers.{i}", 0.9998, 52.3,
                    init_loss=8.0e-6, best_loss=5.2e-6,
                    best_iter=950, total_iters=1000,
                )

            tune_steps = [
                {"setting": "batch_size", "old": 8, "new": 4,
                 "impact": "noisier gradients", "skipped": False},
                {"setting": "nsamples", "old": 512, "new": 512,
                 "impact": "unchanged", "skipped": True},
            ]
            display.end(tune_steps=tune_steps)
            output = sys.stdout.getvalue()
            # Active step shown
            assert "batch_size" in output
            # Skipped step NOT shown
            assert "nsamples" not in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_end_backward_compatible(self):
        """end() without new params works exactly as before."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=2)
            display.begin()
            for i in range(2):
                display.print_sensitivity(
                    f"model.layers.{i}", 0.9998, 52.3,
                    init_loss=8.0e-6, best_loss=5.2e-6,
                    best_iter=950, total_iters=1000,
                )

            display.end(peak_ram_gb=31.25, peak_vram_gb=40.93)
            output = sys.stdout.getvalue()
            assert "Quantization complete" in output
            assert "2/2 blocks" in output
            assert "31.2 GB RAM" in output
            assert "40.9 GB VRAM" in output
            # No quality or auto-tuner lines
            assert "Quality:" not in output
            assert "Auto-tuner" not in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_end_with_warnings(self):
        """end() shows correct warn count when some layers fail."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=4)
            display.begin()
            for i in range(4):
                display.print_sensitivity(
                    f"model.layers.{i}", 0.9998, 52.3,
                    init_loss=8.0e-6, best_loss=5.2e-6,
                    best_iter=950, total_iters=1000,
                )

            quality_summary = {
                "avg_cosine_sim": 0.9990,
                "avg_psnr_db": 51.0,
                "min_cosine_sim": 0.9870,
                "min_psnr_db": 42.0,
                "total": 4,
                "passed": 3,
                "warn": 1,
            }
            display.end(quality_summary=quality_summary)
            output = sys.stdout.getvalue()
            assert "3/4 passed" in output
            assert "1 warnings" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_end_infinite_avg_psnr(self):
        """end() shows ∞ for infinite average PSNR."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=1)
            display.begin()
            display.print_sensitivity(
                "model.layers.0", 1.0, float("inf"),
                init_loss=0.0, best_loss=0.0,
                best_iter=0, total_iters=0,
            )

            quality_summary = {
                "avg_cosine_sim": 1.0,
                "avg_psnr_db": float("inf"),
                "min_cosine_sim": 1.0,
                "min_psnr_db": float("inf"),
                "total": 1,
                "passed": 1,
                "warn": 0,
            }
            display.end(quality_summary=quality_summary)
            output = sys.stdout.getvalue()
            assert "avg PSNR ∞ dB" in output
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled

    def test_end_all_three_sections(self):
        """end() with both quality_summary and tune_steps shows all sections."""
        from auto_round.cli_display import CLIDisplay, Colors

        old_stdout = sys.stdout
        old_enabled = Colors.ENABLED
        sys.stdout = io.StringIO()
        Colors.ENABLED = False
        try:
            display = CLIDisplay(total_blocks=2)
            display.begin()
            for i in range(2):
                display.print_sensitivity(
                    f"model.layers.{i}", 0.9998, 52.3,
                    init_loss=8.0e-6, best_loss=5.2e-6,
                    best_iter=950, total_iters=1000,
                )

            quality_summary = {
                "avg_cosine_sim": 0.9998,
                "avg_psnr_db": 52.3,
                "total": 2, "passed": 2, "warn": 0,
            }
            tune_steps = [
                {"setting": "batch_size", "old": 8, "new": 4,
                 "impact": "noisier gradients", "skipped": False},
            ]
            display.end(
                peak_ram_gb=31.25,
                peak_vram_gb=40.93,
                quality_summary=quality_summary,
                tune_steps=tune_steps,
            )
            output = sys.stdout.getvalue()
            # All three sections present
            assert "Quantization complete" in output
            assert "Auto-tuner adjustments:" in output
            assert "Quality:" in output
            # Verify ordering: completion, then adjustments, then quality
            complete_pos = output.index("Quantization complete")
            adjustments_pos = output.index("Auto-tuner adjustments:")
            quality_pos = output.index("Quality:")
            assert complete_pos < adjustments_pos < quality_pos
        finally:
            sys.stdout = old_stdout
            Colors.ENABLED = old_enabled
