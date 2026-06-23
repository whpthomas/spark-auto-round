# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for auto_round.report — quantization report collection and writing."""

import pytest


class TestQuantizationReport:
    """Test the QuantizationReport class."""

    def test_add_layer_pass(self):
        """Layer above thresholds → passed=True."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="test", version="0.14.3")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3,
                         init_loss=8.0e-6, best_loss=5.2e-6, best_iter=950, total_iters=1000)
        assert len(report.layers) == 1
        assert report.layers[0].passed is True
        assert report.layers[0].init_loss == 8.0e-6
        assert report.layers[0].best_loss == 5.2e-6
        assert report.layers[0].best_iter == 950
        assert report.layers[0].total_iters == 1000

    def test_add_layer_warn_cosine(self):
        """Cosine below threshold → passed=False."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="test", version="0.14.3")
        report.add_layer("model.layers.1", cosine_sim=0.9870, psnr_db=50.1,
                         init_loss=1.0e-5, best_loss=8.0e-6, best_iter=800, total_iters=1000)
        assert report.layers[0].passed is False

    def test_add_layer_warn_psnr(self):
        """PSNR below threshold → passed=False."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="test", version="0.14.3")
        report.add_layer("model.layers.2", cosine_sim=0.9990, psnr_db=42.0,
                         init_loss=2.0e-5, best_loss=1.0e-5, best_iter=700, total_iters=1000)
        assert report.layers[0].passed is False

    def test_add_layer_both_below(self):
        """Both below threshold → passed=False."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="test", version="0.14.3")
        report.add_layer("model.layers.3", cosine_sim=0.95, psnr_db=30.0,
                         init_loss=0.02, best_loss=0.01, best_iter=500, total_iters=1000)
        assert report.layers[0].passed is False

    def test_get_summary(self):
        """Summary counts are correct."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="test", version="0.14.3")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3,
                         init_loss=8.0e-6, best_loss=5.2e-6, best_iter=950, total_iters=1000)
        report.add_layer("model.layers.1", cosine_sim=0.9870, psnr_db=50.1,
                         init_loss=1.0e-5, best_loss=8.0e-6, best_iter=800, total_iters=1000)
        report.add_layer("model.layers.2", cosine_sim=0.9990, psnr_db=52.0,
                         init_loss=2.0e-5, best_loss=1.0e-5, best_iter=700, total_iters=1000)
        summary = report.get_summary()
        assert summary["total"] == 3
        assert summary["passed"] == 2
        assert summary["warn"] == 1

    def test_save_creates_file(self, tmp_path):
        """Report saves to quantization-report.txt."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="Qwen/Qwen3.5-0.8B", version="0.14.3")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3,
                         init_loss=8.0e-6, best_loss=5.2e-6, best_iter=950, total_iters=1000)
        report.set_memory_summary(peak_ram_gb=31.25, peak_vram_gb=40.93)
        path = report.save(str(tmp_path))
        assert path.exists()
        assert path.name == "quantization-report.txt"
        content = path.read_text()
        assert "=== Quantization Report ===" in content
        assert "Qwen/Qwen3.5-0.8B" in content
        assert "0.14.3" in content
        assert "31.25" in content

    def test_report_format_sections(self, tmp_path):
        """Report contains all required sections."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(
            model_name="test-model",
            version="0.14.3",
            cli_args={"batch_size": 8, "iters": 1000, "scheme": "W4A16"},
        )
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3,
                         init_loss=8.0e-6, best_loss=5.2e-6, best_iter=950, total_iters=1000)
        report.add_layer("model.layers.1", cosine_sim=0.9870, psnr_db=50.1,
                         init_loss=1.0e-5, best_loss=8.0e-6, best_iter=800, total_iters=1000)
        report.set_memory_summary(peak_ram_gb=31.25, peak_vram_gb=40.93)
        path = report.save(str(tmp_path))
        content = path.read_text()
        assert "CLI Arguments:" in content
        assert "--batch_size 8" in content
        assert "--iters 1000" in content
        assert "Memory Summary:" in content
        assert "Peak RAM:" in content
        assert "Peak VRAM:" in content
        assert "Sensitivity Analysis:" in content
        assert "🟢" in content
        assert "🟠" in content
        assert "Summary:" in content
        assert "Thresholds:" in content

    def test_report_status_icons(self, tmp_path):
        """Correct emoji icons in report."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="test", version="0.14.3")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3,
                         init_loss=8.0e-6, best_loss=5.2e-6, best_iter=950, total_iters=1000)
        report.add_layer("model.layers.1", cosine_sim=0.9870, psnr_db=50.1,
                         init_loss=1.0e-5, best_loss=8.0e-6, best_iter=800, total_iters=1000)
        path = report.save(str(tmp_path))
        content = path.read_text()
        lines = content.split("\n")
        layer0_line = [l for l in lines if "model.layers.0" in l][0]
        layer1_line = [l for l in lines if "model.layers.1" in l][0]
        assert "🟢" in layer0_line
        assert "PASS" in layer0_line
        assert "🟠" in layer1_line
        assert "WARN" in layer1_line

    def test_report_infinite_psnr(self, tmp_path):
        """Perfect reconstruction → inf PSNR displayed as ∞."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="test", version="0.14.3")
        report.add_layer("model.layers.0", cosine_sim=1.0, psnr_db=float("inf"),
                         init_loss=0.0, best_loss=0.0, best_iter=0, total_iters=0)
        path = report.save(str(tmp_path))
        content = path.read_text()
        assert "∞" in content

    def test_report_empty(self, tmp_path):
        """Empty report (no layers) still writes valid file."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="test", version="0.14.3")
        path = report.save(str(tmp_path))
        content = path.read_text()
        assert "=== Quantization Report ===" in content
        assert "Total blocks: 0" in content

    def test_report_no_memory(self, tmp_path):
        """Report without memory summary still valid."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="test", version="0.14.3")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3,
                         init_loss=8.0e-6, best_loss=5.2e-6, best_iter=950, total_iters=1000)
        path = report.save(str(tmp_path))
        content = path.read_text()
        assert "(not available)" in content

    def test_save_creates_csv(self, tmp_path):
        """save() also creates quantization-report.csv."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="test", version="0.14.3")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3,
                         init_loss=8.0e-6, best_loss=5.2e-6, best_iter=950, total_iters=1000)
        report.save(str(tmp_path))
        csv_path = tmp_path / "quantization-report.csv"
        assert csv_path.exists()

    def test_csv_headers(self, tmp_path):
        """CSV has correct column headers."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="test", version="0.14.3")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3)
        report.save_csv(str(tmp_path))
        csv_path = tmp_path / "quantization-report.csv"
        lines = csv_path.read_text().strip().split("\n")
        headers = lines[0]
        expected = "block_name,cosine_sim,psnr_db,init_loss,best_loss,best_iter,total_iters,passed"
        assert headers == expected

    def test_csv_content(self, tmp_path):
        """CSV rows match added layers."""
        import csv
        from io import StringIO
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="test", version="0.14.3")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3,
                         init_loss=8.0e-6, best_loss=5.2e-6, best_iter=950, total_iters=1000)
        report.add_layer("model.layers.1", cosine_sim=0.9870, psnr_db=50.1,
                         init_loss=1.0e-5, best_loss=8.0e-6, best_iter=800, total_iters=1000)
        report.save_csv(str(tmp_path))
        csv_path = tmp_path / "quantization-report.csv"
        reader = csv.DictReader(csv_path.open())
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["block_name"] == "model.layers.0"
        assert rows[0]["cosine_sim"] == "0.9998"
        assert rows[0]["passed"] == "True"
        assert rows[1]["block_name"] == "model.layers.1"
        assert rows[1]["passed"] == "False"

    def test_csv_empty_report(self, tmp_path):
        """Empty report produces CSV with only headers."""
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="test", version="0.14.3")
        report.save_csv(str(tmp_path))
        csv_path = tmp_path / "quantization-report.csv"
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 1  # header only

    def test_csv_none_losses(self, tmp_path):
        """Layers with None losses write empty CSV fields."""
        import csv
        from auto_round.report import QuantizationReport
        report = QuantizationReport(model_name="test", version="0.14.3")
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3)
        report.save_csv(str(tmp_path))
        csv_path = tmp_path / "quantization-report.csv"
        reader = csv.DictReader(csv_path.open())
        rows = list(reader)
        assert rows[0]["init_loss"] == ""
        assert rows[0]["best_loss"] == ""
        assert rows[0]["best_iter"] == "0"
        assert rows[0]["total_iters"] == "0"
