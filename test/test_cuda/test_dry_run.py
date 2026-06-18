"""Tests for --dry-run mode.

These tests verify that dry-run exercises the full init pipeline and writes
config files without quantizing the model or writing safetensors weights.
"""

import json
import os
import shutil
import tempfile
import time

import pytest
import torch


@pytest.fixture(scope="module")
def dry_run_opt_model_path():
    """Create a tiny OPT model for dry-run testing."""
    from ..helpers import get_model_path, save_tiny_model

    model_name = get_model_path("facebook/opt-125m")
    path = "./tmp/tiny_dry_run_opt"
    path = save_tiny_model(model_name, path, num_layers=2)
    yield path
    shutil.rmtree(path, ignore_errors=True)


class TestDryRunBasic:
    """Test that dry-run produces config files without weights."""

    def test_dry_run_produces_config_files(self, dry_run_opt_model_path):
        """dry-run should create config.json and quantization_config.json."""
        from auto_round import AutoRound

        with tempfile.TemporaryDirectory() as tmpdir:
            ar = AutoRound(
                model=dry_run_opt_model_path,
                format="auto_round",
                iters=0,
                dataset=None,
                dry_run=True,
            )
            # Use quantize_and_save instead of _save_config_dry_run
            model, output_dir = ar.quantize_and_save(tmpdir, format="auto_round")

            # Check config.json exists
            config_json_path = os.path.join(output_dir, "config.json")
            assert os.path.exists(config_json_path), "config.json should exist"

            # Check quantization_config.json exists
            qconfig_path = os.path.join(output_dir, "quantization_config.json")
            assert os.path.exists(qconfig_path), "quantization_config.json should exist"

    def test_dry_run_no_safetensors(self, dry_run_opt_model_path):
        """dry-run should NOT write any .safetensors files."""
        from auto_round import AutoRound

        with tempfile.TemporaryDirectory() as tmpdir:
            ar = AutoRound(
                model=dry_run_opt_model_path,
                format="auto_round",
                iters=0,
                dataset=None,
                dry_run=True,
            )
            model, output_dir = ar.quantize_and_save(tmpdir, format="auto_round")

            # No .safetensors files
            safetensors = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
            assert len(safetensors) == 0, f"Should not write safetensors, found: {safetensors}"

    def test_dry_run_no_quantization_report(self, dry_run_opt_model_path):
        """dry-run should NOT write quantization-report files."""
        from auto_round import AutoRound

        with tempfile.TemporaryDirectory() as tmpdir:
            ar = AutoRound(
                model=dry_run_opt_model_path,
                format="auto_round",
                iters=0,
                dataset=None,
                dry_run=True,
            )
            model, output_dir = ar.quantize_and_save(tmpdir, format="auto_round")

            # No quantization-report files
            report_files = [f for f in os.listdir(output_dir) if "quantization-report" in f]
            assert len(report_files) == 0, f"Should not write report files, found: {report_files}"

    def test_dry_run_quantization_config_content(self, dry_run_opt_model_path):
        """quantization_config.json should have required fields."""
        from auto_round import AutoRound

        with tempfile.TemporaryDirectory() as tmpdir:
            ar = AutoRound(
                model=dry_run_opt_model_path,
                format="auto_round",
                iters=0,
                dataset=None,
                dry_run=True,
            )
            model, output_dir = ar.quantize_and_save(tmpdir, format="auto_round")

            qconfig_path = os.path.join(output_dir, "quantization_config.json")
            with open(qconfig_path) as f:
                qconfig = json.load(f)

            # Required fields
            assert "block_name_to_quantize" in qconfig, "Missing block_name_to_quantize"
            assert "quant_method" in qconfig, "Missing quant_method"
            assert "packing_format" in qconfig, "Missing packing_format"
            assert qconfig["quant_method"] == "auto-round"

            # block_name_to_quantize should not be None
            assert qconfig["block_name_to_quantize"] is not None, "block_name_to_quantize is None"

    def test_dry_run_model_weights_unchanged(self, dry_run_opt_model_path):
        """dry-run should NOT modify model weights."""
        from auto_round import AutoRound
        from transformers import AutoModelForCausalLM

        # Capture weights before dry-run
        model_before = AutoModelForCausalLM.from_pretrained(
            dry_run_opt_model_path, trust_remote_code=True
        )
        weights_before = {
            name: param.clone()
            for name, param in model_before.named_parameters()
        }
        del model_before

        with tempfile.TemporaryDirectory() as tmpdir:
            ar = AutoRound(
                model=dry_run_opt_model_path,
                format="auto_round",
                iters=0,
                dataset=None,
                dry_run=True,
            )
            model, output_dir = ar.quantize_and_save(tmpdir, format="auto_round")

            # Compare weights
            model_after = ar.model_context.model
            for name, param in model_after.named_parameters():
                if name in weights_before:
                    assert torch.equal(param, weights_before[name]), (
                        f"Weight {name} was modified by dry-run"
                    )

    def test_dry_run_speed(self, dry_run_opt_model_path):
        """dry-run should complete in under 15 seconds."""
        from auto_round import AutoRound

        with tempfile.TemporaryDirectory() as tmpdir:
            start = time.time()
            ar = AutoRound(
                model=dry_run_opt_model_path,
                format="auto_round",
                iters=0,
                dataset=None,
                dry_run=True,
            )
            model, output_dir = ar.quantize_and_save(tmpdir, format="auto_round")
            elapsed = time.time() - start

            assert elapsed < 15, f"dry-run took {elapsed:.1f}s, should be < 15s"


class TestDryRunCLI:
    """Test the CLI --dry-run flag."""

    def test_cli_dry_run_flag_exists(self):
        """CLI should accept --dry-run flag."""
        from auto_round.__main__ import BasicArgumentParser

        parser = BasicArgumentParser()

        # Should parse without error
        args = parser.parse_args(["some-model", "--dry-run"])
        assert args.dry_run is True

    def test_cli_dry_run_default_false(self):
        """--dry-run should default to False."""
        from auto_round.__main__ import BasicArgumentParser

        parser = BasicArgumentParser()

        args = parser.parse_args(["some-model"])
        assert args.dry_run is False


class TestDryRunViaQuantizeAndSave:
    """Test dry-run through the public quantize_and_save() API."""

    def test_quantize_and_save_dry_run(self, dry_run_opt_model_path):
        """quantize_and_save with dry_run=True should return early."""
        from auto_round import AutoRound

        with tempfile.TemporaryDirectory() as tmpdir:
            ar = AutoRound(
                model=dry_run_opt_model_path,
                format="auto_round",
                iters=0,
                dataset=None,
                dry_run=True,
            )

            # This should NOT call quantize()
            model, output_dir = ar.quantize_and_save(tmpdir, format="auto_round")

            # Should return model and output_dir
            assert model is not None
            assert output_dir is not None
            assert os.path.isdir(output_dir), f"output_dir should exist: {output_dir}"

            # Config files should exist in the export directory
            qconfig_path = os.path.join(output_dir, "quantization_config.json")
            assert os.path.exists(qconfig_path), (
                f"quantization_config.json should exist at {qconfig_path}"
            )

            config_path = os.path.join(output_dir, "config.json")
            assert os.path.exists(config_path), (
                f"config.json should exist at {config_path}"
            )

            # No safetensors should be written
            safetensors = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
            assert len(safetensors) == 0, f"Should not write safetensors, found: {safetensors}"
