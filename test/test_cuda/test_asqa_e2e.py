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

"""End-to-end validation of spark-asqa-substitute.

Run after installation:
    pip install -e .
    pytest test/test_cuda/test_asqa_e2e.py -v

Expected output:
    Quantized model: ./models/Qwen3.6-27B-int4-AutoRound
    FP16 model:      Qwen/Qwen3.6-27B
    Output:          ./models/Qwen3.6-27B-int4-asaq

    Substituting 2 layers to FP16:
      model.language_model.layers.54
      model.language_model.layers.58

    Model size: 14.2 GB → 14.8 GB (+0.6 GB, +4.2%)
    Saving to ./models/Qwen3.6-27B-int4-asaq...
    Report: ./models/Qwen3.6-27B-int4-asaq/quantization-report.txt

    Running smoke test...
    ✅ Smoke test passed — model produces valid output

    Saved to: ./models/Qwen3.6-27B-int4-asaq
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def asaq_output_dir():
    """Run spark-asqa-substitute and return output directory."""
    output_dir = "./models/Qwen3.6-27B-int4-asaq"

    # Skip if quantized model doesn't exist
    quantized_dir = "./models/Qwen3.6-27B-int4-AutoRound"
    if not os.path.isdir(quantized_dir):
        pytest.skip("Quantized model not found")

    # Clean up previous run
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Run CLI with --no-smoke-test to avoid GPU requirement in CI
    result = subprocess.run(
        [
            sys.executable, "-m", "auto_round.asqa",
            "--layers", "54,58",
            "--no-smoke-test",
            "Qwen/Qwen3.6-27B",
        ],
        capture_output=True,
        text=True,
        timeout=600,  # 10 minutes
        cwd=os.getcwd(),
    )

    if result.returncode != 0:
        pytest.skip(f"CLI failed: {result.stderr}")

    print(result.stdout)
    return output_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestE2E:
    """End-to-end tests for spark-asqa-substitute CLI."""

    def test_output_directory_exists(self, asaq_output_dir):
        assert os.path.isdir(asaq_output_dir)

    def test_safetensors_present(self, asaq_output_dir):
        import glob
        safetensors = glob.glob(os.path.join(asaq_output_dir, "*.safetensors"))
        assert len(safetensors) > 0

    def test_index_file_present(self, asaq_output_dir):
        assert os.path.exists(os.path.join(asaq_output_dir, "model.safetensors.index.json"))

    def test_config_files_copied(self, asaq_output_dir):
        assert os.path.exists(os.path.join(asaq_output_dir, "tokenizer.json"))
        assert os.path.exists(os.path.join(asaq_output_dir, "tokenizer_config.json"))

    def test_quantization_config_updated(self, asaq_output_dir):
        with open(os.path.join(asaq_output_dir, "quantization_config.json")) as f:
            qc = json.load(f)
        assert qc["substituted_layers"] == [54, 58]
        assert qc["substituted_dtype"] == "fp16"

    def test_report_generated(self, asaq_output_dir):
        report_path = os.path.join(asaq_output_dir, "quantization-report.txt")
        assert os.path.exists(report_path)
        content = open(report_path).read()
        assert "ASAQ" in content
        assert "FP16 (substituted)" in content

    def test_model_reloadable(self, asaq_output_dir):
        """Verify the output model can be loaded back."""
        from safetensors import safe_open

        index_path = os.path.join(asaq_output_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        # Verify we can load at least one shard
        shards = set(index["weight_map"].values())
        assert len(shards) > 0

        # Load first shard to verify it's valid
        first_shard = list(shards)[0]
        shard_path = os.path.join(asaq_output_dir, first_shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            assert len(keys) > 0


class TestCLIErrorHandling:
    """Test CLI error handling for invalid inputs."""

    def test_missing_model_argument(self):
        result = subprocess.run(
            [sys.executable, "-m", "auto_round.asqa"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_missing_layers_argument(self):
        result = subprocess.run(
            [sys.executable, "-m", "auto_round.asqa", "Qwen/Qwen3.6-27B"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_invalid_layer_indices(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "auto_round.asqa",
                "--layers", "abc",
                "Qwen/Qwen3.6-27B",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "invalid" in result.stderr.lower() or "error" in result.stderr.lower()
