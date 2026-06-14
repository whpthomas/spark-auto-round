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

"""Integration tests for ASAQ — requires CUDA and Qwen3.6-27B model.

These tests verify the full end-to-end substitution workflow with real model
weights. They require:
- CUDA GPU
- Pre-quantized model at ./models/Qwen3.6-27B-int4-AutoRound
- Access to Qwen/Qwen3.6-27B from HuggingFace cache or Hub

Run with:
    pytest test/test_cuda/test_asqa_integration.py -v
"""

from __future__ import annotations

import json
import os

import pytest

from auto_round.asqa import (
    compute_model_size,
    copy_config_files,
    generate_asaq_report,
    infer_paths,
    load_fp16_layers,
    load_quantized_weights,
    save_model,
    substitute_layers,
    update_quantization_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def quantized_model_path():
    """Path to pre-quantized Qwen3.6-27B."""
    try:
        path, _, _ = infer_paths("Qwen/Qwen3.6-27B")
        return path
    except FileNotFoundError:
        pytest.skip("Quantized model not found at ./models/Qwen3.6-27B-int4-AutoRound")


@pytest.fixture(scope="module")
def quantized_weights(quantized_model_path):
    """Load quantized model weights (module-scoped for efficiency)."""
    weights, config = load_quantized_weights(quantized_model_path)
    return weights, config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInferPathsIntegration:
    """Test path inference with real model directory."""

    def test_infer_paths_returns_correct_structure(self, quantized_model_path):
        q_path, fp16_id, out_dir = infer_paths("Qwen/Qwen3.6-27B")
        assert q_path.endswith("Qwen3.6-27B-int4-AutoRound")
        assert fp16_id == "Qwen/Qwen3.6-27B"
        assert "asaq" in out_dir.lower()


class TestLoadQuantizedWeightsIntegration:
    """Test loading real quantized model weights."""

    def test_loads_weights(self, quantized_weights):
        weights, config = quantized_weights
        assert len(weights) > 0
        assert "model_type" in config or "architectures" in config

    def test_has_quantized_tensors(self, quantized_weights):
        weights, _ = quantized_weights
        qweight_count = sum(1 for k in weights if k.endswith(".qweight"))
        assert qweight_count > 0, "Expected quantized tensors in model"

    def test_has_fp16_tensors(self, quantized_weights):
        weights, _ = quantized_weights
        # Norms should be FP16
        norm_count = sum(1 for k in weights if "layernorm.weight" in k)
        assert norm_count > 0, "Expected layernorm tensors in model"


class TestSubstituteLayersIntegration:
    """Test layer substitution with real weights."""

    def test_substitute_single_layer(self, quantized_weights):
        weights, _ = quantized_weights
        original_keys = set(weights.keys())

        # Load FP16 layer 54
        fp16 = load_fp16_layers("Qwen/Qwen3.6-27B", [54])

        # Substitute
        weights = substitute_layers(weights, fp16, [54])

        # Verify quantized tensors removed
        layer_54_qweight = [k for k in weights if "layers.54." in k and k.endswith(".qweight")]
        assert len(layer_54_qweight) == 0, "Quantized tensors should be removed"

        # Verify FP16 tensors added
        layer_54_weight = [k for k in weights if "layers.54." in k and k.endswith(".weight")]
        assert len(layer_54_weight) > 0, "FP16 weight tensors should be added"

    def test_substitute_preserves_other_layers(self, quantized_weights):
        weights, _ = quantized_weights
        original_q54 = sum(1 for k in weights if "layers.54." in k and k.endswith(".qweight"))
        original_q58 = sum(1 for k in weights if "layers.58." in k and k.endswith(".qweight"))

        # Load FP16 for layer 54 only
        fp16 = load_fp16_layers("Qwen/Qwen3.6-27B", [54])
        weights = substitute_layers(weights, fp16, [54])

        # Layer 54 quantized tensors should be removed
        new_q54 = sum(1 for k in weights if "layers.54." in k and k.endswith(".qweight"))
        assert new_q54 == 0

        # Layer 58 quantized tensors should still be there
        new_q58 = sum(1 for k in weights if "layers.58." in k and k.endswith(".qweight"))
        assert new_q58 == original_q58


class TestSaveModelIntegration:
    """Test saving model with real weights."""

    def test_save_and_reload(self, quantized_weights, tmp_path):
        weights, config = quantized_weights

        # Substitute layer 54
        fp16 = load_fp16_layers("Qwen/Qwen3.6-27B", [54])
        weights = substitute_layers(weights, fp16, [54])

        # Save
        output_dir = str(tmp_path / "test-model")
        save_model(weights, config, output_dir)

        # Reload and verify
        reloaded_weights, reloaded_config = load_quantized_weights(output_dir)
        assert len(reloaded_weights) == len(weights)
        assert reloaded_config == config


class TestCopyConfigFilesIntegration:
    """Test copying config files from real model."""

    def test_copies_tokenizer_files(self, quantized_model_path, tmp_path):
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir)

        copy_config_files(quantized_model_path, output_dir)

        # Check tokenizer files exist
        assert os.path.exists(os.path.join(output_dir, "tokenizer.json"))
        assert os.path.exists(os.path.join(output_dir, "tokenizer_config.json"))


class TestUpdateQuantizationConfigIntegration:
    """Test updating quantization config."""

    def test_creates_config_with_substituted_layers(self, quantized_model_path, tmp_path):
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir)

        # Copy quantization_config.json if it exists
        src_config = os.path.join(quantized_model_path, "quantization_config.json")
        if os.path.exists(src_config):
            import shutil
            shutil.copy2(src_config, output_dir)

        update_quantization_config(output_dir, [54, 58])

        with open(os.path.join(output_dir, "quantization_config.json")) as f:
            qc = json.load(f)
        assert qc["substituted_layers"] == [54, 58]
        assert qc["substituted_dtype"] == "fp16"


class TestGenerateAsaqReportIntegration:
    """Test report generation with real model."""

    def test_generates_report_with_original_content(self, quantized_model_path, tmp_path):
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir)

        report = generate_asaq_report(quantized_model_path, output_dir, [54, 58])

        assert report.exists()
        content = report.read_text()
        assert "ASAQ" in content
        assert "54" in content
        assert "58" in content
        assert "FP16 (substituted)" in content


class TestFullSubstituteFlow:
    """Full end-to-end substitution flow (without smoke test)."""

    def test_full_flow(self, quantized_model_path, tmp_path):
        """Complete substitution flow: load → substitute → save → verify."""
        output_dir = str(tmp_path / "Qwen3.6-27B-int4-asaq")

        # Load
        weights, config = load_quantized_weights(quantized_model_path)
        original_size = compute_model_size(weights)

        # Load FP16 layers
        fp16 = load_fp16_layers("Qwen/Qwen3.6-27B", [54, 58])

        # Substitute
        weights = substitute_layers(weights, fp16, [54, 58])
        new_size = compute_model_size(weights)

        # Size should increase (FP16 > INT4)
        assert new_size > original_size

        # Save
        save_model(weights, config, output_dir)
        copy_config_files(quantized_model_path, output_dir)
        update_quantization_config(output_dir, [54, 58])
        generate_asaq_report(quantized_model_path, output_dir, [54, 58])

        # Verify files exist
        assert os.path.exists(os.path.join(output_dir, "config.json"))
        assert os.path.exists(os.path.join(output_dir, "model.safetensors.index.json"))
        assert os.path.exists(os.path.join(output_dir, "quantization_config.json"))
        assert os.path.exists(os.path.join(output_dir, "quantization-report.txt"))

        # Verify quantization config
        with open(os.path.join(output_dir, "quantization_config.json")) as f:
            qc = json.load(f)
        assert qc["substituted_layers"] == [54, 58]

        # Verify model can be reloaded
        reloaded_weights, _ = load_quantized_weights(output_dir)
        assert len(reloaded_weights) > 0

        # Verify no quantized tensors for substituted layers
        for idx in [54, 58]:
            q_count = sum(
                1 for k in reloaded_weights
                if f"layers.{idx}." in k and k.endswith(".qweight")
            )
            assert q_count == 0, f"Layer {idx} should not have quantized tensors"
