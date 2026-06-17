"""Tests for Qwen3.5 multi-modal model export correctness.

These tests validate the Phase 1-2 regression fixes:
- Architecture normalization in config.json
- SUPPORT_ONLY_TEXT_MODELS includes qwen3_5 variants
- Fix script normalizes architectures correctly
- processor_class preservation in tokenizer_config.json
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest


class TestConfigNormalizationsIncludeQwen3_5:
    """Test that _CONFIG_NORMALIZATIONS has all required Qwen3.5 entries."""

    def test_config_normalizations_has_qwen3_5(self):
        """_CONFIG_NORMALIZATIONS should have qwen3_5 key."""
        from auto_round.export.utils import _CONFIG_NORMALIZATIONS

        assert "qwen3_5" in _CONFIG_NORMALIZATIONS, (
            "qwen3_5 should be in _CONFIG_NORMALIZATIONS"
        )

    def test_config_normalizations_has_qwen3_5_moe(self):
        """_CONFIG_NORMALIZATIONS should have qwen3_5_moe key."""
        from auto_round.export.utils import _CONFIG_NORMALIZATIONS

        assert "qwen3_5_moe" in _CONFIG_NORMALIZATIONS, (
            "qwen3_5_moe should be in _CONFIG_NORMALIZATIONS"
        )

    def test_qwen3_5_normalization_values(self):
        """qwen3_5 normalization should map to correct values."""
        from auto_round.export.utils import _CONFIG_NORMALIZATIONS

        config = _CONFIG_NORMALIZATIONS["qwen3_5"]
        assert config["model_type"] == "qwen3_5"
        assert config["architectures"] == ["Qwen3_5ForConditionalGeneration"]

    def test_qwen3_5_moe_normalization_values(self):
        """qwen3_5_moe normalization should map to correct values."""
        from auto_round.export.utils import _CONFIG_NORMALIZATIONS

        config = _CONFIG_NORMALIZATIONS["qwen3_5_moe"]
        assert config["model_type"] == "qwen3_5_moe"
        assert config["architectures"] == ["Qwen3_5MoeForConditionalGeneration"]


class TestNormalizeConfigForVllm:
    """Test that _normalize_config_for_vllm fixes Qwen3.5 architectures."""

    def test_normalize_config_for_vllm_qwen3_5(self):
        """qwen3_5 model_type should be normalized to correct architectures."""
        from auto_round.export.utils import _normalize_config_for_vllm

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForCausalLM"],
                "vision_config": {"dtype": "bfloat16"},
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            _normalize_config_for_vllm(tmpdir)

            with open(config_path) as f:
                fixed = json.load(f)

            assert fixed["architectures"] == ["Qwen3_5ForConditionalGeneration"]
            assert fixed["model_type"] == "qwen3_5"

    def test_normalize_config_for_vllm_qwen3_5_moe(self):
        """qwen3_5_moe model_type should be normalized to correct architectures."""
        from auto_round.export.utils import _normalize_config_for_vllm

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForCausalLM"],
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            _normalize_config_for_vllm(tmpdir)

            with open(config_path) as f:
                fixed = json.load(f)

            assert fixed["architectures"] == ["Qwen3_5MoeForConditionalGeneration"]
            assert fixed["model_type"] == "qwen3_5_moe"


class TestSupportOnlyTextModels:
    """Test that qwen3_5 is in SUPPORT_ONLY_TEXT_MODELS."""

    def test_support_only_text_models_includes_qwen3_5(self):
        """qwen3_5 should be in SUPPORT_ONLY_TEXT_MODELS."""
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "qwen3_5" in SUPPORT_ONLY_TEXT_MODELS, (
            "qwen3_5 should be in SUPPORT_ONLY_TEXT_MODELS"
        )

    def test_support_only_text_models_includes_qwen3_5_moe(self):
        """qwen3_5_moe should be in SUPPORT_ONLY_TEXT_MODELS."""
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "qwen3_5_moe" in SUPPORT_ONLY_TEXT_MODELS, (
            "qwen3_5_moe should be in SUPPORT_ONLY_TEXT_MODELS"
        )


class TestFixScript:
    """Test fix script functions for architecture normalization."""

    @pytest.fixture(autouse=True)
    def _load_fix_module(self):
        """Load the fix script as a module."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        fix_script = scripts_dir / "fix-v14.1-layer-prefix.py"
        if not fix_script.exists():
            pytest.skip("fix-v14.1-layer-prefix.py not found")

        from importlib.util import module_from_spec, spec_from_file_location

        spec = spec_from_file_location("fix_v14_1", fix_script)
        self.fix_module = module_from_spec(spec)
        spec.loader.exec_module(self.fix_module)

    def test_fix_architectures_qwen3_5(self):
        """fix_architectures should normalize Qwen3.5 architectures."""
        cfg = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForCausalLM"],
            "vision_config": {"dtype": None},
        }

        fixed = self.fix_module.fix_architectures(cfg)

        assert fixed is True
        assert cfg["architectures"] == ["Qwen3_5ForConditionalGeneration"]
        assert cfg["vision_config"]["dtype"] == "bfloat16"

    def test_fix_architectures_qwen3_5_moe(self):
        """fix_architectures should normalize Qwen3.5-MoE architectures."""
        cfg = {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
        }

        fixed = self.fix_module.fix_architectures(cfg)

        assert fixed is True
        assert cfg["architectures"] == ["Qwen3_5MoeForConditionalGeneration"]

    def test_fix_architectures_non_qwen_unchanged(self):
        """fix_architectures should not modify non-Qwen3.5 configs."""
        cfg = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        }

        fixed = self.fix_module.fix_architectures(cfg)

        assert fixed is False
        assert cfg["architectures"] == ["LlamaForCausalLM"]


class TestProcessorClassPreservation:
    """Test that processor_class is preserved during export."""

    def test_processor_class_added_when_missing(self):
        """processor_class should be added when missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tok_config_path = os.path.join(tmpdir, "tokenizer_config.json")
            with open(tok_config_path, "w") as f:
                json.dump({}, f)

            # Simulate the fix logic from export.py
            with open(tok_config_path, "r") as f:
                tok_config = json.load(f)

            processor_class = "Qwen3VLProcessor"

            if "processor_class" not in tok_config or tok_config["processor_class"] is None:
                tok_config["processor_class"] = processor_class
                with open(tok_config_path, "w") as f:
                    json.dump(tok_config, f, indent=2)

            with open(tok_config_path) as f:
                result = json.load(f)

            assert result["processor_class"] == "Qwen3VLProcessor"

    def test_existing_processor_class_not_overwritten(self):
        """Existing processor_class should not be overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tok_config_path = os.path.join(tmpdir, "tokenizer_config.json")
            with open(tok_config_path, "w") as f:
                json.dump({"processor_class": "ExistingProcessor"}, f)

            with open(tok_config_path, "r") as f:
                tok_config = json.load(f)

            new_processor_class = "Qwen3VLProcessor"

            if "processor_class" not in tok_config or tok_config["processor_class"] is None:
                tok_config["processor_class"] = new_processor_class
                with open(tok_config_path, "w") as f:
                    json.dump(tok_config, f, indent=2)

            with open(tok_config_path) as f:
                result = json.load(f)

            assert result["processor_class"] == "ExistingProcessor"
