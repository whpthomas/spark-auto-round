"""Tests for Qwen3.5 multi-modal regression fixes (Phase 1).

These tests validate that:
1. Architecture normalization works for qwen3_5 and qwen3_5_moe model types
2. qwen3_5 and qwen3_5_moe are in SUPPORT_ONLY_TEXT_MODELS (prevents visual layer quantization)
"""

import json
import os
import tempfile

import pytest


class TestArchitectureNormalization:
    """Test that _normalize_config_for_vllm correctly fixes Qwen3.5 configs."""

    def test_qwen3_5_architecture_normalization(self):
        """Raw qwen3_5 model_type should be normalized to correct architectures."""
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

            assert fixed["architectures"] == ["Qwen3_5ForConditionalGeneration"], (
                f"Expected ['Qwen3_5ForConditionalGeneration'], got {fixed['architectures']}"
            )
            assert fixed["model_type"] == "qwen3_5", (
                f"Expected 'qwen3_5', got {fixed['model_type']}"
            )

    def test_qwen3_5_text_architecture_normalization(self):
        """qwen3_5_text model_type should also be normalized."""
        from auto_round.export.utils import _normalize_config_for_vllm

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model_type": "qwen3_5_text",
                "architectures": ["Qwen3_5ForCausalLM"],
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            _normalize_config_for_vllm(tmpdir)

            with open(config_path) as f:
                fixed = json.load(f)

            assert fixed["model_type"] == "qwen3_5"
            assert fixed["architectures"] == ["Qwen3_5ForConditionalGeneration"]

    def test_qwen3_5_moe_architecture_normalization(self):
        """Raw qwen3_5_moe model_type should be normalized to correct architectures."""
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

    def test_qwen3_5_moe_text_architecture_normalization(self):
        """qwen3_5_moe_text model_type should also be normalized."""
        from auto_round.export.utils import _normalize_config_for_vllm

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model_type": "qwen3_5_moe_text",
                "architectures": ["Qwen3_5MoeForCausalLM"],
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            _normalize_config_for_vllm(tmpdir)

            with open(config_path) as f:
                fixed = json.load(f)

            assert fixed["model_type"] == "qwen3_5_moe"
            assert fixed["architectures"] == ["Qwen3_5MoeForConditionalGeneration"]

    def test_already_correct_config_unchanged(self):
        """Config that's already correct should not be modified."""
        from auto_round.export.utils import _normalize_config_for_vllm

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            mtime_before = os.path.getmtime(config_path)
            _normalize_config_for_vllm(tmpdir)
            mtime_after = os.path.getmtime(config_path)

            # File should not be rewritten if already correct
            assert mtime_before == mtime_after

    def test_unrelated_model_type_unaffected(self):
        """Model types not in _CONFIG_NORMALIZATIONS should be unaffected."""
        from auto_round.export.utils import _normalize_config_for_vllm

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            _normalize_config_for_vllm(tmpdir)

            with open(config_path) as f:
                result = json.load(f)

            assert result["model_type"] == "llama"
            assert result["architectures"] == ["LlamaForCausalLM"]


class TestSupportOnlyTextModels:
    """Test that qwen3_5 and qwen3_5_moe are in SUPPORT_ONLY_TEXT_MODELS."""

    def test_qwen3_5_in_support_only_text_models(self):
        """qwen3_5 should be in SUPPORT_ONLY_TEXT_MODELS to prevent visual layer quantization."""
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "qwen3_5" in SUPPORT_ONLY_TEXT_MODELS, (
            "qwen3_5 must be in SUPPORT_ONLY_TEXT_MODELS to prevent visual layer quantization"
        )

    def test_qwen3_5_moe_in_support_only_text_models(self):
        """qwen3_5_moe should be in SUPPORT_ONLY_TEXT_MODELS to prevent visual layer quantization."""
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "qwen3_5_moe" in SUPPORT_ONLY_TEXT_MODELS, (
            "qwen3_5_moe must be in SUPPORT_ONLY_TEXT_MODELS to prevent visual layer quantization"
        )


class TestConfigNormalizationsDict:
    """Test that _CONFIG_NORMALIZATIONS has all required Qwen3.5 entries."""

    def test_config_normalizations_has_qwen3_5(self):
        """_CONFIG_NORMALIZATIONS should have qwen3_5 key."""
        from auto_round.export.utils import _CONFIG_NORMALIZATIONS

        assert "qwen3_5" in _CONFIG_NORMALIZATIONS

    def test_config_normalizations_has_qwen3_5_text(self):
        """_CONFIG_NORMALIZATIONS should have qwen3_5_text key."""
        from auto_round.export.utils import _CONFIG_NORMALIZATIONS

        assert "qwen3_5_text" in _CONFIG_NORMALIZATIONS

    def test_config_normalizations_has_qwen3_5_moe(self):
        """_CONFIG_NORMALIZATIONS should have qwen3_5_moe key."""
        from auto_round.export.utils import _CONFIG_NORMALIZATIONS

        assert "qwen3_5_moe" in _CONFIG_NORMALIZATIONS

    def test_config_normalizations_has_qwen3_5_moe_text(self):
        """_CONFIG_NORMALIZATIONS should have qwen3_5_moe_text key."""
        from auto_round.export.utils import _CONFIG_NORMALIZATIONS

        assert "qwen3_5_moe_text" in _CONFIG_NORMALIZATIONS

    def test_qwen3_5_normalization_values(self):
        """qwen3_5 normalization should map to correct values."""
        from auto_round.export.utils import _CONFIG_NORMALIZATIONS

        entry = _CONFIG_NORMALIZATIONS["qwen3_5"]
        assert entry["model_type"] == "qwen3_5"
        assert entry["architectures"] == ["Qwen3_5ForConditionalGeneration"]

    def test_qwen3_5_moe_normalization_values(self):
        """qwen3_5_moe normalization should map to correct values."""
        from auto_round.export.utils import _CONFIG_NORMALIZATIONS

        entry = _CONFIG_NORMALIZATIONS["qwen3_5_moe"]
        assert entry["model_type"] == "qwen3_5_moe"
        assert entry["architectures"] == ["Qwen3_5MoeForConditionalGeneration"]


class TestIsMllmModel:
    """Test is_mllm_model() multimodal detection function."""

    def test_is_mllm_model_with_vision_config(self, tmp_path):
        """Model with vision_config should be detected as multimodal."""
        config = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "vision_config": {"dtype": "bfloat16"},
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        from auto_round.utils.model.detect import is_mllm_model
        assert is_mllm_model(str(tmp_path)) is True

    def test_is_mllm_model_with_preprocessor_config(self, tmp_path):
        """Model with preprocessor_config.json should be detected as multimodal."""
        config_path = tmp_path / "preprocessor_config.json"
        config_path.write_text("{}")

        from auto_round.utils.model.detect import is_mllm_model
        assert is_mllm_model(str(tmp_path)) is True

    def test_is_mllm_model_with_processor_config(self, tmp_path):
        """Model with processor_config.json should be detected as multimodal."""
        config_path = tmp_path / "processor_config.json"
        config_path.write_text("{}")

        from auto_round.utils.model.detect import is_mllm_model
        assert is_mllm_model(str(tmp_path)) is True

    def test_is_mllm_model_pure_text(self, tmp_path):
        """Pure text model should NOT be detected as multimodal."""
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        from auto_round.utils.model.detect import is_mllm_model
        assert is_mllm_model(str(tmp_path)) is False

    def test_is_mllm_model_caching(self, tmp_path):
        """Results should be cached for performance."""
        config = {"model_type": "qwen3_5", "vision_config": {}}
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        from auto_round.utils.model.detect import is_mllm_model, _is_mllm_model_cache
        path_str = str(tmp_path)
        # Clear cache
        _is_mllm_model_cache.pop(path_str, None)
        result1 = is_mllm_model(path_str)
        result2 = is_mllm_model(path_str)
        assert result1 == result2 is True
        assert path_str in _is_mllm_model_cache

    def test_is_mllm_model_empty_dir(self, tmp_path):
        """Empty directory (no config) should NOT be detected as multimodal."""
        from auto_round.utils.model.detect import is_mllm_model
        assert is_mllm_model(str(tmp_path)) is False

    def test_llm_only_model_type_excluded(self, tmp_path):
        """BAGEL model_type should be excluded from multimodal detection."""
        config = {
            "model_type": "bagel",
            "architectures": ["BagelForConditionalGeneration"],
            "vision_config": {},
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        from auto_round.utils.model.detect import is_mllm_model
        assert is_mllm_model(str(tmp_path)) is False


class TestGetModelNameOrPath:
    """Test get_model_name_or_path() helper."""

    def test_string_input(self):
        """String input should be returned as-is."""
        from auto_round.utils.model.detect import get_model_name_or_path
        assert get_model_name_or_path("Qwen/Qwen3.5-0.8B") == "Qwen/Qwen3.5-0.8B"

    def test_empty_string(self):
        """Empty string should be returned as-is."""
        from auto_round.utils.model.detect import get_model_name_or_path
        assert get_model_name_or_path("") == ""

    def test_none_returns_none(self):
        """None input should return None (getattr fails)."""
        from auto_round.utils.model.detect import get_model_name_or_path
        # None has no _name_or_path attribute, so getattr returns None
        assert get_model_name_or_path(None) is None
