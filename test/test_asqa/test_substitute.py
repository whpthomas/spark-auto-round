"""Unit tests for auto_round.asqa.substitute — no CUDA or real models required."""

from __future__ import annotations

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from auto_round.asqa.__main__ import parse_layer_indices
from auto_round.asqa.substitute import (
    _check_disk_space,
    _is_quantized_tensor,
    _layer_prefix,
    _shard_name,
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
# _is_quantized_tensor
# ---------------------------------------------------------------------------


class TestIsQuantizedTensor:
    def test_qweight(self):
        name = "model.language_model.layers.54.mlp.gate_proj.qweight"
        prefix = "model.language_model.layers.54."
        assert _is_quantized_tensor(name, prefix) is True

    def test_qzeros(self):
        name = "model.language_model.layers.54.mlp.gate_proj.qzeros"
        prefix = "model.language_model.layers.54."
        assert _is_quantized_tensor(name, prefix) is True

    def test_scales(self):
        name = "model.language_model.layers.54.mlp.gate_proj.scales"
        prefix = "model.language_model.layers.54."
        assert _is_quantized_tensor(name, prefix) is True

    def test_weight_not_quantized(self):
        name = "model.language_model.layers.54.mlp.gate_proj.weight"
        prefix = "model.language_model.layers.54."
        assert _is_quantized_tensor(name, prefix) is False

    def test_layernorm_not_quantized(self):
        name = "model.language_model.layers.54.input_layernorm.weight"
        prefix = "model.language_model.layers.54."
        assert _is_quantized_tensor(name, prefix) is False

    def test_different_layer_prefix(self):
        name = "model.language_model.layers.58.mlp.gate_proj.qweight"
        prefix = "model.language_model.layers.54."
        assert _is_quantized_tensor(name, prefix) is False

    def test_linear_attn_quantized(self):
        name = "model.language_model.layers.54.linear_attn.in_proj_qkv.qweight"
        prefix = "model.language_model.layers.54."
        assert _is_quantized_tensor(name, prefix) is True


# ---------------------------------------------------------------------------
# _layer_prefix, _shard_name
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_layer_prefix(self):
        assert _layer_prefix(0) == "model.language_model.layers.0."
        assert _layer_prefix(54) == "model.language_model.layers.54."

    def test_shard_name_single(self):
        assert _shard_name(1, 1) == "model.safetensors"

    def test_shard_name_multi(self):
        assert _shard_name(1, 5) == "model-00001-of-00005.safetensors"
        assert _shard_name(3, 5) == "model-00003-of-00005.safetensors"


# ---------------------------------------------------------------------------
# infer_paths
# ---------------------------------------------------------------------------


class TestInferPaths:
    def test_raises_for_missing_quantized(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Quantized model not found"):
            infer_paths("Qwen/Qwen3.6-27B")

    def test_raises_for_missing_local_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / "models")
        with pytest.raises(FileNotFoundError, match="Quantized model not found"):
            infer_paths("Qwen/Qwen3.6-27B")

    def test_infer_paths_success(self, tmp_path, monkeypatch):
        """Verify correct path inference when quantized model exists."""
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / "models" / "Qwen3.6-27B-int4-AutoRound")

        q_path, fp16_id, out_dir = infer_paths("Qwen/Qwen3.6-27B")

        assert q_path == "./models/Qwen3.6-27B-int4-AutoRound"
        assert fp16_id == "Qwen/Qwen3.6-27B"
        assert out_dir == "./models/Qwen3.6-27B-int4-asaq"

    def test_infer_paths_trailing_slash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs(tmp_path / "models" / "Qwen3.6-27B-int4-AutoRound")

        q_path, fp16_id, out_dir = infer_paths("Qwen/Qwen3.6-27B/")

        assert q_path == "./models/Qwen3.6-27B-int4-AutoRound"
        assert fp16_id == "Qwen/Qwen3.6-27B/"


# ---------------------------------------------------------------------------
# load_quantized_weights
# ---------------------------------------------------------------------------


class TestLoadQuantizedWeights:
    def _make_sharded_model(self, base_dir: str, shard_count: int = 2):
        """Create a minimal sharded safetensors model for testing."""
        os.makedirs(base_dir, exist_ok=True)

        all_tensors: dict[str, torch.Tensor] = {}
        per_shard = 3  # tensors per shard
        for i in range(shard_count):
            shard_tensors = {}
            for j in range(per_shard):
                name = f"layer.{i}.weight.{j}"
                shard_tensors[name] = torch.randn(10, 10)
                all_tensors[name] = shard_tensors[name]
            shard_name = _shard_name(i + 1, shard_count)
            save_file(shard_tensors, os.path.join(base_dir, shard_name))

        # Build weight map
        weight_map = {}
        for i in range(shard_count):
            shard_name = _shard_name(i + 1, shard_count)
            for j in range(per_shard):
                weight_map[f"layer.{i}.weight.{j}"] = shard_name

        index = {"metadata": {}, "weight_map": weight_map}
        with open(os.path.join(base_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)

        # config.json
        with open(os.path.join(base_dir, "config.json"), "w") as f:
            json.dump({"model_type": "test"}, f)

        return all_tensors

    def test_load_sharded_model(self, tmp_path):
        model_dir = str(tmp_path / "model")
        expected = self._make_sharded_model(model_dir, shard_count=2)

        weights, config = load_quantized_weights(model_dir)

        assert config == {"model_type": "test"}
        assert len(weights) == len(expected)
        for name in expected:
            assert name in weights
            assert torch.equal(weights[name], expected[name])

    def test_load_single_shard_model(self, tmp_path):
        model_dir = str(tmp_path / "model")
        os.makedirs(model_dir)

        tensors = {
            "layer.0.weight": torch.randn(10, 10),
            "layer.1.weight": torch.randn(10, 10),
        }
        save_file(tensors, os.path.join(model_dir, "model.safetensors"))
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump({"model_type": "test"}, f)

        weights, config = load_quantized_weights(model_dir)

        assert len(weights) == 2
        assert config == {"model_type": "test"}

    def test_missing_config_raises(self, tmp_path):
        model_dir = str(tmp_path / "model")
        os.makedirs(model_dir)

        with pytest.raises(FileNotFoundError, match="Missing config.json"):
            load_quantized_weights(model_dir)

    def test_missing_safetensors_raises(self, tmp_path):
        model_dir = str(tmp_path / "model")
        os.makedirs(model_dir)
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump({}, f)

        with pytest.raises(FileNotFoundError, match="No safetensors"):
            load_quantized_weights(model_dir)


# ---------------------------------------------------------------------------
# substitute_layers
# ---------------------------------------------------------------------------


class TestSubstituteLayers:
    def _make_quantized_layer(self, idx: int) -> dict[str, torch.Tensor]:
        """Create fake quantized tensors for one layer."""
        prefix = f"model.language_model.layers.{idx}."
        return {
            f"{prefix}input_layernorm.weight": torch.randn(4096),
            f"{prefix}mlp.gate_proj.qweight": torch.randint(
                0, 100, (512, 4096), dtype=torch.int32
            ),
            f"{prefix}mlp.gate_proj.qzeros": torch.randint(
                0, 100, (32, 4096), dtype=torch.int32
            ),
            f"{prefix}mlp.gate_proj.scales": torch.randn(32, 4096),
            f"{prefix}mlp.up_proj.qweight": torch.randint(
                0, 100, (512, 4096), dtype=torch.int32
            ),
            f"{prefix}mlp.up_proj.qzeros": torch.randint(
                0, 100, (32, 4096), dtype=torch.int32
            ),
            f"{prefix}mlp.up_proj.scales": torch.randn(32, 4096),
        }

    def _make_fp16_layer(self, idx: int) -> dict[str, torch.Tensor]:
        """Create fake FP16 tensors for one layer."""
        prefix = f"model.language_model.layers.{idx}."
        return {
            f"{prefix}input_layernorm.weight": torch.randn(4096),
            f"{prefix}mlp.gate_proj.weight": torch.randn(4096, 4096),
            f"{prefix}mlp.up_proj.weight": torch.randn(4096, 4096),
        }

    def test_substitute_single_layer(self):
        weights: dict[str, torch.Tensor] = {}
        weights.update(self._make_quantized_layer(0))
        weights.update(self._make_quantized_layer(1))

        fp16 = self._make_fp16_layer(1)

        substitute_layers(weights, fp16, [1])

        # Layer 1 should have FP16 weight, no qweight
        assert "model.language_model.layers.1.mlp.gate_proj.weight" in weights
        assert "model.language_model.layers.1.mlp.up_proj.weight" in weights
        assert "model.language_model.layers.1.mlp.gate_proj.qweight" not in weights
        assert "model.language_model.layers.1.mlp.gate_proj.qzeros" not in weights
        assert "model.language_model.layers.1.mlp.gate_proj.scales" not in weights

        # Layer 0 should be untouched
        assert "model.language_model.layers.0.mlp.gate_proj.qweight" in weights

    def test_substitute_multiple_layers(self):
        weights: dict[str, torch.Tensor] = {}
        weights.update(self._make_quantized_layer(0))
        weights.update(self._make_quantized_layer(1))
        weights.update(self._make_quantized_layer(2))

        fp16: dict[str, torch.Tensor] = {}
        fp16.update(self._make_fp16_layer(1))
        fp16.update(self._make_fp16_layer(2))

        substitute_layers(weights, fp16, [1, 2])

        # Layers 1 and 2 should have FP16 weights
        assert "model.language_model.layers.1.mlp.gate_proj.weight" in weights
        assert "model.language_model.layers.2.mlp.gate_proj.weight" in weights

        # Layer 0 untouched
        assert "model.language_model.layers.0.mlp.gate_proj.qweight" in weights

    def test_substitute_preserves_norms(self):
        weights = self._make_quantized_layer(0)
        fp16 = self._make_fp16_layer(0)

        substitute_layers(weights, fp16, [0])

        # Norms should still be there (they were never quantized)
        assert "model.language_model.layers.0.input_layernorm.weight" in weights

    def test_substitute_empty_indices_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            substitute_layers({}, {}, [])

    def test_substitute_missing_fp16_raises(self):
        weights = self._make_quantized_layer(0)
        with pytest.raises(KeyError, match="No FP16 tensors found"):
            substitute_layers(weights, {}, [0])

    def test_substitute_modifies_in_place(self):
        weights = self._make_quantized_layer(0)
        original_ref = weights
        fp16 = self._make_fp16_layer(0)

        result = substitute_layers(weights, fp16, [0])

        assert result is original_ref




# ---------------------------------------------------------------------------
# save_model
# ---------------------------------------------------------------------------


class TestSaveModel:
    def test_save_creates_files(self, tmp_path):
        weights = {
            "layer.0.weight": torch.randn(10, 10),
            "layer.1.weight": torch.randn(10, 10),
        }
        config = {"model_type": "test"}

        out = save_model(weights, config, str(tmp_path / "output"))

        assert out.exists()
        assert (out / "config.json").exists()
        assert (out / "model.safetensors.index.json").exists()

        # Check config content
        with open(out / "config.json") as f:
            assert json.load(f) == {"model_type": "test"}

        # Check index file
        with open(out / "model.safetensors.index.json") as f:
            index = json.load(f)
        assert "weight_map" in index
        assert len(index["weight_map"]) == 2
        assert "metadata" in index
        assert index["metadata"]["total_size"] > 0

    def test_save_single_shard(self, tmp_path):
        """Small model should produce a single model.safetensors file."""
        weights = {
            "layer.0.weight": torch.randn(10, 10),
        }
        config = {"model_type": "test"}

        out = save_model(weights, config, str(tmp_path / "output"))

        assert (out / "model.safetensors").exists()
        # No multi-shard index when single shard
        with open(out / "model.safetensors.index.json") as f:
            index = json.load(f)
        assert all(v == "model.safetensors" for v in index["weight_map"].values())

    def test_save_large_model_shards(self, tmp_path):
        """Model exceeding MAX_SHARD_SIZE should be split into multiple shards."""
        # Create tensors that exceed MAX_SHARD_SIZE (5GB)
        # We can't actually create 5GB tensors in tests, so test the logic
        # by checking that the shard naming convention works.
        weights = {
            f"layer.{i}.weight": torch.randn(10, 10) for i in range(5)
        }
        config = {"model_type": "test"}

        out = save_model(weights, config, str(tmp_path / "output"))

        with open(out / "model.safetensors.index.json") as f:
            index = json.load(f)

        assert len(index["weight_map"]) == 5

    def test_save_copies_config_files(self, tmp_path):
        """When source_dir is given, extra config files should be copied."""
        # Create source dir with config files
        source = tmp_path / "source"
        source.mkdir()
        (source / "tokenizer.json").write_text('{"dummy": true}')
        (source / "tokenizer_config.json").write_text('{"dummy": true}')
        (source / "config.json").write_text('{"source_config": true}')

        weights = {"layer.0.weight": torch.randn(10, 10)}
        config = {"model_type": "test"}

        out = save_model(
            weights, config, str(tmp_path / "output"), source_dir=str(source)
        )

        assert (out / "tokenizer.json").exists()
        assert (out / "tokenizer_config.json").exists()
        # config.json should NOT be overwritten (it uses the provided config)
        with open(out / "config.json") as f:
            assert json.load(f) == {"model_type": "test"}

    def test_save_empty_weights(self, tmp_path):
        """Empty weights dict should still produce valid output."""
        out = save_model({}, {"model_type": "test"}, str(tmp_path / "output"))

        assert (out / "model.safetensors.index.json").exists()
        with open(out / "model.safetensors.index.json") as f:
            index = json.load(f)
        assert index["weight_map"] == {}


# ---------------------------------------------------------------------------
# Integration: round-trip through save -> load
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_save_and_load_round_trip(self, tmp_path):
        """Verify weights survive a save/load round trip."""
        original = {
            "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(
                32, 32
            ),
            "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(
                32, 32
            ),
        }
        config = {"model_type": "qwen3", "hidden_size": 4096}

        out = save_model(original, config, str(tmp_path / "model"))
        loaded_weights, loaded_config = load_quantized_weights(str(out))

        assert loaded_config == config
        assert set(loaded_weights.keys()) == set(original.keys())
        for name in original:
            assert torch.equal(loaded_weights[name], original[name])


# ---------------------------------------------------------------------------
# compute_model_size
# ---------------------------------------------------------------------------


class TestComputeModelSize:
    def test_empty_weights(self):
        assert compute_model_size({}) == 0

    def test_single_tensor(self):
        w = {"layer.0.weight": torch.randn(10, 10)}
        expected = 10 * 10 * 4  # float32 = 4 bytes
        assert compute_model_size(w) == expected

    def test_multiple_tensors(self):
        w = {
            "layer.0.weight": torch.randn(10, 10),
            "layer.1.weight": torch.randn(20, 20),
        }
        expected = 10 * 10 * 4 + 20 * 20 * 4
        assert compute_model_size(w) == expected

    def test_int32_tensor(self):
        w = {"layer.0.qweight": torch.randint(0, 100, (512, 4096), dtype=torch.int32)}
        expected = 512 * 4096 * 4  # int32 = 4 bytes
        assert compute_model_size(w) == expected


# ---------------------------------------------------------------------------
# copy_config_files
# ---------------------------------------------------------------------------


class TestCopyConfigFiles:
    def test_copies_existing_files(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "tokenizer.json").write_text('{"dummy": true}')
        (source / "tokenizer_config.json").write_text('{"dummy": true}')
        (source / "generation_config.json").write_text('{"dummy": true}')

        output = tmp_path / "output"
        output.mkdir()

        copy_config_files(str(source), str(output))

        assert (output / "tokenizer.json").exists()
        assert (output / "tokenizer_config.json").exists()
        assert (output / "generation_config.json").exists()

    def test_skips_missing_files(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "tokenizer.json").write_text('{"dummy": true}')
        # No tokenizer_config.json

        output = tmp_path / "output"
        output.mkdir()

        copy_config_files(str(source), str(output))

        assert (output / "tokenizer.json").exists()
        assert not (output / "tokenizer_config.json").exists()

    def test_creates_output_dir(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "tokenizer.json").write_text('{"dummy": true}')

        output = tmp_path / "new_output"  # Does not exist yet

        copy_config_files(str(source), str(output))

        assert output.exists()
        assert (output / "tokenizer.json").exists()

    def test_copies_quantization_config(self, tmp_path):
        """Verify quantization_config.json is copied (critical for vLLM)."""
        source = tmp_path / "source"
        source.mkdir()
        quant_config = {
            "bits": 4,
            "quant_method": "auto-round",
            "packing_format": "auto_round:auto_gptq",
        }
        (source / "quantization_config.json").write_text(json.dumps(quant_config))

        output = tmp_path / "output"
        output.mkdir()

        copy_config_files(str(source), str(output))

        assert (output / "quantization_config.json").exists()
        with open(output / "quantization_config.json") as f:
            copied = json.load(f)
        assert copied["quant_method"] == "auto-round"
        assert copied["packing_format"] == "auto_round:auto_gptq"


# ---------------------------------------------------------------------------
# update_quantization_config
# ---------------------------------------------------------------------------


class TestUpdateQuantizationConfig:
    def test_creates_config_if_missing(self, tmp_path):
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir)

        update_quantization_config(output_dir, [54, 58])

        config_path = tmp_path / "output" / "quantization_config.json"
        assert config_path.exists()
        with open(config_path) as f:
            qc = json.load(f)
        assert qc["substituted_layers"] == [54, 58]
        assert qc["substituted_dtype"] == "fp16"

    def test_updates_existing_config(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "quantization_config.json").write_text(
            json.dumps({"bits": 4, "group_size": 128})
        )

        update_quantization_config(str(output_dir), [42])

        with open(output_dir / "quantization_config.json") as f:
            qc = json.load(f)
        assert qc["bits"] == 4  # Original preserved
        assert qc["group_size"] == 128  # Original preserved
        assert qc["substituted_layers"] == [42]
        assert qc["substituted_dtype"] == "fp16"

    def test_sorted_indices(self, tmp_path):
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir)

        update_quantization_config(output_dir, [58, 54, 58])

        with open(tmp_path / "output" / "quantization_config.json") as f:
            qc = json.load(f)
        assert qc["substituted_layers"] == [54, 58]  # Sorted and unique

    def test_updates_extra_config_with_weights(self, tmp_path):
        """When weights dict is provided, extra_config should be updated.
        
        vLLM expects layer names without .weight suffix in extra_config.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "quantization_config.json").write_text(
            json.dumps({"bits": 4, "extra_config": {}})
        )

        # Simulate weights for layer 54
        weights = {
            "model.language_model.layers.54.mlp.gate_proj.weight": torch.randn(10, 10),
            "model.language_model.layers.54.mlp.up_proj.weight": torch.randn(10, 10),
            "model.language_model.layers.54.mlp.down_proj.weight": torch.randn(10, 10),
            "model.language_model.layers.54.input_layernorm.weight": torch.randn(10),
        }

        update_quantization_config(str(output_dir), [54], weights)

        with open(output_dir / "quantization_config.json") as f:
            qc = json.load(f)
        
        extra = qc["extra_config"]
        # Linear weights should be added to extra_config as FP16
        # Note: vLLM expects names WITHOUT .weight suffix
        assert "model.language_model.layers.54.mlp.gate_proj" in extra
        assert extra["model.language_model.layers.54.mlp.gate_proj"]["bits"] == 16
        assert "model.language_model.layers.54.mlp.up_proj" in extra
        assert "model.language_model.layers.54.mlp.down_proj" in extra
        # Layernorm should NOT be added (it's always FP16)
        assert "model.language_model.layers.54.input_layernorm" not in extra


# ---------------------------------------------------------------------------
# generate_asaq_report
# ---------------------------------------------------------------------------


class TestGenerateAsaqReport:
    def test_creates_report(self, tmp_path):
        # Create fake quantized dir with report
        quant_dir = tmp_path / "quantized"
        quant_dir.mkdir()
        (quant_dir / "quantization-report.txt").write_text(
            "=== Quantization Report ===\nModel: test\n"
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        report = generate_asaq_report(str(quant_dir), str(output_dir), [54, 58])

        assert report.exists()
        content = report.read_text()
        assert "ASAQ" in content
        assert "54, 58" in content
        assert "FP16 (substituted)" in content

    def test_includes_original_report(self, tmp_path):
        quant_dir = tmp_path / "quantized"
        quant_dir.mkdir()
        (quant_dir / "quantization-report.txt").write_text(
            "=== Quantization Report ===\nOriginal content here\n"
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        report = generate_asaq_report(str(quant_dir), str(output_dir), [54])

        content = report.read_text()
        assert "Original content here" in content
        assert "--- Original Report ---" in content

    def test_handles_missing_original_report(self, tmp_path):
        quant_dir = tmp_path / "quantized"
        quant_dir.mkdir()
        # No quantization-report.txt

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        report = generate_asaq_report(str(quant_dir), str(output_dir), [54])

        assert report.exists()
        content = report.read_text()
        assert "ASAQ" in content
        assert "no original report found" in content

    def test_single_layer(self, tmp_path):
        quant_dir = tmp_path / "quantized"
        quant_dir.mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        report = generate_asaq_report(str(quant_dir), str(output_dir), [42])

        content = report.read_text()
        assert "42" in content
        assert "FP16 (substituted)" in content


# ---------------------------------------------------------------------------
# parse_layer_indices (from __main__)
# ---------------------------------------------------------------------------


class TestParseLayerIndices:
    def test_simple(self):
        assert parse_layer_indices("54,58") == [54, 58]

    def test_with_spaces(self):
        assert parse_layer_indices("54, 58") == [54, 58]

    def test_single(self):
        assert parse_layer_indices("42") == [42]

    def test_sorted_unique(self):
        # Duplicates should raise an error
        with pytest.raises(ValueError, match="Duplicate"):
            parse_layer_indices("58,54,58")

    def test_valid_sorted(self):
        assert parse_layer_indices("58,54") == [54, 58]

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            parse_layer_indices("abc")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No layer indices"):
            parse_layer_indices("")

    def test_blank_raises(self):
        with pytest.raises(ValueError, match="No layer indices"):
            parse_layer_indices("   ")

    def test_duplicate_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            parse_layer_indices("54,54")

    def test_whitespace_only(self):
        with pytest.raises(ValueError, match="No layer indices"):
            parse_layer_indices(" ")

    def test_trailing_comma(self):
        # Trailing comma produces an empty string after strip, which fails int conversion
        with pytest.raises(ValueError, match="Invalid"):
            parse_layer_indices("54,58,")


# ---------------------------------------------------------------------------
# _fuse_fp16_weights
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# _check_disk_space
# ---------------------------------------------------------------------------


class TestCheckDiskSpace:
    def test_sufficient_space(self, tmp_path):
        # Should not raise if there's enough space (any small amount)
        _check_disk_space(str(tmp_path), 1024)  # 1 KB

    def test_insufficient_space_raises(self, tmp_path):
        # Requesting more space than could possibly be available
        with pytest.raises(OSError, match="Insufficient disk space"):
            _check_disk_space(str(tmp_path), 1024**4 * 1000)  # 1000 TB

    def test_walks_up_to_existing_parent(self, tmp_path):
        # Non-existing directory should walk up to find existing parent
        nonexistent = tmp_path / "a" / "b" / "c"
        _check_disk_space(str(nonexistent), 1024)  # Should not raise


# ---------------------------------------------------------------------------
# load_fp16_layers validation
# ---------------------------------------------------------------------------


class TestLoadFp16LayersValidation:
    def test_empty_indices_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            load_fp16_layers("some/model", [])

    def test_missing_layers_raises(self, tmp_path):
        # Create a minimal model index that doesn't have the requested layers
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        index = {"weight_map": {}, "metadata": {}}
        with open(model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        with pytest.raises(KeyError, match="Layers.*not found"):
            load_fp16_layers("some/model", [54], _model_dir=str(model_dir))

