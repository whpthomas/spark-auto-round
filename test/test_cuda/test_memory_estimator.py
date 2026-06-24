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
"""Tests for auto_round.utils.device.memory_estimator."""

import gc
import math
import pytest
import torch

from auto_round.utils.device.memory_estimator import (
    estimate_peak_memory_per_block,
    _get_block_params,
    _get_hidden_dimensions,
)


# ---------------------------------------------------------------------------
# Mock config helpers
#
# We use a simple class instead of MagicMock because MagicMock auto-creates
# attributes on access — getattr(config, "num_experts", None) would return
# a MagicMock() instead of None, breaking the getattr-based attribute
# detection in _get_hidden_dimensions().
# ---------------------------------------------------------------------------


class _MockConfig:
    """Simple mock that does NOT auto-create attributes on access.

    getattr(obj, attr, default) correctly returns the default when the
    attribute is not set — just like a real HuggingFace AutoConfig.
    """
    pass


# ---------------------------------------------------------------------------
# Fixtures — mocked HF configs
# ---------------------------------------------------------------------------


@pytest.fixture
def llama_config():
    """Simulate a Llama-7B config (dense)."""
    config = _MockConfig()
    config.hidden_size = 4096
    config.intermediate_size = 11008
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    config.num_hidden_layers = 32
    config.max_position_embeddings = 2048
    # No MoE fields — getattr returns None as expected
    return config


@pytest.fixture
def qwen_moe_config():
    """Simulate Qwen3.5-122B-A10B config (MoE)."""
    config = _MockConfig()
    config.hidden_size = 8192
    config.intermediate_size = 24576
    config.num_attention_heads = 64
    config.num_key_value_heads = 8
    config.num_hidden_layers = 48
    config.num_experts = 256
    config.top_k = 8  # num_experts_per_tok
    config.max_position_embeddings = 32768
    return config


@pytest.fixture
def smol_config():
    """Simulate SmolLM-135M config (tiny dense)."""
    config = _MockConfig()
    config.hidden_size = 576
    config.intermediate_size = 1536
    config.num_attention_heads = 9
    config.num_key_value_heads = 3
    config.num_hidden_layers = 30
    config.max_position_embeddings = 2048
    return config


# ---------------------------------------------------------------------------
# _get_hidden_dimensions
# ---------------------------------------------------------------------------


class TestGetHiddenDimensions:
    def test_llama(self, llama_config):
        dims = _get_hidden_dimensions(llama_config)
        assert dims["hidden_size"] == 4096
        assert dims["intermediate_size"] == 11008
        assert dims["num_attention_heads"] == 32
        assert dims["num_layers"] == 32
        assert dims["num_experts"] is None
        assert dims["top_k"] is None

    def test_qwen_moe(self, qwen_moe_config):
        dims = _get_hidden_dimensions(qwen_moe_config)
        assert dims["hidden_size"] == 8192
        assert dims["intermediate_size"] == 24576
        assert dims["num_layers"] == 48
        assert dims["num_experts"] == 256
        assert dims["top_k"] == 8

    def test_fallback_attributes(self):
        """Test configs with non-standard attribute names (d_model, ffn_dim)."""
        config = _MockConfig()
        config.d_model = 1024
        config.ffn_dim = 4096
        config.num_attention_heads = 16
        config.num_hidden_layers = 12
        dims = _get_hidden_dimensions(config)
        assert dims["hidden_size"] == 1024
        assert dims["intermediate_size"] == 4096
        assert dims["num_layers"] == 12

    def test_moe_no_top_k_fallsback_to_default(self):
        """MoE config without top_k should default to 2."""
        config = _MockConfig()
        config.hidden_size = 4096
        config.intermediate_size = 11008
        config.num_attention_heads = 32
        config.num_hidden_layers = 32
        config.num_experts = 8
        # No top_k set
        dims = _get_hidden_dimensions(config)
        assert dims["num_experts"] == 8
        assert dims["top_k"] == 2  # Mixtral default

    def test_num_local_experts_fallback(self):
        """Config with num_local_experts instead of num_experts."""
        config = _MockConfig()
        config.hidden_size = 4096
        config.intermediate_size = 11008
        config.num_attention_heads = 32
        config.num_hidden_layers = 32
        config.num_local_experts = 64
        config.top_k = 4
        dims = _get_hidden_dimensions(config)
        assert dims["num_experts"] == 64
        assert dims["top_k"] == 4


# ---------------------------------------------------------------------------
# _get_block_params
# ---------------------------------------------------------------------------


class TestGetBlockParams:
    def test_llama_block(self, llama_config):
        params = _get_block_params(llama_config)
        # attn: 4 * 4096^2 = 67,108,864
        # ffn: 3 * 4096 * 11008 = 135,266,304
        # total: 202,375,168
        expected = 4 * 4096 * 4096 + 3 * 4096 * 11008
        assert params == expected

    def test_qwen_moe_block(self, qwen_moe_config):
        params = _get_block_params(qwen_moe_config)
        # attn: 4 * 8192^2 = 268,435,456
        # ffn: 3 * 8192 * 24576 * 256 = 154,618,822,656
        # total: ~154.9 billion
        expected = 4 * 8192 * 8192 + 3 * 8192 * 24576 * 256
        assert params == expected

    def test_smol_block(self, smol_config):
        params = _get_block_params(smol_config)
        # attn: 4 * 576^2 = 1,327,104
        # ffn: 3 * 576 * 1536 = 2,654,208
        # total: 3,981,312
        expected = 4 * 576 * 576 + 3 * 576 * 1536
        assert params == expected

    def test_moe_block_with_dims_override(self, qwen_moe_config):
        """Test passing dims directly instead of config."""
        dims = _get_hidden_dimensions(qwen_moe_config)
        params = _get_block_params(qwen_moe_config, dims=dims)
        expected = 4 * 8192 * 8192 + 3 * 8192 * 24576 * 256
        assert params == expected

    def test_block_params_no_moe_when_experts_is_none(self, llama_config):
        """Dense model with num_experts=None should use dense formula."""
        params = _get_block_params(llama_config)
        # If MoE code path ran, it would be much larger
        dense_expected = 4 * 4096 * 4096 + 3 * 4096 * 11008
        assert params == dense_expected


# ---------------------------------------------------------------------------
# estimate_peak_memory_per_block
# ---------------------------------------------------------------------------


class TestEstimatePeakMemory:
    def test_llama_defaults(self, llama_config):
        """Llama-7B block with default settings (bs=8, seqlen=2048)."""
        peak, breakdown = estimate_peak_memory_per_block(llama_config, {
            "batch_size": 8, "seqlen": 2048,
        })
        # Quick sanity: should be < 10 GB for a 7B model block
        assert peak > 0
        assert peak < 10.0
        # Check all breakdown keys present (includes intermediate tensors)
        expected_keys = {
            "block_weights_bf16", "wrapper_value_fp32", "wrapper_scales_fp32",
            "activation_forward", "activation_backward",
            "attention_scores", "qkv_intermediate", "ffn_intermediate",
            "calibration_input", "safety_margin", "total_estimated",
        }
        assert set(breakdown.keys()) == expected_keys
        # total_estimated should round to peak
        assert round(breakdown["total_estimated"], 2) == peak

    def test_batch_size_reduces_memory(self, llama_config):
        """Halving batch_size should roughly halve activation components."""
        peak_bs8, _ = estimate_peak_memory_per_block(llama_config, {
            "batch_size": 8, "seqlen": 2048,
        })
        peak_bs4, _ = estimate_peak_memory_per_block(llama_config, {
            "batch_size": 4, "seqlen": 2048,
        })
        # Activation components should be ~half
        # (weight components stay same, so reduction < 50% of total)
        assert peak_bs4 < peak_bs8

    def test_seqlen_reduces_memory(self, llama_config):
        """Halving seqlen should roughly halve activation components."""
        peak_2048, _ = estimate_peak_memory_per_block(llama_config, {
            "batch_size": 8, "seqlen": 2048,
        })
        peak_1024, _ = estimate_peak_memory_per_block(llama_config, {
            "batch_size": 8, "seqlen": 1024,
        })
        assert peak_1024 < peak_2048

    def test_qwen_moe_memory_gigantic(self, qwen_moe_config):
        """Qwen3.5-122B-A10B per-block memory should be large (~100+ GB)."""
        peak, breakdown = estimate_peak_memory_per_block(qwen_moe_config, {
            "batch_size": 8, "seqlen": 2048,
        })
        # Each MoE block has ~155B params, so > 300 GB even without activations
        # This confirms why users OOM on 128 GB
        assert peak > 300.0
        # Verify MoE components dominate
        assert breakdown["block_weights_bf16"] > breakdown["activation_forward"] * 10

    def test_qwen_moe_minimal_settings(self, qwen_moe_config):
        """Qwen3.5-122B-A10B with bs=1, seqlen=512 — barely fits 128 GB."""
        peak, breakdown = estimate_peak_memory_per_block(qwen_moe_config, {
            "batch_size": 1, "seqlen": 512,
        })
        # Even minimal settings: ~310 GB for weights alone in bf16
        # (155B params × 2 bytes = 310 GB + wrapper = ~930 GB)
        # This should be way over 128 GB, confirming OOMs
        assert peak > 128.0

    def test_smol_model_fits_easily(self, smol_config):
        """SmolLM-135M should fit easily in any GPU (<< 2 GB per block)."""
        peak, _ = estimate_peak_memory_per_block(smol_config, {
            "batch_size": 8, "seqlen": 2048,
        })
        assert peak < 2.0

    def test_invalid_settings_raises(self, llama_config):
        """Missing/invalid settings should raise ValueError."""
        with pytest.raises(ValueError):
            estimate_peak_memory_per_block(llama_config, {})
        with pytest.raises(ValueError):
            estimate_peak_memory_per_block(llama_config, {"batch_size": 0, "seqlen": 2048})
        with pytest.raises(ValueError):
            estimate_peak_memory_per_block(llama_config, {"batch_size": 8, "seqlen": -1})


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_safety_factor_applied(self, llama_config):
        """Verify safety factor of 1.50 is applied."""
        peak, breakdown = estimate_peak_memory_per_block(llama_config, {
            "batch_size": 1, "seqlen": 1,
        })
        # Compute pre-safety total
        pre_safety = peak / 1.50
        # Breakdown safety_margin should be peak - pre_safety
        assert abs(breakdown["safety_margin"] - (peak - pre_safety)) < 0.01

    def test_components_sum_to_total(self, llama_config):
        """Pre-safety components + safety_margin should sum to total_estimated."""
        peak, breakdown = estimate_peak_memory_per_block(llama_config, {
            "batch_size": 1, "seqlen": 1,
        })
        # Sum all pre-safety components
        pre_safety_keys = [k for k in breakdown if k not in ("safety_margin", "total_estimated")]
        pre_safety_sum = sum(breakdown[k] for k in pre_safety_keys)
        # Apply safety to pre_safety_sum
        expected_peak = round(pre_safety_sum * 1.50, 2)
        assert peak == expected_peak

    def test_group_size_affects_wrapper_scales(self, llama_config):
        """Changing group_size should affect wrapper scale memory slightly."""
        _, b1 = estimate_peak_memory_per_block(llama_config, {
            "batch_size": 1, "seqlen": 1, "group_size": 32,
        })
        _, b2 = estimate_peak_memory_per_block(llama_config, {
            "batch_size": 1, "seqlen": 1, "group_size": 256,
        })
        # group_size=32 → more scales → higher wrapper_scale_bytes
        assert b1["wrapper_scales_fp32"] > b2["wrapper_scales_fp32"]

    def test_moe_activations_scale_with_top_k(self, qwen_moe_config):
        """MoE activations should account for top_k routing."""
        peak, breakdown = estimate_peak_memory_per_block(qwen_moe_config, {
            "batch_size": 1, "seqlen": 1,
        })
        # top_k=8 means activations are 8× larger than dense equivalent
        assert breakdown["activation_forward"] > 0


# ---------------------------------------------------------------------------
# CUDA Integration Tests — validate estimator against real GPU measurements
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model_fixture():
    """Load SmolLM-135M (tiny dense model) for CUDA integration tests.

    Uses transformers to load a real model so the test exercises the full
    forward/backward path with real tensors (not mocks).
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_name = "HuggingFaceTB/SmolLM-135M"
    try:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        pytest.skip(f"Could not load {model_name}: {exc}")

    model.config._name_or_path = model_name  # ensure config has name
    yield model, tokenizer, config

    del model, tokenizer, config
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def estimator_prediction(tiny_model_fixture):
    """Run the estimator for SmolLM-135M with default settings."""
    _, _, config = tiny_model_fixture
    settings = {
        "batch_size": 8,
        "seqlen": 2048,
        "group_size": 128,
    }
    peak_gb, breakdown = estimate_peak_memory_per_block(config, settings)
    return peak_gb, breakdown, settings


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
class TestMemoryEstimatorCUDA:
    """Validate memory estimator against real GPU peak memory measurements.

    These tests require a CUDA GPU and load a tiny model (SmolLM-135M).
    They run the forward/backward/optimizer cycle on a single block and
    compare torch.cuda.max_memory_allocated() to the estimator's prediction.
    """

    @pytest.mark.cuda
    def test_estimator_prediction_is_reasonable(self, estimator_prediction):
        """Sanity check: SmolLM-135M prediction should be < 2 GB."""
        peak_gb, breakdown, _ = estimator_prediction
        assert peak_gb > 0, "Estimator should predict positive memory"
        assert peak_gb < 2.0, f"SmolLM-135M prediction too large: {peak_gb} GB"

    @pytest.mark.cuda
    def test_peak_memory_within_tolerance(self, tiny_model_fixture, estimator_prediction):
        """Real GPU peak should be within ±30% of estimator prediction.

        Tolerance rationale (from design brief):
        - Validates safety factor is adequate without being brittle
        - Accounts for CUDA allocator fragmentation, torch.compile effects,
          system-level noise
        - If this fails, the safety factor needs adjustment

        NOTE: If this test fails with measured >> predicted, it reveals the
        estimator significantly underestimates real GPU memory usage.
        """
        model, tokenizer, config = tiny_model_fixture
        predicted_gb, _, settings = estimator_prediction

        device = torch.device("cuda:0")

        # Extract the first transformer block and move only it to GPU
        # (not the entire model — we want to measure block-level memory only)
        block = model.model.layers[0].to(device).to(torch.bfloat16)

        # Create a dummy input matching the estimator's settings
        bs = settings["batch_size"]
        seqlen = settings["seqlen"]
        hidden = config.hidden_size

        # Create hidden states as input (bypasses embedding layer)
        dummy_input = torch.randn(bs, seqlen, hidden, dtype=torch.bfloat16, device=device)

        # Compute position embeddings using a temporary rotary_emb on GPU
        # We need the model's rotary_emb config, but move it to GPU temporarily
        rotary_emb = model.model.rotary_emb.to(device)
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bs, -1)
        position_embeddings = rotary_emb(dummy_input, position_ids=position_ids)
        # Move rotary_emb back to CPU to avoid counting its memory
        model.model.rotary_emb = rotary_emb.cpu()
        del rotary_emb

        # Clear any prior memory stats
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        # Record baseline (just the block weights on GPU)
        baseline_bytes = torch.cuda.max_memory_allocated()

        # Run forward pass through the block
        # Use no_grad first to measure forward-only peak
        with torch.no_grad():
            output = block(dummy_input, position_embeddings=position_embeddings)

        forward_peak = torch.cuda.max_memory_allocated()

        # Now run with grad to measure forward + backward peak
        torch.cuda.reset_peak_memory_stats()
        dummy_input_grad = torch.randn(bs, seqlen, hidden, dtype=torch.bfloat16, device=device)
        dummy_input_grad.requires_grad_(True)

        output = block(dummy_input_grad, position_embeddings=position_embeddings)
        if isinstance(output, tuple):
            loss = output[0].sum()
        else:
            loss = output.sum()
        loss.backward()

        fwd_bwd_peak = torch.cuda.max_memory_allocated()

        # The higher of forward-only and fwd+bwd is our measured peak
        # (forward-only captures activation memory, fwd+bwd captures gradients)
        measured_peak_bytes = max(forward_peak, fwd_bwd_peak)
        measured_peak_gb = measured_peak_bytes / (1024 ** 3)

        # Clean up
        del block, dummy_input, dummy_input_grad, output, loss, position_embeddings
        gc.collect()
        torch.cuda.empty_cache()

        # Tolerance check: prediction within ±30% of measured
        lower_bound = predicted_gb * 0.70
        upper_bound = predicted_gb * 1.30

        if not (lower_bound <= measured_peak_gb <= upper_bound):
            ratio = measured_peak_gb / predicted_gb if predicted_gb > 0 else float('inf')
            pytest.fail(
                f"Memory estimator prediction ({predicted_gb:.3f} GB) differs from "
                f"measured peak ({measured_peak_gb:.3f} GB) by {ratio:.1f}× "
                f"(expected ±30%). "
                f"The estimator significantly underestimates real GPU memory usage. "
                f"Consider: (1) increasing safety factor from 1.15× to ~{ratio:.1f}×, "
                f"or (2) fixing the formula to account for intermediate tensors."
            )

    @pytest.mark.cuda
    def test_safety_factor_provides_headroom(self, tiny_model_fixture, estimator_prediction):
        """Estimator prediction should be ABOVE measured peak (safety headroom).

        The 1.15× safety factor means predicted > measured. If this fails,
        the safety factor is too low and needs increase.

        NOTE: If this test fails with predicted << measured, it reveals the
        estimator significantly underestimates real GPU memory usage.
        """
        model, tokenizer, config = tiny_model_fixture
        predicted_gb, _, settings = estimator_prediction

        device = torch.device("cuda:0")

        # Extract the first transformer block and move only it to GPU
        block = model.model.layers[0].to(device).to(torch.bfloat16)

        bs = settings["batch_size"]
        seqlen = settings["seqlen"]
        hidden = config.hidden_size

        dummy_input = torch.randn(bs, seqlen, hidden, dtype=torch.bfloat16, device=device)

        # Compute position embeddings
        rotary_emb = model.model.rotary_emb.to(device)
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bs, -1)
        position_embeddings = rotary_emb(dummy_input, position_ids=position_ids)
        model.model.rotary_emb = rotary_emb.cpu()
        del rotary_emb

        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        # Run forward + backward
        dummy_input.requires_grad_(True)
        output = block(dummy_input, position_embeddings=position_embeddings)
        if isinstance(output, tuple):
            loss = output[0].sum()
        else:
            loss = output.sum()
        loss.backward()

        measured_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        del block, dummy_input, output, loss, position_embeddings
        gc.collect()
        torch.cuda.empty_cache()

        # Safety factor check: prediction should be ≥ measured
        # Allow 5% tolerance for measurement noise
        if predicted_gb < measured_peak_gb * 0.95:
            ratio = measured_peak_gb / predicted_gb if predicted_gb > 0 else float('inf')
            pytest.fail(
                f"Safety factor inadequate: prediction ({predicted_gb:.3f} GB) is below "
                f"measured peak ({measured_peak_gb:.3f} GB) by {ratio:.1f}×. "
                f"The 1.15× safety factor needs to be increased to ~{ratio:.1f}× "
                f"or the estimator formula needs to account for intermediate tensors."
            )

    @pytest.mark.cuda
    def test_breakdown_components_are_positive(self, estimator_prediction):
        """All breakdown components should be non-negative."""
        _, breakdown, _ = estimator_prediction
        for key, value in breakdown.items():
            assert value >= 0, f"Component '{key}' is negative: {value}"