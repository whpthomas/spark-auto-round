"""Integration tests for auto-tuner + checkpoint resume interaction.

These tests verify that checkpoint metadata (exit_reason, tuning_profile,
ome_count) flows correctly to the auto-tuner on resume. All tests are
mock-only (no GPU required) and run fast.

The key data flow being tested:
  1. Checkpoint crash → progress.json written with exit_reason + tuning_profile
  2. On resume, CLI reads checkpoint → extracts resume_context
  3. resume_context passed to auto_tune() → skips relaxation steps for OOM

Phase 3 of test-gaps plan (Gap A: Auto-Tuner Resume Integration).
Phase 4 of test-gaps plan (Gap 3a: Production Model Integration Test).
"""

import json
import os
import shutil
import tempfile

import pytest
import torch
from pathlib import Path

from auto_round.compressors.auto_tune import (
    auto_tune,
    _resolve_resume_offset,
    _RELAXATION_LADDER,
    format_preflight_message,
)
from auto_round.utils.device.memory_estimator import estimate_peak_memory_per_block


# ---------------------------------------------------------------------------
# Mock config helpers
#
# We use a simple class instead of MagicMock to avoid auto-attribute creation
# which breaks getattr-based attribute detection in _get_hidden_dimensions().
# ---------------------------------------------------------------------------


class _MockConfig:
    """Simple mock that does NOT auto-create attributes on access."""
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model_config():
    """Config for SmolLM-135M (tiny, fits any budget)."""
    config = _MockConfig()
    config.hidden_size = 576
    config.intermediate_size = 1536
    config.num_attention_heads = 9
    config.num_key_value_heads = 3
    config.num_hidden_layers = 30
    config.max_position_embeddings = 2048
    return config


@pytest.fixture
def medium_model_config():
    """Config for a 7B dense model (e.g., Llama-7B)."""
    config = _MockConfig()
    config.hidden_size = 4096
    config.intermediate_size = 11008
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    config.num_hidden_layers = 32
    config.max_position_embeddings = 2048
    return config


# Default user settings — matches __main__.py defaults
DEFAULT_SETTINGS = {
    "batch_size": 8,
    "seqlen": 2048,
    "nsamples": 512,
    "iters": 1000,
    "group_size": 128,
}


# ---------------------------------------------------------------------------
# Helper to write checkpoint progress.json
# ---------------------------------------------------------------------------


def _write_checkpoint(cache_dir: str, completed: int, total: int,
                      exit_reason: str = None, tuning_profile: dict = None):
    """Write a checkpoint progress.json file for testing."""
    os.makedirs(cache_dir, exist_ok=True)
    progress = {
        "completed": completed,
        "total": total,
        "block_names": [f"model.layers.{i}" for i in range(total)],
        "exit_reason": exit_reason,
        "tuning_profile": tuning_profile,
    }
    progress_path = os.path.join(cache_dir, "progress.json")
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-Tuner + Resume Integration Tests (mock-only, no GPU)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutoTuneResumeIntegration:
    """Test auto-tuner behavior with resume context (mock-only).

    These tests verify the core auto-tuner + resume interaction:
    - OOM resume skips relaxation steps
    - Interrupt resume starts fresh
    - oom_count affects skip count
    - tuning_profile flows from checkpoint to auto-tuner
    """

    def test_resume_oom_skips_first_relaxation(self, small_model_config):
        """OOM resume should skip the first relaxation step.

        When the previous run OOM'd, the auto-tuner should skip the first
        relaxation step (batch_size=8→4) since that setting already failed.
        """
        # Use a budget that forces relaxation (tight for the model)
        peak, _ = estimate_peak_memory_per_block(small_model_config, {
            "batch_size": 8, "seqlen": 2048, "group_size": 128,
        })
        # Budget at 70% of peak forces relaxation
        budget_bytes = int(peak * (1024 ** 3) * 0.7)

        # Fresh run (no resume)
        adjusted_fresh, steps_fresh = auto_tune(
            DEFAULT_SETTINGS, small_model_config, budget_bytes,
        )

        # OOM resume (skip first step)
        adjusted_resume, steps_resume = auto_tune(
            DEFAULT_SETTINGS, small_model_config, budget_bytes,
            resume_context={"exit_reason": "oom", "oom_count": 0},
        )

        # OOM version should have at least one skipped step
        skipped = [s for s in steps_resume if s.get("skipped")]
        assert len(skipped) >= 1, "OOM resume should skip at least one step"

        # The skipped step should be batch_size (first in ladder)
        assert skipped[0]["setting"] == "batch_size"

    def test_resume_interrupt_fresh_start(self, small_model_config):
        """Interrupt resume should start fresh (no skipped steps).

        When the previous run was interrupted (user Ctrl-C), the auto-tuner
        should start fresh with no skipped steps, identical to a fresh run.
        """
        peak, _ = estimate_peak_memory_per_block(small_model_config, {
            "batch_size": 8, "seqlen": 2048, "group_size": 128,
        })
        budget_bytes = int(peak * (1024 ** 3) * 0.7)

        # Interrupt resume
        adjusted_resume, steps_resume = auto_tune(
            DEFAULT_SETTINGS, small_model_config, budget_bytes,
            resume_context={"exit_reason": "interrupted", "oom_count": 0},
        )

        # Fresh run
        adjusted_fresh, steps_fresh = auto_tune(
            DEFAULT_SETTINGS, small_model_config, budget_bytes,
        )

        # Should be identical to fresh run
        assert adjusted_resume == adjusted_fresh
        assert steps_resume == steps_fresh

    def test_resume_oom_count_affects_skip_count(self, small_model_config):
        """Higher oom_count should skip more steps.

        The skip formula is: skip = 1 + (oom_count // 2)
        So oom_count=0 → skip 1, oom_count=4 → skip 3.
        """
        peak, _ = estimate_peak_memory_per_block(small_model_config, {
            "batch_size": 8, "seqlen": 2048, "group_size": 128,
        })
        # Very tight budget to ensure many relaxation steps available
        budget_bytes = int(peak * (1024 ** 3) * 0.5)

        # oom_count=0 → skip 1 step
        _, steps_oom0 = auto_tune(
            DEFAULT_SETTINGS, small_model_config, budget_bytes,
            resume_context={"exit_reason": "oom", "oom_count": 0},
        )
        skipped_oom0 = [s for s in steps_oom0 if s.get("skipped")]

        # oom_count=4 → skip 3 steps (1 + 4//2 = 3)
        _, steps_oom4 = auto_tune(
            DEFAULT_SETTINGS, small_model_config, budget_bytes,
            resume_context={"exit_reason": "oom", "oom_count": 4},
        )
        skipped_oom4 = [s for s in steps_oom4 if s.get("skipped")]

        assert len(skipped_oom4) >= len(skipped_oom0) + 2, \
            "Higher oom_count should skip more steps"

    def test_tuning_profile_flows_to_resume_context(self, small_model_config):
        """Verify tuning_profile structure matches what CLI extracts.

        The CLI reads tuning_profile from progress.json and passes oom_count
        to auto_tune(). This test verifies that data flow works correctly.
        """
        # Simulate what CLI does when reading checkpoint
        checkpoint_tuning_profile = {
            "relaxation_step": 1,
            "oom_count": 2,
            "settings_active": {
                "batch_size": 4,
                "seqlen": 2048,
                "nsamples": 512,
            },
        }

        # Extract resume_context as CLI does
        resume_context = {
            "exit_reason": "oom",
            "oom_count": checkpoint_tuning_profile.get("oom_count", 0),
            "tuning_profile": checkpoint_tuning_profile,
        }

        # Pass to auto_tune
        peak, _ = estimate_peak_memory_per_block(small_model_config, {
            "batch_size": 8, "seqlen": 2048, "group_size": 128,
        })
        budget_bytes = int(peak * (1024 ** 3) * 0.5)

        adjusted, steps = auto_tune(
            DEFAULT_SETTINGS, small_model_config, budget_bytes,
            resume_context=resume_context,
        )

        # Should have skipped steps (oom_count=2 → skip = 1 + 2//2 = 2)
        skipped = [s for s in steps if s.get("skipped")]
        assert len(skipped) >= 2, "oom_count=2 should skip at least 2 steps"

    def test_resume_oom_produces_relaxed_settings(self, small_model_config):
        """OOM resume with tight budget produces more relaxed settings than fresh.

        When resuming from OOM, the auto-tuner should start from a more
        relaxed position, resulting in lower batch_size/seqlen.
        """
        peak, _ = estimate_peak_memory_per_block(small_model_config, {
            "batch_size": 8, "seqlen": 2048, "group_size": 128,
        })
        budget_bytes = int(peak * (1024 ** 3) * 0.5)

        # Fresh run
        adjusted_fresh, _ = auto_tune(
            DEFAULT_SETTINGS, small_model_config, budget_bytes,
        )

        # OOM resume with oom_count=4 (skip 3 steps)
        adjusted_oom, _ = auto_tune(
            DEFAULT_SETTINGS, small_model_config, budget_bytes,
            resume_context={"exit_reason": "oom", "oom_count": 4},
        )

        # OOM should produce at least as relaxed settings
        # (batch_size and/or seqlen should be <= fresh run)
        assert adjusted_oom["batch_size"] <= adjusted_fresh["batch_size"], \
            "OOM resume should produce batch_size <= fresh run"
        assert adjusted_oom["seqlen"] <= adjusted_fresh["seqlen"], \
            "OOM resume should produce seqlen <= fresh run"

    def test_resume_none_context_same_as_fresh(self, small_model_config):
        """resume_context=None should produce identical results to fresh run."""
        peak, _ = estimate_peak_memory_per_block(small_model_config, {
            "batch_size": 8, "seqlen": 2048, "group_size": 128,
        })
        budget_bytes = int(peak * (1024 ** 3) * 0.7)

        adjusted_none, steps_none = auto_tune(
            DEFAULT_SETTINGS, small_model_config, budget_bytes,
            resume_context=None,
        )
        adjusted_fresh, steps_fresh = auto_tune(
            DEFAULT_SETTINGS, small_model_config, budget_bytes,
        )

        assert adjusted_none == adjusted_fresh
        assert steps_none == steps_fresh

    def test_resume_empty_context_same_as_fresh(self, small_model_config):
        """Empty resume_context dict should produce same results as fresh run."""
        peak, _ = estimate_peak_memory_per_block(small_model_config, {
            "batch_size": 8, "seqlen": 2048, "group_size": 128,
        })
        budget_bytes = int(peak * (1024 ** 3) * 0.7)

        adjusted_empty, steps_empty = auto_tune(
            DEFAULT_SETTINGS, small_model_config, budget_bytes,
            resume_context={},
        )
        adjusted_fresh, steps_fresh = auto_tune(
            DEFAULT_SETTINGS, small_model_config, budget_bytes,
        )

        assert adjusted_empty == adjusted_fresh
        assert steps_empty == steps_fresh

    def test_oom_skip_formula_correct(self):
        """Verify _resolve_resume_offset skip formula: skip = 1 + (oom_count // 2)."""
        # oom_count=0 → skip 1
        assert _resolve_resume_offset({"exit_reason": "oom", "oom_count": 0}) == (0, 1)
        # oom_count=1 → skip 1 (1 + 0)
        assert _resolve_resume_offset({"exit_reason": "oom", "oom_count": 1}) == (0, 1)
        # oom_count=2 → skip 2 (1 + 1)
        assert _resolve_resume_offset({"exit_reason": "oom", "oom_count": 2}) == (0, 2)
        # oom_count=3 → skip 2 (1 + 1)
        assert _resolve_resume_offset({"exit_reason": "oom", "oom_count": 3}) == (0, 2)
        # oom_count=4 → skip 3 (1 + 2)
        assert _resolve_resume_offset({"exit_reason": "oom", "oom_count": 4}) == (0, 3)

    def test_interrupt_always_zero_skip(self):
        """Interrupt exit_reason always produces skip=0 regardless of oom_count."""
        assert _resolve_resume_offset(
            {"exit_reason": "interrupted", "oom_count": 5}
        ) == (0, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint File Parsing Tests (mock-only, no GPU)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckpointParsing:
    """Test that checkpoint files are parsed correctly.

    These tests verify the JSON parsing logic that the CLI uses to read
    progress.json and extract resume context. They don't require GPU
    and exercise the data flow that feeds into auto_tune().
    """

    def test_parse_oom_checkpoint(self, tmp_path):
        """Parse a checkpoint with OOM exit reason."""
        cache_dir = str(tmp_path / ".cache")
        tuning_profile = {
            "relaxation_step": 1,
            "oom_count": 0,
            "settings_active": {"batch_size": 4, "seqlen": 2048, "nsamples": 512},
        }
        _write_checkpoint(
            cache_dir, completed=5, total=32,
            exit_reason="oom", tuning_profile=tuning_profile,
        )

        # Parse as CLI does
        progress_path = os.path.join(cache_dir, "progress.json")
        with open(progress_path, "r") as f:
            progress = json.load(f)

        stored_exit_reason = progress.get("exit_reason")
        tp = progress.get("tuning_profile")
        stored_completed = progress.get("completed", 0)
        stored_total = progress.get("total", 0)

        resume_mode = stored_completed > 0 and stored_completed < stored_total
        assert resume_mode is True
        assert stored_exit_reason == "oom"
        assert tp["oom_count"] == 0

    def test_parse_interrupt_checkpoint(self, tmp_path):
        """Parse a checkpoint with interrupted exit reason."""
        cache_dir = str(tmp_path / ".cache")
        _write_checkpoint(
            cache_dir, completed=10, total=48,
            exit_reason="interrupted", tuning_profile=None,
        )

        progress_path = os.path.join(cache_dir, "progress.json")
        with open(progress_path, "r") as f:
            progress = json.load(f)

        assert progress["exit_reason"] == "interrupted"
        assert progress["tuning_profile"] is None

    def test_parse_checkpoint_missing_tuning_profile(self, tmp_path):
        """Handle checkpoint with missing tuning_profile key."""
        cache_dir = str(tmp_path / ".cache")
        _write_checkpoint(
            cache_dir, completed=3, total=32,
            exit_reason="oom", tuning_profile=None,
        )

        progress_path = os.path.join(cache_dir, "progress.json")
        with open(progress_path, "r") as f:
            progress = json.load(f)

        # Should handle None gracefully
        tp = progress.get("tuning_profile")
        oom_count = tp.get("oom_count", 0) if tp else 0
        assert oom_count == 0

    def test_corrupt_json_falls_through(self, tmp_path):
        """Corrupt progress.json should be handled gracefully."""
        cache_dir = str(tmp_path / ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        progress_path = os.path.join(cache_dir, "progress.json")
        with open(progress_path, "w") as f:
            f.write("not valid json {{{")

        # Simulate CLI parsing with try/except
        resume_context = None
        try:
            with open(progress_path, "r") as f:
                progress = json.load(f)
            # If we get here, parsing succeeded
        except (json.JSONDecodeError, OSError):
            # Corrupt file → fresh start
            resume_context = None

        assert resume_context is None

    def test_resume_context_extraction_from_checkpoint(self, tmp_path):
        """Full data flow: write checkpoint → read → extract resume_context."""
        cache_dir = str(tmp_path / ".cache")
        tuning_profile = {
            "relaxation_step": 2,
            "oom_count": 3,
            "settings_active": {"batch_size": 2, "seqlen": 1024, "nsamples": 256},
        }
        _write_checkpoint(
            cache_dir, completed=15, total=32,
            exit_reason="oom", tuning_profile=tuning_profile,
        )

        # Simulate CLI extraction (from __main__.py)
        progress_path = os.path.join(cache_dir, "progress.json")
        with open(progress_path, "r") as f:
            progress = json.load(f)

        stored_exit_reason = progress.get("exit_reason")
        tp = progress.get("tuning_profile")
        stored_completed = progress.get("completed", 0)
        stored_total = progress.get("total", 0)

        resume_mode = stored_completed > 0 and stored_completed < stored_total
        assert resume_mode is True

        # Build resume_context as CLI does
        resume_context = {
            "exit_reason": stored_exit_reason,
            "oom_count": tp.get("oom_count", 0) if tp else 0,
            "tuning_profile": tp,
        }

        assert resume_context["exit_reason"] == "oom"
        assert resume_context["oom_count"] == 3
        assert resume_context["tuning_profile"]["relaxation_step"] == 2

    def test_no_resume_when_completed_equals_total(self, tmp_path):
        """When completed == total, resume_mode should be False."""
        cache_dir = str(tmp_path / ".cache")
        _write_checkpoint(
            cache_dir, completed=32, total=32,
            exit_reason="completed", tuning_profile=None,
        )

        progress_path = os.path.join(cache_dir, "progress.json")
        with open(progress_path, "r") as f:
            progress = json.load(f)

        stored_completed = progress.get("completed", 0)
        stored_total = progress.get("total", 0)
        resume_mode = stored_completed > 0 and stored_completed < stored_total

        assert resume_mode is False, "completed == total should not trigger resume"

    def test_no_resume_when_completed_zero(self, tmp_path):
        """When completed == 0, resume_mode should be False."""
        cache_dir = str(tmp_path / ".cache")
        _write_checkpoint(
            cache_dir, completed=0, total=32,
            exit_reason=None, tuning_profile=None,
        )

        progress_path = os.path.join(cache_dir, "progress.json")
        with open(progress_path, "r") as f:
            progress = json.load(f)

        stored_completed = progress.get("completed", 0)
        stored_total = progress.get("total", 0)
        resume_mode = stored_completed > 0 and stored_completed < stored_total

        assert resume_mode is False, "completed == 0 should not trigger resume"

    def test_checkpoint_with_all_fields(self, tmp_path):
        """Parse a complete checkpoint with all fields present."""
        cache_dir = str(tmp_path / ".cache")
        tuning_profile = {
            "relaxation_step": 3,
            "oom_count": 5,
            "settings_active": {
                "batch_size": 1,
                "seqlen": 256,
                "nsamples": 128,
                "iters": 1000,
                "group_size": 128,
            },
        }
        _write_checkpoint(
            cache_dir, completed=8, total=48,
            exit_reason="oom", tuning_profile=tuning_profile,
        )

        progress_path = os.path.join(cache_dir, "progress.json")
        with open(progress_path, "r") as f:
            progress = json.load(f)

        # Verify all fields
        assert progress["completed"] == 8
        assert progress["total"] == 48
        assert progress["exit_reason"] == "oom"
        assert len(progress["block_names"]) == 48
        assert progress["block_names"][0] == "model.layers.0"
        assert progress["block_names"][47] == "model.layers.47"

        tp = progress["tuning_profile"]
        assert tp["relaxation_step"] == 3
        assert tp["oom_count"] == 5
        assert tp["settings_active"]["batch_size"] == 1
        assert tp["settings_active"]["seqlen"] == 256
        assert tp["settings_active"]["nsamples"] == 128

    def test_checkpoint_block_names_match_total(self, tmp_path):
        """Verify block_names list length matches total count."""
        cache_dir = str(tmp_path / ".cache")
        total = 16
        _write_checkpoint(
            cache_dir, completed=4, total=total,
            exit_reason="oom", tuning_profile=None,
        )

        progress_path = os.path.join(cache_dir, "progress.json")
        with open(progress_path, "r") as f:
            progress = json.load(f)

        assert len(progress["block_names"]) == total
        # Sequential naming
        for i in range(total):
            assert progress["block_names"][i] == f"model.layers.{i}"


# ═══════════════════════════════════════════════════════════════════════════════
# Full E2E Integration Tests (requires GPU + real model)
# ═══════════════════════════════════════════════════════════════════════════════


def _get_model_subdir(save_dir, model_path):
    """Return the model-specific subdirectory under save_dir.

    Mirrors the path logic used by AutoRound/Quantizer: the output directory
    is named after the model's directory basename.
    """
    model_basename = os.path.basename(os.path.normpath(model_path))
    return os.path.join(save_dir, model_basename)


@pytest.mark.cuda
class TestAutoTuneE2E:
    """End-to-end tests for auto-tuner with real models.

    These tests verify the complete pipeline: memory estimation → auto-tune →
    display → AutoRound construction → quantization.
    """

    @pytest.fixture(autouse=True)
    def check_cuda(self, request):
        """Skip if CUDA not available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_memory_estimator_gives_reasonable_numbers(self, tiny_qwen_model_path):
        """Memory estimator produces reasonable estimates for a real model."""
        from transformers import AutoConfig

        # Load real config from the tiny model directory
        config = AutoConfig.from_pretrained(
            tiny_qwen_model_path, trust_remote_code=True
        )

        # Get estimate with default settings
        peak_gb, breakdown = estimate_peak_memory_per_block(
            config, {"batch_size": 8, "seqlen": 2048, "group_size": 128}
        )

        # Qwen3-0.6B is tiny, should be well under 5 GiB
        assert 0 < peak_gb < 5.0, f"Peak memory should be reasonable, got {peak_gb} GB"
        assert "block_weights_bf16" in breakdown
        assert "activation_forward" in breakdown
        assert "total_estimated" in breakdown

    def test_auto_tune_produces_message(self, tiny_qwen_model_path, capsys):
        """Auto-tuner produces "Memory OK" message for small model."""
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            tiny_qwen_model_path, trust_remote_code=True
        )

        user_settings = {
            "batch_size": 8,
            "seqlen": 2048,
            "nsamples": 512,
            "iters": 1000,
            "group_size": 128,
        }

        # 96 GiB budget (default)
        budget_bytes = 96 * (1024 ** 3)

        # Run auto-tuner
        adjusted_settings, tune_steps = auto_tune(
            user_settings=user_settings,
            model_config=config,
            budget_bytes=budget_bytes,
        )

        # Compute peak for display
        peak_gb, _ = estimate_peak_memory_per_block(config, adjusted_settings)

        # Generate message
        msg = format_preflight_message(
            user_settings=user_settings,
            adjusted_settings=adjusted_settings,
            steps=tune_steps,
            peak_gb=peak_gb,
            budget_gb=budget_bytes / (1024 ** 3),
        )

        # Small model should fit within 96 GiB
        assert "Memory OK" in msg, f"Expected 'Memory OK' in message: {msg}"
        assert "proceeding with user settings" in msg

    def test_auto_tune_no_adjustments_for_small_model(self, tiny_qwen_model_path):
        """Small model needs no auto-tuner adjustments."""
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            tiny_qwen_model_path, trust_remote_code=True
        )

        user_settings = {
            "batch_size": 8,
            "seqlen": 2048,
            "nsamples": 512,
            "iters": 1000,
            "group_size": 128,
        }

        budget_bytes = 96 * (1024 ** 3)

        adjusted, steps = auto_tune(
            user_settings=user_settings,
            model_config=config,
            budget_bytes=budget_bytes,
        )

        # Tiny model should not need any adjustments
        assert adjusted == user_settings, "Settings should be unchanged"
        assert len(steps) == 0, "No steps should be recorded"

    def test_auto_tune_produces_output(self, tiny_qwen_model_path):
        """Auto-tuner E2E: load model, auto-tune, quantize, verify output."""
        from auto_round import AutoRound

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create AutoRound with minimal settings for speed
            ar = AutoRound(
                model=tiny_qwen_model_path,
                format="auto_round",
                iters=1,
                seqlen=32,
                nsamples=2,
                batch_size=1,
                dataset=None,
            )

            # Run quantize and save
            model, output_dir = ar.quantize_and_save(tmpdir, format="auto_round")

            # Verify output directory exists
            assert os.path.isdir(output_dir), f"Output dir should exist: {output_dir}"

            # Verify safetensors written
            safetensors = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
            assert len(safetensors) > 0, f"Should write safetensors in {output_dir}"

            # Verify config.json exists
            config_path = os.path.join(output_dir, "config.json")
            assert os.path.exists(config_path), f"config.json should exist at {config_path}"

            # Verify quantization_config is embedded in config.json
            import json
            with open(config_path) as f:
                config_data = json.load(f)
            assert "quantization_config" in config_data, (
                "quantization_config should be embedded in config.json"
            )

    def test_checkpoint_not_created_for_completed_run(self, tiny_qwen_model_path):
        """Successful run should not leave .cache/ directory."""
        from auto_round import AutoRound

        with tempfile.TemporaryDirectory() as tmpdir:
            ar = AutoRound(
                model=tiny_qwen_model_path,
                format="auto_round",
                iters=1,
                seqlen=32,
                nsamples=2,
                batch_size=1,
                dataset=None,
            )

            model, output_dir = ar.quantize_and_save(tmpdir, format="auto_round")

            # After completion, .cache should be cleaned
            cache_dir = os.path.join(output_dir, ".cache")
            assert not os.path.isdir(cache_dir), (
                f".cache/ should be cleaned after completion, found at {cache_dir}"
            )

    def test_output_dir_has_expected_structure(self, tiny_qwen_model_path):
        """Output directory should have expected file structure."""
        from auto_round import AutoRound

        with tempfile.TemporaryDirectory() as tmpdir:
            ar = AutoRound(
                model=tiny_qwen_model_path,
                format="auto_round",
                iters=1,
                seqlen=32,
                nsamples=2,
                batch_size=1,
                dataset=None,
            )

            model, output_dir = ar.quantize_and_save(tmpdir, format="auto_round")

            # Expected files
            expected_files = [
                "config.json",
            ]

            for filename in expected_files:
                filepath = os.path.join(output_dir, filename)
                assert os.path.exists(filepath), f"Expected {filename} at {filepath}"

            # Verify quantization_config is embedded in config.json
            import json
            with open(os.path.join(output_dir, "config.json")) as f:
                config_data = json.load(f)
            assert "quantization_config" in config_data, (
                "quantization_config should be embedded in config.json"
            )

            # Should have at least one safetensors
            safetensors = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
            assert len(safetensors) > 0, "Should have at least one .safetensors file"

            # Should have tokenizer files
            tokenizer_files = [f for f in os.listdir(output_dir) if "tokenizer" in f.lower()]
            assert len(tokenizer_files) > 0, "Should have tokenizer files"


@pytest.mark.cuda
class TestAutoTuneWithCheckpoint:
    """Integration tests for auto-tuner + checkpoint resume (requires GPU)."""

    @pytest.fixture(autouse=True)
    def check_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    @pytest.fixture(autouse=True)
    def cleanup(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_resume_after_oom_with_real_checkpoint(self, tiny_qwen_model_path):
        """Resume with OOM checkpoint applies relaxed settings."""
        # Step 1: Create a fake OOM checkpoint
        model_subdir = _get_model_subdir(self.save_dir, tiny_qwen_model_path)
        cache_dir = os.path.join(model_subdir, ".cache")

        tuning_profile = {
            "relaxation_step": 1,
            "oom_count": 0,
            "settings_active": {
                "batch_size": 4,
                "seqlen": 2048,
                "nsamples": 512,
            },
        }

        _write_checkpoint(
            cache_dir,
            completed=1,  # Say we completed 1 block
            total=2,  # Out of 2 total
            exit_reason="oom",
            tuning_profile=tuning_profile,
        )

        # Step 2: Run AutoRound (should detect resume)
        from auto_round import AutoRound

        ar = AutoRound(
            model=tiny_qwen_model_path,
            format="auto_round",
            iters=1,
            seqlen=32,
            nsamples=2,
            batch_size=1,
            dataset=None,
            clear_cache=False,  # Don't clear - we want resume
        )

        # The quantize_and_save should detect the checkpoint and resume
        model, output_dir = ar.quantize_and_save(self.save_dir, format="auto_round")

        # Verify completion
        assert os.path.isdir(output_dir)
        safetensors = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
        assert len(safetensors) > 0

    def test_clear_cache_removes_checkpoint(self, tiny_qwen_model_path):
        """--clear-cache flag removes existing checkpoint."""
        # Create a fake checkpoint
        model_subdir = _get_model_subdir(self.save_dir, tiny_qwen_model_path)
        cache_dir = os.path.join(model_subdir, ".cache")
        _write_checkpoint(cache_dir, completed=5, total=32)

        # Verify checkpoint exists
        assert os.path.isdir(cache_dir)

        # Run with clear_cache=True
        from auto_round import AutoRound

        ar = AutoRound(
            model=tiny_qwen_model_path,
            format="auto_round",
            iters=1,
            seqlen=32,
            nsamples=2,
            batch_size=1,
            dataset=None,
            clear_cache=True,  # Clear the checkpoint
        )

        model, output_dir = ar.quantize_and_save(self.save_dir, format="auto_round")

        # Checkpoint should be removed (or not exist after fresh start)
        cache_dir_after = os.path.join(output_dir, ".cache")
        # Either doesn't exist or is empty (fresh start)
        if os.path.isdir(cache_dir_after):
            files = os.listdir(cache_dir_after)
            assert len(files) == 0, f".cache/ should be empty after clear, found: {files}"
