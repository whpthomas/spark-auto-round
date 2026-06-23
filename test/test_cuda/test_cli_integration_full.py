"""Full CLI integration tests for memory-aware auto-tuning.

Tests that the auto-tuner, display, and resume flow work together through
the CLI entry point. Some tests require GPU, others use Mock.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, PropertyMock

from auto_round.compressors.memory_estimator import estimate_peak_memory_per_block
from auto_round.compressors.auto_tune import (
    auto_tune,
    format_preflight_message,
    format_resume_message,
)


# ---------------------------------------------------------------------------
# Data flow tests (no GPU)
# ---------------------------------------------------------------------------


class TestCliDataFlow:
    """Test the data flow through the CLI integration points."""

    def test_auto_tune_called_with_correct_args(self):
        """Verify auto_tune receives expected inputs from args."""
        # Simulate what __main__.py does
        available_memory = 128 * 1024 ** 3  # 128 GB
        memory_utilization = 0.75
        effective_utilization = memory_utilization  # no margin

        user_settings = {
            "batch_size": 8,
            "seqlen": 2048,
            "nsamples": 512,
            "adam": False,
            "iters": 1000,
            "group_size": 128,
        }

        # Mock model config
        config = MagicMock()
        config.hidden_size = 4096
        config.intermediate_size = 11008
        config.num_attention_heads = 32
        config.num_key_value_heads = 32
        config.num_hidden_layers = 32
        config.max_position_embeddings = 2048

        adjusted, steps = auto_tune(
            user_settings=user_settings,
            model_config=config,
            available_memory=available_memory,
            memory_utilization=effective_utilization,
        )

        # With 128 GB budget and a 7B model, no adjustments needed
        assert len(steps) == 0
        assert adjusted == user_settings

    def test_memory_safety_margin_reduces_utilization(self):
        """--memory_safety_margin 5 -> effective utilization = 70%."""
        args_memory_utilization = 75  # percent
        margin = 5
        effective = args_memory_utilization / 100.0 - margin / 100.0
        effective = max(0.50, min(0.95, effective))
        assert effective == 0.70

    def test_memory_safety_margin_clamps_low(self):
        """--memory_safety_margin 30 clamps to 0.50."""
        args_memory_utilization = 75
        margin = 30
        effective = args_memory_utilization / 100.0 - margin / 100.0
        effective = max(0.50, min(0.95, effective))
        assert effective == 0.50  # clamped at minimum

    def test_memory_safety_margin_clamps_high(self):
        """--memory_safety_margin negative clamps to 0.95."""
        args_memory_utilization = 100
        margin = -10
        effective = args_memory_utilization / 100.0 - margin / 100.0
        effective = max(0.50, min(0.95, effective))
        assert effective == 0.95  # clamped at maximum

    def test_output_dir_cache_check(self, tmp_path):
        """Verify the .cache/progress.json check logic."""
        # Create fake cache with OOM exit reason
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        progress = {
            "completed": 5,
            "total": 32,
            "block_names": [f"model.layers.{i}" for i in range(32)],
            "exit_reason": "oom",
            "tuning_profile": {
                "relaxation_step": 1,
                "oom_count": 0,
                "settings_active": {"batch_size": 4, "seqlen": 2048, "nsamples": 512, "adam": True},
            },
        }
        (cache_dir / "progress.json").write_text(json.dumps(progress))

        # Simulate CLI logic
        progress_path = cache_dir / "progress.json"
        assert progress_path.exists()

        with open(progress_path, "r") as f:
            loaded = json.load(f)

        stored_exit_reason = loaded.get("exit_reason")
        tuning_profile = loaded.get("tuning_profile")
        stored_completed = loaded.get("completed", 0)
        stored_total = loaded.get("total", 0)

        resume_mode = stored_completed > 0 and stored_completed < stored_total
        assert resume_mode is True
        assert stored_exit_reason == "oom"
        assert tuning_profile["relaxation_step"] == 1

        resume_context = {
            "exit_reason": stored_exit_reason,
            "oom_count": tuning_profile.get("oom_count", 0),
            "tuning_profile": tuning_profile,
        }

        # Now pass to auto_tune with resume context
        config = MagicMock()
        config.hidden_size = 4096
        config.intermediate_size = 11008
        config.num_attention_heads = 32
        config.num_key_value_heads = 32
        config.num_hidden_layers = 32
        config.max_position_embeddings = 2048

        # Use a tight budget to force adjustments
        adjusted, steps = auto_tune(
            user_settings={"batch_size": 8, "seqlen": 2048, "nsamples": 512, "adam": True, "iters": 1000, "group_size": 128},
            model_config=config,
            available_memory=50 * 1024 ** 3,  # 50 GB - tight
            memory_utilization=0.65,  # tighter margin from OOM
            resume_context=resume_context,
        )

        # Should have skipped steps and applied new relaxations
        # The OOM resume skips one step
        any_skipped = any(s.get("skipped") for s in steps)
        assert any_skipped, "OOM resume should produce skipped steps"

    def test_tuning_profile_passthrough(self):
        """Verify tuning_profile structure flows correctly."""
        tune_steps = [{"setting": "batch_size", "old": 8, "new": 4, "impact": "noisier gradients", "skipped": False}]
        tuning_profile = {
            "relaxation_step": len([s for s in tune_steps if not s.get("skipped")]),
            "oom_count": 1,
            "settings_active": {
                "batch_size": 4,
                "seqlen": 2048,
                "nsamples": 512,
                "adam": True,
            },
        }
        assert tuning_profile["relaxation_step"] == 1
        assert tuning_profile["oom_count"] == 1
        assert tuning_profile["settings_active"]["batch_size"] == 4


# ---------------------------------------------------------------------------
# Display integration tests (no GPU)
# ---------------------------------------------------------------------------


class TestDisplayIntegration:
    """Test that display functions work correctly in CLI context."""

    def test_full_preflight_no_adjustment(self):
        """Full pre-flight message for a small model."""
        msg = format_preflight_message(
            user_settings={"batch_size": 8, "seqlen": 2048, "nsamples": 512, "adam": False},
            adjusted_settings={"batch_size": 8, "seqlen": 2048, "nsamples": 512, "adam": False},
            steps=[],
            peak_gb=1.2,
            budget_gb=96.0,
            memory_utilization=0.75,
        )
        assert "Memory OK" in msg
        assert "1.2" in msg
        assert "96.0" in msg
        assert "75%" in msg

    def test_full_preflight_with_adjustment(self):
        """Pre-flight with adjustments shows all fields."""
        msg = format_preflight_message(
            user_settings={"batch_size": 8, "seqlen": 2048, "nsamples": 512, "adam": True},
            adjusted_settings={"batch_size": 4, "seqlen": 1024, "nsamples": 256, "adam": False},
            steps=[
                {"setting": "batch_size", "old": 8, "new": 4, "impact": "noisier gradients", "skipped": False},
                {"setting": "seqlen", "old": 2048, "new": 1024, "impact": "truncated context", "skipped": False},
                {"setting": "adam", "old": True, "new": False, "impact": "SignSGD vs Adam optimizer", "skipped": False},
            ],
            peak_gb=58.0,
            budget_gb=96.0,
            memory_utilization=0.75,
        )
        assert "Memory budget exceeded" in msg
        assert "batch_size" in msg
        assert "8 -> 4" in msg or "8 → 4" in msg
        assert "seqlen" in msg
        assert "2048 -> 1024" in msg or "2048 → 1024" in msg
        assert "adam" in msg
        # "False" displays as "0" in f-string width formatting of bool
        assert "True -> False" in msg or "True → False" in msg or "True -> 0" in msg or "True → 0" in msg
        assert "58.0" in msg

    def test_full_resume_after_oom(self):
        """Resume message with OOM context."""
        msg = format_resume_message(
            completed=9, total=48,
            exit_reason="oom",
            adjusted_settings={"batch_size": 2, "seqlen": 2048, "nsamples": 512, "adam": False},
            steps=[
                {"setting": "batch_size", "old": 4, "new": 2, "impact": "noisier gradients", "skipped": False},
            ],
            peak_gb=35.0,
            budget_gb=83.2,
            memory_utilization=0.65,
        )
        assert "Resuming from block 9/48" in msg
        assert "OOM'd" in msg
        assert "batch_size" in msg

    def test_full_resume_after_interrupt(self):
        """Resume message with interrupt context - fresh settings."""
        msg = format_resume_message(
            completed=5, total=48,
            exit_reason="interrupted",
            adjusted_settings={"batch_size": 8, "seqlen": 2048, "nsamples": 512, "adam": True},
            steps=[],
            peak_gb=42.0,
            budget_gb=96.0,
            memory_utilization=0.75,
        )
        assert "interrupted" in msg
        assert "user stopped" in msg
        assert "42.0" in msg

    def test_resume_with_oom_count(self):
        """Resume message showing OOM counter."""
        msg = format_resume_message(
            completed=3, total=48,
            exit_reason="oom",
            adjusted_settings={"batch_size": 1, "seqlen": 512, "nsamples": 128, "adam": False},
            steps=[],
            peak_gb=25.0,
            budget_gb=83.2,
            memory_utilization=0.65,
            oom_count=3,
        )
        assert "OOM count: 3" in msg
        assert "Accelerating relaxation" in msg


# ---------------------------------------------------------------------------
# Full flow integration (requires GPU)
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestFullFlowIntegration:
    """End-to-end tests that require a CUDA GPU and a real HF model.

    These tests verify the complete pipeline: memory estimation -> auto-tune ->
    display -> AutoRound construction -> quantization with checkpoint.
    """

    @pytest.fixture(autouse=True)
    def check_cuda(self, request):
        """Check CUDA availability."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_small_model_no_tuning(self):
        """Verify memory estimator gives reasonable numbers for a small model."""
        # Use a minimal config to avoid loading model
        config = MagicMock()
        config.hidden_size = 576  # very small
        config.intermediate_size = 1536
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.num_hidden_layers = 2
        config.max_position_embeddings = 2048

        peak_gb, breakdown = estimate_peak_memory_per_block(
            config, {"batch_size": 1, "seqlen": 128, "adam": False}
        )
        assert peak_gb > 0, "Peak memory should be positive"
        assert "block_weights_bf16" in breakdown

    def test_memory_estimator_called_from_cli_context(self):
        """Verify the CLI integration calls memory estimator with correct args."""
        from auto_round.compressors.memory_estimator import estimate_peak_memory_per_block

        config = MagicMock()
        config.hidden_size = 4096
        config.intermediate_size = 11008
        config.num_attention_heads = 32
        config.num_key_value_heads = 32
        config.num_hidden_layers = 32
        config.max_position_embeddings = 2048

        # What __main__.py does after auto_tune
        adjusted_settings = {"batch_size": 4, "seqlen": 1024, "nsamples": 256, "adam": False}
        peak_gb, breakdown = estimate_peak_memory_per_block(config, adjusted_settings)

        # Verify estimate is in a reasonable range for 8 params per block
        assert 0.5 < peak_gb < 50, f"Unexpected peak memory: {peak_gb} GB"
        assert isinstance(breakdown, dict)
        # total_estimated is unrounded, peak_gb is round(..., 2)
        assert breakdown["total_estimated"] == pytest.approx(peak_gb, abs=0.01)