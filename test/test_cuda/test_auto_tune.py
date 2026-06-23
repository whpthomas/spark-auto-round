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
"""Tests for auto_round.compressors.auto_tune.

These tests verify the auto-tuner's relaxation priority ladder, resume
context handling, and display formatting. They do NOT require a GPU —
all memory estimates are computed from config dimensions alone.
"""

import pytest

from auto_round.compressors.auto_tune import (
    auto_tune,
    format_preflight_message,
    format_resume_message,
    _resolve_resume_offset,
    _RELAXATION_LADDER,
)


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
# Fixtures — mocked HF configs
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    """Simulate SmolLM-135M (tiny, easily fits any budget)."""
    config = _MockConfig()
    config.hidden_size = 576
    config.intermediate_size = 1536
    config.num_attention_heads = 9
    config.num_key_value_heads = 3
    config.num_hidden_layers = 30
    config.max_position_embeddings = 2048
    return config


@pytest.fixture
def medium_config():
    """Simulate a 7B dense model (e.g., Llama-7B)."""
    config = _MockConfig()
    config.hidden_size = 4096
    config.intermediate_size = 11008
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    config.num_hidden_layers = 32
    config.max_position_embeddings = 2048
    return config


@pytest.fixture
def large_moe_config():
    """Simulate Qwen3.5-122B-A10B (massive MoE, exceeds any budget)."""
    config = _MockConfig()
    config.hidden_size = 8192
    config.intermediate_size = 24576
    config.num_attention_heads = 64
    config.num_key_value_heads = 8
    config.num_hidden_layers = 48
    config.num_experts = 256
    config.top_k = 8
    config.max_position_embeddings = 32768
    return config


# Default user settings — matches __main__.py defaults
DEFAULT_SETTINGS = {
    "batch_size": 8,
    "seqlen": 2048,
    "nsamples": 512,
    "iters": 1000,
    "group_size": 128,
}

# 128 GB available memory (DGX Spark GB10)
DGX_SPARK_MEMORY = 128 * 1024 ** 3  # 128 GiB in bytes
DEFAULT_BUDGET = 96 * 1024 ** 3  # 96 GiB default budget


# ---------------------------------------------------------------------------
# auto_tune — fresh (no resume context)
# ---------------------------------------------------------------------------


class TestAutoTuneFresh:
    def test_no_adjustment_needed(self, small_config):
        """Small model with generous budget — no settings changed."""
        adjusted, steps = auto_tune(
            DEFAULT_SETTINGS, small_config, DEFAULT_BUDGET,
        )
        assert len(steps) == 0
        assert adjusted == DEFAULT_SETTINGS

    def test_batch_size_relaxed_once(self, medium_config):
        """Moderate pressure — batch_size reduces one step (8→4)."""
        from auto_round.compressors.memory_estimator import (
            estimate_peak_memory_per_block,
        )
        peak_bs8, _ = estimate_peak_memory_per_block(medium_config, {
            "batch_size": 8, "seqlen": 2048,
            "group_size": DEFAULT_SETTINGS["group_size"],
        })
        peak_bs4, _ = estimate_peak_memory_per_block(medium_config, {
            "batch_size": 4, "seqlen": 2048,
            "group_size": DEFAULT_SETTINGS["group_size"],
        })

        # Budget must be between peak(bs=8) and peak(bs=4) so exactly one step
        assert peak_bs8 > peak_bs4, "expected bs=8 to use more memory than bs=4"
        midpoint_gb = (peak_bs8 + peak_bs4) / 2
        budget_bytes = int(midpoint_gb * (1024 ** 3))

        adjusted, steps = auto_tune(
            DEFAULT_SETTINGS, medium_config, budget_bytes,
        )
        assert len(steps) >= 1
        assert steps[0]["setting"] == "batch_size"
        assert steps[0]["old"] == 8
        assert steps[0]["new"] == 4
        assert adjusted["batch_size"] == 4

    def test_multiple_relaxations(self, medium_config):
        """Sequential relaxations across different settings."""
        from auto_round.compressors.memory_estimator import (
            estimate_peak_memory_per_block,
        )

        # Find a budget that forces at least 2 steps
        # Start with bs=1 (minimum), check peak
        peak_min_bs, _ = estimate_peak_memory_per_block(medium_config, {
            "batch_size": 1, "seqlen": 2048,
            "group_size": DEFAULT_SETTINGS["group_size"],
        })

        # Budget just below bs=1 peak — forces at least 2 relaxations
        budget_bytes = int(peak_min_bs * (1024 ** 3))

        adjusted, steps = auto_tune(
            DEFAULT_SETTINGS, medium_config, budget_bytes,
        )
        # Should have at least 2 steps and batch_size should be ≤ 1
        assert len(steps) >= 2
        assert adjusted["batch_size"] <= 1
        # batch_size should appear in the steps
        batch_steps = [s for s in steps if s["setting"] == "batch_size"]
        assert len(batch_steps) >= 1

    def test_max_relaxations(self, large_moe_config):
        """MoE model with 128 GB budget — all settings should hit minimums."""
        adjusted, steps = auto_tune(
            DEFAULT_SETTINGS, large_moe_config, DEFAULT_BUDGET,
        )
        # All relaxations should be applied
        assert adjusted["batch_size"] == 1
        assert adjusted["seqlen"] == 256
        assert adjusted["nsamples"] == 128
        # Should have 3+ steps (one per ladder entry minimum)
        assert len(steps) >= 3
        # Verify iters and group_size untouched
        assert adjusted["iters"] == 1000
        assert adjusted["group_size"] == 128

    def test_never_touches_iters_or_group_size(self, medium_config):
        """Auto-tuner must not modify iters or group_size under any budget."""
        from auto_round.compressors.memory_estimator import (
            estimate_peak_memory_per_block,
        )

        peak_default, _ = estimate_peak_memory_per_block(medium_config, {
            "batch_size": 8, "seqlen": 2048,
            "group_size": DEFAULT_SETTINGS["group_size"],
        })
        # Tight budget forces relaxations
        budget_bytes = int(peak_default * (1024 ** 3) * 0.8)

        adjusted, steps = auto_tune(
            DEFAULT_SETTINGS, medium_config, budget_bytes,
        )
        assert adjusted["iters"] == 1000
        assert adjusted["group_size"] == 128
        # Verify steps don't include iters or group_size
        setting_names = [s["setting"] for s in steps]
        assert "iters" not in setting_names
        assert "group_size" not in setting_names

    def test_nonstandard_initial_values(self, medium_config):
        """User starts with batch_size=2 — tuner starts from that level."""
        from auto_round.compressors.memory_estimator import (
            estimate_peak_memory_per_block,
        )

        settings = dict(DEFAULT_SETTINGS)
        settings["batch_size"] = 2

        peak_bs2, _ = estimate_peak_memory_per_block(medium_config, {
            "batch_size": 2, "seqlen": 2048,
            "group_size": settings["group_size"],
        })
        budget_bytes = int(peak_bs2 * (1024 ** 3) * 0.98)

        adjusted, steps = auto_tune(
            settings, medium_config, budget_bytes,
        )
        # batch_size should go 2→1
        batch_steps = [s for s in steps if s["setting"] == "batch_size"]
        if batch_steps:
            assert batch_steps[0]["old"] == 2
            assert batch_steps[0]["new"] == 1
        assert adjusted["batch_size"] <= 1

    def test_different_budget_levels(self, medium_config):
        """Lower budget should trigger stricter relaxations."""
        from auto_round.compressors.memory_estimator import (
            estimate_peak_memory_per_block,
        )

        peak_default, _ = estimate_peak_memory_per_block(medium_config, {
            "batch_size": 8, "seqlen": 2048,
            "group_size": DEFAULT_SETTINGS["group_size"],
        })

        # Generous budget (fits with no relaxation)
        generous_budget = int(peak_default * (1024 ** 3) * 1.1)
        # Tight budget (forces relaxation)
        tight_budget = int(peak_default * (1024 ** 3) * 0.8)

        adjusted_generous, steps_generous = auto_tune(
            DEFAULT_SETTINGS, medium_config, generous_budget,
        )
        adjusted_tight, steps_tight = auto_tune(
            DEFAULT_SETTINGS, medium_config, tight_budget,
        )
        # Tighter budget = more relaxation steps
        assert len(steps_tight) >= len(steps_generous)

    def test_budget_exceeded_at_max_relaxations(self, large_moe_config):
        """Even at max relaxations the MoE model exceeds budget — best-effort."""
        adjusted, steps = auto_tune(
            DEFAULT_SETTINGS, large_moe_config, DEFAULT_BUDGET,
        )
        # Even with everything at minimum, MoE still likely exceeds 96 GB
        # The tuner should still return settings and steps (best effort)
        assert len(steps) > 0
        # All settings at their minimums
        assert adjusted["batch_size"] == 1
        assert adjusted["seqlen"] == 256
        assert adjusted["nsamples"] == 128


# ---------------------------------------------------------------------------
# auto_tune — resume context
# ---------------------------------------------------------------------------


class TestAutoTuneResume:
    def test_resume_after_interrupt_fresh_start(self, medium_config):
        """Interrupted run — no skip, fresh auto-tune."""
        from auto_round.compressors.memory_estimator import (
            estimate_peak_memory_per_block,
        )

        peak_bs8, _ = estimate_peak_memory_per_block(medium_config, {
            "batch_size": 8, "seqlen": 2048,
            "group_size": DEFAULT_SETTINGS["group_size"],
        })
        # Budget that forces exactly 1 relaxation
        peak_bs4, _ = estimate_peak_memory_per_block(medium_config, {
            "batch_size": 4, "seqlen": 2048,
            "group_size": DEFAULT_SETTINGS["group_size"],
        })
        midpoint_gb = (peak_bs8 + peak_bs4) / 2
        budget_bytes = int(midpoint_gb * (1024 ** 3))

        adjusted, steps = auto_tune(
            DEFAULT_SETTINGS, medium_config, budget_bytes,
            resume_context={"exit_reason": "interrupted", "oom_count": 0},
        )
        # Should be identical to fresh run with same budget
        adjusted_fresh, steps_fresh = auto_tune(
            DEFAULT_SETTINGS, medium_config, budget_bytes,
        )
        assert adjusted == adjusted_fresh
        assert steps == steps_fresh

    def test_resume_after_oom_skip_one_level(self, medium_config):
        """OOM resume — skip one level (start from next relaxation step)."""
        from auto_round.compressors.memory_estimator import (
            estimate_peak_memory_per_block,
        )

        peak_bs8, _ = estimate_peak_memory_per_block(medium_config, {
            "batch_size": 8, "seqlen": 2048,
            "group_size": DEFAULT_SETTINGS["group_size"],
        })
        # Tighter budget
        budget_bytes = int(peak_bs8 * (1024 ** 3) * 0.7)

        adjusted_resume, steps_resume = auto_tune(
            DEFAULT_SETTINGS, medium_config, budget_bytes,
            resume_context={"exit_reason": "oom", "oom_count": 0},
        )
        # OOM resume should have at least one skipped step
        skipped = [s for s in steps_resume if s.get("skipped")]
        assert len(skipped) >= 1

        # Without OOM context, fewer steps (no skip)
        adjusted_fresh, steps_fresh = auto_tune(
            DEFAULT_SETTINGS, medium_config, budget_bytes,
        )
        # OOM version should have same or more total steps but with skip markers
        # (skipped steps are included in steps list with skipped=True)
        assert len(steps_resume) >= len(steps_fresh)

    def test_resume_after_multiple_ooms(self, medium_config):
        """2+ OOMs — skip 2 steps (accelerate through broken configs)."""
        from auto_round.compressors.memory_estimator import (
            estimate_peak_memory_per_block,
        )

        peak_default, _ = estimate_peak_memory_per_block(medium_config, {
            "batch_size": 8, "seqlen": 2048,
            "group_size": DEFAULT_SETTINGS["group_size"],
        })
        budget_bytes = int(peak_default * (1024 ** 3) * 0.6)

        adjusted, steps = auto_tune(
            DEFAULT_SETTINGS, medium_config, budget_bytes,
            resume_context={"exit_reason": "oom", "oom_count": 3},
        )
        # oom_count=3 → skip = 1 + (3//2) = 2
        skipped = [s for s in steps if s.get("skipped")]
        assert len(skipped) >= 2

    def test_resume_no_context(self, medium_config):
        """No resume context — fresh start (same as auto_tune without ctx)."""
        from auto_round.compressors.memory_estimator import (
            estimate_peak_memory_per_block,
        )

        peak_default, _ = estimate_peak_memory_per_block(medium_config, {
            "batch_size": 8, "seqlen": 2048,
            "group_size": DEFAULT_SETTINGS["group_size"],
        })
        budget_bytes = int(peak_default * (1024 ** 3) * 0.7)

        adjusted, steps = auto_tune(
            DEFAULT_SETTINGS, medium_config, budget_bytes,
            resume_context=None,
        )
        adjusted_fresh, _ = auto_tune(
            DEFAULT_SETTINGS, medium_config, budget_bytes,
        )
        assert adjusted == adjusted_fresh


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------


class TestFormatMessages:
    def test_preflight_no_adjustment(self):
        """Happy path — no adjustments needed."""
        msg = format_preflight_message(
            {"batch_size": 8, "seqlen": 2048, "nsamples": 512},
            {"batch_size": 8, "seqlen": 2048, "nsamples": 512},
            steps=[],
            peak_gb=42.5,
            budget_gb=96.0,
        )
        assert "Memory OK" in msg
        assert "42.5" in msg
        assert "96.0" in msg

    def test_preflight_single_adjustment(self):
        """Single adjustment displayed correctly."""
        msg = format_preflight_message(
            {"batch_size": 8, "seqlen": 2048, "nsamples": 512},
            {"batch_size": 4, "seqlen": 2048, "nsamples": 512},
            steps=[
                {
                    "setting": "batch_size", "old": 8, "new": 4,
                    "impact": "noisier gradients", "skipped": False,
                },
            ],
            peak_gb=58.0,
            budget_gb=96.0,
        )
        assert "Memory budget exceeded" in msg
        assert "batch_size" in msg
        assert "8 → 4" in msg
        assert "noisier gradients" in msg
        assert "58.0" in msg
        assert "96.0" in msg

    def test_preflight_multiple_adjustments(self):
        """Multiple adjustments each on their own line."""
        msg = format_preflight_message(
            {"batch_size": 8, "seqlen": 2048, "nsamples": 512},
            {"batch_size": 2, "seqlen": 1024, "nsamples": 512},
            steps=[
                {
                    "setting": "batch_size", "old": 8, "new": 4,
                    "impact": "noisier gradients", "skipped": False,
                },
                {
                    "setting": "batch_size", "old": 4, "new": 2,
                    "impact": "noisier gradients", "skipped": False,
                },
                {
                    "setting": "seqlen", "old": 2048, "new": 1024,
                    "impact": "truncated context", "skipped": False,
                },
            ],
            peak_gb=72.0,
            budget_gb=96.0,
        )
        assert "batch_size" in msg
        assert "seqlen" in msg
        assert msg.count("batch_size") == 2  # two batch_size adjustments

    def test_preflight_with_skipped(self):
        """Skipped steps (OOM resume) shown as additional info."""
        msg = format_preflight_message(
            {"batch_size": 8, "seqlen": 2048, "nsamples": 512},
            {"batch_size": 2, "seqlen": 2048, "nsamples": 512},
            steps=[
                {
                    "setting": "batch_size", "old": 8, "new": 4,
                    "impact": "noisier gradients", "skipped": True,
                },
                {
                    "setting": "batch_size", "old": 4, "new": 2,
                    "impact": "noisier gradients", "skipped": False,
                },
            ],
            peak_gb=65.0,
            budget_gb=96.0,
        )
        assert "additionally" in msg or "skipped" in msg

    def test_preflight_still_exceeds_budget(self):
        """Warning indicator when peak still exceeds budget."""
        msg = format_preflight_message(
            {"batch_size": 1, "seqlen": 256, "nsamples": 128},
            {"batch_size": 1, "seqlen": 256, "nsamples": 128},
            steps=[],
            peak_gb=320.0,
            budget_gb=96.0,
        )
        # No adjustments to show, but peak exceeds budget
        assert "still exceeds budget" in msg or "⚠" in msg

    def test_resume_after_oom(self):
        """Resume message after OOM shows relevant context."""
        msg = format_resume_message(
            completed=9, total=48,
            exit_reason="oom",
            adjusted_settings={
                "batch_size": 2, "seqlen": 2048,
                "nsamples": 512,
            },
            steps=[
                {
                    "setting": "batch_size", "old": 4, "new": 2,
                    "impact": "noisier gradients", "skipped": False,
                },
            ],
            peak_gb=35.0,
            budget_gb=83.2,
        )
        assert "Resuming from block 9/48" in msg
        assert "OOM'd" in msg
        assert "batch_size" in msg
        assert "4 → 2" in msg
        assert "35.0" in msg

    def test_resume_after_interrupt(self):
        """Resume message after interrupt shows user-stopped context."""
        msg = format_resume_message(
            completed=5, total=48,
            exit_reason="interrupted",
            adjusted_settings={
                "batch_size": 8, "seqlen": 2048,
                "nsamples": 512,
            },
            steps=[],
            peak_gb=42.0,
            budget_gb=96.0,
        )
        assert "interrupted" in msg
        assert "user stopped" in msg

    def test_resume_with_oom_count(self):
        """OOM resume with oom_count shows acceleration message."""
        msg = format_resume_message(
            completed=3, total=48,
            exit_reason="oom",
            adjusted_settings={
                "batch_size": 4, "seqlen": 2048,
                "nsamples": 512,
            },
            oom_count=2,
            steps=[
                {
                    "setting": "batch_size", "old": 8, "new": 4,
                    "impact": "noisier gradients", "skipped": False,
                },
            ],
            peak_gb=75.0,
            budget_gb=96.0,
        )
        assert "OOM count" in msg
        assert "Accelerating" in msg

    def test_resume_unknown_exit_reason(self):
        """Unknown exit reason falls through to generic message."""
        msg = format_resume_message(
            completed=1, total=48,
            exit_reason="crash",
            adjusted_settings={
                "batch_size": 8, "seqlen": 2048,
                "nsamples": 512,
            },
            steps=[],
            peak_gb=42.0,
            budget_gb=96.0,
        )
        assert "crash" in msg
        assert "Resuming from block 1/48" in msg


# ---------------------------------------------------------------------------
# _resolve_resume_offset edge cases
# ---------------------------------------------------------------------------


class TestResolveResumeOffset:
    def test_no_context(self):
        assert _resolve_resume_offset(None) == (0, 0)

    def test_interrupted(self):
        assert _resolve_resume_offset(
            {"exit_reason": "interrupted", "oom_count": 0}
        ) == (0, 0)

    def test_oom_first(self):
        assert _resolve_resume_offset(
            {"exit_reason": "oom", "oom_count": 0}
        ) == (0, 1)

    def test_oom_second(self):
        assert _resolve_resume_offset(
            {"exit_reason": "oom", "oom_count": 1}
        ) == (0, 1)

    def test_oom_third(self):
        # oom_count=2 → skip = 1 + (2//2) = 2
        assert _resolve_resume_offset(
            {"exit_reason": "oom", "oom_count": 2}
        ) == (0, 2)

    def test_oom_fifth(self):
        # oom_count=4 → skip = 1 + (4//2) = 3
        assert _resolve_resume_offset(
            {"exit_reason": "oom", "oom_count": 4}
        ) == (0, 3)

    def test_unknown_reason(self):
        assert _resolve_resume_offset(
            {"exit_reason": None, "oom_count": 0}
        ) == (0, 0)

    def test_missing_oom_count(self):
        """Missing oom_count key treated as 0."""
        assert _resolve_resume_offset(
            {"exit_reason": "oom"}
        ) == (0, 1)

    def test_empty_context(self):
        assert _resolve_resume_offset({}) == (0, 0)

    def test_negative_oom_count(self):
        """Negative oom_count is treated as 0 (floor division)."""
        assert _resolve_resume_offset(
            {"exit_reason": "oom", "oom_count": -1}
        ) == (0, 1)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestRelaxationLadder:
    def test_ladder_has_required_entries(self):
        """Ladder must have batch_size, seqlen, nsamples entries."""
        keys = [e["key"] for e in _RELAXATION_LADDER]
        assert "batch_size" in keys
        assert "seqlen" in keys
        assert "nsamples" in keys
        # Must be in this order
        assert keys.index("batch_size") < keys.index("seqlen")
        assert keys.index("seqlen") < keys.index("nsamples")

    def test_each_entry_has_required_fields(self):
        """Each ladder entry must have key, levels, and impact."""
        for entry in _RELAXATION_LADDER:
            assert "key" in entry
            assert "levels" in entry
            assert "impact" in entry
            assert len(entry["levels"]) >= 2  # at least one relaxation possible

    def test_never_auto_tune_not_in_ladder(self):
        """iters, group_size, and adam must NOT be in the ladder."""
        keys = [e["key"] for e in _RELAXATION_LADDER]
        assert "iters" not in keys
        assert "group_size" not in keys
        # adam should also not be in the ladder (dead code)
        assert "adam" not in keys


# ---------------------------------------------------------------------------
# Memory budget tests
# ---------------------------------------------------------------------------


class TestMemoryBudget:
    """Tests for the --memory-budget flag behavior."""

    def test_budget_bytes_is_direct_ceiling(self, small_config):
        """budget_bytes is used directly — no multiplication."""
        # If budget is exactly peak, it should fit (peak <= budget)
        from auto_round.compressors.memory_estimator import (
            estimate_peak_memory_per_block,
        )
        # Use same settings as the auto_tune call
        settings = dict(DEFAULT_SETTINGS)
        peak, _ = estimate_peak_memory_per_block(small_config, {
            "batch_size": 8, "seqlen": 2048,
            "group_size": settings["group_size"],
        })
        budget_bytes = int(peak * (1024 ** 3))
        adjusted, steps = auto_tune(
            settings, small_config, budget_bytes,
        )
        # Should fit with no relaxation
        assert len(steps) == 0

    def test_budget_below_minimum_triggers_all_relaxations(self, large_moe_config):
        """Very small budget forces all settings to minimums."""
        adjusted, steps = auto_tune(
            DEFAULT_SETTINGS, large_moe_config, 1 * (1024 ** 3),  # 1 GiB
        )
        assert adjusted["batch_size"] == 1
        assert adjusted["seqlen"] == 256
        assert adjusted["nsamples"] == 128
        assert len(steps) >= 3
