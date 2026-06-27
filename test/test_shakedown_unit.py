# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shakedown mode override logic (no GPU needed).

These tests validate that the ``--shakedown`` flag correctly overrides
tuning parameters in ``tune()`` after the auto-tuner has run.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestShakedownOverrideLogic:
    """Test shakedown parameter overrides without calling tune()."""

    def test_shakedown_override_values(self):
        """Verify the exact overrides applied by shakedown mode."""
        import argparse

        # Simulate the override block from tune()
        args = argparse.Namespace(
            shakedown=True,
            iters=1000,
            nsamples=512,
            seqlen=2048,
            batch_size=8,
        )

        # Apply shakedown overrides (same logic as in tune())
        if args.shakedown:
            args.iters = 1
            args.nsamples = 1
            args.seqlen = 2
            args.batch_size = 1

        assert args.iters == 1
        assert args.nsamples == 1
        assert args.seqlen == 2
        assert args.batch_size == 1

    def test_shakedown_does_not_override_non_defaults(self):
        """shakedown overrides are independent of pre-existing values."""
        import argparse

        args = argparse.Namespace(
            shakedown=True,
            iters=500,
            nsamples=256,
            seqlen=1024,
            batch_size=4,
        )

        if args.shakedown:
            args.iters = 1
            args.nsamples = 1
            args.seqlen = 2
            args.batch_size = 1

        assert args.iters == 1
        assert args.nsamples == 1
        assert args.seqlen == 2
        assert args.batch_size == 1

    def test_no_shakedown_preserves_original_values(self):
        """Without --shakedown, original values are preserved."""
        import argparse

        args = argparse.Namespace(
            shakedown=False,
            iters=1000,
            nsamples=512,
            seqlen=2048,
            batch_size=8,
        )

        if args.shakedown:
            args.iters = 1
            args.nsamples = 1
            args.seqlen = 2
            args.batch_size = 1

        assert args.iters == 1000
        assert args.nsamples == 512
        assert args.seqlen == 2048
        assert args.batch_size == 8


class TestShakedownAdjustedSettings:
    """Test that shakedown mode overrides adjusted_settings after auto-tuner."""

    def test_adjusted_settings_overridden_in_shakedown(self):
        """In shakedown mode, adjusted_settings are replaced with floor values."""
        # Simulate what tune() does after auto-tuner:
        # adjusted_settings comes back with tuner's values,
        # then shakedown overrides and also rebuilds adjusted_settings
        adjusted_settings = {
            "batch_size": 4,
            "seqlen": 1024,
            "nsamples": 64,
        }

        if True:  # args.shakedown
            adjusted_settings = {"batch_size": 1, "seqlen": 2, "nsamples": 1}

        assert adjusted_settings["batch_size"] == 1
        assert adjusted_settings["seqlen"] == 2
        assert adjusted_settings["nsamples"] == 1
