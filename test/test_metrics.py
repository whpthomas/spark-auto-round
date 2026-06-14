# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for auto_round.metrics — standalone sensitivity metric functions."""

import math

import pytest
import torch


class TestComputePSNR:
    """Test compute_psnr function."""

    def test_identical_outputs(self):
        """Identical tensors → inf PSNR (perfect reconstruction)."""
        from auto_round.metrics import compute_psnr
        ref = [torch.randn(1, 128, 64)]
        q = [t.clone() for t in ref]
        assert compute_psnr(ref, q) == float("inf")

    def test_different_outputs(self):
        """Ones vs zeros → 0 dB PSNR."""
        from auto_round.metrics import compute_psnr
        ref = [torch.ones(1, 128, 64)]
        q = [torch.zeros(1, 128, 64)]
        psnr = compute_psnr(ref, q)
        assert psnr == pytest.approx(0.0, abs=1e-5)

    def test_scaled_output(self):
        """Scaled output: CosSim=1.0 but PSNR catches magnitude collapse."""
        from auto_round.metrics import compute_psnr, compute_block_sensitivity
        ref = [torch.ones(1, 128, 64)]
        q = [0.1 * torch.ones(1, 128, 64)]
        cos_sim, _ = compute_block_sensitivity(ref, q)
        psnr = compute_psnr(ref, q)
        assert cos_sim == pytest.approx(1.0, abs=1e-4)
        assert psnr < 45.0

    def test_empty_inputs(self):
        """Empty lists → inf PSNR."""
        from auto_round.metrics import compute_psnr
        assert compute_psnr([], []) == float("inf")

    def test_multiple_samples(self):
        """PSNR is averaged across multiple samples."""
        from auto_round.metrics import compute_psnr
        ref1 = [torch.ones(1, 64)]
        q1 = [0.5 * torch.ones(1, 64)]  # PSNR ≈ 6.02 dB
        ref2 = [torch.ones(1, 64)]
        q2 = [0.99 * torch.ones(1, 64)]  # PSNR ≈ 46.06 dB
        psnr = compute_psnr(ref1 + ref2, q1 + q2)
        # Should be average of both
        assert psnr > 6.0 and psnr < 47.0

    def test_zero_mixed_with_nonzero(self):
        """Mixed samples: one perfect, one not → average of inf and finite = inf."""
        from auto_round.metrics import compute_psnr
        ref = [torch.ones(1, 64), torch.ones(1, 64)]
        q = [torch.ones(1, 64), torch.zeros(1, 64)]
        psnr = compute_psnr(ref, q)
        # Perfect sample (inf) dominates the average
        assert psnr == float("inf")


class TestComputeBlockSensitivity:
    """Test compute_block_sensitivity function."""

    def test_returns_tuple(self):
        """Returns (float, float) tuple."""
        from auto_round.metrics import compute_block_sensitivity
        ref = [torch.randn(1, 128, 64)]
        q = [t.clone() for t in ref]
        result = compute_block_sensitivity(ref, q)
        assert isinstance(result, tuple)
        assert len(result) == 2
        cos_sim, psnr_db = result
        assert isinstance(cos_sim, float)
        assert isinstance(psnr_db, float)

    def test_perfect_reconstruction(self):
        """Identical tensors → (1.0, inf)."""
        from auto_round.metrics import compute_block_sensitivity
        ref = [torch.randn(1, 128, 64)]
        q = [t.clone() for t in ref]
        cos_sim, psnr_db = compute_block_sensitivity(ref, q)
        assert cos_sim == pytest.approx(1.0, abs=1e-5)
        assert psnr_db == float("inf")

    def test_empty_inputs(self):
        """Empty lists → (1.0, inf)."""
        from auto_round.metrics import compute_block_sensitivity
        cos_sim, psnr_db = compute_block_sensitivity([], [])
        assert cos_sim == 1.0
        assert psnr_db == float("inf")

    def test_known_values(self):
        """Verify against hand-computed values for simple tensors."""
        from auto_round.metrics import compute_block_sensitivity
        ref = [torch.tensor([1.0, 0.0, 0.0])]
        q = [torch.tensor([0.9, 0.1, 0.0])]
        cos_sim, psnr_db = compute_block_sensitivity(ref, q)
        # cos_sim should be close to 0.9938
        assert 0.99 < cos_sim < 1.0
        # PSNR should be positive
        assert psnr_db > 0

    def test_multiple_samples_averaged(self):
        """Metrics are averaged across samples."""
        from auto_round.metrics import compute_block_sensitivity
        # Use non-constant vectors so cosine similarity differs from 1.0
        torch.manual_seed(42)
        ref = [torch.randn(1, 64), torch.randn(1, 64)]
        # High quality: small perturbation
        q_high = [r + 0.001 * torch.randn_like(r) for r in ref]
        # Low quality: large perturbation
        q_low = [r + 0.5 * torch.randn_like(r) for r in ref]
        cos_high, psnr_high = compute_block_sensitivity(ref, q_high)
        cos_low, psnr_low = compute_block_sensitivity(ref, q_low)
        # High quality should have better metrics than low quality
        assert cos_high > cos_low
        assert psnr_high > psnr_low
        # Both should be reasonable values
        assert 0.95 < cos_high <= 1.0
        assert 0 < psnr_low < psnr_high
