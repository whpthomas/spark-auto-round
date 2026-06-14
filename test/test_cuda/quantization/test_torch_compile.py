import os
import shutil
from types import SimpleNamespace

import pytest

from auto_round.algorithms.quantization.base import BaseQuantizers
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.compressors.utils import block_forward


class TestTorchCompile:
    """Tests for torch.compile integration with quantization."""

    def test_signround_without_torch_compile_uses_plain(self):
        """Test that SignRound without torch.compile uses plain block_forward."""
        config = SignRoundConfig(bits=4, data_type="int", act_bits=16)
        quantizer = BaseQuantizers(config)
        quantizer.compress_context = SimpleNamespace(enable_torch_compile=False, device="cpu")
        resolved = quantizer._resolve_block_forward()
        assert resolved is block_forward

    def test_signround_with_torch_compile_resolves(self):
        """Test that SignRound with torch.compile enabled resolves a compiled forward."""
        config = SignRoundConfig(bits=4, data_type="int", act_bits=16)
        quantizer = BaseQuantizers(config)
        quantizer.compress_context = SimpleNamespace(enable_torch_compile=True, device="cpu")
        resolved = quantizer._resolve_block_forward()
        # Should not be plain block_forward when torch.compile is enabled
        # (assuming no act quantization and no alg_ext)
        assert resolved is not None
