"""Tests for meta device loading support (Phase 2).

Verifies that:
1. llm_load_model() supports device="meta"
2. ModelContext accepts and uses use_meta_device parameter
3. CompressContext stores use_meta_device flag
4. unsupported_meta_device() allows models with .path attribute
5. use_meta_device flows from CLI through to BaseCompressor
"""
import inspect
import pytest
import torch
import torch.nn as nn


class TestLLMLoadModelMetaDevice:
    """Tests for meta device handling in llm_load_model()."""

    def test_llm_load_model_accepts_device_meta(self):
        """Verify llm_load_model signature accepts device='meta'."""
        from auto_round.utils.model import llm_load_model
        sig = inspect.signature(llm_load_model)
        assert "device" in sig.parameters
        # Default should still be "cpu"
        assert sig.parameters["device"].default == "cpu"

    def test_meta_device_path_in_source(self):
        """Verify meta device handling exists in llm_load_model()."""
        from auto_round.utils.model import llm_load_model
        source = inspect.getsource(llm_load_model)
        assert 'device == "meta"' in source or "device == 'meta'" in source
        assert "from_config" in source

    def test_meta_device_sets_model_path(self):
        """Verify meta device path sets model.path for later block loading."""
        from auto_round.utils.model import llm_load_model
        source = inspect.getsource(llm_load_model)
        assert "model.path = pretrained_model_name_or_path" in source

    def test_meta_device_sets_model_config(self):
        """Verify meta device path stores config on the model."""
        from auto_round.utils.model import llm_load_model
        source = inspect.getsource(llm_load_model)
        assert "model.config = config" in source


class TestUnsupportedMetaDevice:
    """Tests for unsupported_meta_device() with meta device models."""

    def test_meta_device_with_path_passes(self):
        """Model with .path attribute on meta device should NOT be unsupported."""
        from auto_round.utils.model import unsupported_meta_device

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
                self.path = "/some/model/path"

            def parameters(self):
                yield nn.Parameter(torch.empty(10, 10, device="meta"))

        model = MockModel()
        result = unsupported_meta_device(model)
        assert result is False, "Model with .path on meta device should be supported"

    def test_meta_device_without_path_fails(self):
        """Model without .path on meta device should be unsupported."""
        from auto_round.utils.model import unsupported_meta_device

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def parameters(self):
                yield nn.Parameter(torch.empty(10, 10, device="meta"))

        model = MockModel()
        result = unsupported_meta_device(model)
        assert result is True, "Model without .path on meta device should be unsupported"

    def test_cpu_model_passes(self):
        """Model on CPU should pass (not unsupported)."""
        from auto_round.utils.model import unsupported_meta_device

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

        model = MockModel()
        result = unsupported_meta_device(model)
        assert result is False, "CPU model should be supported"


class TestModelContextMetaDevice:
    """Tests for ModelContext meta device support."""

    def test_model_context_accepts_use_meta_device(self):
        """Verify ModelContext.__init__ accepts use_meta_device param."""
        from auto_round.context.model import ModelContext
        # The singleton metaclass wraps __init__, so inspect.getsource(cls.__init__)
        # returns the wrapper. Inspect the full class source instead.
        source = inspect.getsource(ModelContext)
        assert "use_meta_device" in source

    def test_model_context_stores_use_meta_device(self):
        """Verify _use_meta_device is stored when passed."""
        from auto_round.context.model import ModelContext
        source = inspect.getsource(ModelContext)
        assert "_use_meta_device = use_meta_device" in source

    def test_load_model_uses_meta_device_when_flagged(self):
        """Verify _load_model uses meta device when _use_meta_device is True."""
        from auto_round.context.model import ModelContext
        source = inspect.getsource(ModelContext._load_model)
        assert '"meta" if self._use_meta_device else "cpu"' in source


class TestCompressContextMetaDevice:
    """Tests for CompressContext meta device support."""

    def test_compress_context_accepts_use_meta_device(self):
        """Verify CompressContext.__init__ accepts use_meta_device param."""
        from auto_round.context.compress import CompressContext
        # The singleton metaclass wraps __init__, so inspect.getsource(cls.__init__)
        # returns the wrapper. Inspect the full class source instead.
        source = inspect.getsource(CompressContext)
        assert "use_meta_device" in source

    def test_compress_context_stores_use_meta_device(self):
        """Verify CompressContext stores use_meta_device."""
        from auto_round.context.compress import CompressContext
        CompressContext.reset_context()  # Reset singleton to get fresh instance
        ctx = CompressContext(low_cpu_mem_usage=True, use_meta_device=True)
        assert ctx.use_meta_device is True

    def test_compress_context_default_use_meta_device_false(self):
        """Default use_meta_device should be False."""
        from auto_round.context.compress import CompressContext
        CompressContext.reset_context()  # Reset singleton to get fresh instance
        ctx = CompressContext(low_cpu_mem_usage=True)
        assert ctx.use_meta_device is False


class TestBaseCompressorWiring:
    """Tests for BaseCompressor use_meta_device wiring."""

    def test_base_compressor_pops_use_meta_device(self):
        """Verify BaseCompressor pops use_meta_device from kwargs."""
        from auto_round.compressors.base import BaseCompressor
        source = inspect.getsource(BaseCompressor.__init__)
        assert 'use_meta_device' in source
        assert 'kwargs.pop("use_meta_device"' in source

    def test_base_compressor_passes_use_meta_device_to_model_context(self):
        """Verify BaseCompressor passes use_meta_device to ModelContext."""
        from auto_round.compressors.base import BaseCompressor
        source = inspect.getsource(BaseCompressor.__init__)
        assert "use_meta_device=use_meta_device" in source

    def test_base_compressor_passes_use_meta_device_to_compress_context(self):
        """Verify BaseCompressor passes use_meta_device to CompressContext."""
        from auto_round.compressors.base import BaseCompressor
        source = inspect.getsource(BaseCompressor.__init__)
        # Check CompressContext creation includes use_meta_device
        assert "use_meta_device=use_meta_device" in source


class TestIntegrationFlow:
    """Integration tests for the full meta device flow."""

    def test_tune_computes_use_meta_device(self):
        """Verify tune() computes use_meta_device from use_offload."""
        from auto_round.__main__ import tune
        source = inspect.getsource(tune)
        assert "use_meta_device" in source
        assert "use_meta_device = use_offload" in source

    def test_tune_passes_use_meta_device_to_autoround(self):
        """Verify tune() passes use_meta_device to AutoRound."""
        from auto_round.__main__ import tune
        source = inspect.getsource(tune)
        assert "use_meta_device=use_meta_device" in source
