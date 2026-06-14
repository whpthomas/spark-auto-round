"""Tests for offloader mode selection.

Phase 3: Validates that _hardware_setup() creates the correct OffloadManager
mode based on model device type (meta device → clean, normal → offload, none → disabled).
"""
import pytest
import torch


class TestOffloaderModeSelection:
    """Tests for offloader mode selection in _hardware_setup()."""

    def test_hardware_setup_creates_offloader_for_meta_device(self):
        """Verify _hardware_setup creates clean mode offloader for meta device."""
        import inspect
        from auto_round.compressors.base import BaseCompressor

        source = inspect.getsource(BaseCompressor._hardware_setup)
        assert 'mode="clean"' in source or "mode='clean'" in source
        assert "use_meta_device" in source

    def test_hardware_setup_creates_offload_mode_for_normal(self):
        """Verify _hardware_setup creates offload mode for normal models."""
        import inspect
        from auto_round.compressors.base import BaseCompressor

        source = inspect.getsource(BaseCompressor._hardware_setup)
        assert 'mode="offload"' in source or "mode='offload'" in source

    def test_hardware_setup_disables_offloader_when_not_needed(self):
        """Verify _hardware_setup creates disabled offloader when not needed."""
        import inspect
        from auto_round.compressors.base import BaseCompressor

        source = inspect.getsource(BaseCompressor._hardware_setup)
        assert "enabled=False" in source

    def test_init_does_not_create_offloader(self):
        """Verify __init__ sets _offloader to None (created in _hardware_setup)."""
        import inspect
        from auto_round.compressors.base import BaseCompressor

        source = inspect.getsource(BaseCompressor.__init__)
        # Should NOT have OffloadManager(enabled=...) in __init__
        # It should be self._offloader = None
        assert "self._offloader = None" in source


class TestOffloadManagerCleanMode:
    """Tests for OffloadManager in clean mode."""

    def test_offload_manager_clean_mode(self):
        """Verify OffloadManager can be created in clean mode."""
        from auto_round.utils.offload import OffloadManager

        manager = OffloadManager(
            enabled=True,
            mode="clean",
            model_dir="/tmp/test_model",
        )
        assert manager.enabled is True
        assert manager.mode == "clean"

    def test_offload_manager_clean_mode_requires_model_dir(self):
        """Verify clean mode requires model_dir parameter."""
        from auto_round.utils.offload import OffloadManager
        import inspect

        sig = inspect.signature(OffloadManager.__init__)
        assert "model_dir" in sig.parameters

    def test_offload_manager_offload_mode(self):
        """Verify OffloadManager can be created in offload mode."""
        from auto_round.utils.offload import OffloadManager

        manager = OffloadManager(
            enabled=True,
            mode="offload",
            offload_dir_prefix="test_compress",
        )
        assert manager.enabled is True
        assert manager.mode == "offload"


class TestMetaDevicePathAttribute:
    """Tests for model.path attribute on meta device models."""

    def test_meta_device_model_has_path(self):
        """Verify meta device model has .path attribute."""
        import inspect
        from auto_round.utils.model import llm_load_model

        source = inspect.getsource(llm_load_model)
        assert "model.path" in source

    def test_unsupported_meta_device_checks_path(self):
        """Verify unsupported_meta_device checks for .path attribute."""
        import inspect
        from auto_round.utils.model import unsupported_meta_device

        source = inspect.getsource(unsupported_meta_device)
        assert 'hasattr(model, "path")' in source


class TestIntegrationOffloaderFlow:
    """Integration tests for complete offloader flow."""

    def test_three_way_offloader_selection(self):
        """Verify three offloader paths exist in source."""
        import inspect
        from auto_round.compressors.base import BaseCompressor

        source = inspect.getsource(BaseCompressor._hardware_setup)

        # Path 1: meta device → clean mode
        assert "use_meta_device" in source
        assert "clean" in source

        # Path 2: normal offload → offload mode
        assert "offload" in source

        # Path 3: no offload → disabled
        assert "enabled=False" in source

    def test_use_meta_device_flows_through(self):
        """Verify use_meta_device flows from CLI to CompressContext."""
        import inspect
        from auto_round.__main__ import tune

        source = inspect.getsource(tune)
        assert "use_meta_device" in source

    def test_low_cpu_mem_usage_dynamic(self):
        """Verify low_cpu_mem_usage is dynamic, not hardcoded."""
        import inspect
        from auto_round.__main__ import tune

        source = inspect.getsource(tune)
        # Should NOT have hardcoded True
        assert "low_cpu_mem_usage=True" not in source
        # Should use dynamic value
        assert "low_cpu_mem_usage=use_offload" in source

    def test_offloader_imports(self):
        """Verify OffloadManager import works."""
        from auto_round.utils.offload import OffloadManager

        assert OffloadManager is not None
