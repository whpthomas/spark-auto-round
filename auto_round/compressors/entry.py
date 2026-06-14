# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Callable, Optional, Union

import torch

from auto_round.algorithms.alg_config import AlgConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.transforms import normalize_rotation_config as _normalize_any_rotation_config
from auto_round.algorithms.transforms.base import BaseRotationConfig as _BaseRotationConfig
from auto_round.algorithms.transforms.rotation.config import RotationConfig as _NewArchRotationConfig
from auto_round.compressors.data_driven import DataDrivenCompressor
from auto_round.compressors.utils import check_need_act_calibration
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme, _parse_scheme


def _preview_resolved_attrs(config, scheme=None) -> dict:
    """Resolve scheme attributes without mutating config, for routing decisions.

    Called in ``AutoRound.__new__`` before the concrete compressor class is
    chosen.  ``SchemeMixin.resolve_scheme()`` will do the authoritative
    resolution later; this is just a lightweight preview so routing logic
    (``enable_imatrix``, ``needs_act_calib``, etc.) can use the correct values
    even when the user specified only ``scheme=`` without explicit bit/dtype args.

    Returns:
        dict: resolved attributes (may be empty if scheme cannot be previewed).
    """
    scheme_attr_names = QuantizationScheme.get_attributes()
    user_overrides = {k: getattr(config, k) for k in scheme_attr_names if getattr(config, k, None) is not None}
    try:
        _, _, final_attrs = _parse_scheme(scheme, user_overrides)
        return final_attrs
    except Exception:
        return {}


def _eager_validate_scheme(config, scheme=None) -> None:
    """Eagerly validate scheme/config constraints at construction time.

    Mirrors the old-arch ``_check_configs()`` call in ``BaseCompressor.__init__``.
    Raises ``ValueError`` or ``NotImplementedError`` immediately if the scheme
    contains config-only invalid combinations (e.g. tuple group_size with non-fp8
    weight dtype) so that callers get a fast failure rather than a deferred error
    buried inside ``post_init()``.
    """
    scheme_attr_names = QuantizationScheme.get_attributes()
    user_overrides = {k: getattr(config, k) for k in scheme_attr_names if getattr(config, k, None) is not None}
    try:
        _, _, final_attrs = _parse_scheme(scheme, user_overrides)
    except (ValueError, NotImplementedError):
        raise
    except Exception:
        return  # Other parse errors are deferred to post_init

    import copy

    temp_config = copy.copy(config)
    for key, value in final_attrs.items():
        setattr(temp_config, key, value)
    temp_config.check_config()  # raises ValueError / NotImplementedError if invalid


# ---------------------------------------------------------------------------
# Compressor-class registry
# ---------------------------------------------------------------------------
# Maps (model_type, base_class_name) → combined class, created lazily.
_COMPRESSOR_REGISTRY: dict[tuple[str, str], type] = {}


def _get_compressor_class(model_type: str, base_cls: type) -> type:
    """Return the compressor class for *base_cls* wired with the right model-type Mixin.

    For ``model_type == "llm"`` the bare *base_cls* is returned unchanged.
    For ``"mllm"`` and ``"diffusion"`` the corresponding Mixin is prepended via
    :func:`type` and the result is cached in ``_COMPRESSOR_REGISTRY`` so that
    each ``(model_type, base_cls)`` pair is created at most once per process.
    """
    if model_type == "llm":
        return base_cls
    key = (model_type, base_cls.__name__)
    if key in _COMPRESSOR_REGISTRY:
        return _COMPRESSOR_REGISTRY[key]
    if model_type == "mllm":
        from auto_round.compressors.mllm_mixin import MLLMMixin

        mixin = MLLMMixin
    else:
        return base_cls
    combined = type(f"{model_type.capitalize()}{base_cls.__name__}", (mixin, base_cls), {})
    _COMPRESSOR_REGISTRY[key] = combined
    return combined




class AutoRound(object):
    # Mapping from string alias to config class (and optional defaults override).
    _CONFIG_ALIASES: dict[str, type] = {
        "sign_round": SignRoundConfig,
        "signround": SignRoundConfig,
        "hadamard": _NewArchRotationConfig,
    }

    @classmethod
    def _resolve_config(cls, config: Union[str, AlgConfig, list]) -> Union[AlgConfig, list[AlgConfig]]:
        """Convert string alias(es) to the corresponding config instance(s) with default parameters."""
        if isinstance(config, str):
            key = config.strip().lower()
            # Handle spinquant/quarot via unified normalizer
            if key in ("spinquant", "quarot"):
                return _normalize_any_rotation_config(key)
            if key not in cls._CONFIG_ALIASES:
                raise ValueError(f"Unknown config alias '{config}'. " f"Supported: {list(cls._CONFIG_ALIASES.keys())}")
            return cls._CONFIG_ALIASES[key]()
        if isinstance(config, list):
            return [cls._resolve_config(c) for c in config]
        return config

    def __new__(
        cls,
        alg_configs: Union[str, AlgConfig, list[Union[str, AlgConfig]]],
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform="hf",
        format=None,
        scheme="W4A16",
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        iters: int = None,
        gradient_accumulate_steps: int = 1,
        enable_torch_compile: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        layer_config=None,
        nsamples: int = None,
        seqlen: int = None,
        **kwargs,
    ):
        from auto_round.algorithms.quantization.config import QuantizationConfig

        # Resolve string alias(es) to config instance(s) before routing.
        alg_configs = cls._resolve_config(alg_configs)

        # Extract the single QuantizationConfig from a list; validate at most one exists.
        if isinstance(alg_configs, list):
            quant_configs = [c for c in alg_configs if isinstance(c, QuantizationConfig)]
            if len(quant_configs) == 0:
                raise ValueError("At least one QuantizationConfig (SignRoundConfig / RTNConfig) is required.")
            if len(quant_configs) > 1:
                raise ValueError(
                    f"Only one QuantizationConfig is allowed, but got {len(quant_configs)}: "
                    f"{[type(c).__name__ for c in quant_configs]}"
                )
            quant_config = quant_configs[0]
        else:
            quant_config = alg_configs

        # Eagerly validate scheme constraints that do not require model info.
        # This mirrors old-arch _check_configs() called at __init__ time so that
        # callers get ValueError/NotImplementedError on construction, not deferred.
        _eager_validate_scheme(quant_config, scheme)

        # Explicitly build the dict of constructor args to forward to the
        # compressor.  This avoids the fragile locals()-based approach that
        # required a growing SKIP_ARGS blocklist.
        local_args = dict(
            model=model,
            tokenizer=tokenizer,
            platform=platform,
            format=format,
            scheme=scheme,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            iters=iters,
            gradient_accumulate_steps=gradient_accumulate_steps,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            low_cpu_mem_usage=low_cpu_mem_usage,
            layer_config=layer_config,
            nsamples=nsamples,
            seqlen=seqlen,
        )

        # Detect model type to determine if we need special compressor
        from auto_round.utils.model import detect_model_type

        model_type = detect_model_type(model)

        # If the user explicitly passes processor/image_processor, treat as MLLM even if
        # auto-detection missed it (mirrors the has_multimodal_assets check in autoround.py).
        has_multimodal_assets = kwargs.get("processor") is not None or kwargs.get("image_processor") is not None
        if has_multimodal_assets and model_type != "mllm":
            model_type = "mllm"

        # Pop kwargs that are only consumed by specific Mixins so they don't
        # leak through to BaseCompressor as unrecognized keys.
        if model_type != "mllm":
            for _k in ("processor", "image_processor", "template", "extra_data_dir", "quant_nontext_module"):
                kwargs.pop(_k, None)
        kwargs.pop("disable_opt_rtn", None)  # consumed by RTN routing above, not a compressor param
        kwargs.pop("use_meta_device", None)  # consumed by CompressContext via CompressConfig, not a compressor param

        # Only SignRoundConfig is supported
        if not isinstance(quant_config, SignRoundConfig):
            raise ValueError(
                f"Only SignRoundConfig is supported, but got {type(quant_config).__name__}. "
                f"RTN and AWQ algorithms have been removed."
            )

        return _get_compressor_class(model_type, DataDrivenCompressor)(alg_configs, **local_args, **kwargs)


class AutoRoundCompatible:
    """AutoRoundCompatible wrapper class for backward compatibility.

    This class provides the same API as the old AutoRoundCompatible class but internally
    uses the new AutoRound architecture with Mixin pattern.

    Args:
        model: Model object or model name to load
        tokenizer: Tokenizer for text processing
        platform: Platform to download model ("hf" or "model_scope")
        scheme: Quantization scheme (str, dict, or QuantizationScheme)
        layer_config: Layer-wise quantization config
        dataset: Calibration data
        iters: Optimization iterations
        seqlen: Calibration sequence length
        nsamples: Number of calibration samples
        batch_size: Calibration batch size
        gradient_accumulate_steps: Gradient accumulation steps
        low_gpu_mem_usage: Lower GPU memory mode
        device_map: Device map for each module
        enable_torch_compile: Enable torch.compile
        seed: Random seed
        low_cpu_mem_usage: Lower CPU memory mode
        **kwargs: Additional arguments (bits, group_size, sym, etc.)

    Example:
        >>> # Old API - still works
        >>> from auto_round.compressors.entry import AutoRoundCompatible
        >>> autoround = AutoRoundCompatible(
        ...     model="/models/opt-125m",
        ...     bits=4,
        ...     group_size=128,
        ...     iters=200,
        ... )
        >>> quantized_model, layer_config = autoround.quantize()
    """

    SKIP_ARGS = ("local_args", "kwargs", "cls", "config")

    bits: int | None
    group_size: int | None
    sym: bool | None
    data_type: str | None
    act_bits: int | None
    act_group_size: int | None
    act_sym: bool | None
    act_data_type: str | None
    act_dynamic: bool | None
    super_bits: int | None
    super_group_size: int | None

    @staticmethod
    def _pop_config_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract old-API config kwargs and split them by config type."""
        common_keys = (
            "ignore_layers",
            "quant_lm_head",
            "scale_dtype",
            "super_bits",
            "super_group_size",
            "to_quant_block_names",
        )
        auto_round_only_keys = (
            "nblocks",
            "enable_alg_ext",
            "lr_scheduler",
            "not_use_best_mse",
            "dynamic_max_gap",
            "optimizer",
            "enable_adam",
            "momentum",
        )
        common_kwargs = {}
        auto_round_kwargs = {}
        for key in common_keys:
            if key in kwargs:
                common_kwargs[key] = kwargs.pop(key)
        for key in auto_round_only_keys:
            if key in kwargs:
                auto_round_kwargs[key] = kwargs.pop(key)
        return common_kwargs, auto_round_kwargs

    def __new__(
        cls,
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform: str = "hf",
        scheme: Union[str, dict, QuantizationScheme] = "W4A16",
        layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
        iters: int = 200,
        seqlen: int = 2048,
        nsamples: int = 128,
        batch_size: int = 8,
        gradient_accumulate_steps: int = 1,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        algorithm: str = None,
        **kwargs,
    ):
        """Create AutoRoundCompatible instance using new AutoRound architecture.

        This method translates old AutoRoundCompatible API to new AutoRound API.

        Args:
            algorithm: Quantization algorithm to use. Options:
                - None or "auto_round": SignSGD-based optimization (default when iters > 0)
        """
        from auto_round.utils import is_mllm_model

        device = kwargs.pop("device", None)
        if device is not None:
            logger.warning_once("`device` is deprecated, please use `device_map` instead")
            if device_map in (None, 0):
                device_map = device

        common_config_kwargs, auto_round_config_kwargs = cls._pop_config_kwargs(kwargs)

        # Extract quantization parameters from kwargs or use defaults
        bits = kwargs.pop("bits", None)
        group_size = kwargs.pop("group_size", None)
        sym = kwargs.pop("sym", None)
        data_type = kwargs.pop("data_type", None)
        act_bits = kwargs.pop("act_bits", None)
        act_group_size = kwargs.pop("act_group_size", None)
        act_sym = kwargs.pop("act_sym", None)
        act_data_type = kwargs.pop("act_data_type", None)
        act_dynamic = kwargs.pop("act_dynamic", None)
        enable_opt_rtn = kwargs.pop("enable_opt_rtn", None)
        lr = kwargs.pop("lr", None)
        minmax_lr = kwargs.pop("minmax_lr", None)
        enable_minmax_tuning = kwargs.pop("enable_minmax_tuning", True)
        enable_norm_bias_tuning = kwargs.pop("enable_norm_bias_tuning", False)
        enable_quanted_input = kwargs.pop("enable_quanted_input", True)

        # Always use SignRoundConfig
        config = SignRoundConfig(
            iters=iters,
            gradient_accumulate_steps=gradient_accumulate_steps,
            bits=bits,
            group_size=group_size,
            sym=sym,
            data_type=data_type,
            act_bits=act_bits,
            act_group_size=act_group_size,
            act_sym=act_sym,
            act_data_type=act_data_type,
            act_dynamic=act_dynamic,
            lr=lr,
            minmax_lr=minmax_lr,
            enable_minmax_tuning=enable_minmax_tuning,
            enable_norm_bias_tuning=enable_norm_bias_tuning,
            enable_quanted_input=enable_quanted_input,
            **common_config_kwargs,
            **auto_round_config_kwargs,
        )

        # Determine output format if specified
        format = kwargs.pop("format", None)

        # Extract rotation_config (old-API kwarg) and thread it into alg_configs.
        # In old arch this was a standalone keyword arg; the new arch passes rotation
        # transforms as part of the alg_configs list.  All backends (auto / inplace /
        # transform) are dispatched inside ``HadamardRotation.apply_to_model``.
        # Also supports SpinQuantConfig and string shorthands ("quarot", "spinquant").
        _rotation_config_raw = kwargs.pop("rotation_config", None)
        if _rotation_config_raw is not None:
            if isinstance(_rotation_config_raw, _BaseRotationConfig):
                # Already a valid config (RotationConfig, SpinQuantConfig, etc.)
                _rc = _rotation_config_raw
            elif isinstance(_rotation_config_raw, dict):
                # Use unified normalizer which dispatches by "algorithm" key
                _rc = _normalize_any_rotation_config(_rotation_config_raw)
            elif isinstance(_rotation_config_raw, str):
                # String shorthands: "quarot", "spinquant", "hadamard",
                # "random_hadamard", "default", etc.
                _rc = _normalize_any_rotation_config(_rotation_config_raw)
            else:
                _rc = _NewArchRotationConfig()
            if _rc is not None:
                config = [config, _rc]

        # Extract MLLM-specific parameters
        processor = kwargs.pop("processor", None)
        image_processor = kwargs.pop("image_processor", None)
        template = kwargs.pop("template", None)
        extra_data_dir = kwargs.pop("extra_data_dir", None)
        quant_nontext_module = kwargs.pop("quant_nontext_module", False)

        # Check model type for logging
        if is_mllm_model(model, platform=platform):
            logger.info("Using MLLM mode for multimodal model.")
        else:
            logger.info("Using LLM mode.")

        # Create AutoRound instance using new architecture
        compressor = AutoRound(
            alg_configs=config,
            model=model,
            tokenizer=tokenizer,
            platform=platform,
            format=format,
            scheme=scheme,
            dataset=dataset,
            iters=iters,
            gradient_accumulate_steps=gradient_accumulate_steps,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            low_cpu_mem_usage=low_cpu_mem_usage,
            layer_config=layer_config,
            nsamples=nsamples,
            seqlen=seqlen,
            batch_size=batch_size,
            # MLLM parameters
            processor=processor,
            image_processor=image_processor,
            template=template,
            extra_data_dir=extra_data_dir,
            quant_nontext_module=quant_nontext_module,
            # Pass remaining kwargs
            **kwargs,
        )

        return compressor
