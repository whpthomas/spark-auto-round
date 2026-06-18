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
import copy
import gc
import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional, Union

import torch
from transformers import AutoConfig, set_seed

from auto_round.algorithms.alg_config import AlgConfig
from auto_round.algorithms.quantization import BaseQuantizers, QuantizationConfig
from auto_round.algorithms.transforms import (
    BaseRotationConfig,
    apply_rotation,
)
from auto_round.compressors.shard_writer import ShardWriter
from auto_round.compressors.utils import _get_save_folder_name, set_layer_config
from auto_round.context.compress import CompressContext
from auto_round.context.model import ModelContext
from auto_round.formats import OutputFormat, get_formats
from auto_round.logger import logger
from auto_round.schemes import (
    QuantizationScheme,
    _parse_scheme,
    preset_name_to_scheme,
)
from auto_round.special_model_handler import get_predefined_ignore_layers, update_module
from auto_round.utils import (
    AUDIO_MM_KEYS,
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    TORCH_VERSION_AT_LEAST_2_6,
    VISION_MM_KEYS,
    apply_checkpoint_conversion_mapping,
    compress_layer_names,
    convert_dtype_str2torch,
    extract_block_names_to_str,
    find_matching_blocks,
    get_block_names,
    get_reverse_checkpoint_conversion_mapping,
    is_debug_mode,
    is_quantized_input_module,
    memory_monitor,
    preserve_original_visual_block_name,
    revert_checkpoint_conversion_mapping,
)
from auto_round.utils.device import (
    _force_trim_malloc,
    get_major_device,
    set_non_auto_device_map,
)
from auto_round.utils.offload import OffloadManager


@dataclass
class SerializedCompressorConfig:
    bits: Optional[int] = None
    act_bits: Optional[int] = None
    data_type: Optional[str] = None
    act_data_type: Optional[str] = None
    group_size: Optional[int] = None
    act_group_size: Optional[int] = None
    sym: Optional[bool] = None
    act_sym: Optional[bool] = None
    act_dynamic: Optional[bool] = None
    amp: Optional[bool] = None
    batch_size: Optional[int] = None
    enable_minmax_tuning: Optional[bool] = True
    enable_norm_bias_tuning: Optional[bool] = False
    enable_quanted_input: Optional[bool] = True
    gradient_accumulate_steps: Optional[int] = None
    iters: Optional[int] = None
    lr: Optional[float] = None
    low_gpu_mem_usage: Optional[bool] = None
    minmax_lr: Optional[float] = None
    nsamples: Optional[int] = None
    quant_block_list: Optional[list[str]] = None
    regex_config: Optional[dict[str, Any]] = None
    scale_dtype: Optional[str] = None
    seqlen: Optional[int] = None
    supported_types: Optional[list[str]] = SUPPORTED_LAYER_TYPES
    static_attention_dtype: Optional[str] = None
    static_kv_dtype: Optional[str] = None
    super_bits: Optional[int] = None
    super_group_size: Optional[int] = None
    to_quant_block_names: Optional[list[str]] = None
    rotation_configs: Optional[list[dict[str, Any]]] = None


SERIALIZATION_KEYS = tuple(field.name for field in fields(SerializedCompressorConfig))


class BaseCompressor(object):
    need_calib: bool = True
    compress_context: CompressContext = None
    model_context: ModelContext = None
    shard_writer: ShardWriter = None
    supported_types = SUPPORTED_LAYER_TYPES
    inner_supported_types = INNER_SUPPORTED_LAYER_TYPES

    # ── Scheme state (populated during resolve_scheme / _scheme_post_init) ──
    orig_scheme = None
    scheme = None
    scale_dtype = None
    layer_config = None
    has_qlayer_outside_block: bool = False
    regex_config: dict = None
    quant_block_list: list = None
    to_quant_block_names = None
    ignore_layers: str = ""
    quant_lm_head: bool = False
    _scheme_resolved: bool = False
    scheme_generator = None

    @staticmethod
    def _preload_model_config(model: Union[torch.nn.Module, str], trust_remote_code: bool) -> Optional[AutoConfig]:
        if not isinstance(model, str):
            return None

        try:
            return AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
        except (OSError, EnvironmentError, ValueError) as e:
            logger.debug(
                "Failed to load config via AutoConfig.from_pretrained for %s: %s. "
                "Proceeding without config-based checks.",
                model,
                e,
            )
            return None

    def __init__(
        self,
        config: Union[AlgConfig, list[AlgConfig]],
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform="hf",
        format=None,
        scheme="W4A16",
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        layer_config=None,
        nsamples: int = None,
        seqlen: int = None,
        **kwargs,
    ):
        # ``CalibrationState`` is the single source of truth for calibration
        # runtime state.  Seed every calibration field here in one block so
        # the rest of ``__init__`` only ever interacts with the state object
        # via property forwarders.  ``_resolve_scheme`` later wires this same
        # instance onto the quantizer so the two share state.
        from auto_round.calibration.state import CalibrationState

        self._calibration_state = CalibrationState(
            nsamples=nsamples if nsamples is not None else 128,
            seqlen=seqlen if seqlen is not None else 2048,
            batch_size=kwargs.pop("batch_size", 8),
            gradient_accumulate_steps=kwargs.pop("gradient_accumulate_steps", 1),
        )

        # ``dataset`` is not a named __init__ parameter – it arrives via
        # **kwargs from the compatibility layer.  Pop it early and route
        # through the property setter so CalibrationState owns it.
        _dataset = kwargs.pop("dataset", None)
        if _dataset is not None:
            self.dataset = _dataset

        self.quantize_config = None
        self.rotation_configs: list[BaseRotationConfig] = []
        _config_list = config if isinstance(config, list) else [config]
        for _cfg in _config_list:
            if isinstance(_cfg, QuantizationConfig):
                self.quantize_config = _cfg
            elif isinstance(_cfg, BaseRotationConfig):
                self.rotation_configs.append(_cfg)
        assert self.quantize_config is not None, "QuantizationConfig is required for Compressor"

        # Compressor-level layer params (do not live in QuantizationConfig).
        # Calibration params (nsamples/seqlen/batch_size) are owned by
        # ``self._calibration_state`` (seeded above) and exposed via
        # ``@property`` forwarders.
        self.layer_config = layer_config
        # ``post_init()`` may run before ``quantize_and_save()`` in tests and
        # compatibility paths, so seed the same default used by
        # ``quantize_and_save(..., inplace=True)`` here.
        self.inplace = True

        # Scheme is passed directly to the compressor, not stored in QuantizationConfig.
        self.scheme = scheme

        # Calibrator strategy (auto_round.calibration.base.Calibrator).  Constructed
        # lazily by ``DataDrivenCompressor.post_init`` based on ``_get_calibrator_kind()``;
        # remains ``None`` for ``ZeroShotCompressor`` (RTN does not need data).
        self.calibration = None

        self.formats = format

        # Extra/legacy kwargs for backward compatibility
        # Major version releases may pack them with extra configuration options
        kwargs.pop("iters", None)
        kwargs.pop("enable_alg_ext", None)
        kwargs.pop("vlm", None)
        amp = kwargs.pop("amp", True)
        nblocks = kwargs.pop("nblocks", 1)
        disable_deterministic_algorithms = kwargs.pop("disable_deterministic_algorithms", True)
        enable_deterministic_algorithms = kwargs.pop("enable_deterministic_algorithms", False)

        # Offloader is created in _hardware_setup() with the correct mode
        # (offload for normal models, clean for meta device models).
        self._offloader = None

        # Model related
        model_dtype = kwargs.pop("model_dtype", None)
        trust_remote_code = kwargs.pop("trust_remote_code") if "trust_remote_code" in kwargs else True
        self.dry_run = kwargs.pop("dry_run", False)
        quant_nontext_module = kwargs.pop("quant_nontext_module", False)
        device = kwargs.pop("device", None)
        if device is not None:
            logger.warning("`device` is deprecated, please use `device_map` instead")

        self.static_attention_dtype = kwargs.pop("static_attention_dtype", None)
        # Attention static dtype
        if self.static_attention_dtype is not None:
            logger.warning("The static attention dtype is experimental and currently has limited support.")
        # KV cache, this one does not affect tuning but will collect some infos during tuning
        self.static_kv_dtype = kwargs.pop("static_kv_dtype", None)
        if self.static_kv_dtype is not None:
            logger.warning("The static kv is experimental and currently has limited support.")

        if kwargs:
            logger.warning_once(
                f"unrecognized keys {list(kwargs.keys())} were passed. "
                "Please check them. If you use old api, just ignore this warning."
            )
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # Deprecated, default not to use torch.use_deterministic_algorithms
        if not disable_deterministic_algorithms or enable_deterministic_algorithms:
            if not disable_deterministic_algorithms:
                logger.warning(
                    "default not use deterministic_algorithms. disable_deterministic_algorithms is deprecated,"
                    " please use enable_deterministic_algorithms instead. "
                )

            torch.use_deterministic_algorithms(True, warn_only=False)
        else:
            torch.use_deterministic_algorithms(True, warn_only=True)

        # Tuning hyperparameters
        self.seed = seed
        set_seed(self.seed)

        self.nblocks = nblocks

        self.enable_torch_compile = enable_torch_compile

        # Whether to pack the layer immediately after tuning
        # Managed via self.compress_context.is_immediate_packing / is_immediate_saving

        torch.set_printoptions(precision=3, sci_mode=True)

        # Reset both context singletons before creating fresh instances so that
        # consecutive AutoRound creations don't inherit stale config from earlier ones.
        CompressContext.reset_context()
        ModelContext.reset_context()

        # Resolve the device eagerly so ModelContext can be created before
        # CompressContext.  Creating ModelContext first places the large model
        # allocation early in the heap, matching the OLD arch allocation order
        # and reducing C-heap fragmentation (which is amplified on HPU).
        _device = get_major_device(device_map if device_map is not None else 0)
        model_config = self._preload_model_config(model, trust_remote_code)

        use_meta_device = kwargs.pop("use_meta_device", False)

        self.model_context = ModelContext(
            model,
            tokenizer=tokenizer,
            platform=platform,
            model_dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            config=model_config,
            amp=amp,
            need_calib=self.need_calib,
            device=_device,
            formats=self.formats,
            is_act_quantize=self.quantize_config.is_act_quantize,
            quant_nontext_module=quant_nontext_module,
            use_meta_device=use_meta_device,
        )
        # Alternatively, you can use CompressContext.create_context
        self.compress_context = CompressContext(
            low_cpu_mem_usage,
            low_gpu_mem_usage,
            device_map,
            enable_torch_compile,
            formats=self.formats,
            static_kv_dtype=self.static_kv_dtype,
            static_attention_dtype=self.static_attention_dtype,
            use_meta_device=use_meta_device,
        )
        self.shard_writer = None

        # scale_dtype is resolved in quantizer.resolve_scheme() after scheme resolution,
        # so it is not initialized here to avoid premature evaluation with an unresolved scheme.

        # Flag for post_init idempotency.  Set to False here so post_init() can be called
        # either via quantize_and_save() (preferred, outside inference_mode) or directly
        # from quantize() as a fallback for non-AutoScheme cases.
        self._post_init_done = False

        # Apply torch compile adjustments eagerly so that ar.enable_torch_compile
        # reflects the correct value immediately after construction (not only after post_init).
        self._precheck_torch_compile(enable_torch_compile)
        self.compress_context.enable_torch_compile = self.enable_torch_compile

        # ``self._calibration_state`` was created at the top of __init__ so
        # all calibration-related property writes above (nsamples / seqlen /
        # batch_size from kwargs) have already routed through it.

        self.has_variable_block_shape = False

    # ── Scheme resolution ─────────────────────────────────────────────────────

    def resolve_scheme(self, model_context=None, compress_context=None, dataset: str = None) -> None:
        """Phase-1 init: resolve scheme and bind config attrs (no model structure needed).

        Must be called BEFORE ``get_formats()`` and BEFORE ``_scheme_post_init()``.
        Idempotent: safe to call multiple times.
        """
        if self._scheme_resolved:
            return

        if model_context is not None:
            self.model_context = model_context
        if compress_context is not None:
            self.compress_context = compress_context
        if dataset is not None:
            self.dataset = dataset

        scheme_fields = {f.name for f in fields(QuantizationScheme)}
        user_scheme_overrides = {
            k: getattr(self.quantize_config, k)
            for k in scheme_fields
            if getattr(self.quantize_config, k, None) is not None
        }
        default_scheme, final_attrs = _parse_scheme(self.scheme, user_scheme_overrides)

        for key, value in final_attrs.items():
            setattr(self.quantize_config, key, value)
            if hasattr(self, key):
                setattr(self, key, value)
        self.quantize_config.check_config()

        self.orig_scheme = copy.deepcopy(self.scheme)
        self.scheme = default_scheme

        if self.scale_dtype is None:
            self.scale_dtype = "fp16"
        self.scale_dtype = convert_dtype_str2torch(self.scale_dtype)

        self._scheme_resolved = True

    def _scheme_post_init(self) -> None:
        """Phase-4 init: build layer config on the patched model.

        Requires ``resolve_scheme()`` to have been called first.
        Must be called AFTER ``model_context.apply_patches()``.
        """
        assert self._scheme_resolved, (
            "resolve_scheme() must be called before _scheme_post_init(). "
            "BaseCompressor.post_init() does this automatically."
        )

        if self.quant_block_list is None:
            quant_nontext_module = getattr(self.model_context, "quant_nontext_module", False)
            all_blocks = get_block_names(self.model_context.model, quant_vision=quant_nontext_module)

            if self.dry_run:
                preview = [b[:3] for b in all_blocks[:2]] if all_blocks else []
                logger.info(
                    f"[dry-run] Step 1 — get_block_names():\n"
                    f"  all_blocks[:2] (3 names each) = {preview}"
                )

            self.quant_block_list = find_matching_blocks(
                self.model_context.model, all_blocks, self.to_quant_block_names
            )
            if self.to_quant_block_names is None and self.quant_block_list:
                self.to_quant_block_names = extract_block_names_to_str(self.quant_block_list)
                self.quantize_config.to_quant_block_names = self.to_quant_block_names

        if self.dry_run:
            logger.info(
                f"[dry-run] Step 2 — _scheme_post_init():\n"
                f"  to_quant_block_names = {self.to_quant_block_names!r}\n"
                f"  quant_block_list[:2][0][:3] = {[b[:3] for b in (self.quant_block_list or [])[:2]]}"
            )

        self.configure_layer_config()

    def configure_layer_config(self) -> None:
        """Build ``self.layer_config`` from the resolved scheme on the patched model."""
        predefined_ignore_layers = get_predefined_ignore_layers(self.model_context.model)
        compressed_predefined_ignore_layers = compress_layer_names(predefined_ignore_layers)

        if predefined_ignore_layers:
            logger.info(f"Using predefined ignore_layers: {compressed_predefined_ignore_layers}")
            tmp_str = ",".join(predefined_ignore_layers).replace(" ", "")
            if self.ignore_layers == "":
                self.ignore_layers = tmp_str
            else:
                self.ignore_layers += "," + tmp_str

        self.layer_config, self.has_qlayer_outside_block, self.regex_config = set_layer_config(
            self.model_context.model,
            self.layer_config,
            self.scheme,
            self.scale_dtype,
            SUPPORTED_LAYER_TYPES,
            INNER_SUPPORTED_LAYER_TYPES,
            self.quant_block_list,
            self.ignore_layers,
            self.quant_lm_head,
            is_mllm=self.model_context.is_mllm,
        )

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def mllm(self):
        return self.model_context.is_mllm

    def _get_torch_compile_guard_state(self) -> tuple[bool, bool, int]:
        """Return raw dtype state used by torch.compile guard rules."""
        cfg = self.quantize_config
        raw_scheme = self.scheme if isinstance(self.scheme, str) else ""
        raw_dt = (cfg.data_type or "").lower()
        raw_adt = (cfg.act_data_type or "").lower()
        raw_scheme_upper = raw_scheme.upper()

        is_raw_fp8 = (
            "fp8" in raw_dt
            or "fp8" in raw_adt
            or "FP8" in raw_scheme_upper
            or ("fp" in raw_dt and getattr(cfg, "bits", 16) == 8)
            or ("fp" in raw_adt and getattr(cfg, "act_bits", 16) == 8)
        )

        act_bits = getattr(cfg, "act_bits", 16) or 16
        return is_raw_fp8, False, act_bits  # nvfp not supported

    def _maybe_log_torch_compile_default_hint(self) -> None:
        """Log the default torch.compile hint once final config state is available."""
        is_raw_fp8, _, act_bits = self._get_torch_compile_guard_state()
        if (
            not self.enable_torch_compile
            and TORCH_VERSION_AT_LEAST_2_6
            and act_bits > 8
            and not is_debug_mode()
            and not is_raw_fp8
            and self.need_calib
        ):
            logger.info(
                "%s",
                "'enable_torch_compile' is set to `False` by default. "
                "Enabling it can reduce tuning cost by 20%, but it might throw an exception.",
            )

    def _apply_torch_compile_constraints(self, enable_torch_compile: bool) -> None:
        """Apply torch.compile disabling rules for the current compressor state."""
        self.enable_torch_compile = enable_torch_compile
        cfg = self.quantize_config
        is_raw_fp8, is_raw_nv_fp, _ = self._get_torch_compile_guard_state()

        # FP8 is not used by W4A16; this guard is a no-op for this fork.
        # Kept for upstream compatibility.
        if self.enable_torch_compile and is_raw_fp8:
            self.enable_torch_compile = False
            logger.warning_once("reset enable_torch_compile to `False` as fp8 is enabled")
        # super_group_size = getattr(cfg, "super_group_size", None)
        # enable_alg_ext = getattr(cfg, "enable_alg_ext", False)
        # if self.enable_torch_compile and super_group_size is not None and enable_alg_ext:
        #     self.enable_torch_compile = False
        #     logger.warning_once(
        #         "reset enable_torch_compile to `False` as super_group_size is set for algorithm extension"
        #     )

    def _precheck_torch_compile(self, enable_torch_compile: bool) -> None:
        """Apply early torch.compile adjustments before scheme resolution.

        This runs during ``__init__`` so the compressor exposes a sensible
        ``enable_torch_compile`` value immediately after construction, even
        though scheme resolution has not completed yet.
        """
        self._apply_torch_compile_constraints(enable_torch_compile)

    def _finalize_torch_compile(self) -> None:
        """Re-evaluate torch.compile after scheme resolution with final attrs."""
        requested_enable_torch_compile = self.enable_torch_compile
        self._apply_torch_compile_constraints(requested_enable_torch_compile)
        if not requested_enable_torch_compile:
            self._maybe_log_torch_compile_default_hint()

    def _get_calibration_dataset(self) -> str:
        """Resolve calibration dataset: self.dataset > default."""
        dataset = self._calibration_state.dataset
        if dataset is not None:
            return dataset
        return "NeelNanda/pile-10k"

    def post_init(self) -> None:
        """One-time initialization that requires a loaded model.

        Delegates to ordered pipeline phases; see each `_resolve_scheme`,
        `_resolve_formats`, `_build_quantizer`, `_patch_model`,
        `_build_layer_config`, and `_hardware_setup` for the precise
        preconditions and postconditions.
        """
        if self._post_init_done:
            return

        self._resolve_scheme()

        # After scheme resolution, is_act_quantize is known.  When activation
        # quantization is enabled and the model is in float16, convert to
        # bfloat16 to match the old arch.  This also detaches any parameter
        # tensors that are still backed by safetensors' mmap, preventing
        # per-block RSS growth (~14 MB/block) when .to(device) page-faults
        # the underlying file pages into physical memory.
        if self.quantize_config.is_act_quantize and self.model_context.amp_dtype == torch.float16:
            logger.warning("force to use bf16 for quantization tuning when enabling activation quantization")
            self.model_context.amp_dtype = torch.bfloat16
            if self.model_context.model.dtype != torch.bfloat16:
                self.model_context.model = self.model_context.model.to(torch.bfloat16)

        self._resolve_formats()
        self._build_quantizer()
        self._patch_model()
        self._build_layer_config()
        self._apply_rotations()

        # Reclaim temporaries from Phases 1-4 (scheme resolution, format
        # parsing, model patching, layer-config walk) before Phase 5
        # allocates hardware/compile objects.  This compacts the heap so that
        # the fragmentation gap between live and freed blocks is minimised.
        gc.collect()
        _force_trim_malloc()

        self._hardware_setup()

        # Enable memory-efficient quantization for large MLLM models
        if self.model_context.is_mllm:
            if torch.cuda.device_count() > 1:
                # Use gradient checkpointing to reduce memory
                if hasattr(self.model_context.model, 'gradient_checkpointing_enable'):
                    try:
                        self.model_context.model.gradient_checkpointing_enable()
                        logger.info("Enabled gradient checkpointing for multi-GPU MLLM quantization")
                    except Exception as e:
                        logger.debug(f"Could not enable gradient checkpointing: {e}")

            # Clear cache between operations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final trim after all init phases.
        gc.collect()
        _force_trim_malloc()

        self._post_init_done = True

    # ── Pipeline phase methods ────────────────────────────────────────────────

    def _resolve_scheme(self) -> None:
        """Phase 1 – Scheme resolution.

        Preconditions:
          - ``self.quantize_config`` is a valid :class:`QuantizationConfig`.

        Work performed:
          - Seeds scheme-related attrs (``scale_dtype``, ``ignore_layers``,
            ``quant_lm_head``, ``to_quant_block_names``) from ``quantize_config``.
          - Calls :meth:`resolve_scheme` to derive ``data_type``, ``bits``,
            ``sym``, ``scale_dtype`` etc. and write them back to both ``self``
            and ``self.quantize_config``.

        Postconditions:
          - ``self.scheme`` and ``self.quantize_config`` carry resolved scheme attrs.
        """
        cfg = self.quantize_config
        self.scale_dtype = cfg.scale_dtype
        # self.layer_config is already set from __init__ (direct compressor param).
        self.ignore_layers = cfg.ignore_layers
        self.quant_lm_head = cfg.quant_lm_head
        self.to_quant_block_names = cfg.to_quant_block_names
        if self.to_quant_block_names is None:
            self.to_quant_block_names = getattr(self.model_context.model, "_autoround_to_quant_block_names", None)
            if self.to_quant_block_names is not None:
                self.quantize_config.to_quant_block_names = self.to_quant_block_names

        # Resolve the scheme (pure config work: sets data_type / bits / sym /
        # scale_dtype etc. on both self and self.quantize_config).
        self.resolve_scheme(
            model_context=self.model_context,
            compress_context=self.compress_context,
            dataset=self._get_calibration_dataset(),
        )

    def _build_quantizer(self) -> None:
        """Phase 1b – Quantizer construction and wiring.

        Preconditions:
                    - :meth:`_resolve_scheme` complete: ``self.quantize_config`` carries
                        resolved scheme attrs.
                    - :meth:`_resolve_formats` complete: format-driven overrides have
                        been synced back to ``self.quantize_config``.

        Work performed:
          - Constructs ``self.quantizer`` from the resolved config.
          - Calls ``quantizer.bind(self)`` so the quantizer pulls
            ``model_context`` / ``compress_context`` / ``scale_dtype`` /
            ``CalibrationState`` from this compressor.  ``quantizer.model``
            is a property that reads ``model_context.model``.

        Postconditions:
          - ``self.quantizer`` is ready and shares ``CalibrationState`` with
            the compressor.
        """
        self.quantizer = BaseQuantizers.from_config(self.quantize_config)
        self.quantizer.bind(self)

    def _resolve_formats(self) -> None:
        """Phase 2 – Format resolution and config attr sync.

        Preconditions:
                    - Phase 1 complete: the scheme is resolved (``data_type``, ``bits``,
                        ``sym`` etc. are set on both ``self`` and ``self.quantize_config``).

        Work performed:
          - Converts a string ``self.formats`` to a list of
            :class:`~auto_round.formats.OutputFormat` objects via
            :func:`~auto_round.formats.get_formats`.
          - Initialises :class:`~auto_round.compressors.shard_writer.ShardWriter`
            when formats are present.
                    - **(2b)** Detects format-driven attribute mutations (``bits``, ``sym``,
            ``data_type``, ``group_size``, etc.) that ``_check_format_compat`` may
                        have written onto ``self`` inside ``get_formats``, syncs them back
                        to ``self.quantize_config``, and rebuilds ``self.scheme`` accordingly.
                    - Merges any format-injected entries into ``self.layer_config``.

        Postconditions:
          - ``self.formats`` is a list (or ``None``).
          - ``self.compress_context.formats`` mirrors ``self.formats``.
                    - ``self.quantize_config`` and ``self.scheme`` reflect the final attrs.
        """
        # get_formats() inspects data_type / bits etc. that were just resolved.
        if isinstance(self.formats, str):
            self.formats = get_formats(self.formats, self)
        if self.formats is not None:
            self.compress_context.formats = self.formats
            ShardWriter.reset()
            # Defer ShardWriter construction to _ensure_shard_writer() to avoid
            # heap fragmentation during post_init (parameter iteration).

        # Snapshot the user-specified layer_config before format processing may
        # inject extra per-layer entries.
        _pre_format_layer_config = copy.copy(self.layer_config) or {}

        # ── 2b: propagate format-adjusted attrs back to quantize_config ─────
        # Format resolution may have overridden bits / sym / data_type etc.
        # on this BaseCompressor object via setattr(self, ...).  Sync those
        # changes to self.quantize_config before creating the quantizer.
        _forwarded_attrs = (
            "bits",
            "sym",
            "data_type",
            "super_bits",
            "super_group_size",
            "group_size",
            "act_bits",
            "scale_dtype",
        )
        _any_attr_changed = False
        for _attr in _forwarded_attrs:
            if _attr not in self.__dict__:
                continue
            config_val = getattr(self.quantize_config, _attr, None)
            self_val = self.__dict__[_attr]
            if _attr not in ("scale_dtype", "act_bits") and config_val != self_val:
                _any_attr_changed = True
            if config_val != self_val:
                setattr(self.quantize_config, _attr, self_val)
        # If format resolution changed scheme attrs, rebuild self.scheme.
        if _any_attr_changed:
            from auto_round.schemes import QuantizationScheme as _QS

            _new_scheme_dict = {f.name: getattr(self, f.name, None) for f in fields(_QS)}
            _new_scheme = _QS.from_dict({k: v for k, v in _new_scheme_dict.items() if v is not None})
            self.scheme = _new_scheme

        _format_layer_cfg = {
            k: v for k, v in (self.__dict__.get("layer_config") or {}).items() if k not in (_pre_format_layer_config)
        }
        if _format_layer_cfg:
            if self.layer_config is None:
                self.layer_config = {}
            for _lname, _lval in _format_layer_cfg.items():
                self.layer_config.setdefault(_lname, _lval)

    def _apply_rotations(self) -> None:
        """Phase 4.5 – Apply Hadamard / rotation transforms to the model.

        Preconditions:
          - Phase 3 complete: model topology is final (``apply_patches`` has
            replaced / merged layers, e.g. MoE experts), so rotation operates
            on the same modules that quantization will later see.
          - Phase 4 complete: ``self.layer_config`` is built; rotation only
            transforms weights and does not change layer names, so this
            ordering matches the old arch where rotation ran after
            ``configure_layer_config``.
          - ``self.quantize_config.data_type`` is final (rotation backend
            dispatch depends on it).

        Work performed:
          - Iterates ``self.rotation_configs`` and calls
            :func:`~auto_round.algorithms.transforms.apply_rotation` on the
            model for each config.

        Postconditions:
          - ``self.model_context.model`` carries the rotated weights and any
            inserted online-Hadamard hooks.
        """
        if not self.rotation_configs:
            return
        logger.info("Applying Hadamard transform to the model.")
        for rotation_cfg in self.rotation_configs:
            self.model_context.model = apply_rotation(
                self.model_context.model,
                rotation_cfg,
                data_type=self.quantize_config.data_type,
            )

    def _patch_model(self) -> None:
        """Phase 3 – Model structure patching.

        Preconditions:
          - Phase 2 complete: ``self.formats`` is resolved so that
            ``apply_patches`` can inspect format-specific requirements.

        Work performed:
          - Delegates to :meth:`~auto_round.context.model.ModelContext.apply_patches`
            which may replace or merge layers (e.g. MoE expert merging, adding
            static-kv wrappers) to produce the final model topology.

        Postconditions:
          - ``self.model_context.model`` reflects the definitive topology that
            :meth:`_build_layer_config` will walk.
        """
        # apply_patches() may replace layers (e.g. MoE expert merging); must
        # happen before configure_layer_config() so it sees the final topology.
        self.model_context.apply_patches(self.formats)

    def _build_layer_config(self) -> None:
        """Phase 4 – Layer-config construction and quantizer sync.

        Preconditions:
          - Phase 3 complete: model topology is final.
          - ``self.scheme`` and all scheme-resolved attrs are consistent with
            the scheme-resolved values set in earlier phases.

        Work performed:
          - Calls :meth:`_scheme_post_init` which walks the patched model to
            build ``self.layer_config``, ``self.quant_block_list``, etc.
            On the AutoScheme path this also runs delta-loss forward/backward
            passes to select per-layer schemes.
          - Syncs the fully-resolved ``layer_config`` and related attrs to
            ``self.quantizer`` so quantization methods have the complete view.

        Postconditions:
          - ``self.layer_config`` is fully populated.
          - ``self.quantizer`` mirrors ``layer_config``, ``has_qlayer_outside_block``,
            ``regex_config``, ``quant_block_list``, ``to_quant_block_names``,
            ``scale_dtype``, and ``ignore_layers``.
        """
        # configure_layer_config() walks the patched model; _gen_auto_scheme()
        # (AutoScheme path) runs delta-loss forward+backward passes.
        self._scheme_post_init()

        # Sync the fully-resolved scheme state to the quantizer so that
        # quantization methods (quantize_block, quantize_layer, etc.) have
        # access to layer_config, scale_dtype, quant_block_list, etc.
        self.quantizer.layer_config = self.layer_config
        self.quantizer.has_qlayer_outside_block = self.has_qlayer_outside_block
        self.quantizer.regex_config = self.regex_config
        self.quantizer.quant_block_list = self.quant_block_list
        self.quantizer.to_quant_block_names = self.to_quant_block_names
        self.quantizer.scale_dtype = self.scale_dtype
        self.quantizer.ignore_layers = self.ignore_layers

    def _hardware_setup(self) -> None:
        """Phase 5 – Hardware and compile configuration.

        Preconditions:
          - Phase 4 complete: ``layer_config`` is built and
            ``has_qlayer_outside_block`` is known.
          - ``self.quantize_config.data_type`` is the final resolved value
            (needed by :meth:`_finalize_torch_compile`).

        Work performed:
          - Applies the device map via ``set_non_auto_device_map``.
          - Re-evaluates ``torch.compile`` eligibility.
          - Configures the offloader:
            * Meta device models: "clean" mode (reload from original safetensors)
            * Normal models with offload: "offload" mode (save to temp dir)
            * No offload needed: disabled
          - Disables ``self.inplace`` when quantized layers live outside blocks.
          - Calls ``_adjust_immediate_packing_and_saving``.

        Postconditions:
          - ``compress_context.enable_torch_compile`` is final.
          - ``self._offloader`` is configured for the correct mode.
          - ``self.inplace`` and ``compress_context.is_immediate_packing`` /
            ``compress_context.is_immediate_saving`` are set.
        """
        set_non_auto_device_map(self.model_context.model, self.compress_context.device_map)
        # Re-evaluate torch.compile eligibility now that data_type is resolved.
        self._finalize_torch_compile()
        self.compress_context.enable_torch_compile = self.enable_torch_compile

        # ── Offloader mode selection ──────────────────────────────────────────
        if self.compress_context.low_cpu_mem_usage:
            if self.compress_context.use_meta_device:
                # Meta device: reload blocks from original safetensors files.
                # The model was loaded on meta device (zero memory), so there's
                # nothing to save to temp directory. Blocks are loaded on demand
                # from the original checkpoint.
                model_path = getattr(self.model_context.model, "path", None)
                if model_path is None:
                    model_path = self.model_context.model
                self._offloader = OffloadManager(
                    enabled=True,
                    mode="clean",
                    model_dir=model_path,
                    offload_dir_prefix="compressor",
                )
                logger.info(
                    "OffloadManager: clean mode (reload from original checkpoint: %s)",
                    model_path,
                )
            else:
                # Normal model: offload to temp directory, reload from temp.
                self._offloader = OffloadManager(
                    enabled=True,
                    mode="offload",
                    offload_dir_prefix="compressor",
                )
                self._offloader.reset()
                logger.info("OffloadManager: offload mode (save to temp directory)")
        else:
            # No offloading needed — model fits in memory.
            self._offloader = OffloadManager(enabled=False)
        # ── End offloader mode selection ───────────────────────────────────────

        # Disable inplace when quantized layers live outside transformer blocks.
        if self.has_qlayer_outside_block and self.need_calib:
            self.inplace = False

        if not hasattr(self, "formats"):
            logger.warning("this API is deprecated, please use `quantize_and_save` instead")
        else:
            self._adjust_immediate_packing_and_saving()

    # backward compatible with the legacy API
    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]

        for obj in ["quantizer", "quantize_config", "model_context", "compress_context"]:
            if obj not in self.__dict__:
                continue
            obj = object.__getattribute__(self, obj)
            try:
                return object.__getattribute__(obj, name)
            except AttributeError:
                continue

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ── Forwarding properties to ``self._calibration_state`` ──────────────────
    @property
    def calibration_state(self):
        return self._calibration_state

    @calibration_state.setter
    def calibration_state(self, value) -> None:
        self._calibration_state = value
        # Re-wire quantizer if it already exists so they keep sharing.
        q = self.__dict__.get("quantizer")
        if q is not None:
            q.calibration_state = value

    @property
    def inputs(self) -> dict:
        return self._calibration_state.inputs

    @inputs.setter
    def inputs(self, value: dict) -> None:
        self._calibration_state.inputs = value if value is not None else {}

    @property
    def to_cached_layers(self) -> list:
        return self._calibration_state.to_cached_layers

    @to_cached_layers.setter
    def to_cached_layers(self, value: list) -> None:
        self._calibration_state.to_cached_layers = value if value is not None else []

    @to_cached_layers.deleter
    def to_cached_layers(self) -> None:
        self._calibration_state.to_cached_layers = []

    @property
    def last_cache_name(self):
        return self._calibration_state.last_cache_name

    @last_cache_name.setter
    def last_cache_name(self, value) -> None:
        self._calibration_state.last_cache_name = value

    @last_cache_name.deleter
    def last_cache_name(self) -> None:
        self._calibration_state.last_cache_name = None

    @property
    def blocks_requiring_input_ids(self) -> list:
        return self._calibration_state.blocks_requiring_input_ids

    @blocks_requiring_input_ids.setter
    def blocks_requiring_input_ids(self, value: list) -> None:
        self._calibration_state.blocks_requiring_input_ids = value if value is not None else []

    @property
    def batch_size(self) -> int:
        return self._calibration_state.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self._calibration_state.batch_size = value

    @property
    def gradient_accumulate_steps(self) -> int:
        return self._calibration_state.gradient_accumulate_steps

    @gradient_accumulate_steps.setter
    def gradient_accumulate_steps(self, value: int) -> None:
        if value is not None:
            self._calibration_state.gradient_accumulate_steps = value

    @property
    def nsamples(self) -> int:
        return self._calibration_state.nsamples

    @nsamples.setter
    def nsamples(self, value: int) -> None:
        if value is not None:
            self._calibration_state.nsamples = value

    @property
    def seqlen(self) -> int:
        return self._calibration_state.seqlen

    @seqlen.setter
    def seqlen(self, value: int) -> None:
        if value is not None:
            self._calibration_state.seqlen = value

    @property
    def dataset(self):
        return self._calibration_state.dataset

    @dataset.setter
    def dataset(self, value) -> None:
        self._calibration_state.dataset = value

    @property
    def dataloader(self):
        return self._calibration_state.dataloader

    @dataloader.setter
    def dataloader(self, value) -> None:
        self._calibration_state.dataloader = value

    @dataloader.deleter
    def dataloader(self) -> None:
        self._calibration_state.dataloader = None

    @property
    def optimizer(self):
        """Return the actual optimizer class, converting string to class for backward compat.

        Old API stored ``self.optimizer = torch.optim.AdamW`` (the class itself).
        New arch stores the optimizer name as a string in ``quantize_config.optimizer``.
        This property converts it so that ``ar.optimizer == torch.optim.AdamW`` works.
        """
        if self.quantize_config is None:
            return None
        opt = getattr(self.quantize_config, "optimizer", None)
        if opt is None:
            # Default to AdamW when enable_adam=True and no explicit optimizer was set
            if getattr(self.quantize_config, "enable_adam", False):
                return torch.optim.AdamW
            return None
        if isinstance(opt, str):
            return getattr(torch.optim, opt, None)
        return opt

    def _adjust_immediate_packing_and_saving(self):
        if self.formats is None:
            return

        formats = getattr(self, "formats", [])
        if len(formats) == 1 and not formats[0].is_fake() and self.inplace:
            self.compress_context.is_immediate_packing = True

        if self.has_qlayer_outside_block and self.need_calib:
            self.compress_context.is_immediate_packing = False
        if not ("causallm" in self.model_context.model.__class__.__name__.lower() and not self.model_context.is_mllm):
            # TODO For tied keys, there may some issues, we have not verified this
            tied_weight_keys = getattr(self.model_context.model, "_tied_weight_keys", {})
            if len(tied_weight_keys) > 1:
                self.compress_context.is_immediate_saving = False
                if self.compress_context.low_cpu_mem_usage:
                    logger.warning("reset low_cpu_mem_usage to False due to tied weights")
                return
            if len(tied_weight_keys) == 1:
                key = list(tied_weight_keys.keys())[0]
                if "lm_head" not in key:
                    self.compress_context.is_immediate_saving = False
                    if self.compress_context.low_cpu_mem_usage:
                        logger.warning("reset low_cpu_mem_usage to False due to tied weights")
                    return

        if self.compress_context.low_cpu_mem_usage and self.compress_context.is_immediate_packing:
            self.compress_context.is_immediate_saving = True

        if self.compress_context.low_cpu_mem_usage and self.compress_context.is_immediate_packing:
            if self.has_qlayer_outside_block:
                logger.warning(
                    "`low_cpu_mem_usage` is not fully supported "
                    "when there are quantized layers outside blocks. "
                    "Setting low_cpu_mem_usage to False."
                )
                self.compress_context.low_cpu_mem_usage = False
                self.compress_context.is_immediate_saving = False

        if self.compress_context.is_immediate_saving and not (
            "int" in self.quantize_config.data_type
        ):
            logger.warning("immediate_saving is only supported for int quantization, set to False")
            self.compress_context.is_immediate_saving = False

        if self.output_dir is None:
            self.compress_context.is_immediate_saving = False

        # Create ShardWriter eagerly only when immediate saving is active
        # (it interleaves with the quantize loop).  Otherwise keep it deferred
        # until save_quantized() to avoid heap fragmentation during init.
        if self.compress_context.is_immediate_saving:
            self._ensure_shard_writer()

    def _ensure_shard_writer(self):
        """Lazily create ShardWriter if it hasn't been created yet."""
        if self.shard_writer is None and self.formats is not None:
            self.shard_writer = ShardWriter(self.model_context.model, bits=8)

    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """
        raise NotImplementedError("quantize method must be implemented in subclass")

    def save_quantized(
        self,
        output_dir: str = None,
        format: Union[str, list[OutputFormat]] = None,
        inplace: bool = True,
        return_folders=False,
        **kwargs,
    ) -> torch.nn.Module:
        """Save the quantized model to the specified output directory in the specified format.

        Args:
            output_dir (str, optional): The directory to save the quantized model. Defaults to None.
            format (str, optional): The format in which to save the model. Defaults to "auto_round".
            inplace (bool, optional): Whether to modify the model in place. Defaults to True.
            **kwargs: Additional keyword arguments specific to the export format.

        Returns:
            object: The compressed model object.
        """
        self.output_dir = output_dir
        if output_dir is not None:
            self.compress_context.output_dir = output_dir
        if format is not None:
            if isinstance(format, str) and getattr(self, "formats", None) is None:
                self.formats = get_formats(format, self)
                self.compress_context.formats = self.formats

        if not self.model_context.quantized:
            logger.warning("please run autoround.quantize first")
            return
        folders = []
        if self.formats is None:
            logger.info("format is not set, using default auto_round format.")
            self.formats = "auto_round"
        if isinstance(self.formats, str):
            self.formats = get_formats(self.formats, self)
            self.compress_context.formats = self.formats
        for format in self.formats:
            save_folder = _get_save_folder_name(format)
            if self.act_bits <= 8 and format.is_fake():
                logger.warning(
                    "Support for exporting activation quantization is limited. "
                    "Please ensure that your configuration is supported."
                )

            serialization_dict = self._build_quantization_config(
                backend=kwargs.get("backend", "auto_round:auto_gptq")
            )

            compressed_model = format.save_quantized(
                save_folder,
                model=self.model_context.model,
                layer_config=self.quantizer.layer_config,
                inplace=inplace,
                tokenizer=self.model_context.tokenizer,
                device=self.compress_context.device,
                serialization_dict=serialization_dict,
                **kwargs,
            )
            folders.append(save_folder)

        if return_folders:
            if len(folders) == 1:
                folders = folders[0]
            return compressed_model, folders
        else:
            return compressed_model

    def _get_export_dir(self, output_dir: str, format_str: str) -> str:
        """Derive a descriptive export directory from model name and quantization config.

        Must be called after ``post_init()`` so that scheme-resolved attrs
        (bits, group_size, data_type, etc.) are available on ``self.quantize_config``.

        Mirrors the logic previously in ``__main__.py`` so callers only need to
        pass the base ``output_dir`` and the format string.
        """
        model_name = (getattr(self.model_context.model, "name_or_path", "") or "").rstrip("/")
        cfg = self.quantize_config
        group_size = cfg.group_size
        bits = cfg.bits
        data_type = cfg.data_type or "int"
        act_bits = cfg.act_bits or 16
        act_data_type = cfg.act_data_type or "float"

        last = model_name.split("/")[-1].strip(".")

        if last == "":
            # model path is just '.' or './' – put inside output_dir with suffix
            if group_size <= 0:
                suffix = f"afp{act_bits}" if "fp" in act_data_type else f"a{act_bits}"
            else:
                suffix = f"g{group_size}"
            return os.path.join(output_dir, f"w{bits}{suffix}")

        # Use spark-auto-round naming convention: {model}-int4-AutoRound
        return os.path.join(
            output_dir,
            model_name.split("/")[-1] + "-int4-AutoRound",
        )

    def _build_quantization_config(self, backend: str = "auto_round:auto_gptq") -> dict:
        """Build the complete quantization_config dict from serialization state.

        This is the single source of truth for config construction, used by both
        ``save_quantized()`` and ``_save_config_dry_run()``.

        Args:
            backend: The packing backend to use. Defaults to "auto_round:auto_gptq".

        Returns:
            The complete quantization_config dict ready for serialization.
        """
        from auto_round.export.utils import filter_quantization_config
        from auto_round.utils import (
            check_start_with_block_name,
            to_standard_regex,
        )
        from auto_round.export.export_to_autoround.utils import check_neq_config

        # Step 1: Build serialization dict from SerializedCompressorConfig
        serialization_dict = asdict(SerializedCompressorConfig())
        for key in serialization_dict:
            serialization_dict[key] = getattr(self, key, serialization_dict[key])
        from auto_round.version import __version__
        serialization_dict["autoround_version"] = __version__

        # Fallback: extract block names if to_quant_block_names is None
        if serialization_dict.get("to_quant_block_names") is None and self.quantizer.quant_block_list:
            serialization_dict["to_quant_block_names"] = extract_block_names_to_str(
                self.quantizer.quant_block_list
            )

        # Convert scale_dtype to string
        if "scale_dtype" in serialization_dict:
            serialization_dict["scale_dtype"] = str(serialization_dict["scale_dtype"])

        # Step 2: Revert block names to checkpoint format
        original_to_quant_block_names = serialization_dict.get("to_quant_block_names")
        if isinstance(original_to_quant_block_names, list):
            original_to_quant_block_names = original_to_quant_block_names[:]

        reverse_mapping = get_reverse_checkpoint_conversion_mapping(self.model)

        if isinstance(serialization_dict["to_quant_block_names"], str):
            reverted = revert_checkpoint_conversion_mapping(
                serialization_dict["to_quant_block_names"], reverse_mapping
            )
            serialization_dict["to_quant_block_names"] = preserve_original_visual_block_name(
                original_to_quant_block_names, reverted
            )
        elif isinstance(serialization_dict["to_quant_block_names"], list):
            for idx in range(len(serialization_dict["to_quant_block_names"])):
                reverted = revert_checkpoint_conversion_mapping(
                    serialization_dict["to_quant_block_names"][idx], reverse_mapping
                )
                orig = None
                if isinstance(original_to_quant_block_names, list) and idx < len(original_to_quant_block_names):
                    orig = original_to_quant_block_names[idx]
                serialization_dict["to_quant_block_names"][idx] = preserve_original_visual_block_name(
                    orig, reverted
                )

        # Step 3: Build quantization_config
        quantization_config = serialization_dict

        # Backend fix for sym
        if (
            (quantization_config.get("sym") is None or quantization_config.get("sym"))
            and ("gptq" not in backend and "awq" not in backend)
        ):
            backend = backend.replace("auto_round", "auto_round:auto_gptq")

        quantization_config["block_name_to_quantize"] = quantization_config.pop("to_quant_block_names", None)
        quantization_config["quant_method"] = "auto-round"
        quantization_config["packing_format"] = backend

        # Step 4: Build extra_config from layer_config
        extra_config = {}
        block_name_to_quantize = quantization_config["block_name_to_quantize"]
        if isinstance(block_name_to_quantize, str):
            block_name_to_quantize = [name.strip() for name in block_name_to_quantize.split(",")]
        elif isinstance(block_name_to_quantize, list):
            block_name_to_quantize = [
                os.path.commonprefix(item).rstrip(".") if isinstance(item, list) else item
                for item in block_name_to_quantize
            ]

        scheme_keys = [f.name for f in fields(QuantizationScheme)]
        # Apply checkpoint conversion mapping to layer names for extra_config
        forward_mapping = get_reverse_checkpoint_conversion_mapping(self.model)
        for layer_name, cfg in self.quantizer.layer_config.items():
            # Convert layer name from PyTorch format to checkpoint format
            ckpt_layer_name = apply_checkpoint_conversion_mapping(layer_name, forward_mapping)
            if not cfg["in_blocks"] and cfg["bits"] <= 8:  # lm head
                extra_config[ckpt_layer_name] = {key: cfg.get(key) for key in scheme_keys}
            elif cfg["in_blocks"] or (
                block_name_to_quantize is not None
                and check_start_with_block_name(layer_name, block_name_to_quantize)
            ):
                neq_keys = check_neq_config(cfg, **{k: quantization_config.get(k) for k in scheme_keys})
                if len(neq_keys) > 0:
                    extra_config[ckpt_layer_name] = {}
                    for key in neq_keys:
                        if cfg.get(key) is not None:
                            extra_config[ckpt_layer_name][key] = cfg[key]

        # Handle regex_config
        regex_config = quantization_config.pop("regex_config", None)
        if regex_config is not None:
            for name, cfg in regex_config.items():
                regex_name = to_standard_regex(name)
                neq_keys = check_neq_config(cfg, **{k: quantization_config.get(k) for k in scheme_keys})
                if len(neq_keys) > 0:
                    extra_config[regex_name] = {}
                    for key in neq_keys:
                        if cfg.get(key) is not None:
                            extra_config[regex_name][key] = cfg[key]

        if len(extra_config) > 0:
            quantization_config["extra_config"] = extra_config

        # Step 5: Clean up config (remove defaults)
        filter_quantization_config(quantization_config)

        return quantization_config

    def quantize_and_save(
        self, output_dir: str = "tmp_autoround", format: str = None, inplace: bool = True, **kwargs
    ) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantizes the model and saves it in the specified format(s).

        This function checks the validity of the requested format(s), quantizes
        the model accordingly, and saves it to the specified output directory.
        If multiple formats are provided, the model is saved separately for each format.

        Args:
            output_dir (str, optional): The directory where the quantized model
                will be saved. Defaults to "tmp_autoround".
            format (str, optional): The quantization format(s) to use, separated
                by commas if multiple. Defaults to "auto_round".
            inplace (bool, optional): Whether to modify the model in place if only
                one format is used. Defaults to True.
            **kwargs: Additional arguments for the quantization and saving process.

        Returns:
            model: A qdq model or packed model based on the configurations
            folders: The folder paths where the quantized models are saved.

        Raises:
            ValueError: If an unsupported format is specified.
        """
        # Validate and process the specified formats
        self.output_dir = output_dir
        self.compress_context.output_dir = output_dir

        # check and update the format based on the current configuration
        if format and self.formats is None:
            self.formats = format
        if self.formats is None:
            logger.info("format is not set, using default auto_round format.")
            self.formats = "auto_round"

        # If multiple formats are specified, enforce inplace=False
        if len(self.formats.split(",")) > 1:
            inplace = False
        self.inplace = kwargs.get("inplace", inplace)
        kwargs.pop("inplace", None)

        # Perform model quantization
        # IMPORTANT: post_init() must run outside any @torch.inference_mode() context
        # because AutoScheme's delta-loss selection requires gradient tracking.
        self.post_init()
        # If post_init() was called manually before quantize_and_save() (e.g. ar.post_init()
        # in tests), _resolve_formats saw formats=None and was a no-op.  Now that we have set
        # self.formats to a default string above, resolve it into OutputFormat objects so that
        # quantize() and save_quantized() receive proper objects, not a raw string.
        if isinstance(self.formats, str):
            self.formats = get_formats(self.formats, self)
            self.compress_context.formats = self.formats
        # Derive descriptive export dir after post_init so scheme-resolved attrs are available.
        _fmt_str = format or (self.formats if isinstance(self.formats, str) else "")
        output_dir = self._get_export_dir(output_dir, _fmt_str)
        self.output_dir = output_dir
        self.compress_context.output_dir = output_dir

        # ── DRY-RUN: write config files only, skip quantization and weight saving ──
        if self.dry_run:
            logger.info("Dry-run mode: skipping quantization, writing config files only.")

            # Build quantization config using the SAME code path as save_quantized()
            quantization_config = self._build_quantization_config(
                backend=kwargs.get("backend", "auto_round:auto_gptq")
            )

            # Discover MTP layers from source checkpoint
            self._discover_mtp_layers(quantization_config)

            # Set config on model (same as save_quantized_as_autoround does)
            model = self.model_context.model
            if hasattr(model, "config"):
                model.config.quantization_config = quantization_config

            # Write config.json (model.config.to_dict() includes quantization_config)
            os.makedirs(output_dir, exist_ok=True)
            config_path = os.path.join(output_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(model.config.to_dict(), f, indent=2, default=str)
            logger.info(f"[dry-run] Wrote {config_path}")

            # Write quantization_config.json (standalone copy)
            qconfig_path = os.path.join(output_dir, "quantization_config.json")
            with open(qconfig_path, "w", encoding="utf-8") as f:
                json.dump(quantization_config, f, indent=2, default=str)
            logger.info(f"[dry-run] Wrote {qconfig_path}")

            # Save tokenizer files
            tokenizer = self.model_context.tokenizer
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
                logger.info(f"[dry-run] Saved tokenizer to {output_dir}")

            logger.info("[dry-run] DRY-RUN COMPLETE: config files written, no weights saved")
            return self.model, output_dir

        if self.static_attention_dtype is not None:
            logger.warning("static_attention_dtype is not supported in this build; ignoring.")
        if self.static_kv_dtype is not None:
            logger.warning("static_kv_dtype is not supported in this build; ignoring.")
        self.quantize()
        self.model_context.quantized = True

        # Ensure ShardWriter is ready before saving (deferred from post_init).
        self._ensure_shard_writer()

        # Save the quantized model in the specified format_list
        model, folders = self.save_quantized(output_dir, inplace=inplace, return_folders=True, **kwargs)
        memory_monitor.log_summary()

        return model, folders

    # ── Dry-run helpers ───────────────────────────────────────────────────────

    def _discover_mtp_layers(self, quantization_config: dict) -> None:
        """Discover MTP layers from source checkpoint and update config.

        Mirrors the logic in missing_tensors._woq_quantize_missing_tensors that
        discovers new block prefixes (like mtp.layers) and adds ignored layers
        (like mtp.fc) to extra_config.
        """
        import json
        import os

        model = self.model_context.model

        # Get source model directory
        source_dir = None
        for attr in ["name_or_path"]:
            val = getattr(model, attr, None)
            if isinstance(val, str) and val:
                source_dir = val
                break
        if source_dir is None:
            config = getattr(model, "config", None)
            if config is not None:
                for attr in ["_name_or_path", "name_or_path"]:
                    val = getattr(config, attr, None)
                    if isinstance(val, str) and val:
                        source_dir = val
                        break

        if not source_dir or not os.path.isdir(source_dir):
            # Try to resolve from HuggingFace cache
            try:
                from huggingface_hub import try_to_load_from_cache

                cached = try_to_load_from_cache(source_dir, "model.safetensors.index.json")
                if isinstance(cached, str) and os.path.exists(cached):
                    source_dir = os.path.dirname(cached)
            except Exception:
                pass

        if not source_dir or not os.path.isdir(source_dir):
            logger.debug("[dry-run] Cannot resolve source directory for MTP discovery")
            return

        # Read source checkpoint to discover weight prefixes
        index_file = os.path.join(source_dir, "model.safetensors.index.json")
        if not os.path.exists(index_file):
            return

        try:
            with open(index_file) as f:
                src_index = json.load(f)
        except Exception:
            return

        src_tensor_names = set(src_index.get("weight_map", {}).keys())

        # Discover new block prefixes from source tensors
        existing_blocks = quantization_config.get("block_name_to_quantize")
        if existing_blocks is None:
            return
        if isinstance(existing_blocks, str):
            existing_blocks = [b.strip() for b in existing_blocks.split(",") if b.strip()]
        existing_set = set(existing_blocks)

        # Find prefixes that have numeric layer indices
        new_prefixes = set()
        for tensor_name in src_tensor_names:
            if not tensor_name.endswith(".weight"):
                continue
            parts = tensor_name.split(".")
            for i, part in enumerate(parts):
                if part.isdigit():
                    prefix = ".".join(parts[:i])
                    if prefix and prefix not in existing_set:
                        new_prefixes.add(prefix)

        added = sorted(new_prefixes - existing_set)
        # Filter out vision/audio blocks - only keep language model blocks
        VISION_AUDIO_KEYS = ["visual", "audio", "vision"]
        added = [p for p in added if not any(k in p.lower() for k in VISION_AUDIO_KEYS)]
        if added:
            merged = existing_blocks + added
            # Remove duplicates and nested prefixes
            merged = [b for b in merged if not any(b != other and b.startswith(other + ".") for other in merged)]
            quantization_config["block_name_to_quantize"] = merged
            logger.info(f"[dry-run] Discovered MTP blocks: {added}")
            logger.info(f"[dry-run] Updated block_name_to_quantize: {merged}")

        # Add ignored layers (like mtp.fc) to extra_config
        BLOCK_NAME_TO_IGNORE = [".shared_expert_gate.", ".mlp.gate.", ".g_proj.", "mtp.fc."]
        extra_config = quantization_config.get("extra_config", {})
        for tensor_name in src_tensor_names:
            if not tensor_name.endswith(".weight"):
                continue
            layer_name = tensor_name[:-len(".weight")]
            if layer_name in extra_config:
                continue
            if any(block in tensor_name for block in BLOCK_NAME_TO_IGNORE):
                extra_config[layer_name] = {"bits": 16, "data_type": "fp"}
                logger.info(f"[dry-run] Added ignored layer to extra_config: {layer_name}")

        if extra_config:
            quantization_config["extra_config"] = extra_config
