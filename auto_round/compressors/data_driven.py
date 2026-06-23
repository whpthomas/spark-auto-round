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
import shutil
import time
import traceback
from functools import partial
from typing import Any, Callable, Optional, Union

import accelerate
import torch
from accelerate.big_modeling import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory, get_max_memory
from auto_round import envs
from auto_round.cli_display import CLIDisplay
from auto_round.metrics import compute_block_sensitivity
from auto_round.report import QuantizationReport
from auto_round.algorithms.alg_config import AlgConfig
from auto_round.calibration.utils import (
    _infer_last_cache_name,
    _update_inputs,
)
from auto_round.compressors.base import BaseCompressor
from auto_round.compressors.utils import (
    _get_quantized_layer_names_outside_blocks,
    check_skippable_keywords,
    immediate_pack,
    init_cache,
    is_nv_fp,
    is_static_wfp8afp8,
    reset_params,
)
from auto_round.logger import logger
from auto_round.modeling.fused_moe.replace_modules import materialize_model_, safe_to_cpu_
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_seqlen_compatible,
    check_to_quantized,
    clear_memory,
    compress_layer_names,
    convert_module_to_hp_if_necessary,
    flatten_list,
    get_block_names,
    get_module,
    hook_ngram_embeddings_on_cpu,
    is_auto_device_mapping,
    is_quantized_input_module,
    memory_monitor,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
    to_device,
    to_dtype,
    wrap_block_forward_positional_to_kwargs,
)
from auto_round.version import __version__
from auto_round.utils.device import (
    _force_trim_malloc,
    parse_available_devices,
)
from auto_round.wrapper import WrapperMultiblock


class DataDrivenCompressor(BaseCompressor):
    need_calib: bool = True

    def __init__(
        self,
        config: Union[AlgConfig, list[AlgConfig]],
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform="hf",
        format=None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
        iters: int = 200,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        tuning_profile: Optional[dict] = None,
        **kwargs,
    ):
        self.iters = iters
        self.clear_cache = kwargs.pop("clear_cache", False)
        self._checkpoint_block_idx = 0
        self._exit_reason: Optional[str] = None
        self._tuning_profile = tuning_profile
        super().__init__(
            config=config,
            model=model,
            tokenizer=tokenizer,
            platform=platform,
            format=format,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            low_cpu_mem_usage=low_cpu_mem_usage,
            **kwargs,
        )
        # Routed to ``self._calibration_state.dataset`` via @property.
        # Set after ``super().__init__()`` because the state object is created there.
        self.dataset = dataset
        if iters == 0:
            self.lr = 5e-3

    def post_init(self) -> None:
        """Run base post-init then attach the registered calibrator strategy.

        Subclasses (MLLM/Diffusion) override ``calib`` directly on the
        Compressor; the calibrator owns ``try_cache_inter_data_gpucpu`` /
        ``cache_inter_data`` orchestration plus the LLM ``calib`` body.
        """
        if self._post_init_done:
            return
        super().post_init()
        if self.calibration is None:
            from auto_round.calibration import get_calibrator

            kind = self._get_calibrator_kind()
            self.calibration = get_calibrator(kind)(self)

    def _get_calibrator_kind(self) -> str:
        """Return the registry name of the calibrator to use.

        Default ``"llm"``.  ``MLLMMixin`` / ``DiffusionMixin`` override this
        to select ``"mllm"`` / ``"diffusion"``.
        """
        return "llm"

    @torch.no_grad()
    def try_cache_inter_data_gpucpu(self, block_names, nsamples, layer_names=None, last_cache_name=None):
        """Thin wrapper around ``self.calibration.collect``.

        Public API kept for backward compatibility (entry.py and
        LLM-Compressor integration).
        """
        if self.calibration is None:
            self.post_init()
        return self.calibration.collect(block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name)

    @torch.no_grad()
    def cache_inter_data(self, block_names, nsamples, layer_names=None, last_cache_name=None):
        """Thin wrapper around ``self.calibration.cache_inter_data``.

        Public API kept for backward compatibility.
        """
        if self.calibration is None:
            self.post_init()
        return self.calibration.cache_inter_data(
            block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name
        )

    @torch.no_grad()
    def calib(self, nsamples, bs):
        """Thin wrapper around ``self.calibration.calib``.

        ``MLLMMixin`` and ``DiffusionMixin`` override this method directly via
        Python MRO; for plain LLM models this routes into ``LLMCalibrator.calib``.
        """
        if self.calibration is None:
            self.post_init()
        return self.calibration.calib(nsamples, bs)

    @torch.no_grad()
    def _get_block_forward_func(self, name: str) -> Callable:
        """Build the block-forward replacement, then let the calibrator wrap it.

        ``Calibrator.wrap_block_forward`` defaults to passthrough; the
        Diffusion calibrator overrides it to convert positional → kwargs.
        """
        from auto_round.calibration.hooks import make_block_forward_func

        fn = make_block_forward_func(self, name)
        if self.calibration is not None:
            fn = self.calibration.wrap_block_forward(fn)
        return fn

    @torch.no_grad()
    def _get_cache_data_hook_for_layer(self, name):
        """Thin wrapper around ``auto_round.calibration.hooks.make_layer_cache_hook``."""
        from auto_round.calibration.hooks import make_layer_cache_hook

        return make_layer_cache_hook(self, name)

    def _replace_forward(self):
        """Thin wrapper around ``auto_round.calibration.hooks.replace_forward_with_hooks``."""
        from auto_round.calibration.hooks import replace_forward_with_hooks

        replace_forward_with_hooks(self)

    def _should_stop_cache_forward(self, name: str) -> bool:
        """Delegate the early-stop policy to the active calibrator.

        Falls back to the default helper when the calibrator has not been
        constructed yet (very early init code paths).
        """
        if self.calibration is not None:
            return self.calibration.should_stop(name)
        from auto_round.calibration.hooks import should_stop_cache_forward

        return should_stop_cache_forward(self, name)

    def _preprocess_block_inputs(self, inputs, first_input_name="input_ids"):
        # Thin wrapper around auto_round.calibration.inputs.preprocess_block_inputs.
        from auto_round.calibration.inputs import preprocess_block_inputs

        return preprocess_block_inputs(
            inputs,
            model_context=self.model_context,
            compress_context=self.compress_context,
            first_input_name=first_input_name,
        )

    def _split_inputs(self, inputs: dict, first_input_name: str) -> tuple[torch.Tensor, dict]:
        # Thin wrapper around auto_round.calibration.inputs.split_inputs.
        from auto_round.calibration.inputs import split_inputs

        return split_inputs(
            inputs,
            first_input_name,
            is_diffusion=self.model_context.is_diffusion,
            shared_cache_keys=self.model_context.shared_cache_keys,
        )

    def normalize_decoding_layer_inputs_(self, decoding_layer_inputs: list[tuple[tuple[Any, dict[str, Any]]]]) -> None:
        """Replay captured decoding-layer calls to populate ``self.inputs``.

        Converts the raw ``(args, kwargs)`` tuples captured by LLM-Compressor's
        input hook into the ``self.inputs`` dict format expected by
        :meth:`quantize_block`.  The logic mirrors the old-arch implementation in
        ``compressors/base.py``.

        Args:
            decoding_layer_inputs:
                A list of entries captured by a forward hook on the decoding layer.
                Each element is a tuple whose first item is ``(args, kwargs)``.
        """
        first_block_name = self.quant_block_list[0][0]

        class _FakeDecodingLayer(torch.nn.Module):

            def forward(self, *args, **kwargs):
                return args, kwargs

        fake_layer = _FakeDecodingLayer()
        fake_layer.orig_forward = fake_layer.forward
        fake_layer._true_orig_forward = lambda *a, **kw: (a, kw)
        fake_layer.forward = partial(self._get_block_forward_func(first_block_name), fake_layer)

        self.inputs = {}
        self.last_cache_name = None
        for step_input in decoding_layer_inputs:
            args, kwargs = step_input[0]
            fake_layer(*args, **kwargs)

    def quantize_block(
        self,
        block: torch.nn.Module,
        inputs,
        q_input: Union[torch.Tensor, dict, None] = None,
        device: Union[str, torch.device] = "cpu",
        auto_offload: bool = True,
    ):
        """Quantize a single decoded block of the model (public API for LLM-Compressor).

        This method is the new-arch equivalent of the old ``BaseCompressor.quantize_block``
        (see ``compressors/base.py``).  It is primarily consumed by LLM-Compressor:
        https://github.com/vllm-project/llm-compressor/pull/1994

        The method normalizes the raw decoding-layer inputs provided by LLM-Compressor,
        runs the full infrastructure pipeline (device placement, act-max collection,
        reference-output caching) for the given *block*, delegates the pure-algorithm
        weight optimization to ``self.quantizer.quantize_block``, then returns the
        quantized-block outputs.

        Args:
            block: The transformer block (decoder layer) to quantize.
            inputs: Either:

                - the raw decoding-layer inputs captured by
                  LLM-Compressor's hook (list of ``((args, kwargs),)`` tuples),
                  in which case they are normalized via
                  :meth:`normalize_decoding_layer_inputs_`; **or**
                - a :class:`~auto_round.calibration.state.CalibrationState`
                  instance produced by a :class:`~auto_round.calibration.base.Calibrator`,
                  which is bound directly without re-normalization.
            q_input: Optional quantized input from the previous block.  ``None`` on
                the first block.
            device: Target device for quantization (e.g. ``"cuda:0"``).
            auto_offload: When *True*, use the device-map-aware offloading path;
                otherwise move ``block`` directly to ``device``.

        Returns:
            tuple: ``(q_outputs, reference_output)`` where *q_outputs* is the
            block's output after quantization (or ``None`` when
            ``enable_quanted_input`` is ``False``), and *reference_output* is the
            full-precision reference output collected before optimization.
        """
        from auto_round.calibration.state import CalibrationState

        if self.diffusion:
            raise NotImplementedError(
                f"Currently, {self.__class__.__name__} does not support quantize_block for diffusion models."
            )

        # Ensure post_init has been called (sets up model_context, compress_context,
        # quantizer, layer_config, etc.).
        if not self._post_init_done:
            self.post_init()

        # When called from LLM-Compressor, `wrapped_model` is a single decoder layer
        # (not the full VL model), so it must not be treated as an MLLM regardless of
        # whether the original model had multimodal assets.  Force is_mllm=False for
        # the duration of this call to stay on the standard LLM quantize_block path.
        orig_is_mllm = self.model_context.is_mllm
        self.model_context.is_mllm = False

        try:
            if isinstance(inputs, CalibrationState):
                # Caller already produced a CalibrationState (typically via
                # ``Calibrator.collect``).  Bind it as the authoritative store so
                # the quantizer reads the same ``inputs`` / ``attention_mask`` /
                # ``batch_dim``.
                self.calibration_state = inputs
            else:
                self.normalize_decoding_layer_inputs_(inputs)
            block_inputs = self.inputs[self.quant_block_list[0][0]]
            input_ids, input_others = self._preprocess_block_inputs(block_inputs, "hidden_states")

            # ── Infrastructure: materialize, dtype convert, device placement ──────
            materialize_model_(block)
            convert_module_to_hp_if_necessary(block, self.model_context.amp_dtype, device)

            if auto_offload:
                if (
                    is_auto_device_mapping(self.compress_context.device_map)
                    and len(self.compress_context.device_list) > 1
                    and not self.model_context.is_diffusion
                ):
                    from auto_round.utils.device import set_auto_device_map_for_block_with_tuning

                    card_0_in_high_risk, loss_device = set_auto_device_map_for_block_with_tuning(
                        block,
                        self.compress_context.device_map,
                        input_ids,
                        self.compress_context.low_gpu_mem_usage,
                        self.quantizer.batch_size,
                        device,
                    )
                else:
                    block = block.to(device)
                    card_0_in_high_risk, loss_device = False, device
            else:
                card_0_in_high_risk, loss_device = False, device

            if len(self.compress_context.device_list) > 1 and auto_offload:
                from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                for n, m in block.named_modules():
                    if len(list(m.children())) != 0 or not hasattr(m, "tuning_device"):
                        continue
                    add_hook_to_module(m, AlignDevicesHook(m.tuning_device, io_same_device=True), True)

            # ── Infrastructure: collect reference output and act_max ──────────────
            bs = self.quantizer.batch_size * self.quantizer.infer_bs_coeff
            if q_input is None:
                hook_handles = self.quantizer.register_calibration_hooks(block)
                reference_output = self.quantizer._get_block_outputs(block, input_ids, input_others, bs)
                for h in hook_handles:
                    h.remove()
            else:
                reference_output = self.quantizer._get_block_outputs(block, input_ids, input_others, bs)
                hook_handles = self.quantizer.register_calibration_hooks(block)
                if hook_handles:
                    self.quantizer._get_block_outputs(block, q_input, input_others, bs, save_output=False)
                for h in hook_handles:
                    h.remove()
                if input_ids is not q_input:
                    clear_memory(input_ids, device_list=self.compress_context.device_list)
                else:
                    clear_memory(device_list=self.compress_context.device_list)
                input_ids = q_input

            # ── Pure algorithm: delegates to quantizer ────────────────────────────
            mid_iter_mem_check = self.compress_context.low_gpu_mem_usage and card_0_in_high_risk
            _tune_result = self.quantizer.quantize_block(
                block,
                input_ids,
                input_others,
                reference_output,
                loss_device=loss_device,
                mid_iter_mem_check=mid_iter_mem_check,
            )

            # ── MoE scale alignment for FP8 dispatch efficiency ────────────────
            if is_nv_fp(self.quantizer.act_data_type) or is_static_wfp8afp8(self.quantizer):
                set_amax_for_all_moe_layers(block, attr_name="act_max")

            # ── Collect quantized-block outputs ───────────────────────────────────
            if self.quantizer.enable_quanted_input:
                q_outputs = self.quantizer._get_block_outputs(block, input_ids, input_others, bs)
            else:
                q_outputs = None

            # ── Cleanup ───────────────────────────────────────────────────────────
            if len(self.compress_context.device_list) > 1:
                accelerate.hooks.remove_hook_from_submodules(block)
            mv_module_from_gpu(block)
            return q_outputs, reference_output
        finally:
            self.model_context.is_mllm = orig_is_mllm

    def _quantize_blocks(
        self,
        model: torch.nn.Module,
        inputs: dict,
        block_names: list,
        q_input: torch.Tensor = None,
        nblocks: int = 1,
        input_others_extra_blocks: dict = None,
    ):
        """Quantize and dequantize the weights of the specified blocks in the model.

        Args:
        model: The PyTorch model to be quantized.
        inputs: The input data for quantization.
        block_names: The names of the blocks to be quantized and dequantized.
        nblocks: The number of blocks to quantize and dequantize.
        device: The device for quantization and dequantization.

        Returns:
        None
        """
        # Checkpoint support for nblocks > 1 is not implemented
        if nblocks > 1:
            logger.warning_once(
                "Checkpointing with nblocks=%d is not supported. "
                "Checkpoints will not be saved for this run.",
                nblocks,
            )

        clear_memory(device_list=self.compress_context.device_list)
        for n, m in model.named_parameters():
            m.requires_grad_(False)

        input_ids, input_others = self._preprocess_block_inputs(inputs)

        # For diffusion models, the heuristic split ("hidden_state" in key) may
        # place keys like encoder_hidden_states in input_ids even though they are
        # not block outputs.  Move those to input_others so they persist across
        # blocks (only output keys get refreshed via reference_output each iteration).
        if self.model_context.is_diffusion and isinstance(input_ids, dict):
            first_block = get_module(model, block_names[0])
            output_config = self.quantizer.DIFFUSION_OUTPUT_CONFIGS.get(
                first_block.__class__.__name__, ["hidden_states"]
            )
            extra_keys = [k for k in list(input_ids.keys()) if k not in output_config]
            for k in extra_keys:
                input_others[k] = input_ids.pop(k)

        for i in range(0, len(block_names), nblocks):
            if input_others_extra_blocks and block_names[i] in input_others_extra_blocks:
                input_others = input_others_extra_blocks[block_names[i]]
                _, input_others = self._preprocess_block_inputs(input_others)
                input_others_extra_blocks.pop(block_names[i])
            if nblocks == 1:
                n = block_names[i]
                m = get_module(model, n)
            else:
                names = block_names[i : min(i + nblocks, len(block_names))]
                modules = [get_module(model, n) for n in names]
                m = WrapperMultiblock(modules)

            if self.compress_context.low_cpu_mem_usage:
                if nblocks == 1:
                    self._offloader.reload(model, n)
                else:
                    self._offloader.reload(model, names)

            block_name_or_names = n if nblocks == 1 else names

            # ── Infrastructure: materialize, dtype convert, device placement ──
            materialize_model_(m)
            convert_module_to_hp_if_necessary(m, self.model_context.amp_dtype, self.compress_context.device)

            if (
                is_auto_device_mapping(self.compress_context.device_map)
                and len(self.compress_context.device_list) > 1
                and not self.model_context.is_diffusion
            ):
                from auto_round.utils.device import set_auto_device_map_for_block_with_tuning

                card_0_in_high_risk, loss_device = set_auto_device_map_for_block_with_tuning(
                    m,
                    self.compress_context.device_map,
                    input_ids,
                    self.compress_context.low_gpu_mem_usage,
                    self.quantizer.batch_size,
                    self.compress_context.device,
                )
            else:
                m = m.to(self.compress_context.device)
                card_0_in_high_risk, loss_device = False, self.compress_context.device

            if len(self.compress_context.device_list) > 1 and not self.model_context.is_diffusion:
                from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                for _n, _mod in m.named_modules():
                    if len(list(_mod.children())) != 0 or not hasattr(_mod, "tuning_device"):
                        continue
                    add_hook_to_module(_mod, AlignDevicesHook(_mod.tuning_device, io_same_device=True), True)

            # ── Infrastructure: collect reference output and act_max ──────────
            bs = self.quantizer.batch_size * self.quantizer.infer_bs_coeff
            if q_input is None:
                hook_handles = self.quantizer.register_calibration_hooks(m)
                reference_output = self.quantizer._get_block_outputs(
                    m, input_ids, input_others, bs, device_override=loss_device
                )
                for h in hook_handles:
                    h.remove()
            else:
                reference_output = self.quantizer._get_block_outputs(
                    m, input_ids, input_others, bs, device_override=loss_device
                )
                hook_handles = self.quantizer.register_calibration_hooks(m)
                if hook_handles:
                    self.quantizer._get_block_outputs(
                        m, q_input, input_others, bs, save_output=False, device_override=loss_device
                    )
                for h in hook_handles:
                    h.remove()

            # ── Infrastructure: swap q_input ──────────────────────────────────
            if q_input is not None:
                if input_ids is not q_input:
                    clear_memory(input_ids, device_list=self.compress_context.device_list)
                else:
                    clear_memory(device_list=self.compress_context.device_list)
                input_ids = q_input

            # ── Pure algorithm: delegates to quantizer ────────────────────────
            mid_iter_mem_check = self.compress_context.low_gpu_mem_usage and card_0_in_high_risk
            _tune_result = self.quantizer.quantize_block(
                m,
                input_ids,
                input_others,
                reference_output,
                loss_device=loss_device,
                mid_iter_mem_check=mid_iter_mem_check,
            )

            # ── MoE scale alignment for FP8 dispatch efficiency ────────────────
            if is_nv_fp(self.quantizer.act_data_type) or is_static_wfp8afp8(self.quantizer):
                set_amax_for_all_moe_layers(m, attr_name="act_max")

            # ── Infrastructure: collect q_outputs if needed ───────────────────
            if self.quantizer.enable_quanted_input:
                q_input = self.quantizer._get_block_outputs(m, input_ids, input_others, bs)
            else:
                q_input = None

            # ── Infrastructure: hook removal, device cleanup, logging ─────────
            if len(self.compress_context.device_list) > 1 and not self.model_context.is_diffusion:
                accelerate.hooks.remove_hook_from_submodules(m)
            mv_module_from_gpu(m)
            # if self.enable_torch_compile:
            #     torch._dynamo.reset()
            #     self.quantizer._invalidate_block_forward_cache()
            # Keep old-arch semantics: the next block's FP reference input comes
            # from the current block's reference output, while q_input (when
            # enabled) is only used as the quantized-input companion for the
            # next block.
            next_input_ids = reference_output
            clear_memory(
                input_ids if input_ids is not next_input_ids else None, device_list=self.compress_context.device_list
            )
            memory_monitor.log_summary()

            # ── Infrastructure: immediate_pack / shard write ──────────────────
            if self.compress_context.is_immediate_packing:
                for _n, _mod in m.named_modules():
                    if hasattr(_mod, "bits") and check_to_quantized(_mod):
                        from auto_round.compressors.utils import immediate_pack as _immediate_pack

                        _immediate_pack(_mod.global_name, self.quantizer.layer_config)

            input_ids = next_input_ids

            if self.compress_context.is_immediate_saving:
                self.shard_writer.write(m, is_finalize=False)

            if self.compress_context.low_cpu_mem_usage and not self.compress_context.is_immediate_saving:
                if nblocks == 1:
                    self._offloader(model, n, overwrite=True)
                else:
                    for name in names:
                        self._offloader(model, name, overwrite=True)

            # ── Checkpoint: save quantized state after each block ──────────
            if nblocks == 1:
                self._save_checkpoint(self._checkpoint_block_idx, block_name_or_names, m)
                self._checkpoint_block_idx += 1
            else:
                logger.debug(
                    f"Skipping checkpoint for block_idx={self._checkpoint_block_idx}: "
                    f"nblocks={nblocks} > 1 not supported for checkpointing"
                )

            # ── Sensitivity metrics ───────────────────────────────────────────
            if q_input is not None and hasattr(self, '_display') and self._display is not None:
                cos_sim, psnr_db = compute_block_sensitivity(reference_output, q_input)
                block_label = n if nblocks == 1 else f"[{i+1}-{min(i+nblocks, len(block_names))}]/{len(block_names)}"
                self._display.print_sensitivity(
                    block_label, cos_sim, psnr_db,
                    init_loss=_tune_result.get("init_loss"),
                    best_loss=_tune_result.get("best_loss"),
                    best_iter=_tune_result.get("best_iter", 0),
                    total_iters=_tune_result.get("total_iters", 0),
                )
                if hasattr(self, '_report') and self._report is not None:
                    self._report.add_layer(
                        block_label, cos_sim, psnr_db,
                        init_loss=_tune_result.get("init_loss"),
                        best_loss=_tune_result.get("best_loss"),
                        best_iter=_tune_result.get("best_iter", 0),
                        total_iters=_tune_result.get("total_iters", 0),
                    )


        if not self.compress_context.is_immediate_saving:
            self.model = mv_module_from_gpu(self.model)
        for n, m in self.model.named_modules():
            if hasattr(m, "name"):
                delattr(m, "name")

        del q_input
        del input_ids
        del input_others
        del inputs

        clear_memory(device_list=self.compress_context.device_list)

    def _collect_cli_args(self) -> dict[str, str | int | float]:
        """Collect CLI-relevant arguments for the report header."""
        args = {}
        if hasattr(self.quantizer, 'batch_size'):
            args['batch_size'] = self.quantizer.batch_size
        if hasattr(self.quantizer, 'iters'):
            args['iters'] = self.quantizer.iters
        if hasattr(self.quantizer, 'nsamples'):
            args['nsamples'] = self.quantizer.nsamples
        if hasattr(self.quantizer, 'seqlen'):
            args['seqlen'] = self.quantizer.seqlen
        if hasattr(self.compress_context, 'group_size'):
            args['group_size'] = self.compress_context.group_size
        if hasattr(self, 'dataset') and self.dataset:
            args['dataset'] = self.dataset
        if hasattr(self.compress_context, 'output_dir'):
            args['output_dir'] = self.compress_context.output_dir
        return args

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    @property
    def _checkpoint_dir(self) -> Optional[str]:
        """Return the .cache checkpoint directory path, or None if output_dir is not set."""
        if self.compress_context and self.compress_context.output_dir:
            return os.path.join(self.compress_context.output_dir, ".cache")
        return None

    def _check_resume_state(self) -> tuple:
        """Check for existing checkpoint and return resume state.

        Validates:
          - .cache/progress.json exists and is valid JSON
          - completed count is within expected range (0 < completed <= total)
          - All block_{i:05d}.pt files for i in 0..completed-1 exist
          - The last checkpoint file loads successfully (sanity check)

        Returns:
            (resume_mode: bool, completed: int, total: int, block_names: list[list[str]],
             exit_reason: Optional[str], tuning_profile: Optional[dict])
        """
        ckpt_dir = self._checkpoint_dir
        if ckpt_dir is None:
            return False, 0, 0, [], None, None
        if not os.path.isdir(ckpt_dir):
            return False, 0, 0, [], None, None

        progress_path = os.path.join(ckpt_dir, "progress.json")
        if not os.path.isfile(progress_path):
            logger.debug("No progress.json found in %s, starting fresh", ckpt_dir)
            return False, 0, 0, [], None, None

        # Parse progress.json
        try:
            with open(progress_path, "r") as f:
                progress = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt progress.json in %s (%s), starting fresh", ckpt_dir, exc)
            return False, 0, 0, [], None, None

        completed = progress.get("completed", 0)
        total = progress.get("total", 0)
        saved_block_names = progress.get("block_names", [])

        # Parse exit_reason and tuning_profile for stateful resume
        exit_reason = progress.get("exit_reason", None)
        tuning_profile = progress.get("tuning_profile", None)

        # Validate completed count
        if not isinstance(completed, int) or completed <= 0:
            logger.debug("progress.json has completed=%s, starting fresh", completed)
            return False, 0, 0, [], None, None
        if not isinstance(total, int) or total < completed:
            logger.warning(
                "progress.json has completed=%d > total=%d, starting fresh",
                completed, total,
            )
            return False, 0, 0, [], None, None

        # Validate that all block checkpoint files exist
        missing_files = []
        for i in range(completed):
            block_path = os.path.join(ckpt_dir, f"block_{i:05d}.pt")
            if not os.path.isfile(block_path):
                missing_files.append(f"block_{i:05d}.pt")
        if missing_files:
            logger.warning(
                "Missing checkpoint files: %s. Starting fresh.",
                ", ".join(missing_files),
            )
            return False, 0, 0, [], None, None

        # Try loading the last checkpoint to verify validity
        try:
            last_block_path = os.path.join(ckpt_dir, f"block_{completed-1:05d}.pt")
            sample = torch.load(last_block_path, map_location="cpu", weights_only=True)
            if not isinstance(sample, dict) or len(sample) == 0:
                logger.warning(
                    "Last checkpoint %s is empty or invalid, starting fresh",
                    last_block_path,
                )
                return False, 0, 0, [], None, None
            del sample
        except Exception as exc:
            logger.warning(
                "Last checkpoint %s failed to load (%s), starting fresh",
                last_block_path, exc,
            )
            return False, 0, 0, [], None, None

        logger.info(
            "Found valid checkpoint: %d/%d blocks completed in %s",
            completed, total, ckpt_dir,
        )
        return True, completed, total, saved_block_names, exit_reason, tuning_profile

    def _checkpoint_block_path(self, block_idx: int) -> str:
        """Return the full path to a checkpoint block file.

        Args:
            block_idx: Zero-based block index.

        Returns:
            Full path to the block checkpoint file.

        Raises:
            ValueError: If checkpoint_dir is not set (output_dir not configured).
        """
        ckpt_dir = self._checkpoint_dir
        if ckpt_dir is None:
            raise ValueError(
                "Cannot build checkpoint path: output_dir is not set. "
                "Call quantize_and_save() with an output_dir first."
            )
        return os.path.join(ckpt_dir, f"block_{block_idx:05d}.pt")

    def _save_checkpoint(self, block_idx: int, block_name: str, module: torch.nn.Module) -> None:
        """Save quantized state dict for a single completed block.

        Saves:
          - block_{block_idx:05d}.pt: module state dict (all tensors moved to CPU)
          - progress.json: updated metadata

        Only saves when nblocks == 1 (the hardcoded GB10 setting).
        """
        ckpt_dir = self._checkpoint_dir
        if ckpt_dir is None:
            return
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save block state dict (tensors on CPU)
        state_dict = {
            k: v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else v
            for k, v in module.state_dict().items()
            if isinstance(v, torch.Tensor)
        }
        block_path = self._checkpoint_block_path(block_idx)
        torch.save(state_dict, block_path)
        logger.debug(f"Checkpoint saved: {block_path}")

        # Update progress.json
        self._save_checkpoint_progress(block_idx)

    def _save_checkpoint_progress(self, completed: int) -> None:
        """Write or update progress.json with the number of completed blocks."""
        ckpt_dir = self._checkpoint_dir
        if ckpt_dir is None:
            return

        # Ensure cache directory exists (normally created by _save_checkpoint first)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Build block names list from the full block list
        # Guard: quantizer may not be available before post_init()
        all_block_names = []
        try:
            if hasattr(self, "quantizer") and self.quantizer is not None:
                if bool(self.quantizer.quant_block_list):
                    all_block_names = self.quantizer.quant_block_list
                else:
                    all_block_names = get_block_names(self.model_context.model)
        except (AttributeError, RuntimeError):
            pass

        progress = {
            "completed": completed,
            "total": len(all_block_names),
            "block_names": [b[0] if isinstance(b, list) else b for b in all_block_names],
            "exit_reason": self._exit_reason,
            "tuning_profile": self._tuning_profile,
        }
        progress_path = os.path.join(ckpt_dir, "progress.json")
        tmp_path = progress_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(progress, f, indent=2)
            os.replace(tmp_path, progress_path)  # atomic on POSIX
            logger.debug(f"Checkpoint progress updated: {progress_path}")
        except OSError as exc:
            logger.warning("Failed to write checkpoint progress to %s: %s", progress_path, exc)
            # Clean up temp file if rename failed
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _write_exit_reason(self, exit_reason: Optional[str]) -> None:
        """Write exit_reason and tuning_profile to progress.json (atomic write).

        Parameters
        ----------
        exit_reason : str or None
            "completed", "oom", "interrupted", or None (preserve existing).
        """
        self._exit_reason = exit_reason
        ckpt_dir = self._checkpoint_dir
        if ckpt_dir is None:
            return

        progress_path = os.path.join(ckpt_dir, "progress.json")
        if not os.path.isfile(progress_path):
            return  # nothing to annotate

        try:
            with open(progress_path, "r") as f:
                progress = json.load(f)
        except (json.JSONDecodeError, OSError):
            return  # corrupt or unreadable

        if exit_reason is not None:
            progress["exit_reason"] = exit_reason
        progress["tuning_profile"] = self._tuning_profile

        # Atomic write: .tmp + os.replace
        tmp_path = progress_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(progress, f, indent=2)
            os.replace(tmp_path, progress_path)
        except OSError:
            pass  # best-effort

    def _load_checkpoint_block(self, block_idx: int, block_name: str, model: torch.nn.Module) -> None:
        """Load a checkpointed block state dict into the model.

        Handles both regular and meta-device modules:
          - Meta device: materializes the module, loads state dict, then re-offloads
          - Regular device: directly calls _load_state_dict_into_module

        Args:
            block_idx: Zero-based block index
            block_name: Name of the block module (e.g. "model.layers.0")
            model: The full model (module hierarchy)
        """
        ckpt_dir = self._checkpoint_dir
        if ckpt_dir is None:
            logger.warning("Cannot load checkpoint block %d: no checkpoint directory", block_idx)
            return

        try:
            block_path = self._checkpoint_block_path(block_idx)
        except ValueError:
            logger.warning("Cannot load checkpoint block %d: output_dir not set", block_idx)
            return

        if not os.path.isfile(block_path):
            logger.warning("Checkpoint file not found: %s", block_path)
            return

        # Load state dict from disk
        state_dict = torch.load(block_path, map_location="cpu", weights_only=True)

        try:
            # Get the module
            from auto_round.utils.model import get_module
            module = get_module(model, block_name)
            if module is None:
                logger.warning("Cannot find module %s in model, skipping checkpoint load", block_name)
                return

            # Check if module is on meta device
            is_meta = False
            for p in module.parameters():
                if p.device.type == "meta":
                    is_meta = True
                    break
            if not is_meta:
                for b in module.buffers():
                    if b.device.type == "meta":
                        is_meta = True
                        break

            if is_meta:
                # Materialize from meta: create real tensors, then load state dict
                logger.debug("Materializing meta-device module %s for checkpoint load", block_name)

                materialize_model_(module)
                # Move to CPU for state dict loading
                module = module.to("cpu")
                # Load state dict using the offload helper
                from auto_round.utils.offload import _load_state_dict_into_module

                _load_state_dict_into_module(state_dict, module)

                # Re-offload if low_cpu_mem_usage is active
                if self.compress_context and self.compress_context.low_cpu_mem_usage and self._offloader is not None:
                    self._offloader(model, block_name, overwrite=True)
            else:
                # Module is already materialized — load state dict directly on CPU
                from auto_round.utils.offload import _load_state_dict_into_module

                _load_state_dict_into_module(state_dict, module)

            logger.debug("Loaded checkpoint block %d (%s) from %s", block_idx, block_name, block_path)
        finally:
            # Free CPU memory from loaded state dict
            del state_dict
            gc.collect()

    def _clear_cache(self) -> None:
        """Remove the .cache checkpoint directory if it exists.

        Safety: validates the cache directory is a proper .cache
        subdirectory of output_dir to prevent accidental deletion of
        arbitrary paths.
        """
        ckpt_dir = self._checkpoint_dir
        if ckpt_dir is None:
            return

        # Safety check: ensure ckpt_dir is actually a .cache subdirectory
        # of output_dir (not "/" or "/.cache" or some other path)
        output_dir = self.compress_context.output_dir
        if not output_dir:
            return
        expected_parent = os.path.normpath(os.path.join(output_dir, ".cache"))
        actual_path = os.path.normpath(ckpt_dir)
        if actual_path != expected_parent:
            logger.error(
                "Safety check failed: ckpt_dir %s is not the expected .cache path %s. "
                "Refusing to delete.",
                actual_path, expected_parent,
            )
            return

        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            logger.info("Removed checkpoint cache: %s", ckpt_dir)

    def _check_and_clear_cache_flag(self) -> None:
        """If the clear_cache flag is set, remove existing checkpoints.

        Note: shutil.rmtree follows symlinks. If .cache/ is a symlink,
        it will delete the target directory content.
        """
        if getattr(self, "clear_cache", False):
            ckpt_dir = self._checkpoint_dir
            if ckpt_dir and os.path.isdir(ckpt_dir):
                if os.path.islink(ckpt_dir):
                    logger.warning(
                        ".cache/ is a symlink to %s. Removing symlink only, "
                        "not the target.",
                        os.readlink(ckpt_dir),
                    )
                    os.unlink(ckpt_dir)  # Remove symlink without following
                else:
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
                    logger.info("Removed checkpoint cache: %s", ckpt_dir)

    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """
        self.post_init()

        # Reclaim heap fragmentation from init/post_init before the memory-intensive quantize loop.
        gc.collect()
        _force_trim_malloc()

        self._check_compatibility()

        if bool(self.quantizer.quant_block_list):
            all_blocks = self.quantizer.quant_block_list
        else:
            all_blocks = get_block_names(self.model_context.model)

        if len(all_blocks) == 0:
            logger.warning("could not find blocks, exit with original model")
            return self.model_context.model, self.quantizer.layer_config

        layer_names = _get_quantized_layer_names_outside_blocks(
            model=self.model_context.model,
            layer_config=self.quantizer.layer_config,
            supported_types=SUPPORTED_LAYER_TYPES,
            quant_block_list=self.quantizer.quant_block_list,
        )
        if not self.has_variable_block_shape:
            to_cache_block_names = [block[0] for block in all_blocks]
        else:
            to_cache_block_names = flatten_list(all_blocks)
        _last_cache_name = to_cache_block_names[-1] if len(to_cache_block_names) > 1 else None
        to_cache_layer_names = layer_names
        if self.super_group_size is not None:
            to_cache_layer_names = []
        if len(layer_names) > 0:
            logger.info(
                "Starting to cache block inputs. This may be slow due to external block layers: %s", layer_names
            )
        else:
            logger.info("start to cache block inputs")
        all_inputs = self.try_cache_inter_data_gpucpu(
            to_cache_block_names,
            self.nsamples,
            to_cache_layer_names,
            last_cache_name=_last_cache_name,
        )
        self.inputs = all_inputs
        is_quantized_embedding = self._quantize_embedding_layer()
        clear_memory(device_list=self.compress_context.device_list)
        all_q_inputs = None
        if is_quantized_embedding:
            all_inputs = copy.deepcopy(self.inputs)
            clear_memory(self.inputs, device_list=self.compress_context.device_list)
            all_q_inputs = self.try_cache_inter_data_gpucpu(
                to_cache_block_names, self.nsamples, to_cache_layer_names, last_cache_name=_last_cache_name
            )
        # Remove accelerate dispatch hooks before moving parameters.
        # hf_device_map is kept for reference but hooks are no longer needed.
        if hasattr(self.model_context.model, "hf_device_map") and len(self.model_context.model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(self.model_context.model)
        self.model_context.model = mv_module_from_gpu(self.model_context.model)
        clear_memory(device_list=self.compress_context.device_list)
        logger.info("caching done")
        if self.compress_context.low_cpu_mem_usage:
            if self.model_context.is_model_patched and not self.compress_context.is_immediate_saving:
                self._offloader(
                    self.model_context.model,
                    all_blocks,
                    clear_memory=True,
                    device_list=self.compress_context.device_list,
                )
                if not self._offloader.enabled:
                    self.compress_context.low_cpu_mem_usage = False
            else:
                self.compress_context.low_cpu_mem_usage = False
        # Count total blocks for display
        total_blocks = sum(len(b) for b in all_blocks) if len(all_blocks) > 1 else len(all_blocks[0])

        # Create display and report
        self._display = CLIDisplay(total_blocks=total_blocks)
        self._report = QuantizationReport(
            model_name=getattr(self.model_context.model, 'name_or_path', 'unknown'),
            version=__version__,
            cli_args=self._collect_cli_args(),
        )
        self._display.begin()

        # ── Reset checkpoint counter and check resume state ──────────────────
        self._checkpoint_block_idx = 0
        resume_mode, completed, total_saved, saved_block_names, exit_reason, tuning_profile = (
            self._check_resume_state()
        )
        self._exit_reason = exit_reason
        self._tuning_profile = tuning_profile

        # ── Resume mode: load completed blocks into model ───────────────────
        if resume_mode and completed > 0:
            logger.info("Loading %d completed blocks from checkpoint...", completed)
            for i in range(completed):
                block_name = saved_block_names[i] if i < len(saved_block_names) else None
                if block_name is None:
                    if i < len(all_blocks):
                        block_entry = all_blocks[i]
                        block_name = block_entry[0] if isinstance(block_entry, list) else block_entry
                if block_name:
                    self._load_checkpoint_block(i, block_name, self.model_context.model)
                else:
                    logger.warning(
                        "Cannot resolve block name for index %d, skipping checkpoint load", i
                    )
            logger.info("Completed block loading. Resuming from block %d.", completed)
        elif not resume_mode:
            logger.info("Fresh quantization run: no existing checkpoint found.")

        # ── All blocks already completed? Skip quantization loop ────────────
        all_done = resume_mode and completed >= len(all_blocks)
        if all_done:
            logger.info(
                "All %d blocks already completed in checkpoint. "
                "Skipping quantization tuning.",
                completed,
            )

        start_time = time.time()
        try:
            for block_names in all_blocks:
                if all_done:
                    break
                # In resume mode, skip blocks up to completed count
                if resume_mode and self._checkpoint_block_idx < completed:
                    logger.info(f"Skipping already-completed block {self._checkpoint_block_idx}: {block_names}")
                    inputs = all_inputs[block_names[0]]
                    all_inputs.pop(block_names[0])
                    if all_q_inputs is not None:
                        all_q_inputs.pop(block_names[0], None)
                    self._checkpoint_block_idx += len(block_names)
                    continue

                inputs = all_inputs[block_names[0]]
                all_inputs.pop(block_names[0])
                q_inputs = None
                if all_q_inputs is not None:
                    q_inputs = all_q_inputs[block_names[0]]
                    all_q_inputs.pop(block_names[0])

                inputs, q_inputs = _update_inputs(inputs, q_inputs, self.model_context)

                clear_memory(self.inputs, device_list=self.compress_context.device_list)

                if "input_ids" in inputs.keys():
                    total_samples = len(inputs["input_ids"])
                    if total_samples < self.quantizer.batch_size:
                        self.quantizer.batch_size = total_samples
                        logger.warning(f"force the train batch size to {total_samples}")

                self._quantize_blocks(
                    self.model_context.model,
                    inputs,
                    block_names,
                    q_input=q_inputs if q_inputs is not None else None,
                    nblocks=self.nblocks,
                    input_others_extra_blocks=all_inputs,
                )
                if self.compress_context.is_immediate_packing and len(self.formats) != 1:
                    raise ValueError(
                        f"Expected exactly one packing format when 'immediate_packing' is True, "
                        f"but got {len(self.formats)} formats."
                    )

            # ── Post-loop: finalize display, save report, extra layers ────
            peak_ram = getattr(memory_monitor, 'peak_ram', None)
            peak_vram = None
            if hasattr(memory_monitor, 'peak_vram') and memory_monitor.peak_vram:
                peak_vram = max(memory_monitor.peak_vram.values())
            self._display.end(peak_ram_gb=peak_ram, peak_vram_gb=peak_vram)
            self._report.set_memory_summary(peak_ram_gb=peak_ram, peak_vram_gb=peak_vram)
            if hasattr(self.compress_context, 'output_dir') and self.compress_context.output_dir:
                report_path = self._report.save(self.compress_context.output_dir)
                logger.info(f"Quantization report saved to {report_path}")
            if self.compress_context.low_cpu_mem_usage:
                self._offloader.reload(self.model_context.model)
            self._quantize_layers(layer_names, all_inputs)

            convert_module_to_hp_if_necessary(
                self.model_context.model, self.model_context.amp_dtype, self.compress_context.device, to_cpu=True
            )
            if self.compress_context.is_immediate_saving:
                self.shard_writer.write(is_finalize=True)

            end_time = time.time()
            cost_time = end_time - start_time
            logger.info(f"quantization tuning time {cost_time}")

            # Dump a summary
            quantized_layers = []
            unquantized_layers = []
            for n, m in self.model_context.model.named_modules():
                if isinstance(m, tuple(SUPPORTED_LAYER_TYPES)):
                    if check_to_quantized(m):
                        quantized_layers.append(n)
                    else:
                        unquantized_layers.append(n)
                elif hasattr(m, "scales") or hasattr(m, "scale"):  # packing_immediately
                    quantized_layers.append(n)
            summary_info = (
                f"Summary: quantized {len(quantized_layers)}/{len(quantized_layers) + len(unquantized_layers)} in the model"
            )
            if len(unquantized_layers) > 0:
                compressed_unquantized_layers = compress_layer_names(unquantized_layers)
                summary_info += f", unquantized layers: {compressed_unquantized_layers}"
            logger.info(summary_info)

            self.model_context.quantized = True

        except KeyboardInterrupt:
            logger.warning(
                "Quantization interrupted by user. Checkpoint files are preserved in .cache/."
            )
            logger.warning(
                "  To resume, re-run with the same --output_dir. "
                "To start fresh, add --clear-cache."
            )
            self._write_exit_reason("interrupted")
            raise

        except torch.OutOfMemoryError as e:
            logger.error(f"Out of memory during quantization: {e}")
            self._write_exit_reason("oom")
            raise

        except BaseException as exc:
            # Check for OOM wrapped in RuntimeError (sometimes CUDA OOM wraps as RuntimeError)
            err_str = str(exc).lower()
            oom_patterns = ["out of memory", "cuda out of memory", "cuda oom", "cuda error"]
            if any(p in err_str for p in oom_patterns):
                logger.error(f"CUDA OOM detected in error: {exc}")
                self._write_exit_reason("oom")
            else:
                logger.warning(
                    "Quantization failed with %s: %s. Checkpoint files are preserved in .cache/.",
                    type(exc).__name__, exc,
                )
                logger.warning(
                    "  To resume, fix the issue and re-run with the same --output_dir. "
                    "To start fresh, add --clear-cache."
                )
                # Don't overwrite a more specific exit_reason that may be on disk
                if self._exit_reason is None:
                    self._write_exit_reason(None)
            raise

        else:
            # Success: write exit_reason before clearing cache
            self._write_exit_reason("completed")
            # No exception: clean up .cache/ on successful completion
            if self.compress_context.output_dir:
                self._clear_cache()
                logger.info("Checkpoint cache cleaned up after successful quantization.")

        return self.model_context.model, self.quantizer.layer_config

    def _quantize_layers(self, layer_names: list, layer_inputs: dict) -> None:
        """Quantizes specified layers based on inputs and configuration.

        Args:
            layer_names (list): list of layer names to quantize.
            layer_inputs (dict): Dictionary mapping layer names to input data.

        Returns:
            None
        """
        # TODO currently we take all the layers outside blocks as post block layers which is not optimal
        # if there is no input for layer, we use rtn
        for layer_name in copy.deepcopy(layer_names):
            if layer_name not in layer_inputs:
                if self.act_bits < 16 and not self.act_dynamic:
                    if "lm_head" in layer_name:
                        logger.warning_once(
                            "Static activation quantization for lm_head is not fully supported yet. "
                            "If lm_head calibration inputs are missing, activation scale may fall back to unit scale "
                            "or quantization may be skipped."
                        )
                    # Activation quantization requires collected inputs
                    msg_prefix = (
                        f"Activation max hook for layer '{layer_name}' is unavailable due to "
                        f"insufficient collected inputs. "
                    )
                    if "fp8_e5m2" in self.act_data_type:
                        logger.warning(msg_prefix + "Please notes that unit scale is used for this layer.")
                    else:
                        logger.warning(
                            msg_prefix + "Static activation quantization is not supported or ineffective, "
                            "Skipping quantization for this layer."
                        )
                        layer_names.remove(layer_name)
                        continue
                self.quantizer.quantize_layer_outside_block(
                    layer_name,
                    input_ids=None,
                    device=self.compress_context.device,
                    disable_opt_rtn=getattr(self, "disable_opt_rtn", False),
                )
                layer_names.remove(layer_name)
        if len(layer_names) == 0:
            memory_monitor.update()
            memory_monitor.log_summary()
            return
        q_layer_inputs = None
        enable_quanted_input = self.enable_quanted_input

        if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1 and enable_quanted_input:
            dispatch_model(self.model, self.model.hf_device_map)

        if enable_quanted_input:
            logger.info("starting to cache layer inputs for %s, this may be quite slow ", layer_names)
            q_layer_inputs = self.try_cache_inter_data_gpucpu([], self.nsamples, layer_names=layer_names)
            if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                accelerate.hooks.remove_hook_from_submodules(
                    self.model
                )  # self.model.hf_device_map has not been changed
        if not self.compress_context.is_immediate_saving:
            self.model = mv_module_from_gpu(self.model)
        clear_memory(device_list=self.compress_context.device_list)
        quant_layer = self.quantizer.quantize_layer_outside_block
        for layer_name in layer_names:
            layer_input = layer_inputs[layer_name]
            layer_input = to_device(layer_input, self.compress_context.cache_device)
            q_layer_input = q_layer_inputs.get(layer_name, None) if q_layer_inputs is not None else None
            q_layer_input = to_device(q_layer_input, self.compress_context.cache_device)
            quant_layer(layer_name, layer_input, q_layer_input, device=self.compress_context.device)
            if self.compress_context.is_immediate_packing:
                immediate_pack(layer_name, self.quantizer.layer_config)

            if self.compress_context.is_immediate_saving:
                m = get_module(self.model, layer_name)
                self.shard_writer.write(m, name=layer_name, is_finalize=False)
            del layer_input
            clear_memory(q_layer_input, device_list=self.compress_context.device_list)
            memory_monitor.log_summary()

    def _check_compatibility(self) -> None:
        """Checks compatibility of the configurations and model."""
        # ``seqlen`` clamping is owned by ``CalibrationState``.
        self._calibration_state.clamp_seqlen(self.model_context)

        if self.group_size == 0 and "fp8" not in self.data_type:
            logger.warning("`group_size==0` is not supported for data_type other than fp8 ")

        if (
            self.bits <= 2
            and (self.iters < 1000 or not getattr(self.quantize_config, "enable_alg_ext", False))
            and self.super_group_size is None
        ):
            logger.warning(
                "for bits <= 2, it is recommended to enable `auto-round-best` " "and turn on `--enable_alg_ext` "
            )


class CalibratedRTNCompressor(DataDrivenCompressor):
    """DataDrivenCompressor variant for iters=0 RTN that needs calibration data.

    Handles two cases that require forward passes through the model:
      - Weight quantization with imatrix (importance-matrix statistics for
        improved RTN accuracy on INT / weight-only schemes).
      - Activation quantization with static scales (e.g. FP8_STATIC)
        where per-tensor or per-channel scale factors must be collected before
        the actual quantization step.

    Both cases use OptimizedRTNQuantizer and need a calibration dataset,
    which is why they cannot be handled by the zero-shot (no-data) path.
    """

    need_calib: bool = True

    def __init__(
        self,
        config: AlgConfig,
        model: torch.nn.Module,
        **kwargs,
    ):
        kwargs["iters"] = 0
        super().__init__(
            config,
            model,
            **kwargs,
        )

    def _quantize_via_rtn_blockwise(self) -> None:
        """Quantize model layers block by block using cached inputs and imatrix."""

        all_blocks = self.quantizer.quant_block_list or get_block_names(self.model)
        if not all_blocks:
            raise ValueError("Could not find any blocks. Check the model or quant_block_list.")

        if not self.has_variable_block_shape:
            to_cache_block_names = [block[0] for block in all_blocks]
        else:
            to_cache_block_names = flatten_list(all_blocks)
        layer_names = _get_quantized_layer_names_outside_blocks(
            model=self.model_context.model,
            layer_config=self.quantizer.layer_config,
            supported_types=SUPPORTED_LAYER_TYPES,
            quant_block_list=self.quantizer.quant_block_list,
        )
        if (
            self.quantize_config.is_act_quantize
            and (not self.quantize_config.act_dynamic or len(layer_names) > 0)
            or self.has_variable_block_shape
        ):
            if len(layer_names) > 0:
                logger.warning(
                    "quantize layers outside blocks for static activation quantizaiton"
                    " will significantly increase calibration time"
                )
            all_inputs = self.try_cache_inter_data_gpucpu(to_cache_block_names, self.nsamples, layer_names)
        else:
            all_inputs = self.cache_inter_data(to_cache_block_names, self.nsamples)

        # Clear hooks for multi-GPU setups
        if hasattr(self.model_context.model, "hf_device_map") and len(self.model_context.model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(self.model_context.model)

        from tqdm import tqdm as _tqdm
        pbar = _tqdm(range(sum(len(block) for block in all_blocks)))

        for block_names in all_blocks:
            first_block = block_names[0]
            inputs = all_inputs.pop(first_block)
            input_keys = [k for k in inputs if k.startswith("hidden_state")]
            if len(input_keys) != 1:
                raise RuntimeError(
                    "hidden_states arg mismatch. Please file an issue at https://github.com/intel/auto-round/issues"
                )
            inputs["input_ids"] = inputs.pop(input_keys[0])

            clear_memory(self.inputs, device_list=self.compress_context.device_list)

            total_samples = len(inputs["input_ids"])
            if total_samples < self.batch_size:
                self.batch_size = total_samples
                logger.warning(f"Forcing batch size to {total_samples}")

            tmp_dtype = self.model_context.amp_dtype if self.model_context.amp else torch.float32

            input_ids = to_device(inputs.pop("input_ids"), self.compress_context.cache_device)
            input_ids = [id_.to(tmp_dtype) for id_ in input_ids]

            def process_input_others(input_others):
                input_others = to_device(input_others, self.compress_context.cache_device)
                # Unwrap single-element list/tuple so they are passed as bare values.
                for key in list(input_others.keys()):
                    val = input_others[key]
                    if isinstance(val, (list, tuple)) and len(val) == 1:
                        input_others[key] = val[0]
                for key, val in input_others.items():
                    if isinstance(val, torch.Tensor) and val.dtype in (torch.float16, torch.bfloat16):
                        input_others[key] = val.to(tmp_dtype)
                    elif isinstance(val, list):
                        input_others[key] = [
                            to_dtype(v, tmp_dtype)
                            for v in val
                            if not (isinstance(v, torch.Tensor) and v.dtype in (torch.int32, torch.int64))
                        ]
                return input_others

            input_others = inputs
            input_others = process_input_others(input_others)
            for block_name in block_names:
                if block_name in all_inputs.keys():
                    input_others = all_inputs[block_name]
                    input_others = process_input_others(input_others)
                    all_inputs.pop(block_name)
                pbar.set_description(f"Quantizing {block_name}")
                block = get_module(self.model_context.model, block_name)

                # ── Infrastructure: materialize, dtype convert, device placement ──
                materialize_model_(block)
                block.to("cpu")
                block = convert_module_to_hp_if_necessary(
                    block, dtype=self.model_context.amp_dtype, device=self.compress_context.device
                )
                if (
                    is_auto_device_mapping(self.compress_context.device_map)
                    and len(self.compress_context.device_list) > 1
                    and not self.model_context.is_diffusion
                ):
                    from auto_round.utils.device import set_auto_device_map_for_block_with_tuning

                    set_auto_device_map_for_block_with_tuning(
                        block,
                        self.compress_context.device_map,
                        input_ids,
                        self.compress_context.low_gpu_mem_usage,
                        self.quantizer.batch_size,
                        self.compress_context.device,
                    )
                    if len(self.compress_context.device_list) > 1 and not self.model_context.is_diffusion:
                        from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                        for _, _mod in block.named_modules():
                            if len(list(_mod.children())) != 0 or not hasattr(_mod, "tuning_device"):
                                continue
                            add_hook_to_module(_mod, AlignDevicesHook(_mod.tuning_device, io_same_device=True), True)
                else:
                    block = block.to(self.compress_context.device)

                # ── Infrastructure: register act_max hook and run forward pass ──
                hook_handles = self.quantizer.register_calibration_hooks(block, imatrix=False)
                block_input_ids = input_ids  # keep reference for quantize_block
                input_ids = self.quantizer._get_block_outputs(
                    block,
                    input_ids,
                    input_others,
                    self.quantizer.batch_size * self.quantizer.infer_bs_coeff,
                )
                for h in hook_handles:
                    h.remove()

                if len(self.compress_context.device_list) > 1:
                    accelerate.hooks.remove_hook_from_submodules(block)

                if self.compress_context.low_gpu_mem_usage:
                    block.to("cpu")
                    self.compress_context.clear_memory()

                # ── Pure algorithm ────────────────────────────────────────────
                self.quantizer.quantize_block(block, block_input_ids, input_others, block_name=block_name)

                # ── Infrastructure: cleanup ───────────────────────────────────
                mv_module_from_gpu(block)

                if self.compress_context.low_cpu_mem_usage and not self.compress_context.is_immediate_saving:
                    self._offloader(self.model_context.model, block_name)
                if block_name == block_names[-1]:
                    clear_memory(input_ids, device_list=self.compress_context.device_list)
                else:
                    clear_memory(device_list=self.compress_context.device_list)

                memory_monitor.log_summary()
                pbar.update(1)
        pbar.close()
        # Process remaining layers not in blocks
        # Collect names of quantizable layers not belonging to any block
        remain_layer_names = []
        block_name_set = set(name for block in all_blocks for name in block)
        for n, m in self.model_context.model.named_modules():
            if not check_to_quantized(m):
                continue
            # Skip if this layer is part of any block (by prefix match)
            if any(n == block_name or n.startswith(f"{block_name}.") for block_name in block_name_set):
                continue
            remain_layer_names.append(n)

        for name in remain_layer_names:
            dtype = None
            if self.super_group_size is not None:
                dtype = torch.float32
            self.quantizer.quantize_layer_outside_block(name, dtype=dtype)
            # clear_memory(device_list=self.compress_context.device_list)
        # if self.compress_context.is_immediate_saving:
        #     shard_writer(self, is_finalize=True)

    def _quant_rtn_with_imatrix(self) -> None:
        """Performs RTN quantization using input activation statistics (imatrix).

        OptimizedRTNQuantizer owns imatrix hook registration. This method only
        enables the quantizer-side collection path and keeps the OOM fallback.

        Returns:
            None
        """
        logger.info("start to compute imatrix")
        self.quantizer.enable_imatrix = True

        # Dataloader resolution is owned by ``CalibrationState``.
        self._calibration_state.ensure_dataloader(self.model_context, self.seed)

        model = self.model_context.model

        # Dispatch multi-GPU model if necessary
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            dispatch_model(model, model.hf_device_map)

        hooks = self.quantizer.register_calibration_hooks(model, act_max=False)
        try:
            if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                import accelerate

                accelerate.hooks.remove_hook_from_submodules(model)
            safe_to_cpu_(model)
            clear_memory(device_list=self.compress_context.device_list)
            self._quantize_via_rtn_blockwise()
        except torch.OutOfMemoryError:
            cuda_error_msg = traceback.format_exc()
            try:
                logger.error(cuda_error_msg)
                # Final fallback: warn and use CPU-only quantization
                logger.warning(
                    "Fallback to CPU. "
                    "Consider enabling `low_gpu_mem_usage` or using more GPUs via `--device 0,1,2,3`."
                )
                safe_to_cpu_(model)
                clear_memory(device_list=self.compress_context.device_list)
                if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                    import accelerate

                    accelerate.hooks.remove_hook_from_submodules(model)

                orig_device = self.compress_context.device
                self.compress_context.device = "cpu"
                self._quantize_via_rtn_blockwise()
                self.compress_context.device = orig_device
            except Exception as e:
                raise
        finally:
            for hook in hooks:
                hook.remove()
            self.quantizer.enable_imatrix = False

    def quantize(self):
        """Quantize all modules in the model.

        Returns:
            tuple[nn.Module, Dict[str, Any]]: The quantized model and the layer configuration.
        """
        # post_init must be called OUTSIDE @torch.inference_mode() because
        # AutoScheme delta-loss selection requires autograd (backward pass).
        self.post_init()
        return self._quantize_impl()

    # Use no_grad instead of inference_mode
    # https://github.com/intel/auto-round/issues/1620
    @torch.no_grad()
    def _quantize_impl(self):

        formats = getattr(self, "formats", None) or []
        if self.super_bits is None:
            self._quantize_embedding_layer()

        # Release memory
        clear_memory(device_list=self.compress_context.device_list)

        enable_imatrix = False
        if not getattr(self, "disable_opt_rtn", True):
            formats = getattr(self, "formats", None) or []
            has_gguf_k = self.super_bits is not None
            if has_gguf_k:
                enable_imatrix = True
            elif self.data_type == "int" and self.sym and self.bits < 8:
                enable_imatrix = True

        if enable_imatrix:
            self._quant_rtn_with_imatrix()
        else:
            self._quantize_via_rtn_blockwise()

        convert_module_to_hp_if_necessary(
            self.model_context.model,
            self.model_context.amp_dtype,
            self.compress_context.device,
        )
        if self.compress_context.low_cpu_mem_usage:
            self._offloader.reload(self.model_context.model)
        if self.compress_context.is_immediate_saving:
            self.shard_writer.write(is_finalize=True)

        self.model_context.quantized = True
        return self.model_context.model, self.quantizer.layer_config
