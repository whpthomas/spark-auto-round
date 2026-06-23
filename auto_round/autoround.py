# Copyright (c) 2025 Intel Corporation
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
"""AutoRound — public API entry point for W4A16 quantization.

This module provides the ``AutoRound()`` factory function, which is the
recommended entry point for quantizing LLMs.  It delegates to
``DataDrivenCompressor`` via the internal ``auto_round_factory()`` function.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import torch

from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme

if TYPE_CHECKING:
    from auto_round.compressors.data_driven import DataDrivenCompressor
    from auto_round.compressors.config import ExtraConfig


def AutoRound(
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
    # enable_adam removed — Adam optimizer not supported in this fork
    extra_config: "ExtraConfig" = None,
    enable_alg_ext: bool = False,
    disable_opt_rtn: bool | None = None,
    low_cpu_mem_usage: bool = True,
    auto_tuner_steps: list | None = None,
    **kwargs,
) -> "DataDrivenCompressor":
    """Create a quantizer for W4A16 LLM quantization.

    This is a factory function that returns a ``DataDrivenCompressor`` instance.
    Despite the class-like name, this is a function — calling ``isinstance(obj, AutoRound)``
    will not work.  Use ``isinstance(obj, DataDrivenCompressor)`` instead.

    Args:
        model: Model object or model name/path to load.
        tokenizer: Tokenizer for text processing.
        platform: Platform to download model ("hf" or "model_scope").
        scheme: Quantization scheme (str, dict, or QuantizationScheme).
        layer_config: Layer-wise quantization config.
        dataset: Calibration data.
        iters: Optimization iterations.
        seqlen: Calibration sequence length.
        nsamples: Number of calibration samples.
        batch_size: Calibration batch size.
        gradient_accumulate_steps: Gradient accumulation steps.
        low_gpu_mem_usage: Lower GPU memory mode.
        device_map: Device map for each module.
        enable_torch_compile: Enable torch.compile.
        seed: Random seed.
        tuning_profile: Optional dict of tuning metadata for checkpoint.
            Contains relaxation_step, oom_count, settings_active. Set to
            None for fresh runs; populated from progress.json on resume.
        extra_config: Extra configuration object.
        enable_alg_ext: Enable algorithm extension.
        disable_opt_rtn: Disable RTN optimization.
        low_cpu_mem_usage: Lower CPU memory mode.
        auto_tuner_steps: Optional list of auto-tuner adjustment steps from
            the pre-flight tuning. Used to display auto-tuner summary at
            end-of-run. Defaults to None (no summary shown).

    Returns:
        DataDrivenCompressor: Configured compressor ready for quantization.
    """
    from auto_round.compressors.entry import auto_round_factory

    return auto_round_factory(
        model=model,
        tokenizer=tokenizer,
        platform=platform,
        scheme=scheme,
        layer_config=layer_config,
        dataset=dataset,
        iters=iters,
        seqlen=seqlen,
        nsamples=nsamples,
        batch_size=batch_size,
        gradient_accumulate_steps=gradient_accumulate_steps,
        low_gpu_mem_usage=low_gpu_mem_usage,
        device_map=device_map,
        enable_torch_compile=enable_torch_compile,
        seed=seed,
        extra_config=extra_config,
        enable_alg_ext=enable_alg_ext,
        disable_opt_rtn=disable_opt_rtn,
        low_cpu_mem_usage=low_cpu_mem_usage,
        auto_tuner_steps=auto_tuner_steps,
        **kwargs,
    )


@torch.no_grad()
def _sampling_inputs(
    input_ids: list[torch.Tensor],
    input_others: dict,
    indices: list[int],
    seqlen: int,
    batch_dim: int = 0,
    share_cache_keys: tuple = (),
):
    """Samples inputs based on the given indices and sequence length.

    Args:
    input_ids: The list of input tensor containing  input_ids.
    input_others: A dictionary containing other input data.
    indices: The indices to sample from the input.
    seqlen: The sequence length.

    Returns:
    current_input_ids: The sampled input IDs.
    current_input_others: The sampled other input data.
    """
    current_input_ids = [input_ids[i] for i in indices]

    current_input_ids = torch.cat(current_input_ids, dim=batch_dim)

    current_input_others = {"positional_inputs": input_others["positional_inputs"]}
    for key in input_others.keys():
        if "positional_inputs" in key:
            continue
        if key in share_cache_keys or isinstance(input_others[key], (str, bool, type(None))):
            current_input_others[key] = input_others[key]
        elif input_others[key] is not None:
            current_input_others[key] = [input_others[key][i] for i in indices]
            if len(indices) == 1:
                current_input_others[key] = current_input_others[key][0]
            else:
                try:
                    current_input_others[key] = torch.cat(current_input_others[key], dim=0)
                except TypeError as err:
                    logger.warning_once("Please check the model cache inputs or try setting batch_size to 1.")
        else:
            current_input_others[key] = None

    return current_input_ids, current_input_others
