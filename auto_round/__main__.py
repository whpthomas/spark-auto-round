# Copyright (c) 2024 Intel Corporation
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
"""CLI for spark-auto-round: W4A16 quantization on CUDA (DGX Spark GB10)."""
import argparse
import json
import os
import re
import sys

import torch

from auto_round.schemes import PRESET_SCHEMES
from auto_round.utils import (
    clear_memory,
    get_device_and_parallelism,
    parse_layer_config_arg,
)


class BasicArgumentParser(argparse.ArgumentParser):
    """Minimal argument parser for W4A16 quantization on GB10."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "model",
            default=None,
            nargs="?",
            help="Path to the pre-trained model or model identifier from huggingface.co/models.",
        )
        basic = self.add_argument_group("Basic Arguments")
        basic.add_argument("--model_dtype", default=None, help="Model dtype used to load the pre-trained model")
        basic.add_argument(
            "--batch_size", "--bs", default=8, type=int, help="Batch size for tuning/calibration."
        )
        basic.add_argument(
            "--iters", default=1000, type=int, help="Number of iterations to tune each block."
        )
        basic.add_argument(
            "--seqlen", default=2048, type=int, help="Sequence length of the calibration samples."
        )
        basic.add_argument(
            "--nsamples", default=512, type=int, help="Number of calibration samples to use."
        )
        basic.add_argument(
            "--group_size", default=128, type=int, help="Group size for weight quantization."
        )
        basic.add_argument(
            "--dataset", default="github-code-clean", type=str,
            help="Calibration dataset. Available: github-code-clean, pile-10k, "
                 "pile-val-backup, CCI3-HQ, ultrachat_200k, "
                 "openbmb/Ultra-FineWeb, new-title-chinese, mbpp, audiocaps, "
                 "local (loads from local directory)."
        )
        basic.add_argument(
            "--output_dir", default="./models", type=str, help="Directory to save the quantized model."
        )
        basic.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
        basic.add_argument(
            "--adam", action="store_true", help="Use Adam optimizer instead of SignSGD."
        )
        basic.add_argument(
            "--disable_torch_compile",
            action="store_true",
            help="Disable torch.compile (enabled by default).",
        )
        basic.add_argument(
            "--memory_utilization",
            default=75,
            type=int,
            help="Memory utilization threshold (50-95). Models using more than "
                 "this percentage of available memory trigger block-by-block "
                 "offloading to disk. Default: 75.",
        )
        basic.add_argument(
            "--mllm",
            action="store_true",
            help="Force multimodal mode (auto-detected by default).",
        )

        tuning = self.add_argument_group("Tuning Arguments")
        tuning.add_argument(
            "--lr",
            default=None,
            type=float,
            help="Learning rate (optional, auto-calculated as 1.0/iters).",
        )
        tuning.add_argument(
            "--minmax_lr",
            default=None,
            type=float,
            help="MinMax learning rate (optional, uses --lr).",
        )

        scheme = self.add_argument_group("Scheme Arguments")
        scheme.add_argument(
            "--quant_lm_head",
            action="store_true",
            help="Quantize the lm_head.",
        )
        scheme.add_argument(
            "--ignore_layers",
            default="",
            type=str,
            help="Layers to skip quantization, separated by commas.",
        )
        scheme.add_argument(
            "--layer_config",
            default=None,
            type=str,
            help="Per-layer quantization config JSON string. "
            'Example: \'{"mtp": {"bits": 8, "data_type": "int"}}\'.',
        )


def start():
    """Parse arguments and run quantization."""
    parser = BasicArgumentParser()
    args = parser.parse_args()
    tune(args)


def tune(args):
    """Run the quantization pipeline with hardcoded W4A16 defaults for GB10."""
    assert args.model, "[model] positional argument must be set."

    from auto_round.utils import detect_device, get_library_version, logger
    from auto_round.version import __version__

    logger.info(
        f"Spark Auto Round version {__version__}\n"
        f"  --batch_size {args.batch_size}\n"
        f"  --nsamples {args.nsamples}\n"
        f"  --seqlen {args.seqlen}\n"
        f"  --group_size {args.group_size}\n"
        f"  --iters {args.iters}\n"
        f"  --dataset {args.dataset}\n"
        f"  --output_dir {args.output_dir}"
    )

    # --- Hardcoded values for GB10 ---
    device_map = "cuda:0"
    format = "auto_round"
    scheme = "W4A16"
    platform = "hf"
    enable_torch_compile = not args.disable_torch_compile

    # --- Clamp memory_utilization to safe range ---
    memory_utilization = max(50, min(95, args.memory_utilization)) / 100.0

    # Validate scheme
    if scheme not in PRESET_SCHEMES:
        raise ValueError(f"{scheme} is not supported. Supported: {list(PRESET_SCHEMES.keys())}")

    if enable_torch_compile:
        logger.info(
            "`torch.compile` is enabled to reduce tuning costs. "
            "Disable with --disable_torch_compile if it causes issues."
        )

    model_name = args.model
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    logger.info(f"start to quantize {model_name}")

    # --- Estimate memory strategy BEFORE model load ---
    from auto_round.utils.device import estimate_memory_strategy, log_memory_analysis
    use_offload, memory_info = estimate_memory_strategy(
        model_name, memory_utilization=memory_utilization
    )
    log_memory_analysis(memory_info, memory_utilization)

    # If model exceeds memory threshold, load on meta device (zero memory)
    # and load blocks on demand during quantization.
    use_meta_device = use_offload

    from auto_round import AutoRound

    from auto_round.compressors import (
        ExtraConfig,
        MLLMExtraConfig,
        SchemeExtraConfig,
        TuningExtraConfig,
    )

    extra_config = ExtraConfig()

    # Tuning config with hardcoded values
    tuning_config = TuningExtraConfig(
        amp=True,
        disable_opt_rtn=None,
        enable_alg_ext=False,
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        enable_quanted_input=True,
        enable_deterministic_algorithms=False,
        lr=args.lr,
        minmax_lr=args.minmax_lr,
        nblocks=1,
        to_quant_block_names=None,
        scale_dtype="bf16",
    )

    # Scheme config with hardcoded W4A16 values
    layer_config = {}
    if args.layer_config:
        layer_config = parse_layer_config_arg(args.layer_config)

    scheme_config = SchemeExtraConfig(
        bits=4,
        group_size=args.group_size,
        sym=True,
        data_type="int",
        act_bits=16,
        act_group_size=None,
        act_data_type="int",
        act_dynamic=None,
        act_sym=None,
        super_bits=None,
        super_group_size=None,
        quant_lm_head=args.quant_lm_head,
        ignore_layers=args.ignore_layers,
        static_kv_dtype=None,
        static_attention_dtype=None,
    )

    mllm_config = MLLMExtraConfig(
        quant_nontext_module=False, extra_data_dir=None, template=None
    )

    extra_config.tuning_config = tuning_config
    extra_config.scheme_config = scheme_config
    extra_config.mllm_config = mllm_config

    # AutoRound with hardcoded values
    autoround = AutoRound(
        model=model_name,
        platform=platform,
        format=format,
        scheme=scheme,
        dataset=args.dataset,
        iters=args.iters,
        seqlen=args.seqlen,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        gradient_accumulate_steps=1,
        low_gpu_mem_usage=False,
        low_cpu_mem_usage=use_offload,
        device_map=device_map,
        enable_torch_compile=enable_torch_compile,
        seed=args.seed,
        not_use_best_mse=False,
        enable_adam=args.adam,
        extra_config=extra_config,
        layer_config=layer_config,
        model_dtype=args.model_dtype,
        momentum=0,
        trust_remote_code=True,
        rotation_config=None,
        algorithm=None,
        use_meta_device=use_meta_device,
    )

    # Quantize and save
    model, folders = autoround.quantize_and_save(args.output_dir, format=format)
    tokenizer = autoround.tokenizer
    clear_memory()


def run():
    """CLI entry point."""
    start()


if __name__ == "__main__":
    run()
