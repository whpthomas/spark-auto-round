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
from pathlib import Path

import torch
from transformers import AutoConfig

from auto_round.compressors.auto_tune import (
    auto_tune,
    format_preflight_message,
    format_resume_message,
)
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
            "--dataset", default="opencode-instruct", type=str,
            help="Calibration dataset. Available: opencode-instruct, github-code-clean, "
                 "pile-10k, pile-val-backup, CCI3-HQ, ultrachat_200k, "
                 "openbmb/Ultra-FineWeb, new-title-chinese, mbpp, audiocaps, "
                 "local (loads from local directory)."
        )
        basic.add_argument(
            "--output_dir", default="./models", type=str, help="Directory to save the quantized model."
        )
        basic.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
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
            "--memory_budget",
            default=96,
            type=int,
            help="Per-block memory budget in GiB for the auto-tuner. "
                 "Default: 96 (75%% of 128 GiB). Max: 120. "
                 "Sets a hard ceiling on estimated peak memory per block.",
        )
        basic.add_argument(
            "--trust-remote-code",
            action="store_true",
            default=True,
            help="Trust remote code when loading models (default: True). "
                 "Disable with --no-trust-remote-code for security.",
        )
        basic.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Run the full init pipeline except quantization tuning; "
                 "write config files only for inspection.",
        )
        basic.add_argument(
            "--clear-cache",
            action="store_true",
            default=False,
            help="Delete checkpoint cache (.cache/) before starting quantization. "
                 "Use this to force a fresh run if the existing checkpoint is stale or corrupt.",
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
    if args.dry_run:
        logger.info("  ** DRY-RUN mode: will skip quantization, write config files only **")
    if args.clear_cache:
        logger.info("  --clear-cache: will delete existing checkpoint cache before starting")
        # Optionally, verify output_dir exists for safety
        if os.path.isdir(args.output_dir):
            cache_path = os.path.join(args.output_dir, ".cache")
            if os.path.isdir(cache_path):
                logger.info("  Found existing .cache/ at %s \u2014 will be removed.", cache_path)

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

    # -------------------------------------------------------------------
    # Auto-tune: adjust per-block settings if peak memory exceeds budget
    # -------------------------------------------------------------------
    # Load model config for the estimator (no full model load needed)
    try:
        model_config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=args.trust_remote_code
        )
    except Exception as exc:
        logger.warning("Could not load model config for auto-tuner: %s", exc)
        model_config = None

    if model_config is not None:
        # Budget from --memory-budget flag (direct GiB ceiling)
        budget_bytes = min(args.memory_budget, 120) * (1024 ** 3)
        if args.memory_budget < 16:
            logger.warning(
                "Very low --memory_budget (%d GiB) — aggressive relaxations "
                "will be applied.", args.memory_budget
            )

        # Build user_settings dict for the auto-tuner
        user_settings = {
            "batch_size": args.batch_size,
            "seqlen": args.seqlen,
            "nsamples": args.nsamples,
            "iters": args.iters,
            "group_size": args.group_size,
        }

        # Check for resume context from checkpoint
        resume_context = None
        resume_mode = False
        completed = 0
        cache_dir = Path(args.output_dir) / ".cache"
        progress_path = cache_dir / "progress.json"
        if progress_path.exists():
            try:
                with open(progress_path, "r") as f:
                    progress = json.load(f)
                stored_exit_reason = progress.get("exit_reason")
                tuning_profile = progress.get("tuning_profile")
                stored_block_names = progress.get("block_names", [])
                stored_total = progress.get("total", 0)
                stored_completed = progress.get("completed", 0)

                if stored_completed > 0 and stored_completed < stored_total:
                    resume_mode = True
                    completed = stored_completed
                    resume_context = {
                        "exit_reason": stored_exit_reason,
                        "oom_count": tuning_profile.get("oom_count", 0) if tuning_profile else 0,
                        "tuning_profile": tuning_profile,
                    }
            except (json.JSONDecodeError, OSError):
                pass

        # Run auto-tuner with budget_bytes
        adjusted_settings, tune_steps = auto_tune(
            user_settings=user_settings,
            model_config=model_config,
            budget_bytes=budget_bytes,
            resume_context=resume_context,
        )

        # Compute peak for display
        from auto_round.compressors.memory_estimator import estimate_peak_memory_per_block
        peak_gb, _ = estimate_peak_memory_per_block(
            model_config, adjusted_settings,
        )

        # Display message
        if resume_mode:
            oom_count = resume_context.get("oom_count", 0) if resume_context else 0
            msg = format_resume_message(
                completed=completed,
                total=stored_total if stored_total else 0,
                exit_reason=resume_context["exit_reason"],
                adjusted_settings=adjusted_settings,
                steps=tune_steps,
                peak_gb=peak_gb,
                budget_gb=budget_bytes / (1024 ** 3),
                oom_count=oom_count,
            )
        else:
            msg = format_preflight_message(
                user_settings=user_settings,
                adjusted_settings=adjusted_settings,
                steps=tune_steps,
                peak_gb=peak_gb,
                budget_gb=budget_bytes / (1024 ** 3),
            )
        print(msg)
        logger.info(msg)

        # Override args with adjusted settings
        args.batch_size = adjusted_settings.get("batch_size", args.batch_size)
        args.seqlen = adjusted_settings.get("seqlen", args.seqlen)
        args.nsamples = adjusted_settings.get("nsamples", args.nsamples)

        # Build tuning_profile for checkpoint metadata (Phase 3)
        tuning_profile = {
            "relaxation_step": len([s for s in tune_steps if not s.get("skipped")]),
            "oom_count": (resume_context.get("oom_count", 0) + 1)
                         if resume_context and resume_context.get("exit_reason") == "oom"
                         else 0,
            "settings_active": {
                "batch_size": args.batch_size,
                "seqlen": args.seqlen,
                "nsamples": args.nsamples,
            },
        }
    else:
        # If model_config could not be loaded, use unmodified settings
        adjusted_settings = {}
        tune_steps = []
        peak_gb = 0.0
        tuning_profile = None

    from auto_round import AutoRound

    from auto_round.compressors.config import SARConfig

    # Scheme config with hardcoded W4A16 values
    layer_config = {}
    if args.layer_config:
        layer_config = parse_layer_config_arg(args.layer_config)

    extra_config = SARConfig(
        # Tuning
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
        # Scheme
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

    # AutoRound with hardcoded values (and potentially adjusted settings)
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
        extra_config=extra_config,
        layer_config=layer_config,
        model_dtype=args.model_dtype,
        momentum=0,
        trust_remote_code=args.trust_remote_code,
        dry_run=args.dry_run,
        clear_cache=args.clear_cache,
        rotation_config=None,
        algorithm=None,
        use_meta_device=use_meta_device,
        tuning_profile=tuning_profile,
        auto_tuner_steps=tune_steps,
    )

    # Reset exit_reason for fresh start (state initialization, not duck-typing)
    if hasattr(autoround, '_exit_reason'):
        autoround._exit_reason = None

    # Quantize and save
    model, folders = autoround.quantize_and_save(args.output_dir, format=format)
    tokenizer = autoround.tokenizer
    clear_memory()


def run():
    """CLI entry point."""
    start()


if __name__ == "__main__":
    run()