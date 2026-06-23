# AGENTS.md

## What this is

Fork of [Intel auto-round](https://github.com/intel/auto-round) trimmed to CUDA + `torch.compile` + W4A16 quantization for **DGX Spark GB10 (128 GiB unified memory)**. Not a general-purpose quantization toolkit — most upstream features are stubbed, commented out, or removed.

## Dev setup

```bash
source ~/spark-auto-round-venv/bin/activate
uv pip install -e .          # editable install
```

## Key commands

| Task | Command |
|------|---------|
| CLI entry point | `spark-auto-round <model> --output_dir ./models` |
| All tests (needs CUDA + HF models) | `pytest test/ -v` |
| CUDA-specific tests only | `pytest test/test_cuda/ -v` |
| Single test | `pytest test/test_cuda/quantization/test_asym.py::TestAutoRoundAsym::test_asym_group_size -v` |
| Filter by keyword | `pytest -k "torch_compile" -v` |
| Resume tests (fast, no GPU) | `pytest test/test_cuda/quantization/test_resume.py -v -m "not slow"` |
| Resume tests (GPU integration) | `pytest test/test_cuda/quantization/test_resume.py -v -m cuda` |
| CLI arg tests (no GPU) | `pytest test/test_cuda/quantization/test_resume.py::TestCliIntegration -v` |
| Auto-tuner tests (no GPU) | `pytest test/test_cuda/test_auto_tune.py -v -m "not slow"` |
| Memory estimator tests (no GPU) | `pytest test/test_cuda/test_memory_estimator.py -v -m "not slow"` |

**No linting, typecheck, formatter, or CI workflows are configured.** There is no `Makefile`, `.github/` directory, or pre-commit hooks.

## Architecture

```
auto_round/              # Main package (installed as `spark_auto_round`)
├── __main__.py          # CLI: spark-auto-round entrypoint → start() → tune()
├── __init__.py          # Exports AutoRound; calls monkey_patch() on import
├── autoround.py         # AutoRound() factory function → delegates to auto_round_factory()
├── schemes.py           # QuantizationScheme dataclass + W4A16 preset
├── formats.py           # OutputFormat (auto_round, fake)
├── envs.py              # Env vars (AR_LOG_LEVEL, AR_DYNAMO_CACHE_SIZE_LIMIT, etc.)
├── wrapper.py           # WrapperLinear, WrapperMultiblock for tuning
├── cli_display.py       # CLIDisplay: progress bar, sensitivity lines, auto-tune messages
├── metrics.py           # Quantization metrics (PSNR, cosine similarity)
├── report.py            # QuantizationReport (per-layer pass/warn/fail)
├── logger.py            # Logging setup
├── calib_dataset.py     # Dataset loading and preprocessing
├── special_model_handler.py  # Model-specific logic (Gemma 4, Qwen, DeepSeek, etc.)
├── version.py           # __version__ (dynamic from setuptools)
├── compressors/         # Core quantization engine
│   ├── entry.py         # auto_round_factory() + AutoRoundCompatible() (backward compat)
│   ├── data_driven.py   # DataDrivenCompressor — main quantization loop + checkpointing
│   ├── base.py          # BaseCompressor abstract class
│   ├── config.py        # SARConfig (SAR-specific configuration)
│   ├── auto_tune.py     # Memory-aware auto-tuner (relaxation ladder)
│   ├── memory_estimator.py  # Per-block peak memory estimation
│   ├── shard_writer.py  # Model shard writer for export
│   ├── utils.py         # Compressor utility functions
│   ├── mllm_mixin.py    # Multimodal model mixin for DataDrivenCompressor
│   └── mllm/            # Multimodal compression support
├── algorithms/          # Quantization algorithms
│   ├── quantization/    # sign_round (SignSGD), sign_roundv2
│   └── transforms/      # Rotation transforms, normalization
├── data_type/           # INT/FP quantization kernels (int.py, w4fp8.py)
├── export/              # Export to auto_round format (export_to_autoround/)
├── calibration/         # LLM/MLLM calibration data loading (dataset.py, processor.py)
├── context/             # Compression context (model.py, compress.py, base.py)
├── modeling/            # Multimodal/MoE layer handling (fused_moe/, unfused_moe/)
├── asqa/                # ASAQ layer substitution utility
│   ├── __main__.py      # CLI: spark-asqa-substitute entrypoint
│   ├── router_jaccard.py  # Router Jaccard Similarity for MOE models
│   └── substitute.py    # Core substitution engine
└── utils/               # Device detection, model loading, monkey patches
    ├── device/          # Device detection, memory estimation, patches
    ├── model/           # Model loading, detection, slicing
    └── common.py        # monkey_patch_transformers(), monkey_patch()

auto_round_extension/    # Low-level quantization kernels
├── cuda/                # Marlin/GPTQ kernels (gptqmodel_marlin.py)
├── triton/              # Triton quantized linear layers (qlinear_tritonv2.py)
└── torch/               # Pure-torch quantized linear layers (qlinear_torch.py)

test/
├── conftest.py          # Test configuration (adds parent to sys.path)
├── fixtures.py          # Session-scoped tiny model fixtures
├── helpers.py           # get_model_path(), model_infer(), DataLoader, get_tiny_model()
├── envs.py              # Test decorators (require_gptqmodel, etc.)
├── test_cli_display.py  # CLI display tests (no GPU)
├── test_cli_integration.py  # CLI integration tests (no GPU)
├── test_dry_run_unit.py # Dry-run unit tests (no GPU)
├── test_metrics.py      # Quantization metrics tests (no GPU)
├── test_report.py       # QuantizationReport tests (no GPU)
├── test_shard_writer.py # Shard writer tests (no GPU)
├── test_qwen3_5_regression.py  # Qwen 3.5 regression tests
├── test_asqa/           # ASAQ unit tests (no GPU)
│   ├── test_router_jaccard.py
│   └── test_substitute.py
├── test_utils/          # Utility tests (no GPU)
│   └── test_revert_checkpoint.py
└── test_cuda/           # CUDA tests (needs GPU)
    ├── quantization/    # test_asym.py, test_packing.py, test_torch_compile.py, test_resume.py
    ├── algorithms/      # (empty, reserved)
    ├── test_auto_tune.py  # Auto-tuner tests
    ├── test_memory_estimator.py  # Memory estimator tests
    ├── test_memory_strategy.py
    ├── test_cli_integration_full.py  # Full CLI integration tests
    ├── test_asqa_integration.py
    ├── test_asqa_e2e.py
    ├── test_dry_run.py
    ├── test_meta_device.py
    ├── test_offloader_modes.py
    ├── test_qwen3_5_export.py
    └── requirements.txt

### Checkpoint & Resume (.cache/)

| Aspect | Details |
|--------|---------|
| Storage | `{output_dir}/.cache/progress.json` + `block_NNNNN.pt` files |
| Save | After each block completes quantization (inside `_quantize_blocks()` inner loop) |
| Resume | Automatic — detected by presence of valid `.cache/progress.json` |
| Cleanup | Removed on successful completion, preserved on crash/interrupt |
| Force fresh | `spark-auto-round --clear-cache ...` |
| Reliable | Atomic writes (`progress.json.tmp` → rename), no optimizer state saved |

**Checkpoint lifecycle:**
```
quantize() start
  ├── _check_resume_state() → (resume_mode, completed, total, names)
  │     ├── No .cache/ → fresh start
  │     ├── Corrupt → warning, fresh start
  │     └── Valid → load completed blocks, skip in loop
  ├── [For each remaining block]
  │     ├── _quantize_blocks(block_idx=i, ...)
  │     └── _save_checkpoint(i, name, module)
  │           ├── block_{i:05d}.pt  (state dict, CPU tensors)
  │           └── progress.json     (atomic write)
  └── [On success]
        └── _clear_cache() → remove .cache/
```

**Key methods** (on `DataDrivenCompressor`):
- `_checkpoint_dir` — property returning `.cache/` path
- `_check_resume_state()` — returns `(bool, int, int, list)`
- `_save_checkpoint(block_idx, block_name, module)` — save state dict + progress
- `_save_checkpoint_progress(completed)` — atomic progress.json write
- `_load_checkpoint_block(block_idx, block_name, model)` — load block from disk
- `_checkpoint_block_path(block_idx)` — returns full path to block file
- `_clear_cache()` — remove `.cache/` with safety check
- `_check_and_clear_cache_flag()` — handle `--clear-cache` flag

**Edge cases handled:**
- Missing/empty `.cache/` → fresh start
- Corrupt `progress.json` → fresh start
- Missing block files → fresh start
- `completed > total` → fresh start
- KeyboardInterrupt → `.cache/` preserved for resume
- Exception → `.cache/` preserved for resume
- `--clear-cache` on non-existent dir → no-op
- `--clear-cache` on symlink → removes symlink only
- `nblocks > 1` → checkpointing disabled with warning
- Meta device → materialize, load state dict, re-offload
```

