# AGENTS.md

## What this is

Fork of [Intel auto-round](https://github.com/intel/auto-round) trimmed to CUDA + `torch.compile` + W4A16 quantization for **DGX Spark GB10 (128 GiB unified memory)**. Not a general-purpose quantization toolkit — most upstream features are stubbed, commented out, or removed.

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
uv pip install -e .          # editable install
uv pip install pytest        # test runner (not in install_requires)
```

Optional test extras live in `test/test_cuda/requirements.txt` (gptqmodel, lm_eval, sentencepiece, etc.).

## Key commands

| Task | Command |
|------|---------|
| CLI entry point | `spark-auto-round <model> --output_dir ./models` |
| Dry-run (validates setup) | `spark-auto-round <model> --dry-run` |
| All tests (needs CUDA + HF models) | `pytest test/ -v` |
| CUDA-specific tests only | `pytest test/test_cuda/ -v` |
| Single test | `pytest test/test_cuda/quantization/test_asym.py::TestAutoRoundAsym::test_asym_group_size -v` |
| Filter by keyword | `pytest -k "torch_compile" -v` |

**No linting, typecheck, formatter, or CI workflows are configured.** There is no `pyproject.toml`, `Makefile`, or `.github/` directory.

## Architecture

```
auto_round/              # Main package (installed as `spark_auto_round`)
├── __main__.py          # CLI: spark-auto-round entrypoint → start() → tune()
├── __init__.py          # Exports AutoRound; calls monkey_patch() on import
├── autoround.py         # AutoRound facade — delegates to AutoRoundCompatible via __new__
├── schemes.py           # QuantizationScheme dataclass + W4A16 preset
├── formats.py           # OutputFormat (auto_round, fake)
├── compressors/         # Core: entry.py (router), data_driven.py, zero_shot.py
├── algorithms/          # sign_round (SignSGD)
├── data_type/           # INT/FP quantization kernels
├── export/              # Export to auto_round format (export_to_autoround/)
├── calibration/         # LLM calibration data loading
├── modeling/            # Multimodal/MoE layer handling
├── special_model_handler.py  # Model-specific logic (Gemma 4, Qwen, DeepSeek, etc.)
├── envs.py              # Env vars (AR_LOG_LEVEL, AR_DYNAMO_CACHE_SIZE_LIMIT, etc.)
├── wrapper.py           # WrapperLinear for tuning
├── cli_display.py       # CLI progress bar and sensitivity lines
├── metrics.py           # Quantization metrics (PSNR, cosine similarity)
├── report.py            # QuantizationReport (per-layer pass/warn/fail)
├── logger.py            # Logging setup
├── asqa/                # ASAQ layer substitution utility
│   ├── __init__.py      # Package exports
│   ├── __main__.py      # CLI: spark-asqa-substitute entrypoint
│   └── substitute.py    # Core substitution engine
└── utils/               # Device detection, model loading, monkey patches

auto_round_extension/    # Low-level quantization kernels
├── cuda/                # Marlin/GPTQ kernels
├── triton/              # Triton quantized linear layers
└── torch/               # Pure-torch quantized linear layers

test/
├── conftest.py          # Test configuration (adds parent to sys.path)
├── fixtures.py          # Session-scoped tiny model fixtures
├── helpers.py           # get_model_path(), model_infer(), DataLoader
├── envs.py              # Test decorators (require_gptqmodel, etc.)
├── test_asqa/           # ASAQ unit tests (no GPU needed)
│   ├── __init__.py
│   └── test_substitute.py
└── test_cuda/           # CUDA tests (quantization, algorithms, packing)
    └── test_asqa_integration.py  # ASAQ integration tests (needs CUDA + model)
```

## Important gotchas

- **`monkey_patch()` runs at import time** when you do `from auto_round import AutoRound`. This patches transformers internals. Don't import auto_round for side effects in test scaffolding.
- **Only `auto_round:auto_gptq` (aka `auto_round`) output format** is actually supported. Don't try marlin, gguf, mlx, etc.
- **Test fixtures download models from HuggingFace** (session-scoped). First run requires network access. Models are tiny (2-layer slices) and saved to `test/tmp/`.
- **Default CLI recipe**: `--scheme W4A16`, `--iters 1000`, `--nsamples 512`, `--seqlen 2048`, `--batch_size 8`, `--dataset github-code-clean`. torch.compile is enabled by default.
- **The `spark-auto-round` shortcut** (from setup.py entry_points) calls `auto_round.__main__:run`.

## CLI defaults

| Argument | Default | Description |
|----------|---------|-------------|
| `model` | (required) | Model path or HuggingFace model ID |
| `--group_size` | 128 | Group size for weight quantization |
| `--iters` | 1000 | Tuning iterations per block |
| `--nsamples` | 512 | Number of calibration samples |
| `--seqlen` | 2048 | Calibration sequence length |
| `--batch_size` | 8 | Calibration batch size |
| `--output_dir` | ./models | Output directory |
| `--dataset` | github-code-clean | Calibration dataset |
| `--disable_torch_compile` | (disabled) | Disable torch.compile (enabled by default) |
| `--memory_utilization` | 75 | Memory threshold (50-95). Models exceeding this trigger block-by-block offloading to disk. |
| `--mllm` | false | Force multimodal mode (auto-detected by default). |

Hardcoded values: device=cuda:0, format=auto_round, scheme=W4A16, platform=hf, scale_dtype=bf16, amp=True, minmax_tuning=True, norm_bias_tuning=False, quanted_input=True, not_use_best_mse=False.

## Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AR_LOG_LEVEL` | INFO | Logging verbosity |
| `AR_DYNAMO_CACHE_SIZE_LIMIT` | 16 | torch._dynamo cache size limit (bumped to cover all distinct linear shapes per block) |
| `AR_DISABLE_OFFLOAD` | 0 | Disable block-by-block offloading |
| `AR_DISABLE_DATASET_SUBPROCESS` | 0 | Load dataset in main process |
| `AR_ENABLE_COMPILE_PACKING` | 0 | Enable compiled packing kernels |

## ASAQ Layer Substitution

The `spark-asqa-substitute` utility takes a working quantized model (all W4A16) and substitutes specified layers back to FP16. This produces an "ASAQ" copy for A/B evaluation.

**Usage**:
```bash
spark-asqa-substitute --layers 54,58 Qwen/Qwen3.6-27B
```

**Arguments**:
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `model` | Yes | — | Model name (positional) |
| `--layers` | Yes | — | Comma-separated layer indices (e.g. `54,58`) |
| `--output_dir` | No | `./models/{name}-int4-asaq` | Output directory override |
| `--no-smoke-test` | No | (test runs) | Skip inference smoke test |

**Path inference** (given `model = "Qwen/Qwen3.6-27B"`):
- Quantized model: `./models/Qwen3.6-27B-int4-AutoRound`
- FP16 model: `Qwen/Qwen3.6-27B` (from HuggingFace cache)
- Output dir: `./models/Qwen3.6-27B-int4-asaq`

**Output**: Modified safetensors model + copied configs + updated quantization report.

**Code location**: `auto_round/asqa/` package.

## Testing patterns

- **Top-level tests** (`test/test_*.py`): Pure Python, no GPU. CLI display, metrics, report.
- **ASAQ tests** (`test/test_asqa/`): No GPU. Core substitution logic, CLI parsing, report generation.
- **CUDA tests** (`test/test_cuda/`): Require CUDA GPU. Quantization, algorithms, packing, ASAQ integration.
- **Fixtures**: Session-scoped, auto-download tiny model slices from HuggingFace. Saved to `test/tmp/`, cleaned up after session.
- **`get_model_path()`**: Checks `/tf_dataset/auto_round/models/`, `/models/`, `/dataset/`, then falls back to HuggingFace name.
- **Test README** at `test/README.md` has detailed fixture/usage docs.

## Codebase origin

The cleanup from upstream auto-round is documented in `thoughts/01-lean-and-mean/design-brief.md` and the multi-phase pruning in `thoughts/02-prune/`. Consult them before re-enabling any upstream feature to understand what was changed and why. (Note: `thoughts/` is in `.gitignore`.)
