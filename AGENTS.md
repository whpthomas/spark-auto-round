# AGENTS.md

## What this is

Fork of [Intel auto-round](https://github.com/intel/auto-round) trimmed to CUDA + `torch.compile` + W4A16 quantization for **DGX Spark GB10 (128 GiB unified memory)**. Not a general-purpose quantization toolkit — most upstream features are stubbed, commented out, or removed.

## Dev setup

```bash
python -m venv ~/spark-auto-round-venv && source ~/spark-auto-round-venv/bin/activate
uv pip install -e .          # editable install
uv pip install pytest        # test runner (not in install_requires)
```

Optional test extras: `uv pip install gptqmodel lm_eval` or see `test/test_cuda/requirements.txt`.

## Key commands

| Task | Command |
|------|---------|
| CLI entry point | `spark-auto-round <model> --output_dir ./models` |
| All tests (needs CUDA + HF models) | `pytest test/ -v` |
| CUDA-specific tests only | `pytest test/test_cuda/ -v` |
| Single test | `pytest test/test_cuda/quantization/test_asym.py::TestAutoRoundAsym::test_asym_group_size -v` |
| Filter by keyword | `pytest -k "torch_compile" -v` |

**No linting, typecheck, formatter, or CI workflows are configured.** There is no `Makefile`, `.github/` directory, or pre-commit hooks.

## Architecture

```
auto_round/              # Main package (installed as `spark_auto_round`)
├── __main__.py          # CLI: spark-auto-round entrypoint → start() → tune()
├── __init__.py          # Exports AutoRound; calls monkey_patch() on import
├── autoround.py         # AutoRound facade — delegates to AutoRoundCompatible via __new__
├── schemes.py           # QuantizationScheme dataclass + W4A16 preset
├── formats.py           # OutputFormat (auto_round, fake)
├── compressors/         # Core: entry.py (factory), data_driven.py, config.py (SARConfig)
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
│   ├── router_jaccard.py  # Router Jaccard Similarity for MOE models
│   └── substitute.py    # Core substitution engine
└── utils/               # Device detection, model loading, monkey patches

auto_round_extension/    # Low-level quantization kernels
├── cuda/                # Marlin/GPTQ kernels
├── triton/              # Triton quantized linear layers
└── torch/               # Pure-torch quantized linear layers

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
├── test_asqa/           # ASAQ unit tests (no GPU)
│   ├── test_router_jaccard.py
│   └── test_substitute.py
├── test_utils/          # Utility tests (no GPU)
│   └── test_revert_checkpoint.py
└── test_cuda/           # CUDA tests (needs GPU)
    ├── quantization/    # test_asym.py, test_packing.py, test_torch_compile.py
    ├── algorithms/      # (empty, reserved)
    ├── test_asqa_integration.py
    ├── test_dry_run.py
    ├── test_memory_strategy.py
    ├── test_meta_device.py
    ├── test_offloader_modes.py
    ├── test_qwen3_5_export.py
    └── requirements.txt
```

## Important gotchas

- **`monkey_patch()` runs at import time** when you do `from auto_round import AutoRound`. This patches transformers internals (notably `AutoModelForCausalLM`). Don't import auto_round for side effects in test scaffolding — `conftest.py` imports fixtures before auto_round for exactly this reason.
- **Only `auto_round:auto_gptq` (aka `auto_round`) output format** is actually supported. Don't try marlin, gguf, mlx, etc.
- **Test fixtures download models from HuggingFace** (session-scoped). First run requires network access. Models are tiny (2-layer slices) and saved to `test/tmp/`.
- **`get_tiny_model()` in `test/helpers.py`** supports `from_config=True` for fast config-only model creation (no download). Use this when you don't need real weights.
- **Default CLI recipe**: `--iters 1000`, `--nsamples 512`, `--seqlen 2048`, `--batch_size 8`, `--dataset github-code-clean`. torch.compile is enabled by default. Scheme is hardcoded to W4A16.
- **The `spark-auto-round` shortcut** (from pyproject.toml `[project.scripts]`) calls `auto_round.__main__:run`.
- **Two build files exist** (`pyproject.toml` and `setup.py`) with slightly different dependency versions. `pyproject.toml` is canonical: Python >= 3.10, torch >= 2.4.

## Avoiding the config-fix loop

This fork has repeatedly gotten stuck in a cycle: broken quantized configs → patch scripts → new divergences → more broken configs. The upstream `~/auto-round` codebase produces correct output. Almost every config bug in this fork traces back to diverging from upstream. Follow these rules to avoid the loop:

1. **Upstream is the reference.** If a function in `auto_round/` exists in `~/auto-round/auto_round/`, read the upstream version first. The upstream `revert_checkpoint_conversion_mapping` works correctly; the spark fork's "guard logic" broke it. Always diff against upstream before modifying shared code.

2. **Post-hoc fix scripts are red flags, not solutions.** `fix_revert.py`, `fix-v14.1-layer-prefix.py` — these patch symptoms at the file-system level instead of fixing the code that writes wrong configs. If you reach for a post-hoc script, stop and fix the source function instead. The `thoughts/10-regression/prompt.md` and `thoughts/12-config-fix/prompt.md` documents trace this exact trap.

3. **Duplicated logic drifts.** If `save_quantized()` and `_save_config_dry_run()` build the same config independently, they *will* diverge. Extract a shared helper (`_build_quantization_config()` in `base.py`). This was the root cause of the config divergence traced in `thoughts/13-review/design.md`.

4. **Verify before you ship.** Use `--dry-run` to check config files in seconds instead of waiting 90 minutes for a full quantize. Compare the dry-run output against upstream's output (`~/models/Qwen3.5-0.8B-w4g128/` is a known-good reference). The `thoughts/11-dry-run/prompt.md` documents the exact debug workflow.

5. **Understand before you patch.** The "double-conversion guard" added to `revert_checkpoint_conversion_mapping` was dead logic — it stripped `\d+` from target patterns but not backreferences, so the condition was almost always true. Adding complexity to code you don't fully understand usually makes things worse. Read the upstream implementation, understand *why* it works, then decide if your fork needs changes.

6. **Config JSON is the contract.** The output files (`config.json`, `quantization_config.json`, `model.safetensors.index.json`) are what vLLM and other consumers read. Treat wrong values in these files as correctness bugs, not cosmetic issues. The `thoughts/10-regression/prompt.md` documents the exact vLLM errors caused by wrong configs.

7. **One fix at a time.** When multiple things are broken, fix them sequentially and verify after each change. Phase 1 → Phase 2 → Phase 3 (as documented in `thoughts/13-review/`). Bundling multiple changes makes it impossible to identify which fix caused a regression.

**Key reference documents** (in `thoughts/`, gitignored):
- `thoughts/01-lean-and-mean/design-brief.md` — what was pruned and why
- `thoughts/10-regression/prompt.md` — the Qwen3.5 config regression symptoms
- `thoughts/11-dry-run/prompt.md` — how to debug configs without full quantize
- `thoughts/12-config-fix/prompt.md` — the iteration workflow
- `thoughts/13-review/design.md` — root cause analysis and fix plan
- `thoughts/13-review/research.md` — verified findings across 14 issues
- `thoughts/13-review/report-{1,2,3}.md` — phased implementation results

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

Hardcoded values: device=cuda:0, format=auto_round, scheme=W4A16, platform=hf, scale_dtype=bf16, amp=True, minmax_tuning=True, norm_bias_tuning=False, quanted_input=True, not_use_best_mse=False.

Additional CLI args: `--dry-run` (skip quantization, write config files only), `--trust-remote-code` (default True, disable with `--no-trust-remote-code`).

## Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AR_LOG_LEVEL` | INFO | Logging verbosity |
| `AR_DYNAMO_CACHE_SIZE_LIMIT` | 16 | torch._dynamo cache size limit (bumped to cover all distinct linear shapes per block) |
| `AR_DISABLE_OFFLOAD` | 0 | Disable block-by-block offloading |
| `AR_DISABLE_DATASET_SUBPROCESS` | 0 | Load dataset in main process |
| `AR_ENABLE_COMPILE_PACKING` | 0 | Enable compiled packing kernels |
| `AR_WORK_SPACE` | ar_work_space | Working directory for intermediate files |
| `AR_DISABLE_COPY_MTP_WEIGHTS` | 0 | Disable MTP weight copying |
| `AR_ACT_SCALE` | 1.0 | Activation scale factor |
| `AR_ENABLE_ACT_MINMAX_TUNING` | 0 | Enable activation min/max tuning |
| `AR_FUSE_ONLINE_ROTATION` | 0 | Enable fused online rotation |
| `AR_SEARCH_SCALE_RATIO` | None | Search range ratio for symmetric int scale |
| `AR_ENABLE_UNIFY_MOE_INPUT_SCALE` | False | Unify MoE input scales |
| `AR_OMP_NUM_THREADS` | None | Override OMP thread count |
| `AUTO_ROUND_CACHE` | None | Cache directory override |

## Test quality warnings

Some tests give false confidence. Know which ones to trust:

**`inspect.getsource()` tests are not tests.** Several test files verify that source code *contains expected strings* rather than testing *behavior*. These tests pass even if the function is completely broken, as long as the text pattern exists in the source. Affected files and their inspect-test ratios:

| File | Total tests | inspect-based | Behavioral |
|------|-------------|---------------|------------|
| `test/test_dry_run_unit.py` | 11 | 8 (73%) | 3 |
| `test/test_cuda/test_offloader_modes.py` | 13 | 10 (77%) | 3 |
| `test/test_cuda/test_meta_device.py` | 18 | 13 (72%) | 5 |

When an agent writes new tests, **never use `inspect.getsource()` or `inspect.signature()` to verify code structure**. Instead, call the function and assert on its return value, side effects, or output files.

**Config value tests are missing.** The dry-run tests check that `block_name_to_quantize` *exists* and *isn't None*, but never check *what the value is*. The actual bug was `model.layers` instead of `model.language_model.layers` — the existing tests pass with either value. If you modify config-building code, add a test that asserts the exact expected values (compare against upstream output or a known-good reference like `~/models/Qwen3.5-0.8B-w4g128/`).

**No dry-run vs. real-quantize parity test.** The `--dry-run` mode was built to debug config files, but there's no test that runs dry-run and real quantization on the same model and compares configs. This is the single most valuable test you could add for preventing config regressions.

**Most CUDA tests are skipped in CI.** Tests marked `@pytest.mark.skip_ci` in `test/test_cuda/quantization/test_asym.py` don't run in CI. The only tests that exercise the full quantize→export pipeline require a GPU and a real model download.

## ASAQ Layer Substitution

The `spark-asqa-substitute` utility takes a working quantized model (all W4A16) and substitutes specified layers back to FP16. This produces an "ASAQ" copy for A/B evaluation.

**Layer selection precedence**:
1. `--layers 54,58` — explicit manual indices
2. `--top-n 5` — pick N worst layers from quantization report
3. (default) — substitute all layers that failed quality checks

**Usage**:
```bash
# Default: fix all broken layers from report
spark-asqa-substitute Qwen/Qwen3.6-27B

# Pick 5 worst layers
spark-asqa-substitute --top-n 5 Qwen/Qwen3.6-27B

# Manual override
spark-asqa-substitute --layers 54,58 Qwen/Qwen3.6-27B

# Use PSNR instead of cosine similarity
spark-asqa-substitute --top-n 5 --metric psnr Qwen/Qwen3.6-27B
```

**Arguments**:
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `model` | Yes | — | Model name (positional) |
| `--layers` | No | (from report) | Comma-separated layer indices (e.g. `54,58`) |
| `--top-n` | No | (from report) | Pick N worst layers from quantization report |
| `--metric` | No | cosine | Ranking metric: `cosine` or `psnr` |
| `--threshold` | No | — | Select all layers below this quality threshold |
| `--output_dir` | No | `./models/{name}-int4-ASAQ` | Output directory override |
| `--no-smoke-test` | No | (test runs) | Skip inference smoke test |

**Path inference** (given `model = "Qwen/Qwen3.6-27B"`):
- Quantized model: `./models/Qwen3.6-27B-int4-AutoRound`
- FP16 model: `Qwen/Qwen3.6-27B` (from HuggingFace cache)
- Output dir: `./models/Qwen3.6-27B-int4-ASAQ`

**Output**: Modified safetensors model + copied configs + updated quantization report.

**Code location**: `auto_round/asqa/` package.

## Testing patterns

- **Top-level tests** (`test/test_*.py`): Pure Python, no GPU. CLI display, metrics, report.
- **ASAQ tests** (`test/test_asqa/`): No GPU. Core substitution logic, CLI parsing, report generation.
- **CUDA tests** (`test/test_cuda/`): Require CUDA GPU. Quantization, algorithms, packing, ASAQ integration.
- **Fixtures**: Session-scoped, auto-download tiny model slices from HuggingFace. Saved to `test/tmp/`, cleaned up after session.
- **`get_model_path()`** (`test/helpers.py`): Checks `/tf_dataset/auto_round/models/`, `/models/`, `/dataset/`, then falls back to HuggingFace name.

## Version note

`pyproject.toml` is canonical (Python >= 3.10, torch >= 2.4). `setup.py` still says `>=3.9` / `>=2.1.0` — these are stale and should not be trusted.

## Codebase origin

The cleanup from upstream auto-round is documented in `thoughts/01-lean-and-mean/design-brief.md` and the multi-phase pruning in `thoughts/02-prune/`. Consult them before re-enabling any upstream feature to understand what was changed and why. (Note: `thoughts/` is in `.gitignore`.)

## Architecture details

- **`SARConfig`** (`auto_round/compressors/config.py`): Flat dataclass replacing the old ExtraConfig/TuningExtraConfig/SchemeExtraConfig hierarchy. All tuning + scheme params live here.
- **`auto_round_factory()`** (`auto_round/compressors/entry.py`): Factory function that creates the appropriate compressor. The `DataDrivenCompressor` is the main quantization workhorse.
- **`compressors/`** is the core package: `entry.py` (factory/router), `data_driven.py` (quantization logic), `base.py` (base class).
- **`auto_round_extension/`**: Low-level kernels. `cuda/` has Marlin/GPTQ kernels, `triton/` has Triton quantized linear layers, `torch/` has pure-torch fallback.
- **Multimodal/MoE**: Handled by `auto_round/modeling/` and `auto_round/special_model_handler.py`. Model-specific logic for Gemma 4, Qwen, DeepSeek, etc.
- **Export**: Only `auto_round:auto_gptq` format is supported. Export logic in `auto_round/export/export_to_autoround/`.
