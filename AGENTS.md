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
| Tests in container (against shipped NGC stack) | see below |

### Running tests in the container

The NGC image (`spark-auto-round:local`) already ships `pytest`, `transformers`,
and `torch` — no `[test]` extra needed for CPU-only suites. Mount the live source
tree over the editable-install path so you test current changes without rebuilding
the ~20 GB image:

```sh
docker run --rm --entrypoint bash \
  -v "$PWD":/opt/spark-auto-round \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /opt/spark-auto-round \
  spark-auto-round:local \
  -c "python -m pytest test/test_resume.py -q"
```

CPU-only tests (e.g. `test_resume.py`) need no GPU reservation, so plain
`docker run` is simpler than `docker compose run`.

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

