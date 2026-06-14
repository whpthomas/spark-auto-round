# spark-auto-round

![Version](https://img.shields.io/badge/version-0.14.0-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![CUDA](https://img.shields.io/badge/CUDA-required-orange)
![GB10](https://img.shields.io/badge/hardware-GB10-purple)

> Your best Int4 AutoRound quantization for GB10 hardware

## What is this?

**Spark Auto Round** is an optimally pre-configured Int4 AutoRound quantization command line tool that is straightforward to use -- no tweaking necessary. This is a trimmed-down version of Intel's [auto-round](https://github.com/intel/auto-round) focused on **CUDA**, `torch.compile`, and **Int4 AutoRound (W4A16)** targeting the **DGX Spark - GB10 128GiB unified memory** architecture.

## Who it's for?

Intel’s AutoRound works exceptionally well on the DGX Spark and its GB10 siblings. AutoRound has been a popular go-to quantization method because of its combination of memory footprint, vllm support, performance and inference quality. However, the original [auto-round](https://github.com/intel/auto-round) codebase is more of a research project than a production codebase. This fork attempts to provide GB10 users a version of `auto-round` that is focused on their architecture and quality expectations, and tuned for the models they typically run as daily drivers.

## What is AutoRound?

Intel’s AutoRound is a technique used to quantize 16-bit models down to 4-bit. AutoRound uses signed gradient descent to jointly optimize weight rounding and clipping ranges. Mixture-of-Experts models are notoriously sensitive to quantization. AutoRound preserves the “distribution” of the weights rather than just the values, keeping the MoE logic intact even at 4-bit. The weights effectively halve the model size compared to FP8. Subsequently the Blackwell GPU needs less bandwidth to pull these weights from the unified pool. Once they reach the GPU, the Tensor Cores dequantizes INT4 weights into bfloat16 on-the-fly for the actual math, giving the speed of 4-bit with the precision of 16-bit calculations. Int4 AutoRound quantization allows large models to run with ample room for speculative decoding and the KV cache.

## Why not NVFP4?

To run comparative benchmarks and compare and contrast quantized models we need the best version of each quantization technique for reference. This is my attempt to provide the GB10 community with optimal Int4 AutoRound models.

## Features

- **Simple CLI**: Easy-to-use command-line interface i.e. `spark-auto-round <model>`
- **GB10 Optimized**: Whole-model quantization with 128GB unified memory, or automatic fallback to block-by-block loading for large models that don't fit in memory
- **torch.compile**: Always enabled for faster quantization on CUDA

## Iterative Optimization

The dense *Qwen 3.5 0.8B* model was used as a testbed to optimize Spark AutoRound (SAR). Using this [test setup and methodology](OPTIMIZATION.md) we achieved Tool Eval Bench score parity with the unquantized bf16 model.

| # | Model | Scheme | Dataset | Score | Rating | P/F | Tokens | Runs |
|---|-------|--------|---------|-------|--------|-----|--------|------|
|🥇 | **qwen3.5-0.8b-sar** | **Int4** | OpenCode Instruct | **69** | ★★★  | 41/13/15 | 516K | 3 |
|🥈 | qwen3.5-0.8b-sar | Int4 | github-code-clean | 67 | ★★★  | 39/14/16 | 516K | 3 |
|🥉 | **qwen3.5-0.8b** | **bf16** | - | **67** | ★★★  | 40/13/16 | 571K | 3 |
| 4 | qwen3.5-0.8b-ar | Int4 | pile-10k | 62 | ★★★ⓢ | 37/11/21 | 486K | 4 |
| 5 | qwen3.5-0.8b-sar | Int4 | pile-10k | 62 | ★★★  | 37/11/21 | 537K | 11 |

*`-sar` Spark AutoRound, `-ar` Intel AutoRound*

These results should **NOT** be interpreted to mean that SAR quantized models are equivalent bf16. It only demonstrates that for one 0.8B model, optimal settings were found that achieved test score parity with the original bf16 model. While these results are encouraging, whether these optimal settings generalize to other models requires further research and is under active investigation.

## Installation

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Install from GitHub
uv pip install git+https://github.com/whpthomas/spark-auto-round.git

# Or for development
git clone https://github.com/whpthomas/spark-auto-round.git
cd spark-auto-round
uv pip install -e .
```

## Quick Start

```bash
spark-auto-round <model> --output_dir ./models
```

The quantized model is saved to `{output_dir}/{model}-int4-AutoRound` by default. For example, quantizing `Qwen/Qwen3.6-27B` with `--output_dir ./models` produces `./models/Qwen3.6-27B-int4-AutoRound/`.

### Examples

```bash
# Optimal quantization
spark-auto-round Qwen/Qwen3.6-27B

# Fast parameters
spark-auto-round Qwen/Qwen3.5-122B-A10B \
    --iters 200 \
    --nsamples 128 \
    --output_dir ./models/fast

# Disable torch.compile (if causing issues)
spark-auto-round Qwen/Qwen3.6-35B-A3B --disable_torch_compile
```

## CLI Reference

```
spark-auto-round <model> [options]
```

### Basic Arguments

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

### Tuning Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | auto | Learning rate (auto-calculated if not set) |
| `--minmax_lr` | auto | MinMax learning rate (uses --lr if not set) |

### Scheme Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--quant_lm_head` | false | Quantize the lm_head layer |
| `--ignore_layers` | "" | Layers to skip (comma-separated) |
| `--layer_config` | null | Per-layer config JSON |

### Other Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_dtype` | null | Model dtype for loading |
| `--seed` | 42 | Random seed |
| `--adam` | false | Use Adam optimizer |
| `--mllm` | false | Force multimodal mode |

## Architecture

```
auto_round/              # Main package
├── __main__.py          # CLI: spark-auto-round entrypoint
├── __init__.py          # Exports AutoRound
├── autoround.py         # AutoRound facade
├── schemes.py           # QuantizationScheme + W4A16 preset
├── formats.py           # OutputFormat (auto_round, fake)
├── compressors/         # Core quantization logic
├── algorithms/          # SignSGD algorithm
├── data_type/           # INT/FP quantization kernels
├── export/              # Export to auto_round format
├── calibration/         # LLM calibration data loading
├── modeling/            # Multimodal/MoE layer handling
├── special_model_handler.py  # Model-specific logic
├── utils/               # Device detection, model loading
└── wrapper.py           # WrapperLinear for tuning

auto_round_extension/    # Low-level quantization kernels
├── cuda/                # Marlin/GPTQ kernels
├── triton/              # Triton quantized linear layers
└── torch/               # Pure-torch quantized linear layers

test/
├── conftest.py          # Test configuration
├── fixtures.py          # Model fixtures
├── helpers.py           # Test utilities
└── test_cuda/           # CUDA tests
```

## Supported Format

- `auto_round` (default) — HuggingFace-compatible format using `auto_round:auto_gptq` backend

## Development

```bash
python -m venv .venv && source .venv/bin/activate
uv pip install -e .
uv pip install pytest
pytest test/ -v
```

**Test prerequisites**: A CUDA GPU is required. First-time test runs download small model slices from HuggingFace (~2-layer 0.5B models saved to `test/tmp/`), so network access is needed on the first run. Optional test dependencies (gptqmodel, lm_eval, sentencepiece, etc.) are listed in `test/test_cuda/requirements.txt`.

### Key Commands

| Task | Command |
|------|---------|
| CLI entry point | `spark-auto-round <model> --output_dir ./models` |
| All tests | `pytest test/ -v` |
| CUDA tests only | `pytest test/test_cuda/ -v` |
| Single test | `pytest test/test_cuda/quantization/test_asym.py::TestAutoRoundAsym::test_asym_group_size -v` |
| Filter by keyword | `pytest -k "torch_compile" -v` |

## Requirements

- Python >= 3.9
- PyTorch >= 2.1.0
- CUDA GPU required (DGX Spark GB10 recommended)
- 128 GB unified memory recommended for large models

Quantization runs on GPU — there is no CPU fallback. The CLI hardcodes `device=cuda:0`.

## License

Apache License 2.0

## Spark-Auto-Round Contributions

- [@whpthomas](https://github.com/whpthomas)

## Acknowledgments

Based on [auto-round](https://github.com/intel/auto-round) by Intel.
