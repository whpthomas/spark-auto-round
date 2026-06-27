# spark-auto-round

![Version](https://img.shields.io/badge/version-0.14.2-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![CUDA](https://img.shields.io/badge/CUDA-required-orange)
![GB10](https://img.shields.io/badge/hardware-GB10-purple)

> int4 AutoRound quantization for GB10 hardware

**NOTE**

This is new software under active development. I am working my way up from Qwen 0.8b -> 27B -> 35B, 122B -> Gemma etc. I will post updates with verified results on specific models here:  

| Model | Tested | Tool Eval Score |
|-------|--------|-------|
| Qwen 3.5 0.8b      | ✔︎ | 67 |
| Qwen 3.6 27b       | ✔︎ | 92 |
| Qwen 3.6 35b a3b   | ✔︎ | 92 |
| Qwen 3.5 122b a10b | ✔︎ |    |
| Gemma 4 12b        |   |    |

Spark ASAQ Substitute has problems and only works dense dense models at the moment. MOE loading in INC is broken upstream.  

## What is this?

**Spark Auto Round** is an optimally pre-configured int4 AutoRound quantization command line tool that is straightforward to use -- no tweaking necessary. This is a trimmed-down version of Intel's [auto-round](https://github.com/intel/auto-round) focused on **CUDA**, `torch.compile`, and **int4 AutoRound (W4A16)** targeting the **DGX Spark - GB10 128GiB unified memory** architecture.

**Spark ASAQ Substitute** is an experimental companion tool that performs Adaptive Sensitivity-Aware Quantization by taking layer-wise Cosine Similarity, Peak Signal-to-Noise Ratio and for MOE models Router Jaccard Similarity, to replace sensitive layers with FP16 layers from the original model.

## Who it's for?

Intel’s AutoRound works exceptionally well on the DGX Spark and its GB10 siblings. AutoRound has been a popular go-to quantization method because of its combination of memory footprint, vllm support, performance and inference quality. However, the original [auto-round](https://github.com/intel/auto-round) codebase is more of a research project than a production codebase. This fork attempts to provide GB10 users a version of `auto-round` that is focused on their architecture and quality expectations, and tuned for the models they typically run as daily drivers.

## What is AutoRound?

Intel’s AutoRound is a technique used to quantize 16-bit models down to 4-bit. AutoRound uses signed gradient descent to jointly optimize weight rounding and clipping ranges. Mixture-of-Experts models are notoriously sensitive to quantization. AutoRound preserves the “distribution” of the weights rather than just the values, keeping the MoE logic intact even at 4-bit. The weights effectively halve the model size compared to FP8. Subsequently the Blackwell GPU needs less bandwidth to pull these weights from the unified pool. Once they reach the GPU, the Tensor Cores dequantizes INT4 weights into bfloat16 on-the-fly for the actual math, giving the speed of 4-bit with the precision of 16-bit calculations. int4 AutoRound quantization allows large models to run with ample room for speculative decoding and the KV cache.

## Why not NVFP4?

To run comparative benchmarks and compare and contrast quantized models we need the best version of each quantization technique for reference. This is my attempt to provide the GB10 community with optimal int4 AutoRound models.

## Features

- **Simple CLI**: Easy-to-use command-line interface i.e. `spark-auto-round <model>`
- **GB10 Optimized**: Whole-model quantization with 128GB unified memory, or automatic fallback to block-by-block loading for large models that don't fit in memory
- **Memory-Aware Auto-Tuner**: Pre-flight peak memory estimation automatically adjusts `--batch_size`, `--seqlen`, and `--nsamples` when the per-block peak exceeds the `--max_model_mem` ceiling. Relaxes the least quality-damaging setting first.
- **Stateful Resume**: If quantization is interrupted (Ctrl-C) or crashes (OOM), re-running the same command resumes from the last completed block. On OOM resume the auto-tuner tightens its budget to avoid re-crashing.
- **torch.compile**: Always enabled for faster quantization on CUDA
- **New Datasets** including OpenCode Instruct, CUAD, FinQA, LegalBench, and updated Github Code Clean
- **Adaptive Sensitivity-Aware Quantization**: A companion tool that replaces sensitive layers with fp16 layers from the original model.

## Installation

### Docker (recommended)

Easy way to get an environment set up on DGX spark without installing dependencies into the host system.


**Prerequisites:** Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
git clone https://github.com/whpthomas/spark-auto-round.git
cd spark-auto-round
docker compose build
docker compose run --rm sar Qwen/Qwen3.6-27B
```

Quantised output lands in `~/models/` on the host. The HuggingFace cache is persisted to `~/.cache/huggingface/` so weights are not re-downloaded between runs. Both paths are configurable via `MODELS_DIR` and `HF_CACHE` environment variables (or a `.env` file).

To use a different NGC PyTorch image (e.g. a newer monthly tag):

```bash
SAR_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3 docker compose build
```

### Host venv (advanced)

A host venv requires manual CUDA wrangling: you must install torch from the `cu130` index, set `CUDA_HOME`, and install `causal-conv1d` with `--no-build-isolation`. Only use this if you have a reason to avoid Docker.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
CUDA_HOME=/usr/local/cuda pip install --no-build-isolation -e .
```

## Quick Start

```bash
spark-auto-round <model>
spark-asaq-substitute <model>
```

The quantized model is saved to `./models/{model}-int4-AutoRound` by default. For example, quantizing `Qwen/Qwen3.6-27B` produces `./models/Qwen3.6-27B-int4-AutoRound/`. The ASAQ model is saved to `./models/{model}-int4-ASAQ` by default.

## Iteratively optimized using Qwen 3.5 0.8b

The dense *Qwen 3.5 0.8B* model was used as a testbed to optimize Spark Auto Round (SAR). Using this [test setup and methodology](docs/optimization.md) we achieved Tool Eval Bench score parity with the unquantized bf16 model. While these results are encouraging, these are complex system and there are many confounding factors that need to be considered. They only demonstrate that for one 0.8B model, optimal settings were found that achieved test score parity with the original bf16 model. Whether these optimal settings generalize to other models requires further research and is under active investigation.

## Performance with Qwen 3.6 27b

Spark auto round repeatedly achieved a [92/100](docs/test-score.md) tool-eval-bench score with the Nvidia's OpenCode Instruct dataset.

- Quantization command: `spark-auto-round --dataset "opencode-instruct" Qwen/Qwen3.6-27B`
- MTP averages ~26.4 t/s with `num_speculative_tokens: 3` for longer context and agentic coding
- DFlash averages ~38.1 t/s with `num_speculative_tokens: 6` for shorter context and instruction following

| # | Model | Scheme | Dataset | Score | t/s | Rating | P/F | Tokens |
|---|-------|--------|---------|-------|-----|--------|-----|--------|
|🥇 | **qwen3.6-27b-sar-oc-mpt** | **int4** | OpenCode Instruct | **92** | 26.4 | ★★★★★ | 59/9/1 | 284K |
|🥈 | qwen3.6-27b-sar-oc-dflash | int4 | OpenCode Instruct | 90 | **38.1** | ★★★★★ | 57/10/2 | 265K |
|🥉 | qwen/qwen3.6-27b-fp8 | fp8 | - | 88 | 18.1 | ★★★★ | 57/8/4 | 275K |
| 4 | qwen3.6-27b-sar-oc | int4 | OpenCode Instruct | 88 | 12.5 | ★★★★ | 57/8/4 | 275K |
| 5 | qwen3.6-27b-sar-git-mtp | int4 | Github Code Clean | 86 | 26.2 | ★★★★ | 54/10/5 | 268K |
| 6 | qwen/qwen3.6-27b | bf16 | - | 83 | 11.4 | ★★★★ | 53/9/7 | 243K |

## Performance with Qwen 3.6 35b a3b

| # | Model | Quant | Score | Min | Max | t/s | P/F | Weakest | Runs |
|---|-------|-------|------:|----:|----:|-----|-----|---------|-----:|
| 🥇 | qwen3.6-35b-sar-pc | int4 | 92.00 | 91 | 93 | 64 | 178/25/4 | 🟡 Instruct 80.00% | 3 |
| 🥈 | qwen3.6-35b-sar-bf16 | int4 | 91.67 | 91 | 93 | 65 | 176/27/4 | 🟡 Plan 77.67% | 3 |
| 🥉 | qwen3.6-35b-sar | int4 | 91.00 | 90 | 92 | 65 | 176/25/6 | 🟡 Create 77.67% | 3 |
|  4 | qwen/qwen3.6-35b-fp8 | FP8 | 91.00 | 91 | 91 | 49 | 171/33/3 | 🟡 Scale 75.00% | 3 |
|  5 | qwen3.6-35b-sar-mtp | int4 | 90.67 | 88 | 93 | 77 | 174/28/5 | 🟡 Instruct 80.00% | 3 |
|  6 | qwen3.6-35b-sar-dflash | int4 | 88.00 | 88 | 88 | 97 | 164/37/6 | 🟠 Schema 52.67% | 3 |

### Scripts and Recipes

For transparency and convenience, and so my results can be independently replicated and verified. All test scripts and recipes are shared in the [spark-vllm-docker/](spark-vllm-docker) sub-directory. These can be used with the DGX [Spark vllm docker](https://github.com/eugr/spark-vllm-docker) community supported tool.

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

# Perform adaptive sensitivity-aware quantization
spark-asaq-substitute Qwen/Qwen3.6-27B
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
| `--output_dir` | ./models | Output directory (also stores checkpoints under `.cache/`) |
| `--dataset` | opencode-instruct | Calibration dataset |
| `--disable_torch_compile` | (disabled) | Disable torch.compile (enabled by default) |

### Memory Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_model_mem` | 96 | Memory budget in GiB. Two roles: (1) hard ceiling on estimated peak per-block memory for the auto-tuner, and (2) threshold for whole-model vs block-offload — if the model-at-rest exceeds this many GiB, blocks are loaded on demand from meta device. Default is 75% of 128 GiB. |

### Speed/Debug Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--shakedown` | false | Override iters=1, nsamples=1, seqlen=2, batch_size=1 for fastest end-to-end test. Quantized model will be very low quality. |
| `--halt-after N` | -1 | Simulate a KeyboardInterrupt after saving the N-th block's checkpoint. Works with or without `--shakedown`. |

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
| `--clear-cache` | false | Delete checkpoint cache before starting. Forces a fresh run even if a valid checkpoint exists. |
| `--dry-run` | false | Run the full init pipeline except quantization tuning; write config files only for inspection. |
| `--trust-remote-code` | true | Trust remote code when loading models (disable with `--no-trust-remote-code`). |
| `--bs` | — | Alias for `--batch_size`. |
| `--mllm` | false | Force multimodal mode |

## Memory and Resume

### Auto-Tuner

Before quantization begins, the CLI estimates per-block peak GPU memory using your model's configuration and the current settings (`--batch_size`, `--seqlen`, `--nsamples`). The estimator accounts for block weights, wrapper parameters, activation outputs, gradient tensors, calibration inputs, attention scores, QKV intermediates, FFN intermediates, and a 1.50× safety factor for CUDA allocator overhead.

If the estimated peak exceeds the `--max_model_mem` ceiling (default 96 GiB), the auto-tuner relaxes settings in priority order to bring peak below budget:

| Step | Setting | Relief | Quality Impact |
|------|---------|--------|----------------|
| 1 | `--batch_size` 8 → 4 → 2 → 1 | 2× per step | Moderate (noisier gradients) |
| 2 | `--seqlen` 2048 → 1024 → 512 → 256 | 2× per step | Larger (truncated context) |
| 3 | `--nsamples` 512 → 256 → 128 | CPU RAM relief | Smaller (less coverage) |

The tuner stops as soon as the budget is satisfied, preserving the highest-quality settings possible. On a fresh run you will see:

```
Memory OK: est. peak 42.1 GB / 96.0 GB (75%) — proceeding with user settings.
```

Or, if adjustments are needed:

```
Memory budget exceeded. Adjusting settings:
  batch_size   8 → 4        (noisier gradients)
  seqlen       2048 → 1024     (truncated context)
Estimated peak: 58.0 GB / 96.0 GB (75%) ✓
```

### Checkpoint Resume

During quantization, a checkpoint is automatically saved after each block completes:

```
{output_dir}/.cache/
  progress.json         # Metadata: completed count, block names, exit reason
  block_00000.pt        # Quantized state for block 0
  block_00001.pt        # Quantized state for block 1
  ...
```

If the run is interrupted (Ctrl-C) or crashes (CUDA OOM), the `.cache/` directory is preserved. Re-running the **same command** with the same `--output_dir` detects the checkpoint and resumes:

```
Resuming from block 9/48.
Previous run OOM'd on block 9. Tightening settings:
  batch_size  4 → 2  (noisier gradients)
Estimated peak: 35.0 GB / 83.2 GB (65%) ✓
```

The exit reason (`"interrupted"` vs `"oom"`) changes the auto-tuner's behaviour on resume:

| Previous Exit | Auto-Tuner Behaviour |
|---------------|----------------------|
| `interrupted` | Fresh auto-tune with original budget — user chose to stop, settings may be fine |
| `oom` | Skips one relaxation step — prevents re-crashing on the same settings |

On successful completion, `.cache/` is automatically cleaned up. To force a fresh run despite an existing checkpoint, use `--clear-cache`.

## Datasets

| Alias | Content | Notes |
|-------|---------|-------------|
| [opencode-instruct](https://huggingface.co/nvidia/OpenCodeInstruct) | **(default)** Code instructions + responses | Packs short sequences; best for coding models |
| [github-code-clean](https://huggingface.co/codeparrot/github-code-clean) | Source code | Downloads random parquet shards for diversity |
| [pile-10k](https://huggingface.co/NeelNanda/pile-10k) | English general text | Classic calibration dataset |
| [CCI3-HQ](https://huggingface.co/BAAI/CCI3-HQ) | Chinese web text | Streaming; good for Chinese models |
| [pile-val-backup](https://modelscope.cn/datasets/swift/pile-val-backup) | English general text | Requires `pip install modelscope` |
| [ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | Chat dialogues | Splits: `train_sft`, `test_sft` |
| [Ultra-FineWeb](https://huggingface.co/openbmb/Ultra-FineWeb) | Web pages | Splits: `en`, `zh` |
| [mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) | Python problems + code | Text + code concatenated |
| [AudioCaps](https://github.com/cdjkim/audiocaps) | Sound/music captions | Good for audio-related models |
| [new-title-chinese](https://huggingface.co/datasets/madao33/new-title-chinese) | Chinese news headlines | For Chinese NLP tasks |
| [cuad](https://huggingface.co/datasets/whpthomas/cuad-parquet) | Legal contract QA | Context + question; packs short sequences |
| [finqa](https://huggingface.co/datasets/whpthomas/finqa-parquet) | Financial QA | Pre-text + question + post-text; packs short sequences |
| [legalbench](https://huggingface.co/datasets/nguha/legalbench) | Trademark classification | Short legal phrases; test split (95 samples) |
| [legalbench-instruct](https://huggingface.co/datasets/nguha/legalbench) | LegalBench + OpenCode Instruct | 95 legal + ~416 code = ~511 total; balanced coverage |

**Tip:** Use `opencode-instruct` for coding models and `pile-10k` or `CCI3-HQ` for general-purpose models. Use `cuad` or `finqa` for domain-specific legal/financial calibration.

## Supported Format

- `auto_round` (default) — HuggingFace-compatible format using `auto_round:auto_gptq` backend

## Requirements

- Python >= 3.9
- PyTorch >= 2.1.0
- CUDA GPU required (DGX Spark GB10 recommended)
- 128 GB unified memory recommended for large models

Quantization runs on single GB10 GPU — there is no CPU fallback. The CLI hardcodes `device=cuda:0`.

## License

Apache License 2.0

## Spark-Auto-Round Contributions

- [@whpthomas](https://github.com/whpthomas)

## Acknowledgments

Based on [auto-round](https://github.com/intel/auto-round) by Intel.

## References

- [auto-round](https://github.com/intel/auto-round) - *Advanced quantization toolkit designed for Large Language Models*
- [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) - *Docker configuration and startup scripts to run vLLM on DGX Spark*
- [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench/) - *A tool-calling quality benchmark for evaluating LLM tool-use in agentic workflows*
- [chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix) - *Stable tool calling with enhanced chat template*
