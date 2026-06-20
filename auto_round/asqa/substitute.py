# Copyright (c) 2026 Dr Henry Thomas
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

"""Core layer substitution engine for ASAQ.

This module provides functions for substituting quantized (W4A16) layers
with their FP16 equivalents in a post-processing step. The intended
workflow is:

1. ``infer_paths`` — Resolve paths from the model name.
2. ``load_quantized_weights`` — Load the full quantized model weights.
3. ``load_fp16_layers`` — Lazy-load only the FP16 tensors for target layers.
4. ``substitute_layers`` — Swap quantized tensors for FP16 versions.
5. ``save_model`` — Persist the modified model to disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANT_SUFFIXES = (".qweight", ".qzeros", ".scales")

MAX_SHARD_SIZE = 5 * 1024**3  # 5 GB per shard

# File names to copy from the quantized model directory (when they exist).
_CONFIG_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_quantized_tensor(tensor_name: str, layer_prefix: str) -> bool:
    """Check if *tensor_name* is a quantized tensor belonging to *layer_prefix*.

    A tensor is considered quantized if its name starts with *layer_prefix*
    and ends with one of ``('.qweight', '.qzeros', '.scales')``.

    >>> _is_quantized_tensor(
    ...     "model.language_model.layers.54.mlp.gate_proj.qweight",
    ...     "model.language_model.layers.54.",
    ... )
    True
    >>> _is_quantized_tensor(
    ...     "model.language_model.layers.54.mlp.gate_proj.weight",
    ...     "model.language_model.layers.54.",
    ... )
    False
    """
    if not tensor_name.startswith(layer_prefix):
        return False
    suffix = tensor_name[len(layer_prefix) :]
    return any(suffix.endswith(qs) for qs in QUANT_SUFFIXES)


def _layer_prefix(idx: int) -> str:
    """Return the canonical tensor-name prefix for layer *idx*.

    >>> _layer_prefix(54)
    'model.language_model.layers.54.'
    """
    return f"model.language_model.layers.{idx}."


def _shard_name(idx: int, total: int) -> str:
    """Return the safetensors shard file name for shard *idx* (1-based)."""
    if total == 1:
        return "model.safetensors"
    return f"model-{idx:05d}-of-{total:05d}.safetensors"


def _is_moe_expert_tensor(tensor_name: str, layer_prefix: str) -> bool:
    """Check if tensor is a MoE expert weight (not shared expert, not individual).

    Returns True for fused MoE weights like:
        model.language_model.layers.23.mlp.experts.gate_up_proj
        model.language_model.layers.23.mlp.experts.down_proj

    Returns False for:
        model.language_model.layers.23.mlp.experts.252.gate_proj  (individual)
        model.language_model.layers.23.mlp.shared_expert.*  (shared)
    """
    if not tensor_name.startswith(layer_prefix):
        return False
    suffix = tensor_name[len(layer_prefix):]
    # Match: mlp.experts.gate_up_proj or mlp.experts.down_proj
    # But NOT: mlp.experts.NNN.gate_proj (individual expert)
    # And NOT: mlp.shared_expert.*
    return (
        suffix.startswith("mlp.experts.gate_up_proj")
        or suffix.startswith("mlp.experts.down_proj")
    ) and not any(c.isdigit() for c in suffix.split("mlp.experts.")[1].split(".")[0])


def _fuse_gate_up_to_w13(gate_up_proj: torch.Tensor) -> torch.Tensor:
    """Fuse gate_proj and up_proj into w13_weight.

    In HuggingFace format, gate_up_proj has shape [num_experts, 2*intermediate, hidden]
    where gate_proj and up_proj are concatenated along dim=1.

    vllm's w13_weight expects the same format (fused gate and up projections).
    """
    # gate_up_proj is already fused in the checkpoint format
    # Shape: [num_experts, 2*intermediate_size, hidden_size]
    return gate_up_proj


# ---------------------------------------------------------------------------
# 1. infer_paths
# ---------------------------------------------------------------------------


def infer_paths(model_name: str) -> tuple[str, str, str]:
    """Infer the quantized model path, FP16 model ID, and output directory.

    Given ``model_name = "Qwen/Qwen3.6-27B"``::

        quantized_path = ./models/Qwen3.6-27B-int4-AutoRound
        fp16_model_id  = Qwen/Qwen3.6-27B
        output_dir     = ./models/Qwen3.6-27B-int4-ASAQ

    Args:
        model_name: HuggingFace model ID or local path
            (e.g. ``"Qwen/Qwen3.6-27B"``).

    Returns:
        ``(quantized_path, fp16_model_id, output_dir)``

    Raises:
        FileNotFoundError: If the quantized model directory does not exist.
    """
    # Extract short name: "Qwen/Qwen3.6-27B" -> "Qwen3.6-27B"
    short_name = model_name.rstrip("/").split("/")[-1]

    # Quantized model convention: ./models/{name}-int4-AutoRound
    quantized_path = f"./models/{short_name}-int4-AutoRound"
    if not os.path.isdir(quantized_path):
        raise FileNotFoundError(
            f"Quantized model not found: {quantized_path}\n"
            "Run spark-auto-round first to quantize the model."
        )

    # FP16 model: same as input (HuggingFace ID or local path)
    fp16_model_id = model_name

    # Output convention: ./models/{name}-int4-ASAQ
    output_dir = f"./models/{short_name}-int4-ASAQ"

    return quantized_path, fp16_model_id, output_dir


# ---------------------------------------------------------------------------
# 2. load_quantized_weights
# ---------------------------------------------------------------------------


def load_quantized_weights(
    model_dir: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load all weights and config from a quantized model directory.

    Handles both single-file and multi-shard safetensors models.

    Args:
        model_dir: Path to quantized model directory.

    Returns:
        ``(weights_dict, config_dict)`` where *weights_dict* maps tensor
        names to ``torch.Tensor`` and *config_dict* is the parsed
        ``config.json``.

    Raises:
        FileNotFoundError: If the model directory or required files are
            missing.
        RuntimeError: If safetensors files are corrupted.
    """
    # -- config.json --
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    with open(config_path) as f:
        config: dict[str, Any] = json.load(f)

    # -- weights --
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")

    weights: dict[str, torch.Tensor] = {}

    if os.path.exists(index_path):
        # Multi-shard model
        with open(index_path) as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]

        # Group tensors by shard file
        shards: dict[str, list[str]] = {}
        for tensor_name, shard_file in weight_map.items():
            shards.setdefault(shard_file, []).append(tensor_name)

        for shard_file, tensor_names in shards.items():
            shard_path = os.path.join(model_dir, shard_file)
            try:
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for name in tensor_names:
                        weights[name] = f.get_tensor(name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load shard {shard_file}: {e}. "
                    f"The safetensors file may be corrupted."
                ) from e

    elif os.path.exists(single_path):
        try:
            with safe_open(single_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    weights[name] = f.get_tensor(name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {single_path}: {e}. "
                f"The safetensors file may be corrupted."
            ) from e
    else:
        raise FileNotFoundError(
            f"No safetensors files found in {model_dir}. "
            "Expected model.safetensors or model.safetensors.index.json"
        )

    return weights, config


# ---------------------------------------------------------------------------
# 3. load_fp16_layers
# ---------------------------------------------------------------------------


def load_fp16_layers(
    model_id: str,
    layer_indices: list[int],
    *,
    _model_dir: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load FP16 weights for specified layers from the original model.

    Uses **lazy loading**: only loads the shards containing target layers,
    avoiding loading the full FP16 model into memory.  For
    ``layer_indices=[54, 58]`` with a 55 GB model, this loads ~12 GB instead.

    Args:
        model_id: HuggingFace model ID or local path.
        layer_indices: List of layer indices to load (e.g. ``[54, 58]``).

    Keyword Args:
        _model_dir: (Testing helper) Override the resolved model directory.
            When *None* (the default), the function resolves the directory
            from *model_id* (local path or HuggingFace Hub download).

    Returns:
        Dict mapping full tensor names to their FP16 values.  Only includes
        non-quantized tensors (weight, bias, norms, linear_attn params, etc.).

    Raises:
        ValueError: If *layer_indices* is empty.
        KeyError: If requested layers are not found in the FP16 model.
        ConnectionError: If network download fails.
    """
    if not layer_indices:
        raise ValueError("layer_indices must not be empty")

    import os as _os

    from huggingface_hub import hf_hub_download

    # -- Resolve model directory and index file --
    if _model_dir is not None:
        model_dir = _model_dir
    elif _os.path.isdir(model_id):
        model_dir = model_id
    else:
        model_dir = model_id  # safe_open can handle HF repo refs

    index_path = _os.path.join(model_dir, "model.safetensors.index.json")
    if not _os.path.exists(index_path):
        try:
            local_index = hf_hub_download(
                repo_id=model_id, filename="model.safetensors.index.json"
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to download model index from {model_id}: {e}. "
                f"Check network connection and model ID."
            ) from e
        model_dir = _os.path.dirname(local_index)
        index_path = local_index

    with open(index_path) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]

    # Build layer prefixes for target layers
    layer_prefixes = [_layer_prefix(idx) for idx in layer_indices]

    # Find which shards contain our target tensors (non-quantized only)
    target_shards: dict[str, list[str]] = {}
    for tensor_name, shard_file in weight_map.items():
        for prefix in layer_prefixes:
            if tensor_name.startswith(prefix) and not _is_quantized_tensor(
                tensor_name, prefix
            ):
                target_shards.setdefault(shard_file, []).append(tensor_name)
                break

    # Load only needed shards
    fp16_weights: dict[str, torch.Tensor] = {}
    for shard_file, tensor_names in target_shards.items():
        shard_path = _os.path.join(model_dir, shard_file)
        if not _os.path.exists(shard_path):
            try:
                shard_path = hf_hub_download(repo_id=model_id, filename=shard_file)
            except Exception as e:
                raise ConnectionError(
                    f"Failed to download {shard_file} from {model_id}: {e}. "
                    f"Check network connection."
                ) from e
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in tensor_names:
                fp16_weights[name] = f.get_tensor(name)

    # Validate that we found tensors for all requested layers
    found_layers = set()
    for name in fp16_weights:
        for idx in layer_indices:
            if f"model.language_model.layers.{idx}." in name:
                found_layers.add(idx)

    missing = set(layer_indices) - found_layers
    if missing:
        raise KeyError(
            f"Layers {sorted(missing)} not found in FP16 model. "
            f"Available layers may be 0-{max(layer_indices)+10}. "
            f"Check model architecture."
        )

    return fp16_weights


# ---------------------------------------------------------------------------
# 4. substitute_layers
# ---------------------------------------------------------------------------


def _remap_moe_weights(
    fp16_prefix_tensors: dict[str, torch.Tensor], layer_prefix: str
) -> dict[str, torch.Tensor]:
    """Remap MoE expert weights to the format vllm expects for unquantized layers.

    vllm expects unquantized MoE layers to have:
        mlp.experts.routed_experts.w13_weight  (gate_up_proj)
        mlp.experts.routed_experts.w2_weight   (down_proj)

    But HuggingFace checkpoint format has:
        mlp.experts.gate_up_proj  (fused gate + up)
        mlp.experts.down_proj

    This function remaps the names and returns new tensors.

    Args:
        fp16_prefix_tensors: Tensors for this layer (modified in place).
        layer_prefix: Layer prefix like "model.language_model.layers.23."

    Returns:
        Updated dict with remapped names (same reference, modified in place).
    """
    routed_prefix = layer_prefix + "mlp.experts.routed_experts."
    moe_prefix = layer_prefix + "mlp.experts."

    # Check if this layer has MoE expert weights
    has_gate_up = any("mlp.experts.gate_up_proj" in k for k in fp16_prefix_tensors)
    has_down = any("mlp.experts.down_proj" in k for k in fp16_prefix_tensors)

    if not (has_gate_up and has_down):
        # Not a MoE layer or already in correct format
        return fp16_prefix_tensors

    # Remap gate_up_proj -> w13_weight
    for k in list(fp16_prefix_tensors.keys()):
        if "mlp.experts.gate_up_proj" in k:
            new_key = k.replace("mlp.experts.gate_up_proj", "mlp.experts.routed_experts.w13_weight")
            fp16_prefix_tensors[new_key] = fp16_prefix_tensors.pop(k)

    # Remap down_proj -> w2_weight
    for k in list(fp16_prefix_tensors.keys()):
        if "mlp.experts.down_proj" in k:
            new_key = k.replace("mlp.experts.down_proj", "mlp.experts.routed_experts.w2_weight")
            fp16_prefix_tensors[new_key] = fp16_prefix_tensors.pop(k)

    return fp16_prefix_tensors


def _unfuse_moe_weights(
    weights: dict[str, torch.Tensor], layer_prefix: str, num_experts: int
) -> dict[str, torch.Tensor]:
    """Unfuse MoE expert weights from fused to individual expert format.

    HuggingFace checkpoint format has fused weights:
        mlp.experts.gate_up_proj  (shape: [num_experts, 2*intermediate, hidden])
        mlp.experts.down_proj     (shape: [num_experts, hidden, intermediate])

    vllm expects individual expert weights:
        mlp.experts.{expert_id}.gate_proj  (shape: [intermediate, hidden])
        mlp.experts.{expert_id}.up_proj    (shape: [intermediate, hidden])
        mlp.experts.{expert_id}.down_proj  (shape: [hidden, intermediate])

    Args:
        weights: Dict of weights (modified in place).
        layer_prefix: Layer prefix like "model.language_model.layers.23."
        num_experts: Number of experts in the MoE layer.

    Returns:
        Updated dict (same reference, modified in place).
    """
    experts_prefix = layer_prefix + "mlp.experts."
    
    # Find and unfuse gate_up_proj
    gate_up_key = None
    for k in weights:
        if k.startswith(experts_prefix) and k.endswith("gate_up_proj") and "." not in k[len(experts_prefix):-len("gate_up_proj")]:
            gate_up_key = k
            break
    
    if gate_up_key is None:
        return weights  # Already in individual format or not found
    
    gate_up_tensor = weights.pop(gate_up_key)
    
    # gate_up_proj shape: [num_experts, 2*intermediate, hidden]
    # Split into gate_proj and up_proj along dim=1
    intermediate_size = gate_up_tensor.shape[1] // 2
    
    for expert_id in range(num_experts):
        expert_gate_up = gate_up_tensor[expert_id]  # [2*intermediate, hidden]
        gate_proj = expert_gate_up[:intermediate_size]  # [intermediate, hidden]
        up_proj = expert_gate_up[intermediate_size:]  # [intermediate, hidden]
        
        weights[f"{experts_prefix}{expert_id}.gate_proj"] = gate_proj
        weights[f"{experts_prefix}{expert_id}.up_proj"] = up_proj
    
    # Find and unfuse down_proj  
    down_key = None
    for k in weights:
        if k.startswith(experts_prefix) and k.endswith("down_proj") and "." not in k[len(experts_prefix):-len("down_proj")]:
            down_key = k
            break
    
    if down_key is not None:
        down_tensor = weights.pop(down_key)
        # down_tensor shape: [num_experts, hidden, intermediate]
        for expert_id in range(num_experts):
            weights[f"{experts_prefix}{expert_id}.down_proj"] = down_tensor[expert_id]
    
    return weights


def substitute_layers(
    weights: dict[str, torch.Tensor],
    fp16_layers: dict[str, torch.Tensor],
    layer_indices: list[int],
) -> dict[str, torch.Tensor]:
    """Substitute quantized layers with FP16 weights.

    For each specified layer:

    * **Remove** quantized tensors: ``{layer}.*.qweight``,
      ``{layer}.*.qzeros``, ``{layer}.*.scales``
    * **Add** FP16 tensors from *fp16_layers*: ``{layer}.*.weight``,
      ``{layer}.*.bias``
    * **Keep** existing FP16 tensors (norms, linear_attn params)

    vllm handles weight fusion internally via ``packed_modules_mapping``,
    so we keep the original unfused tensor names.

    Args:
        weights: Full quantized model weights dict (**modified in place**).
        fp16_layers: FP16 weights for target layers
            (from :func:`load_fp16_layers`).
        layer_indices: Layer indices to substitute (e.g. ``[54, 58]``).

    Returns:
        Modified *weights* dict (same reference as input, modified in place).

    Raises:
        KeyError: If expected FP16 tensors are missing for a layer.
        ValueError: If *layer_indices* is empty.
    """
    if not layer_indices:
        raise ValueError("layer_indices must not be empty")

    for idx in layer_indices:
        prefix = _layer_prefix(idx)

        # Step 1: Remove quantized tensors for this layer
        keys_to_remove = [
            k for k in weights if k.startswith(prefix) and _is_quantized_tensor(k, prefix)
        ]
        for k in keys_to_remove:
            del weights[k]

        # Step 2: Add FP16 tensors for this layer
        fp16_prefix_tensors = {
            k: v for k, v in fp16_layers.items() if k.startswith(prefix)
        }
        if not fp16_prefix_tensors:
            raise KeyError(
                f"No FP16 tensors found for layer {idx}. "
                "Check that layer exists in the FP16 model."
            )

        weights.update(fp16_prefix_tensors)

        # NOTE: We do NOT rename MoE weights.
        # vllm's weight loading code handles gate_up_proj and down_proj by:
        # 1. Chunking gate_up_proj into two parts (w1 and w3)
        # 2. Calling load_fused_expert_weights for each part
        # 
        # The weight names must contain 'experts.gate_up_proj' or 'experts.down_proj'
        # for vllm to recognize them as fused expert weights.
        # Renaming to 'routed_experts.w13_weight' would break this logic.
        # vllm's weight loading code maps:
        #   experts.gate_up_proj -> w13_weight
        #   experts.down_proj -> w2_weight

    return weights


# ---------------------------------------------------------------------------
# 5. save_model
# ---------------------------------------------------------------------------


def _check_disk_space(output_dir: str, required_bytes: int) -> None:
    """Check if enough disk space is available for the output.

    Args:
        output_dir: Target directory (will walk up to find existing parent).
        required_bytes: Minimum required free space in bytes.

    Raises:
        OSError: If insufficient disk space.
    """
    import shutil

    parent = output_dir
    while not os.path.exists(parent):
        parent = os.path.dirname(parent)
        if not parent:
            parent = "."
            break

    usage = shutil.disk_usage(parent)
    if usage.free < required_bytes:
        required_gb = required_bytes / (1024**3)
        free_gb = usage.free / (1024**3)
        raise OSError(
            f"Insufficient disk space: need {required_gb:.1f} GB, "
            f"have {free_gb:.1f} GB free at {parent}"
        )


def save_model(
    weights: dict[str, torch.Tensor],
    config: dict[str, Any],
    output_dir: str,
    *,
    source_dir: str | None = None,
) -> Path:
    """Save modified weights as safetensors with index file.

    Creates a sharded safetensors model directory with ``config.json`` and
    an updated ``model.safetensors.index.json``.

    Args:
        weights: Modified weights dict.
        config: Model config dict (from ``config.json``).
        output_dir: Directory to save to.

    Keyword Args:
        source_dir: If given, copy extra config files (tokenizer, etc.)
            from this directory.

    Returns:
        :class:`~pathlib.Path` to the output directory.

    Raises:
        OSError: If insufficient disk space.
    """
    # Pre-check disk space (estimate: 2x the weight size for safety)
    estimated_size = sum(t.nelement() * t.element_size() for t in weights.values()) * 2
    _check_disk_space(output_dir, estimated_size)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # -- config.json --
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # -- Copy extra config files from source (if provided) --
    if source_dir is not None:
        src = Path(source_dir)
        for fname in _CONFIG_FILES:
            src_file = src / fname
            if src_file.exists() and not (output_path / fname).exists():
                import shutil

                shutil.copy2(src_file, output_path / fname)

    # -- Split weights into shards --
    shards: list[dict[str, torch.Tensor]] = []
    current_shard: dict[str, torch.Tensor] = {}
    current_size = 0

    for name in sorted(weights):
        tensor = weights[name]
        tensor_size = tensor.nelement() * tensor.element_size()
        if current_size + tensor_size > MAX_SHARD_SIZE and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    # -- Save shards and build weight_map --
    weight_map: dict[str, str] = {}
    total_size = 0

    for i, shard in enumerate(shards, start=1):
        shard_name = _shard_name(i, len(shards))
        shard_path = output_path / shard_name
        save_file(shard, str(shard_path))
        for name in shard:
            weight_map[name] = shard_name
            t = shard[name]
            total_size += t.nelement() * t.element_size()

    # -- Index file --
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    return output_path


# ---------------------------------------------------------------------------
# 6. copy_config_files
# ---------------------------------------------------------------------------

CONFIG_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    # Quantization config (critical for vLLM weight loading)
    "quantization_config.json",
    # Extra weight files (e.g., MTP head for Qwen3.5 models)
    "model_extra_tensors.safetensors",
)


def copy_config_files(source_dir: str, output_dir: str) -> None:
    """Copy non-weight config files from quantized model to output.

    Args:
        source_dir: Quantized model directory.
        output_dir: Output directory.
    """
    import shutil

    os.makedirs(output_dir, exist_ok=True)

    for filename in CONFIG_FILES:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# 7. update_quantization_config
# ---------------------------------------------------------------------------


def update_quantization_config(
    output_dir: str,
    layer_indices: list[int],
    weights: dict[str, torch.Tensor] | None = None,
) -> None:
    """Update quantization_config.json to mark substituted layers as FP16.

    Adds a ``substituted_layers`` key with the layer indices that were
    converted back to FP16, and updates ``extra_config`` to mark all
    components of substituted layers as FP16 (bits=16) so vLLM loads
    them correctly.

    Args:
        output_dir: Directory containing the output model.
        layer_indices: Layer indices that were substituted.
        weights: Optional weights dict. When provided, used to detect
            which layer components exist and need to be marked as FP16.
    """
    config_path = os.path.join(output_dir, "quantization_config.json")

    if os.path.exists(config_path):
        with open(config_path) as f:
            quant_config: dict[str, Any] = json.load(f)
    else:
        # Fall back to config.json's quantization_config (auto-round stores
        # the config there, not in a separate quantization_config.json).
        config_json_path = os.path.join(output_dir, "config.json")
        if os.path.exists(config_json_path):
            with open(config_json_path) as f:
                model_config: dict[str, Any] = json.load(f)
            quant_config = model_config.get("quantization_config", {})
        else:
            quant_config = {}

    quant_config["substituted_layers"] = sorted(set(layer_indices))
    quant_config["substituted_dtype"] = "fp16"

    # Keep quant_method as "auto-round" (INC format).
    #
    # Why not "auto_gptq": AutoGPTQConfig.get_config_filenames() returns
    # ["quantize_config.json"] but we have "quantization_config.json".
    # INCConfig.get_config_filenames() returns ["quantization_config.json"]
    # which matches our file.
    #
    # The latest nightly vLLM INC loader properly handles BF16 MoE layers:
    # it checks isinstance(layer, RoutedExperts) and returns
    # UnquantizedFusedMoEMethod (creates w13_weight, w2_weight) for
    # layers marked with bits=16 in extra_config.
    #
    # No dynamic patterns needed — extra_config entries with bits=16
    # are sufficient for INC to detect BF16 layers.

    # Clean up any old auto_gptq artifacts from previous ASAQ runs
    quant_config.pop("quant_method", None)  # keep auto-round from source
    quant_config.pop("desc_act", None)  # not needed for INC
    quant_config.pop("dynamic", None)  # not needed for INC

    # Update extra_config to mark all substituted layer components as FP16.
    # This is critical for vLLM: without these entries, vLLM assumes the
    # layers are quantized and tries to load qweight/qzeros/scales tensors
    # that no longer exist.
    if weights is not None:
        extra_config = quant_config.get("extra_config", {})
        if not isinstance(extra_config, dict):
            extra_config = {}

        # CRITICAL: Remove ALL old vLLM format MoE keys from extra_config.
        # Previous ASAQ runs may have left keys like "language_model.model.layers.X.mlp.experts"
        # (vLLM format) which conflict with the correct HF format keys.
        # These MUST be removed before adding new HF format keys.
        _vllm_moe_prefix = "language_model.model.layers."
        old_vllm_moe_keys = [
            k for k in extra_config
            if k.startswith(_vllm_moe_prefix) and "mlp.experts" in k
        ]
        for key in old_vllm_moe_keys:
            del extra_config[key]

        # Add extra_config entries for individual sub-layers (weight, bias)
        # IMPORTANT: Store keys in HuggingFace format (model.language_model.layers.X...)
        # vLLM's mapper will transform them to vLLM format (language_model.model.layers.X...)
        # This avoids double-transformation issues.
        #
        # CRITICAL: Do NOT add MoE-related keys (mlp.experts, mlp.experts.gate_up_proj, etc.)
        # The original quantized model has 0 MoE-related extra_config keys and works correctly.
        # Adding MoE keys interferes with vllm's normal quantization detection and weight loading.
        # Only add keys for non-MoE layers (attention, shared_expert, etc.)
        _HF_PREFIX = "model.language_model."
        # Skip MoE-related tensors. The original model has 0 MoE-related extra_config keys.
        # We need to skip:
        # - mlp.experts (the MoE layer itself)
        # - mlp.experts.N.gate_proj/up_proj/down_proj (individual expert weights)
        # - mlp.gate_proj/mlp.up_proj/mlp.down_proj (when not under shared_expert)
        # But allow:
        # - mlp.shared_expert.gate_proj/up_proj/down_proj
        # - mlp.shared_expert_gate
        _MOE_SKIP_PATTERNS = ("mlp.experts",)
        _MOE_PROJ_PATTERNS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

        for idx in layer_indices:
            prefix = _layer_prefix(idx)
            # Find all non-quantized tensors for this layer (weight, bias, etc.)
            for tensor_name in weights:
                if (
                    tensor_name.startswith(prefix)
                    and not _is_quantized_tensor(tensor_name, prefix)
                ):
                    # Skip MoE-related tensors (experts, gate_up_proj, down_proj, etc.)
                    # The original model has no MoE extra_config keys and works correctly
                    if any(keyword in tensor_name for keyword in _MOE_SKIP_PATTERNS):
                        continue
                    # Skip mlp.gate_proj/mlp.up_proj/mlp.down_proj unless they're under shared_expert
                    if any(proj in tensor_name for proj in _MOE_PROJ_PATTERNS):
                        if "shared_expert" not in tensor_name:
                            continue
                    # Skip norms and other non-linear params (they're always FP16)
                    if any(
                        skip in tensor_name
                        for skip in ("layernorm", "A_log", "conv1d", "dt_bias", "norm.weight")
                    ):
                        continue
                    # Strip .weight / .bias suffix for vLLM compatibility
                    # vLLM checks: layer_name == prefix (without .weight)
                    layer_key = tensor_name
                    for suffix in (".weight", ".bias"):
                        if layer_key.endswith(suffix):
                            layer_key = layer_key[: -len(suffix)]
                            break
                    # Keep key in HuggingFace format - vLLM's mapper will transform it
                    # HF: model.language_model.layers.X.mlp.shared_expert.gate_proj
                    # vLLM mapper transforms to: language_model.model.layers.X.mlp.shared_expert.gate_proj
                    # Add/update entry in extra_config (keep in HF format)
                    # Note: Old vLLM format keys are already cleaned up at the start of this function
                    if layer_key not in extra_config or (
                        isinstance(extra_config.get(layer_key), dict)
                        and extra_config[layer_key].get("bits", 16) != 16
                    ):
                        extra_config[layer_key] = {"bits": 16, "data_type": "fp"}

            # CRITICAL: Add MoE layer prefix key for substituted layers!
            # vllm needs this key to detect the MoE layer as unquantized.
            # Without this key, vllm detects the layer as quantized (via block_name_to_quantize)
            # and creates quantized parameters (w13_qweight, w2_qweight) instead of
            # unquantized parameters (w13_weight, w2_weight).
            #
            # Store in HF format so mapper transforms it to vLLM format:
            # HF: model.language_model.layers.X.mlp.experts
            # vLLM: language_model.model.layers.X.mlp.experts
            moe_prefix_hf = f"model.language_model.layers.{idx}.mlp.experts"
            if moe_prefix_hf not in extra_config:
                extra_config[moe_prefix_hf] = {"bits": 16, "data_type": "fp"}

        # We add extra_config keys for SUBSTITUTED layers (with bits=16)
        # to tell vllm they're unquantized. Non-substituted layers are left
        # alone so vllm detects them as quantized normally.

        quant_config["extra_config"] = extra_config

    with open(config_path, "w") as f:
        json.dump(quant_config, f, indent=2)

    # Also update config.json's quantization_config key.
    # vLLM reads from config.json (not the separate quantization_config.json),
    # so the substituted layer info must be there for vLLM to detect which
    # layers are unquantized.
    config_json_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_json_path):
        with open(config_json_path) as f:
            model_config: dict[str, Any] = json.load(f)

        # Merge our extra_config into config.json's quantization_config.
        # Keep quant_method as "auto-round" (INC format) — do NOT change to
        # "auto_gptq" as that requires a different config file name.
        model_qc = model_config.get("quantization_config", {})
        if model_qc:
            model_ec = model_qc.get("extra_config", {})
            if not isinstance(model_ec, dict):
                model_ec = {}
            # Update with our substitutions (bits=16 entries win)
            # Remove old vLLM format MoE keys that may be left from previous runs
            # These interfere with vllm's normal quantization detection
            old_vllm_keys = [k for k in model_ec.keys() if k.startswith("language_model.model.layers.") and "mlp.experts" in k]
            for key in old_vllm_keys:
                del model_ec[key]
            # Now merge our extra_config (which has HF format keys)
            model_ec.update(quant_config.get("extra_config", {}))
            model_qc["extra_config"] = model_ec
            # Clean up auto_gptq artifacts from previous runs
            model_qc.pop("desc_act", None)
            model_qc.pop("dynamic", None)
            # Ensure quant_method stays as auto-round (not auto_gptq)
            if model_qc.get("quant_method") == "auto_gptq":
                model_qc["quant_method"] = "auto-round"
            model_config["quantization_config"] = model_qc

            with open(config_json_path, "w") as f:
                json.dump(model_config, f, indent=2)


# ---------------------------------------------------------------------------
# 8. compute_model_size
# ---------------------------------------------------------------------------


def compute_model_size(weights: dict[str, torch.Tensor]) -> int:
    """Compute total size of weights dict in bytes.

    Args:
        weights: Dict mapping tensor names to tensors.

    Returns:
        Total size in bytes.
    """
    total = 0
    for tensor in weights.values():
        total += tensor.nelement() * tensor.element_size()
    return total


# ---------------------------------------------------------------------------
# 9. select_layers_from_report
# ---------------------------------------------------------------------------


def select_layers_from_report(
    quantized_dir: str,
    *,
    top_n: int | None = None,
    threshold: float | None = None,
    metric: str = "cosine",
) -> list[int]:
    """Read quantization-report.csv and select layers for FP16 substitution.

    Selection strategies (in priority order):

    1. ``top_n`` — Pick the N worst layers by the specified metric.
    2. ``threshold`` — Select all layers below the threshold.
    3. Neither — Select all layers that failed (``passed=false``).

    Args:
        quantized_dir: Directory containing quantization-report.csv.

    Keyword Args:
        top_n: Number of worst layers to select.
        threshold: Quality threshold. Layers below this are selected.
            For ``cosine`` metric: layers with cosine_sim < threshold.
            For ``psnr`` metric: layers with psnr_db < threshold.
        metric: Which metric to sort/select by (``"cosine"`` or ``"psnr"``).

    Returns:
        Sorted list of layer indices.

    Raises:
        FileNotFoundError: If quantization-report.csv is missing.
        ValueError: If the CSV is empty or has invalid format.
    """
    import csv as _csv

    csv_path = os.path.join(quantized_dir, "quantization-report.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Quantization report not found: {csv_path}\n"
            "Run spark-auto-round first to generate the report."
        )

    # Read CSV
    rows: list[dict[str, str]] = []
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Quantization report is empty: {csv_path}")

    # Parse layer index from block_name (e.g. "model.layers.54" -> 54)
    def _parse_layer_index(block_name: str) -> int:
        # Handle both "model.layers.54" and "model.language_model.layers.54"
        parts = block_name.rstrip(".").split(".")
        return int(parts[-1])

    # Parse numeric fields
    for row in rows:
        row["_index"] = _parse_layer_index(row["block_name"])
        row["_cosine"] = float(row["cosine_sim"])
        row["_psnr"] = float(row["psnr_db"])
        row["_passed"] = row["passed"].strip().lower() == "true"

    # Select layers based on strategy
    if top_n is not None:
        # Pick N worst layers by metric (lower = worse for both)
        key = "_cosine" if metric == "cosine" else "_psnr"
        rows.sort(key=lambda r: r[key])  # ascending: worst first
        selected = rows[:top_n]
    elif threshold is not None:
        # Select all layers below threshold
        if metric == "cosine":
            selected = [r for r in rows if r["_cosine"] < threshold]
        else:
            selected = [r for r in rows if r["_psnr"] < threshold]
    else:
        # Default: select all layers that failed
        selected = [r for r in rows if not r["_passed"]]

    if not selected:
        print("All layers passed quality checks — nothing to substitute.")
        return []

    # Sort by layer index and return
    selected.sort(key=lambda r: r["_index"])
    return [r["_index"] for r in selected]


# ---------------------------------------------------------------------------
# 11. generate_asaq_report
# ---------------------------------------------------------------------------


def generate_asaq_report(
    quantized_dir: str,
    output_dir: str,
    layer_indices: list[int],
) -> Path:
    """Generate an ASAQ-aware quantization report.

    Reads the original ``quantization-report.txt`` (if available), adds an
    ASAQ header, and marks substituted layers with an "FP16 (substituted)"
    action.

    Args:
        quantized_dir: Directory of original quantized model.
        output_dir: Output directory for new report.
        layer_indices: Layer indices that were substituted.

    Returns:
        :class:`~pathlib.Path` to generated report.
    """
    from datetime import datetime

    # Read original report if available
    original_report_path = os.path.join(quantized_dir, "quantization-report.txt")
    if os.path.exists(original_report_path):
        with open(original_report_path) as f:
            original_lines = f.readlines()
    else:
        original_lines = ["(no original report found)\n"]

    # Build ASAQ report
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    layers_str = ", ".join(str(i) for i in sorted(layer_indices))

    lines = [
        "=== Quantization Report (ASAQ) ===",
        f"Date: {now}",
        f"Substituted layers: {layers_str}",
        "",
        "--- Original Report ---",
    ]
    lines.extend(original_lines)

    # Mark substituted layers
    lines.append("")
    lines.append("ASAQ Substitutions:")
    for idx in sorted(layer_indices):
        lines.append(
            f"  🟢 model.language_model.layers.{idx:<35} FP16 (substituted)"
        )
    lines.append("")

    # Write report
    output_path = Path(output_dir) / "quantization-report.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")

    return output_path

