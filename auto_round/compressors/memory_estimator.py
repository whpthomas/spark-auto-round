# Copyright (c) 2026 Intel Corporation
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
"""Per-block peak GPU memory estimation for spark-auto-round quantization.

Computes an estimate of peak GPU memory consumed during a single quantization
block's forward/backward/optimize cycle. Used by the auto-tuner (auto_tune.py)
to detect memory pressure before a run begins.

Usage:
    from auto_round.compressors.memory_estimator import estimate_peak_memory_per_block

    peak_gb, breakdown = estimate_peak_memory_per_block(config, {
        "batch_size": 8, "seqlen": 2048,
    })
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_peak_memory_per_block(
    config: "AutoConfig",  # noqa: F821 — HuggingFace AutoConfig
    user_settings: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Compute peak GPU memory (GB) for a single quantization block.

    Parameters
    ----------
    config : AutoConfig
        HuggingFace model configuration (AutoConfig.from_pretrained).
    user_settings : dict
        Must contain keys:
            batch_size : int
            seqlen : int
        Optionally:
            nsamples : int      (informational; does not affect GPU peak)
            group_size : int    (default 128)

    Returns
    -------
    peak_gb : float
        Estimated peak GPU memory in gibibytes (GiB), including 1.15× safety.
    breakdown : dict
        Per-component sizes in GiB for debugging.

    Raises
    ------
    ValueError
        If required keys are missing from user_settings or values are invalid.
    """
    _validate_settings(user_settings)

    bs = user_settings["batch_size"]
    seqlen = user_settings["seqlen"]

    # -- Extract model dimensions -------------------------------------------
    dims = _get_hidden_dimensions(config)
    hidden = dims["hidden_size"]
    num_layers = dims["num_layers"]  # noqa: F841 — available for future use

    block_params = _get_block_params(config, dims)

    # -- Component sizes in bytes -------------------------------------------
    # Each weight is stored as bf16 (2 bytes)
    block_weight_bytes = block_params * 2

    # Wrapper duplicates params as fp32 (4 bytes) — value + min_scale + max_scale
    # value (round) has same element count as weights
    # min_scale / max_scale: hidden × ceil(hidden / group_size) each — negligible
    # Conservative: block_params × 4 for value, plus small overhead for scales
    wrapper_value_bytes = block_params * 4
    # Scale params: ~2 × hidden × ceil(hidden / 128) × 4 — typically << 1% of weights
    group_size = user_settings.get("group_size", 128)
    scale_params_per_scale = hidden * math.ceil(hidden / group_size)
    wrapper_scale_bytes = 2 * scale_params_per_scale * 4  # min_scale + max_scale

    # Activations during forward: bs × seqlen × hidden × 2 (bf16)
    # During backward: same size (gradients are same shape as activations)
    # Peak activation memory = forward + backward = 2 × the batch footprint
    # For MoE: activations scale with top_k experts during MoE FFN
    top_k = dims.get("top_k")
    if not isinstance(top_k, (int, float)):
        top_k = dims.get("num_experts_per_tok", 1)
    if not isinstance(top_k, (int, float)):
        top_k = 1
    activation_bytes = bs * seqlen * hidden * 2 * top_k  # forward
    gradient_bytes = activation_bytes  # backward (same shape)

    # Attention scores: bs × num_heads × seqlen × seqlen × 2 (bf16)
    # This is the Q×K^T matrix — dominates memory for long sequences.
    # For GQA (grouped query attention), use num_key_value_heads.
    num_heads = dims.get("num_key_value_heads") or dims.get("num_attention_heads", 1)
    attention_scores_bytes = bs * num_heads * seqlen * seqlen * 2

    # QKV intermediates: stored for backward pass gradient computation
    # Q, K, V each: bs × seqlen × hidden × 2 (bf16)
    qkv_intermediate_bytes = 3 * bs * seqlen * hidden * 2

    # FFN intermediates: gate and up projection outputs, stored for backward
    # Each: bs × seqlen × intermediate × 2 (bf16)
    intermediate = dims["intermediate_size"]
    ffn_intermediate_bytes = 2 * bs * seqlen * intermediate * 2

    # Calibration input batch (on GPU during forward)
    calibration_bytes = bs * seqlen * hidden * 2

    # -- Total --------------------------------------------------------------
    components = {
        "block_weights_bf16": block_weight_bytes,
        "wrapper_value_fp32": wrapper_value_bytes,
        "wrapper_scales_fp32": wrapper_scale_bytes,
        "activation_forward": activation_bytes,
        "activation_backward": gradient_bytes,
        "attention_scores": attention_scores_bytes,
        "qkv_intermediate": qkv_intermediate_bytes,
        "ffn_intermediate": ffn_intermediate_bytes,
        "calibration_input": calibration_bytes,
    }

    total_bytes = sum(components.values())
    # 1.50× safety factor accounts for:
    # - PyTorch CUDA allocator fragmentation overhead (~5-10%)
    # - Autograd computation graph metadata (~15-20%)
    # - Additional intermediate tensors not explicitly modeled
    # - torch.compile kernel fusion effects
    # - System-level memory overhead
    safety_factor = 1.50
    peak_gb = (total_bytes * safety_factor) / (1024 ** 3)

    # Return breakdown in GiB for readability
    breakdown_gb = {
        k: v / (1024 ** 3) for k, v in components.items()
    }
    breakdown_gb["safety_margin"] = peak_gb - (total_bytes / (1024 ** 3))
    breakdown_gb["total_estimated"] = peak_gb

    return round(peak_gb, 2), breakdown_gb


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_hidden_dimensions(config: "AutoConfig") -> Dict[str, Any]:
    """Extract hidden dimensions from a HuggingFace AutoConfig.

    Handles common HF config formats (LlamaConfig, Qwen3.5Config, MixtralConfig,
    DeepseekConfig, etc.) and multimodal models with text_config sub-configs.
    Uses getattr with fallback for flexibility.

    Returns a dict with keys:
        hidden_size, intermediate_size, num_attention_heads,
        num_key_value_heads (optional), num_layers, num_experts (optional),
        top_k (optional, also known as num_experts_per_tok),
        max_position_embeddings
    """

    def _real_val(val):
        """Return the value if it's a real int/float, None otherwise.

        This handles MagicMock auto-created attributes: when a MagicMock is
        used as config, accessing an undefined attribute returns a MagicMock
        instead of None. This function normalises that to None.
        """
        if isinstance(val, (int, float, type(None))):
            return val
        return None

    def _getattr_real(obj, attr, default=None):
        """Like getattr, but return default even for MagicMock auto-creations."""
        val = getattr(obj, attr, None)
        return _real_val(val) if _real_val(val) is not None else default

    hidden_size = _getattr_real(config, "hidden_size")
    intermediate_size = _getattr_real(config, "intermediate_size")
    if intermediate_size is None:
        intermediate_size = _getattr_real(config, "ffn_dim")
    if intermediate_size is None:
        intermediate_size = _getattr_real(config, "moe_intermediate_size")

    # Fallback: multimodal models use text_config sub-config
    if hidden_size is None or intermediate_size is None:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            hidden_size = hidden_size or _getattr_real(text_cfg, "hidden_size")
            if intermediate_size is None:
                intermediate_size = _getattr_real(text_cfg, "intermediate_size")
            if intermediate_size is None:
                intermediate_size = _getattr_real(text_cfg, "ffn_dim")
            if intermediate_size is None:
                intermediate_size = _getattr_real(text_cfg, "moe_intermediate_size")

    # Last-resort fallbacks if still not found
    if hidden_size is None:
        hidden_size = _getattr_real(config, "d_model") or 4096
    if intermediate_size is None:
        intermediate_size = hidden_size * 4

    num_attention_heads = _getattr_real(config, "num_attention_heads") or 32
    num_key_value_heads = _getattr_real(config, "num_key_value_heads") or num_attention_heads
    num_layers = _getattr_real(config, "num_hidden_layers") or _getattr_real(config, "num_layers") or 32

    # MoE fields: try config first, then text_config
    num_experts = _getattr_real(config, "num_experts")
    if num_experts is None:
        num_experts = _getattr_real(config, "num_local_experts")
    if num_experts is None:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            num_experts = _getattr_real(text_cfg, "num_experts")
            if num_experts is None:
                num_experts = _getattr_real(text_cfg, "num_local_experts")

    top_k = _getattr_real(config, "top_k")
    if top_k is None:
        top_k = _getattr_real(config, "num_experts_per_tok")
    if top_k is None:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            top_k = _getattr_real(text_cfg, "top_k")
            if top_k is None:
                top_k = _getattr_real(text_cfg, "num_experts_per_tok")

    if top_k is None and num_experts is not None:
        top_k = 2  # default for Mixtral-style

    max_position_embeddings = _getattr_real(config, "max_position_embeddings") or 2048

    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "max_position_embeddings": max_position_embeddings,
    }


def _get_block_params(config: "AutoConfig",
                      dims: Dict[str, Any] = None) -> int:
    """Return the number of parameters in a single transformer block.

    Reimplements the logic from auto_round/utils/device/memory.py
    _estimate_param_count_from_config() but returns per-block (not total).

    Formula for a dense block:
        block_params = 4 * hidden_size^2 + 3 * hidden_size * intermediate_size

        - 4 * hidden_size^2 : Q, K, V, O projection matrices (each hidden×hidden)
        - 3 * hidden_size * intermediate_size : gate, up, down FFN projections

    For MoE:
        block_params = 4 * hidden_size^2  (self-attention)
                       + 3 * hidden_size * intermediate_size * num_experts  (expert FFNs)
    """
    if dims is None:
        dims = _get_hidden_dimensions(config)

    hidden = dims["hidden_size"]
    inter = dims["intermediate_size"]
    num_experts = dims.get("num_experts")

    # Self-attention: Q, K, V, O = 4 × hidden × hidden
    attn_params = 4 * hidden * hidden

    # FFN: gate, up, down = 3 × hidden × intermediate
    if num_experts is not None and isinstance(num_experts, int) and num_experts > 0:
        # MoE: each expert has its own FFN
        ffn_params = 3 * hidden * inter * num_experts
    else:
        ffn_params = 3 * hidden * inter

    # RMS norm / layer norm params: 2 × hidden (per sublayer) — negligible
    # Skip for simplicity (<< 0.1% of total)

    return attn_params + ffn_params


def _validate_settings(settings: Dict[str, Any]) -> None:
    """Validate user_settings keys and values. Raises ValueError on invalid."""
    required = ["batch_size", "seqlen"]
    for key in required:
        if key not in settings:
            raise ValueError(f"Missing required setting: '{key}'")
        val = settings[key]
        if not isinstance(val, int) or val < 1:
            raise ValueError(f"'{key}' must be a positive int, got {val}")