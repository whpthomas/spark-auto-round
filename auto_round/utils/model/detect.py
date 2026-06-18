# Copyright (c) 2025 Intel Corporation
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
"""Model type detection, block names, and expert helper utilities."""
import collections
import inspect
import json
import os
import re
from typing import TYPE_CHECKING, Union

import torch

from auto_round import envs
from auto_round.logger import logger
from auto_round.utils.common import AUDIO_MM_KEYS, VISION_MM_KEYS
from auto_round.utils.model.utils import (
    check_to_quantized,
    get_common_prefix,
    get_layer_features,
    get_module,
    get_nested_attr,
    set_nested_attr,
)

if TYPE_CHECKING:
    from auto_round.schemes import QuantizationScheme


# Maps architecture class names to virtual model_type keys.
ARCHITECTURE_MODEL_TYPE_MAP = {
    "MiMoAudioModel": "mimo_audio",
    "MiMoAudioForCausalLM": "mimo_audio",
}

FIX_MISTRAL_REGEX_MODEL_TYPE_LIST = ["longcat_next"]

_is_mllm_model_cache: dict = {}
# Model types that have multimodal components but should use LLM compressor
# (text-only calibration, non-text modules excluded from quantization).
_LLM_ONLY_MODEL_TYPES = {"bagel"}


def resolve_model_type(model):
    """Resolve the effective model type using architecture class name as primary source.

    Args:
        model: A model instance with optional config attribute.

    Returns:
        str or None: The resolved model type identifier, or None if config is missing.
    """
    config = getattr(model, "config", None)
    if config is None:
        return None
    archs = getattr(config, "architectures", None)
    if archs:
        for arch in archs:
            if arch in ARCHITECTURE_MODEL_TYPE_MAP:
                return ARCHITECTURE_MODEL_TYPE_MAP[arch]
    return getattr(config, "model_type", None)


def is_moe_layer(module: torch.nn.Module) -> bool:
    """Returns whether the module is an MOE layer."""
    return "moe" in type(module).__name__.lower() or any(
        key in type(module).__name__.lower()
        for key in [
            "MixtralSparseMoeBlock".lower(),
            "ArcticMoE".lower(),
            "DbrxFFN".lower(),
            "MoELayer".lower(),
            "PhimoeSparseMoeBlock".lower(),
            "DeepseekMoE".lower(),
            "DeepseekV2MoE".lower(),
            "DeepseekV3MoE".lower(),
            "Qwen2MoeSparseMoeBlock".lower(),
            "Qwen3MoeSparseMoeBlock".lower(),
            "Qwen3VLMoeTextSparseMoeBlock".lower(),
            "Qwen3OmniMoeThinkerTextSparseMoeBlock".lower(),
            "Qwen3OmniMoeTalkerTextSparseMoeBlock".lower(),
        ]
    )


def get_model_name_or_path(model_or_path: Union[str, "torch.nn.Module"]) -> "str | None":
    """Extract the model name/path from a model object or string."""
    if isinstance(model_or_path, str):
        return model_or_path
    return getattr(model_or_path, "_name_or_path", None) or getattr(model_or_path, "name_or_path", None)


def is_mllm_model(model_or_path: Union[str, "torch.nn.Module"], platform: str = None) -> bool:
    """Check if model is a multimodal model (vision, audio, etc.).

    Ported from upstream auto-round auto_round/utils/model.py:958-1009.
    """
    from auto_round.utils.common import MM_KEYS
    from auto_round.utils.model.load import download_hf_model

    model_path = get_model_name_or_path(model_or_path)

    # Fast path: return cached result for already-seen paths
    if model_path in _is_mllm_model_cache:
        return _is_mllm_model_cache[model_path]

    # Check model_type exclusion: some models have multimodal components
    # but should be quantized as LLM (e.g., BAGEL MoT).
    _model_type = None
    if isinstance(model_or_path, torch.nn.Module) and hasattr(model_or_path, "config"):
        _model_type = getattr(model_or_path.config, "model_type", None)
    elif isinstance(model_path, str) and os.path.isdir(model_path):
        _cfg_path = os.path.join(model_path, "config.json")
        if os.path.exists(_cfg_path):
            with open(_cfg_path) as _f:
                _model_type = json.load(_f).get("model_type")
    if _model_type in _LLM_ONLY_MODEL_TYPES:
        return False

    # For dummy model, model_path could be "".
    # Only try to download if the path looks like a HF repo id (not a local filesystem path).
    _is_local_path = os.path.isabs(model_path) or model_path.startswith("./") or model_path.startswith("../")
    if model_path and not os.path.isdir(model_path) and not _is_local_path:
        model_path = download_hf_model(model_path)

    result = False
    if isinstance(model_path, str):
        if os.path.exists(os.path.join(model_path, "preprocessor_config.json")):
            result = True
        elif os.path.exists(os.path.join(model_path, "processor_config.json")):
            result = True
        elif os.path.exists(os.path.join(model_path, "config.json")):
            with open(os.path.join(model_path, "config.json")) as f:
                config = json.load(f)
            for key in config.keys():
                if any([k in key for k in MM_KEYS]):
                    result = True
                    break

    if not result and isinstance(model_or_path, torch.nn.Module):
        for name, module in model_or_path.named_modules():
            if any([k in name for k in MM_KEYS]):
                result = True
                break

    # Cache by the original path key
    original_key = get_model_name_or_path(model_or_path)
    _is_mllm_model_cache[original_key] = result
    return result


def is_pure_text_model(model) -> bool:
    """Check if model is a pure text model (no vision/audio modules)."""
    for n, m in model.named_modules():
        for key in VISION_MM_KEYS + AUDIO_MM_KEYS:
            if key in n.lower():
                return False
    return True


def get_block_names(model, quant_vision=False):
    """Get the block names for transformers-like networks.

    Args:
        model: The model.
        quant_vision: Whether to quantize vision blocks.

    Returns:
        block_names: A list whose elements are list of block's layer names
    """
    from auto_round.special_model_handler import SPECIAL_MULTIMODAL_BLOCK

    def _search_block(name, module):
        if hasattr(type(module), "__name__") and "ModuleList" in type(module).__name__:
            return [(name, module)]
        target_modules = []
        for n, m in module.named_children():
            if hasattr(type(m), "__name__") and "NgramEmbedding" in type(m).__name__:
                continue
            if hasattr(type(m), "__name__") and "ModuleList" in type(m).__name__:
                target_modules.append((".".join(filter(None, (name, n))), m))
            else:
                target_modules.extend(_search_block(".".join(filter(None, (name, n))), m))
        return target_modules

    def _get_llm_block_names(model):
        block_names = []
        target_modules = _search_block("", model)

        for i, target_m in enumerate(target_modules):
            block_names.append([])
            for n, m in target_m[1].named_children():
                block_names[i].append(target_m[0] + "." + n)
        return block_names

    def _get_vlm_block_names(model, quant_vision=False, ignore_audio=True):
        effective_type = resolve_model_type(model)
        if effective_type and effective_type in SPECIAL_MULTIMODAL_BLOCK:
            return SPECIAL_MULTIMODAL_BLOCK[effective_type](model, quant_vision=quant_vision)
        block_names = []
        target_modules = _search_block("", model)

        for i, target_m in enumerate(target_modules):
            if quant_vision or all(key not in target_m[0].lower() for key in VISION_MM_KEYS):
                if ignore_audio and any(key in target_m[0].lower() for key in AUDIO_MM_KEYS):
                    continue
                block_names.append([])
                for n, m in target_m[1].named_children():
                    block_names[-1].append(target_m[0] + "." + n)
        return block_names

    effective_type = resolve_model_type(model)
    if effective_type and effective_type in SPECIAL_MULTIMODAL_BLOCK:
        return SPECIAL_MULTIMODAL_BLOCK[effective_type](model, quant_vision=quant_vision)

    if quant_vision or not is_pure_text_model(model):
        return _get_vlm_block_names(model, quant_vision=quant_vision)
    else:
        return _get_llm_block_names(model)


def get_lm_head_name(model):
    """Get the name of the lm_head layer."""
    block_names = get_block_names(model, True)
    last_name = None
    for n, m in model.named_modules():
        if any(m.children()):
            continue
        last_name = n
    for l in block_names:
        if last_name in l:
            last_name = None
            break
    return last_name


def get_expert_linear_names(module: torch.nn.Module) -> list[str]:
    """Get the list of linear names for the experts."""

    def module_match_name_list(module, name_list):
        return any(name.lower() in type(module).__name__.lower() for name in name_list)

    if module_match_name_list(
        module,
        [
            "Qwen2MoeSparseMoeBlock",
            "Qwen3MoeSparseMoeBlock",
            "DeepseekMoE",
            "DeepseekV2MoE",
            "DeepseekV3MoE",
            "Qwen3VLMoeTextSparseMoeBlock",
            "Qwen3OmniMoeThinkerTextSparseMoeBlock",
            "Qwen3OmniMoeTalkerTextSparseMoeBlock",
        ],
    ):
        return ["gate_proj", "down_proj", "up_proj"]
    elif module_match_name_list(module, ["MixtralMoeSparseMoeBlock"]):
        return ["linear_fc1", "linear_fc2"]
    elif module_match_name_list(module, ["DBRXMoeSparseMoeBlock"]):
        return ["w1_linear", "w2_linear", "v1_linear"]
    else:
        return ["w1", "w2", "w3"]


def get_expert_input_proj_names(module: torch.nn.Module) -> list[str]:
    """Get the list of input projection names for MoE experts."""

    def module_match_name_list(module, name_list):
        return any(name.lower() in type(module).__name__.lower() for name in name_list)

    if module_match_name_list(
        module,
        [
            "Qwen2MoeSparseMoeBlock",
            "Qwen3MoeSparseMoeBlock",
            "Qwen3VLMoeTextSparseMoeBlock",
            "Qwen3OmniMoeThinkerTextSparseMoeBlock",
            "Qwen3OmniMoeTalkerTextSparseMoeBlock",
            "DeepseekMoE",
            "DeepseekV2MoE",
            "DeepseekV3MoE",
        ],
    ):
        return ["gate_proj", "up_proj"]
    elif module_match_name_list(module, ["MixtralMoeSparseMoeBlock"]):
        return ["linear_fc1"]
    elif module_match_name_list(module, ["DBRXMoeSparseMoeBlock"]):
        return ["w1_linear", "v1_linear"]
    else:
        logger.warning_once("Using default input projection names ['w1', 'w3'] for MoE expert alignment. ")
        return ["w1", "w3"]


def get_layer_names_in_block(
    model: torch.nn.Module,
    supported_types=(torch.nn.Linear,),
    quant_block_list: list = None,
    class_names: tuple = None,
) -> list[str]:
    """Retrieves the names of layers within each block of the model."""
    import transformers

    if class_names is None:
        class_names = []
    for n, m in model.named_modules():
        if type(m) in supported_types or (class_names is not None and m.__class__.__name__ in class_names):
            m.bk_global_name = n
    layers_in_block = []
    if bool(quant_block_list):
        all_blocks = quant_block_list
    else:
        all_blocks = get_block_names(model)
    for block_names in all_blocks:
        for block_name in block_names:
            block = get_module(model, block_name)
            for n, m in block.named_modules():
                if hasattr(m, "bk_global_name"):
                    layers_in_block.append(m.bk_global_name)
                    delattr(m, "bk_global_name")
    return layers_in_block


def is_moe_model(model: torch.nn.Module) -> bool:
    """Check if the model is a MoE model."""
    if hasattr(model, "config") and hasattr(model.config, "to_dict"):
        for key in model.config.to_dict().keys():
            if "moe" in key or "expert" in key:
                return True
    for n, m in model.named_modules():
        if "expert" in n:
            return True
    return False


def is_moe_model_via_config(config) -> bool:
    """Check if config indicates a MoE model."""
    try:
        config_str = str(config).lower()
    except Exception:
        config_str = str(config.to_dict()).lower() if hasattr(config, "to_dict") else ""
    if "moe" in config_str or "expert" in config_str:
        return True
    return False


def is_separate_lm_head(model: torch.nn.Module) -> bool:
    """Check if lm_head weight is in a separate file."""
    from auto_round.utils.model.load import download_hf_model

    dir_path = model.name_or_path
    if not os.path.isdir(dir_path):
        dir_path = download_hf_model(dir_path)
    lm_head_name: str = get_lm_head_name(model)
    lm_head_name += ".weight"

    if "model.safetensors.index.json" in os.listdir(dir_path):
        with open(os.path.join(dir_path, "model.safetensors.index.json")) as f:
            index_mapping = json.load(f)
            if lm_head_name in index_mapping["weight_map"]:
                return True
            else:
                return False
    else:
        from safetensors import safe_open

        f = safe_open(os.path.join(dir_path, "model.safetensors"), framework="pt")
        if lm_head_name in f.keys():
            return True
        else:
            return False


def is_separate_tensor(model: torch.nn.Module, tensor_name: str) -> bool:
    """Check if a tensor is in a separate file."""
    from auto_round.utils.model.load import download_hf_model

    dir_path = model.name_or_path
    if not os.path.isdir(dir_path):
        dir_path = download_hf_model(dir_path)
    if not tensor_name.endswith(".weight"):
        tensor_name += ".weight"

    if "model.safetensors.index.json" in os.listdir(dir_path):
        with open(os.path.join(dir_path, "model.safetensors.index.json")) as f:
            index_mapping = json.load(f)
            if tensor_name in index_mapping["weight_map"]:
                return True
            else:
                return False
    else:
        from safetensors import safe_open

        f = safe_open(os.path.join(dir_path, "model.safetensors"), framework="pt")
        if tensor_name in f.keys():
            return True
        else:
            return False


def extract_block_names_to_str(quant_block_list):
    """Extract block names to a comma-separated string."""
    if not isinstance(quant_block_list, (list, tuple)):
        return None
    prefixes = [get_common_prefix(blocks) for blocks in quant_block_list]
    return ",".join(prefixes)


def find_matching_blocks(model, all_blocks, to_quant_block_names):
    """Find and return matching blocks in the model based on to_quant_block_names."""
    if not to_quant_block_names:
        return all_blocks
    to_quant_block_list = to_quant_block_names
    if isinstance(to_quant_block_names, list) or isinstance(to_quant_block_names, tuple):
        return to_quant_block_names
    if isinstance(to_quant_block_names, str):
        to_quant_block_list = [name.strip() for name in to_quant_block_names.split(",")]
    target_blocks = []
    for block_list in all_blocks:
        matched_sublist = []
        for name in to_quant_block_list:
            matches = [block for block in block_list if re.search(name, block)]
            if matches:
                matched_sublist.extend(matches)
        if matched_sublist:
            target_blocks.append(matched_sublist)
    if not target_blocks:
        raise ValueError(
            "No block names matched. Please check the input for to_quant_block_name,"
            "or set to_quant_block_name to None to automatically match quantizable blocks."
        )
    return target_blocks


def merge_block_output_keys(block, input_others, extra_keys):
    """Merge block output keys into input_others, resolving positional/keyword conflicts."""
    positional_inputs = input_others.get("positional_inputs")
    if not positional_inputs or not extra_keys:
        input_others.update(extra_keys)
        return

    try:
        sig = inspect.signature(block.forward)
    except (ValueError, TypeError):
        input_others.update(extra_keys)
        return

    params = [p for p in sig.parameters.keys() if p != "self"]

    positional_inputs = list(positional_inputs)
    for key, value in extra_keys.items():
        if key in params:
            pos_idx = params.index(key) - 1
            if 0 <= pos_idx < len(positional_inputs):
                positional_inputs[pos_idx] = value
                continue
        input_others[key] = value
    input_others["positional_inputs"] = tuple(positional_inputs)


def wrap_block_forward_positional_to_kwargs(base_hook):
    """Wrap a block forward hook to convert positional inputs to keyword args."""
    _param_names_cache: dict = {}

    def forward(m, hidden_states=None, *positional_inputs, **kwargs):
        if positional_inputs:
            m_id = id(m)
            if m_id not in _param_names_cache:
                sig_target = getattr(m, "_true_orig_forward", None) or m.orig_forward
                sig = inspect.signature(sig_target)
                _param_names_cache[m_id] = [p for p in sig.parameters.keys() if p != "self"]
            _param_names = _param_names_cache[m_id]
            for i, val in enumerate(positional_inputs):
                param_idx = i + 1
                if param_idx < len(_param_names):
                    param_name = _param_names[param_idx]
                    if param_name not in kwargs:
                        kwargs[param_name] = val
            positional_inputs = ()
        return base_hook(m, hidden_states, *positional_inputs, **kwargs)

    return forward


def set_amax_for_uncalibrated_experts(
    experts: torch.nn.Module, set_amax_value: float | None = None, attr_name="act_max", unify_all: bool = False
):
    """Set amax of uncalibrated experts to a given value or the max of existing amax value from other experts."""
    uncalibrated_experts = []

    def _get_attr(module, name):
        if hasattr(module, name):
            return getattr(module, name)
        if hasattr(module, "orig_layer") and hasattr(module.orig_layer, name):
            return getattr(module.orig_layer, name)
        return None

    def _get_amax_value(module):
        value = get_nested_attr(module, attr_name)
        if value is None and hasattr(module, "orig_layer"):
            value = get_nested_attr(module.orig_layer, attr_name)
        return value

    if set_amax_value is None:
        amax_values = [_get_amax_value(m) for m in experts if _get_amax_value(m) is not None]
        if len(amax_values) == 0:
            sample = next((m for m in experts if m is not None), None)
            if sample is not None:
                act_bits = _get_attr(sample, "act_bits")
                act_dynamic = _get_attr(sample, "act_dynamic")
                is_quantized = "Quant" in sample.__class__.__name__ or hasattr(sample, "is_mx")
                needs_warning = (
                    not is_quantized and isinstance(act_bits, (int, float)) and act_bits < 8 and not act_dynamic
                )
                if needs_warning:
                    logger.warning_once(
                        f"All {len(experts)} expert layers are missing '{attr_name}' values. "
                        f"This may indicate calibration hooks were not attached to expert layers."
                    )
            return uncalibrated_experts
        flat_values = [t.reshape(-1) for t in amax_values]
        all_values = torch.cat(flat_values)
        set_amax_value = torch.max(all_values)
        set_amax_value = set_amax_value.unsqueeze(0) if set_amax_value.dim() == 0 else set_amax_value

    for module in experts:
        current_amax = _get_amax_value(module)

        if current_amax is None or unify_all:
            if not isinstance(set_amax_value, torch.Tensor):
                set_amax_value = torch.tensor(set_amax_value, dtype=torch.float32)
            set_nested_attr(module, attr_name, set_amax_value.clone())
            if current_amax is None:
                uncalibrated_experts.append(module)

    if uncalibrated_experts:
        logger.info_once(
            f"Found {len(uncalibrated_experts)} uncalibrated expert layers. "
            "Using max amax from calibrated experts to fill missing values. "
        )

    return uncalibrated_experts


def _is_fused_experts_module(module: torch.nn.Module) -> bool:
    """Check if the module is a fused experts module (has 3D Parameter gate_up_proj/down_proj)."""
    if not hasattr(module, "gate_up_proj") or not hasattr(module, "down_proj"):
        return False
    return (
        isinstance(module.gate_up_proj, torch.nn.Parameter)
        and isinstance(module.down_proj, torch.nn.Parameter)
        and module.gate_up_proj.dim() == 3
        and module.down_proj.dim() == 3
    )


def _set_amax_for_moe_auxiliary_layers(moe_module: torch.nn.Module, attr_name: str = "act_max"):
    """Set amax for auxiliary layers in MOE modules (gate/router, shared_experts)."""
    layers_needing_amax = []

    if hasattr(moe_module, "gate") and isinstance(moe_module.gate, torch.nn.Linear):
        gate = moe_module.gate
        if hasattr(gate, "act_bits") and gate.act_bits < 8:
            if get_nested_attr(gate, attr_name) is None:
                layers_needing_amax.append(gate)

    if hasattr(moe_module, "shared_experts"):
        shared_experts = moe_module.shared_experts
        if shared_experts is not None:
            for child_name, child in shared_experts.named_modules():
                if isinstance(child, torch.nn.Linear):
                    if hasattr(child, "act_bits") and child.act_bits < 8:
                        if get_nested_attr(child, attr_name) is None:
                            layers_needing_amax.append(child)

    if not layers_needing_amax:
        return

    reference_amax = _get_reference_amax_from_experts(moe_module, attr_name)

    if reference_amax is not None:
        for layer in layers_needing_amax:
            if not isinstance(reference_amax, torch.Tensor):
                reference_amax = torch.tensor(reference_amax, dtype=torch.float32)
            set_nested_attr(layer, attr_name, reference_amax.clone())
        logger.info_once(
            f"Set act_max for {len(layers_needing_amax)} MOE auxiliary layers (gate/shared_experts) "
            f"using reference value from calibrated experts."
        )
    else:
        logger.warning_once(
            f"Cannot set act_max for {len(layers_needing_amax)} MOE auxiliary layers: "
            f"no calibrated experts found to use as reference."
        )


def _get_reference_amax_from_experts(moe_module: torch.nn.Module, attr_name: str = "act_max"):
    """Get a reference amax value from calibrated expert layers."""
    amax_values = []

    if not hasattr(moe_module, "experts"):
        return None

    experts = moe_module.experts

    if isinstance(experts, collections.abc.Iterable):
        expert_linear_names = get_expert_linear_names(moe_module)
        for expert in experts:
            for linear_name in expert_linear_names:
                layer = getattr(expert, linear_name, None)
                if layer is not None:
                    amax = get_nested_attr(layer, attr_name)
                    if amax is not None:
                        amax_values.append(amax)

    if not amax_values:
        return None

    flat_values = [t.reshape(-1) for t in amax_values]
    all_values = torch.cat(flat_values)
    return torch.max(all_values)


def set_amax_for_all_moe_layers(model: torch.nn.Module, layer_name=None, attr_name="act_max"):
    """Set amax for all MoE layers in the model."""
    if layer_name is not None:
        parts = layer_name.split(".")
        if "experts" not in parts:
            raise ValueError
        idx = parts.index("experts")
        moe_name = ".".join(parts[:idx])
        model = get_module(model, moe_name)
    for name, sub_module in model.named_modules():
        if not (is_moe_layer(sub_module) and hasattr(sub_module, "experts")):
            continue

        _set_amax_for_moe_auxiliary_layers(sub_module, attr_name=attr_name)

        expert_linear_names = get_expert_linear_names(sub_module)
        expert_input_proj_names = get_expert_input_proj_names(sub_module)

        if _is_fused_experts_module(sub_module.experts):
            logger.debug(
                f"Skipping act_max setting for fused experts module '{name}': "
                f"fused experts use parent module's act_max"
            )
            continue
        elif isinstance(sub_module.experts, collections.abc.Iterable):
            for linear_name in expert_linear_names:
                try:
                    unify_scale = linear_name in expert_input_proj_names and envs.AR_ENABLE_UNIFY_MOE_INPUT_SCALE

                    set_amax_for_uncalibrated_experts(
                        [getattr(expert, linear_name, None) for expert in sub_module.experts],
                        attr_name=attr_name,
                        unify_all=unify_scale,
                    )
                except AttributeError as e:
                    expert_types = list(set(type(expert).__name__ for expert in sub_module.experts))
                    raise AttributeError(
                        f"Failed to access attribute '{linear_name}' on experts. "
                        f"MoE module type: {type(sub_module).__name__}, "
                        f"Expert types: {expert_types}, "
                        f"Expected linear names: {expert_linear_names}. "
                        f"This suggests the get_expert_linear_names function may need "
                        f"to be updated for this model architecture. "
                        f"Original error: {e}"
                    ) from e
        else:
            logger.warning(
                f"Unknown experts structure in '{name}': type={type(sub_module.experts).__name__}. "
                f"Skipping act_max setting. This may cause issues during export."
            )
