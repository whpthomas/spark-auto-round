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
import copy
import json
import os
import random
import re
import sys
from dataclasses import asdict, fields
from enum import Enum
from typing import TYPE_CHECKING, Callable, Union

import torch
import transformers
from torch.amp import autocast

from auto_round.logger import logger
from auto_round.utils import (
    check_to_quantized,
    compress_layer_names,
    get_layer_names_in_block,
    get_module,
    to_standard_regex,
)

if TYPE_CHECKING:
    from auto_round.schemes import QuantizationScheme


class BackendDataType(str, Enum):
    """Data type identifiers for quantization backends."""
    STANDARD_FP = "fp"
    MX_FP = "mx_fp"   # kept for defensive checks; MXFP quantization not supported
    NV_FP = "nv_fp"   # kept for defensive checks; NVFP quantization not supported


def is_standard_fp(backend):
    backend = backend.lower()
    return BackendDataType.STANDARD_FP in backend and not is_mx_fp(backend) and not is_nv_fp(backend)


def is_mx_fp(backend):
    backend = backend.lower()
    return BackendDataType.MX_FP in backend


def is_mx_int(backend):
    """MX_INT format is not supported in this build. Always returns False."""
    return False


def is_nv_fp(backend):
    backend = backend.lower()
    return BackendDataType.NV_FP in backend


def is_wint_woq(ar):
    """Returns True for integer weight-only quantization with non-quantized activations (`act_bits >= 16`)."""
    return "int" in ar.data_type and ar.act_bits >= 16 and ar.super_group_size is None


def is_wint_a16(ar):
    """Backward-compatible alias for `is_wint_woq()`."""
    return is_wint_woq(ar)


def _is_weight_fp8_activation_static_fp8(
    bit: int, group_size: int, sym: bool, data_type: str, act_dynamic: bool
) -> bool:
    return bit == 8 and group_size == -1 and sym and data_type == "fp" and not act_dynamic


def is_wfp8afp8(ar):
    if (
        ("fp8" in ar.act_data_type or ("fp" in ar.act_data_type and ar.act_bits == 8))
        and ("fp8" in ar.data_type or ("fp" in ar.data_type and ar.bits == 8))
        and is_standard_fp(ar.act_data_type)
        and is_standard_fp(ar.data_type)
    ):
        return True
    else:
        return False


def is_wint8aint8(ar):
    if ("int8" in ar.act_data_type or ("int" in ar.act_data_type and ar.act_bits == 8)) and (
        "int8" in ar.data_type or ("int" in ar.data_type and ar.bits == 8)
    ):
        return True
    else:
        return False


def is_static_wfp8afp8(ar_or_format: Union[str, Callable]) -> bool:
    if isinstance(ar_or_format, str):
        return "fp8_static" in ar_or_format.lower()
    if ar_or_format.act_dynamic:
        return False
    if is_wfp8afp8(ar_or_format):
        return True
    return False


def is_dynamic_wint8aint8(ar_or_format: Union[str, Callable]) -> bool:
    if isinstance(ar_or_format, str):
        return "int8_w8a8" in ar_or_format.lower()
    if not ar_or_format.act_dynamic:
        return False
    if is_wint8aint8(ar_or_format):
        return True
    return False


def is_wint4aint4(ar_or_scheme: Union[str, Callable]):
    if isinstance(ar_or_scheme, str):
        return "int4" in ar_or_scheme.lower()
    elif (
        "int4" in ar_or_scheme.act_data_type or ("int" in ar_or_scheme.act_data_type and ar_or_scheme.act_bits == 4)
    ) and ("int4" in ar_or_scheme.data_type or ("int" in ar_or_scheme.data_type and ar_or_scheme.bits == 4)):
        return True
    return False


def is_dynamic_afp8(ar_or_format: Callable) -> bool:
    return ar_or_format.act_dynamic and ar_or_format.act_data_type.startswith("fp") and ar_or_format.act_bits == 8


def is_block_wfp8(ar_or_format: Callable) -> bool:
    return (
        isinstance(ar_or_format.group_size, tuple)
        and len(ar_or_format.group_size) == 2
        and ar_or_format.data_type.startswith("fp")
        and ar_or_format.bits == 8
    )


def block_forward(
    block: torch.nn.Module,
    input_ids: torch.Tensor,
    input_others: dict,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cpu"),
    output_return_id: int = 0,
) -> Union[torch.Tensor, dict]:
    """Performs a forward pass through a block with the given inputs.

    Args:
    block: The block to perform the forward pass on.
    input_ids: The input IDs.
    input_others: A dictionary containing other input data.
    amp: A boolean indicating whether to use automatic mixed precision.
    amp_dtype: The data type for automatic mixed precision.
    device: The target device.
    output_return_id: if the output has more than one tenor, return the specified idx tensor.

    Returns:
    output: The output of the forward pass.
    """
    from auto_round.utils.model import to_device

    if input_ids.device != device:
        input_ids = to_device(input_ids, device)
        input_others = to_device(input_others, device)
    input_tuple = input_others.pop("positional_inputs", None)
    if "alibi" in input_others.keys() and input_others["alibi"] is not None:
        alibi = input_others["alibi"]
        input_others["alibi"] = alibi.reshape(-1, alibi.shape[2], alibi.shape[3])

    from auto_round.special_model_handler import prepare_special_model_block_inputs

    input_others, input_tuple = prepare_special_model_block_inputs(block, input_ids, input_others, input_tuple)

    # Convert 2D attention_mask to 4D causal mask for decoder layers that expect it.
    # The block (decoder layer) doesn't call create_causal_mask itself — that lives
    # in the parent TextModel.  When the block receives a 2D mask (batch, seq_len)
    # from cached calibration data, SDPA rejects it.  Convert here.
    if "attention_mask" in input_others and input_others["attention_mask"] is not None:
        am = input_others["attention_mask"]
        if isinstance(am, torch.Tensor) and am.ndim == 2:
            from transformers.masking_utils import create_causal_mask

            # Find the config — decoder layers store it on sub-modules (e.g. self_attn.config)
            block_config = getattr(block, "config", None)
            if block_config is None:
                for sub in block.modules():
                    if hasattr(sub, "config") and hasattr(sub.config, "_attn_implementation"):
                        block_config = sub.config
                        break
            if block_config is not None:
                inputs_embeds = input_ids if input_ids.dtype.is_floating_point else None
                if inputs_embeds is not None:
                    causal_mask = create_causal_mask(
                        config=block_config,
                        inputs_embeds=inputs_embeds,
                        attention_mask=am,
                        past_key_values=input_others.get("past_key_values"),
                        position_ids=input_others.get("position_ids"),
                    )
                    if causal_mask is not None:
                        input_others["attention_mask"] = causal_mask

    # Resample shared-cache position_embeddings to match the current batch size.
    # position_embeddings is stored as a shared key (one value, not per-sample),
    # but hidden_states may have been resampled to a different batch size by
    # _sampling_inputs.  Slice the embeddings to match.
    if "position_embeddings" in input_others and input_others["position_embeddings"] is not None:
        pe = input_others["position_embeddings"]
        batch_size = input_ids.shape[0]
        if isinstance(pe, (tuple, list)) and len(pe) >= 2:
            cos, sin = pe[0], pe[1]
            if isinstance(cos, torch.Tensor) and cos.shape[0] != batch_size:
                input_others["position_embeddings"] = (cos[:batch_size], sin[:batch_size])
        elif isinstance(pe, torch.Tensor) and pe.shape[0] != batch_size:
            input_others["position_embeddings"] = pe[:batch_size]

    # Use the block's actual parameter name for the first positional argument.
    import inspect as _inspect

    param_names = [p for p in _inspect.signature(block.forward).parameters.keys() if p != "self"]
    block_input_kwarg = param_names[0] if param_names else "hidden_states"
    if block_input_kwarg not in input_others:
        input_others[block_input_kwarg] = input_ids

    # Convert positional inputs to keyword args for any remaining positional parameters.
    positional_inputs = input_tuple or ()
    if positional_inputs:
        for i, val in enumerate(positional_inputs):
            param_idx = i + 1  # hidden_states is params[0]
            if param_idx < len(param_names):
                param_name = param_names[param_idx]
                if param_name not in input_others:
                    input_others[param_name] = val
        positional_inputs = ()

    if amp:
        with autocast(device_type=str(device).split(":")[0], dtype=amp_dtype):  # pragma: no cover
            output = block(**input_others)
    else:
        output = block(**input_others)
    if isinstance(output_return_id, int) and (isinstance(output, list) or isinstance(output, tuple)):
        output = output[output_return_id]
    return output


def check_skippable_keywords(key):
    """
    Prints a reminder if a key is not stored during quantization fine-tuning.
    """
    skippable_cache_keys = ("past_key_value",)
    for cache_key in skippable_cache_keys:
        if cache_key not in key:
            return True
    return False


def check_need_act_calibration(
    is_act_dynamic: Union[bool, None],
    act_data_type: Union[str, None] = None,
    act_bits: Union[int, None] = 16,
    static_kv_dtype: Union[str, None] = None,
    static_attention_dtype: Union[str, None] = None,
) -> bool:
    if static_kv_dtype is not None or static_attention_dtype is not None:
        return True
    if act_bits is None or act_bits > 8:
        return False
    # None is dynamic
    if is_act_dynamic is not None and not is_act_dynamic:
        return True
    if act_data_type is not None and "static" in act_data_type:
        return True
    return False


def collect_best_params(block, cache_device="cpu"):
    """Collect the best parameters from the block to the specified device."""
    params = {}
    if hasattr(block, "orig_layer"):
        for key in block.params.keys():
            params[key] = block.params[key].data.to(cache_device, copy=True)
    else:
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                params[n] = {}
                for key in m.params.keys():
                    params[n][key] = m.params[key].data.to(cache_device, copy=True)
    return params


def infer_bits_by_data_type(data_type: str):
    """Infer bits by data_type

    Args:
        data_type (str): data_type

    Returns:
        int: bits inferred by data_type, None means cannot infer correct bits by data_type
    """
    from auto_round.utils import SUPPORTED_DTYPES

    if data_type is None:
        return 16
    for supported_dtype in SUPPORTED_DTYPES:
        if data_type.startswith(supported_dtype) and len(data_type) > len(supported_dtype):
            ##first check the following two bits
            suc_2str = data_type[len(supported_dtype) : len(supported_dtype) + 2]
            if str.isdigit(suc_2str):
                return int(suc_2str)
            if str.isdigit(data_type[len(supported_dtype)]):
                return int(data_type[len(supported_dtype)])
    return None


def _get_safetensor_layer_names_not_in_model(model, all_module_names: list) -> list:
    """Collect layer names from safetensor files that are not loaded into the model.

    Some tensors (e.g. MTP layers) exist in the original checkpoint but are not
    instantiated by ``transformers``.  This function discovers them so that regex
    patterns in ``layer_config`` can still match them.

    Returns:
        List of layer names (the path without the ``.weight`` suffix) for weight
        tensors present in the safetensor files but absent from *all_module_names*.
    """
    name_or_path = None
    if hasattr(model, "config") and hasattr(model.config, "name_or_path"):
        name_or_path = model.config.name_or_path
    if not name_or_path:
        return []

    if not os.path.isdir(name_or_path):
        try:
            from auto_round.utils.model import download_hf_model

            name_or_path = download_hf_model(name_or_path)
        except Exception as e:
            logger.debug(f"Could not resolve source model path to check for missing tensors: {e}")
            return []

    try:
        from safetensors import safe_open
    except ImportError:
        return []

    # Build tensor-name list from the safetensors index or single file
    source_index_file = os.path.join(name_or_path, "model.safetensors.index.json")
    source_single_file = os.path.join(name_or_path, "model.safetensors")

    tensor_names: list = []
    if os.path.exists(source_index_file):
        with open(source_index_file) as f:
            src_index = json.load(f)
        tensor_names = list(src_index["weight_map"].keys())
    elif os.path.exists(source_single_file):
        with safe_open(source_single_file, framework="pt", device="cpu") as f:
            tensor_names = list(f.keys())
    else:
        return []

    module_name_set = set(all_module_names)
    extra_layer_names = []
    for tensor_name in tensor_names:
        if not tensor_name.endswith(".weight"):
            continue
        layer_name = tensor_name[: -len(".weight")]
        if layer_name not in module_name_set:
            extra_layer_names.append(layer_name)
    return extra_layer_names


def set_layer_config(
    model: torch.nn.Module,
    layer_config: dict[str, Union[str, dict, "QuantizationScheme"]],
    default_scheme: Union[str, "QuantizationScheme"],
    default_scale_dtype: torch.dtype | str,
    supported_types: tuple,
    inner_supported_types: tuple,
    quant_block_list=None,
    ignore_layers: str = "",
    quant_lm_head: bool = False,
    is_mllm: bool = False,
    fill_default_value=True,
) -> tuple[dict, bool, dict]:
    """
    Normalize, validate, and expand layer-specific quantization configs.
    Returns (final_layer_config, has_quant_layer_outside_block)
    """

    from auto_round.schemes import QuantizationScheme, preset_name_to_scheme
    from auto_round.utils.model import get_layer_names_in_block, get_lm_head_name, get_module

    # ---- helpers -------------------------------------------------
    def dispatch_layer_config(layer_config: dict[str, dict]) -> None:
        """Assign scheme values as attributes to matched modules."""
        for layer_name, scheme in layer_config.items():
            module = get_module(model, layer_name)
            if module is None:
                # Layer exists in safetensor files but is not loaded into the model
                # (e.g. MTP layers that transformers does not instantiate). Skip.
                continue
            for attr, value in scheme.items():
                setattr(module, attr, value)

    def normalize_item(item: Union[str, dict, "QuantizationScheme"], layer_name: str) -> dict:
        """Convert config entry into dict and validate keys."""
        if isinstance(item, str):
            config = asdict(preset_name_to_scheme(item.upper()))
        elif isinstance(item, QuantizationScheme):
            config = asdict(item)
        elif isinstance(item, dict):
            # "in_blocks" is an internal bookkeeping key injected by LLM-Compressor;
            # silently drop it before validation.
            item = {k: v for k, v in item.items() if k != "in_blocks"}
            scheme_name = item.pop("scheme", None)
            config = asdict(preset_name_to_scheme(scheme_name.upper())) if scheme_name is not None else {}
            invalid = set(item) - set(scheme_keys + ("fixed_by_user", "scale_dtype"))
            if invalid:
                raise ValueError(
                    f"Invalid keys {invalid} in layer_config for '{layer_name}'. " f"Allowed keys: {scheme_keys}"
                )
            config.update(item)
        else:
            raise TypeError(
                f"Unsupported type for layer_config[{layer_name}]: {type(item)}. "
                f"Expected str, dict, or QuantizationScheme."
            )
        # Clean up
        config = {k: v for k, v in config.items() if v is not None}
        config["fixed_by_user"] = True
        return config

    # ---- main logic ----------------------------------------------
    extra_scheme_keys = ("scale_dtype",)
    scheme_keys = tuple(f.name for f in fields(QuantizationScheme)) + ("scale_dtype",)
    layer_config = copy.deepcopy(layer_config) or {}
    ignore_layer_patterns = set()
    if ignore_layers:
        ignore_layers = ignore_layers.replace(" ", "").split(",")
        ignore_layers = [name + "." if name[-1].isdigit() else name for name in ignore_layers]
        ignore_layer_patterns = set(ignore_layers)

    # 1. ignore_layers -> force 16
    for name in get_fp_layer_names(model, ignore_layers):
        layer_config[name] = {
            "bits": 16,
            "act_bits": 16,
            "data_type": "float",
            "act_data_type": "float",
            "fixed_by_user": True,
        }

    # 2. normalize
    layer_config = {k: normalize_item(v, k) for k, v in layer_config.items()}

    # 3. infer missing bits
    for cfg in layer_config.values():
        if "data_type" in cfg and "bits" not in cfg:
            if (b := infer_bits_by_data_type(cfg["data_type"])) is not None:
                cfg["bits"] = b
        if "act_data_type" in cfg and "act_bits" not in cfg:
            if (b := infer_bits_by_data_type(cfg["act_data_type"])) is not None:
                cfg["act_bits"] = b

    # 4. fill defaults
    if isinstance(default_scheme, str):
        default_dict = asdict(preset_name_to_scheme(default_scheme.upper()))
    else:
        default_dict = asdict(default_scheme)
    default_dict["scale_dtype"] = default_scale_dtype

    # Fill missing scheme keys with defaults (skip None values for non-default-value mode)
    for cfg in layer_config.values():
        for key in scheme_keys:
            if fill_default_value:
                cfg.setdefault(key, copy.deepcopy(default_dict.get(key)))
            else:
                if key in extra_scheme_keys:
                    cfg.setdefault(key, copy.deepcopy(default_dict.get(key)))
                else:
                    cfg.setdefault(key, None)

    # 5. collect supported modules
    embedding_layer_types = (torch.nn.Embedding,)
    all_supported_layer_names, embedding_layer_names = [], []
    all_module_names = []
    for n, m in model.named_modules():
        all_module_names.append(n)
        # cleanup stale attributes
        for key in scheme_keys:
            # `rotation_config` on the root model carries the active
            # Hadamard rotation state (weights + hooks)
            if n == "" and key == "rotation_config":
                continue
            if hasattr(m, key):
                delattr(m, key)
        if type(m) not in supported_types and m.__class__.__name__ not in inner_supported_types:
            continue
        all_supported_layer_names.append(n)
        if isinstance(m, embedding_layer_types) or m.__class__.__name__.endswith("Embedding"):
            embedding_layer_names.append(n)
    # Also include layer names from safetensor files not loaded into the model
    # (e.g. MTP layers that transformers does not instantiate).
    safetensor_only_names = _get_safetensor_layer_names_not_in_model(model, all_module_names)

    # 6. expand regex configs
    regex_config = {}
    for name in list(layer_config.keys()):
        if name in all_supported_layer_names:
            continue
        if name in all_module_names:
            m = get_module(model, name)
            if len(list(m.children())) == 0 and type(m) not in supported_types:
                val = layer_config.pop(name)
                if name in ignore_layer_patterns:
                    # Keep unsupported ignore_layers entries so export can serialize
                    # them into regex-based extra_config for loaders like vLLM INC.
                    regex_config[name] = val
                else:
                    logger.warning(
                        f"'{name}' exists in the model but is not a supported quantization target "
                        f"in the current scheme, ignoring its setting in `layer_config`"
                    )
                continue

        regex = re.compile(to_standard_regex(name))
        matched = [ln for ln in all_supported_layer_names if regex.search(ln)]
        safetensor_only_matched = [ln for ln in safetensor_only_names if regex.search(ln)]
        # skip it for mtp layers not loaded in transformers
        if not matched and not safetensor_only_matched:
            # type(mlp.gate) is Qwen3VLMoeTextTopKRouter instead of Linear
            logger.warning_once(
                f"Layer name or regex '{name}' in layer_config does not match any supported layers. "
                + "Please check for typos or update the regex pattern, ignore it for now"
            )
        val = layer_config.pop(name)
        regex_config[name] = val  # keep regex config
        for match in matched:
            layer_config[match] = val

    # 7. lm_head
    lm_head_name = get_lm_head_name(model)
    tie_word_embeddings = False
    if hasattr(model, "config") and hasattr(model.config, "tie_word_embeddings"):
        tie_word_embeddings = model.config.tie_word_embeddings

    if lm_head_name in layer_config:
        quant_lm_head = True

    if quant_lm_head and tie_word_embeddings:
        quant_lm_head = False
        logger.warning(
            "reset `quant_lm_head` to false as quantizing " "lm_head with tied weights has not been supported currently"
        )

    if lm_head_name not in layer_config and quant_lm_head:
        layer_config[lm_head_name] = copy.deepcopy(default_dict)

    if not quant_lm_head:
        layer_config.pop(lm_head_name, None)

    # 8. enforce shape divisibility for int weight-only
    if default_dict["data_type"] == "int" and default_dict["act_bits"] >= 16:
        for n, m in model.named_modules():
            if type(m) in supported_types or m.__class__.__name__ in inner_supported_types:
                if m.weight.shape[0] % 32 or m.weight.shape[1] % 32:
                    layer_config.setdefault(n, copy.deepcopy(default_dict))
                    layer_config[n].update({"bits": 16, "data_type": "fp", "fixed_by_user": True})

    # 9. block layers: mark as in_blocks=True
    for name in get_layer_names_in_block(model, supported_types, quant_block_list, inner_supported_types):
        if name not in layer_config:
            layer_config[name] = copy.deepcopy(default_dict)
            layer_config[name]["fixed_by_user"] = False
        layer_config[name]["in_blocks"] = True

    # ---- restore: ensure missing in_blocks are set to False and compute flag ----
    has_qlayer_outside_block = False
    for cfg in layer_config.values():
        if "in_blocks" not in cfg:
            cfg["in_blocks"] = False
        # mark layer outside block
        if not cfg["in_blocks"] and check_to_quantized(cfg):
            has_qlayer_outside_block = True

    dispatch_layer_config(layer_config)
    return layer_config, has_qlayer_outside_block, regex_config



def get_fp_layer_names(model: torch.nn.Module, ignore_layers: str):
    """Identifies and returns layers in the model to exclude from quantization.

    This function processes a comma-separated list of fully precision (FP) layers,
    matches them to the names of layers in the model, and returns a list of such
    layers to exclude from quantization.

    Args:
        model (torch.nn.Module): The model whose layers will be inspected.
        ignore_layers (str): A comma-separated string of layer names to be excluded
            from quantization. Whitespace is ignored in this string.

    Returns:
        list: A list of layer names that match the specified FP layers or are
        subcomponents of those layers.
    """
    from auto_round.utils import SUPPORTED_LAYER_TYPES

    if not ignore_layers:
        return []

    all_layer_names = []
    for n, m in model.named_modules():
        if type(m) in SUPPORTED_LAYER_TYPES:
            all_layer_names.append(n)
    not_to_quantized_layers = []

    for fp_layer in ignore_layers:
        if fp_layer == "":
            continue
        if fp_layer in all_layer_names:
            not_to_quantized_layers.append(fp_layer)
            continue
        for name in all_layer_names:
            if fp_layer in name:
                not_to_quantized_layers.append(name)
    not_to_quantized_layers.extend(ignore_layers)  # keep regex name for later use
    if not_to_quantized_layers:
        logger.info(f"Ignored layers: {compress_layer_names(not_to_quantized_layers)}")
    return not_to_quantized_layers


def get_shared_keys(model):
    """
    Retrieves shared keys from the model's state dictionary.

    Args:
        model (torch.nn.Module): The model to retrieve shared keys from.

    Returns:
        tuple: tuple of shared keys.
    """
    from auto_round.special_model_handler import SPECIAL_SHARED_CACHE_KEYS
    from auto_round.utils import SHARED_CACHE_KEYS

    shared_keys = SHARED_CACHE_KEYS
    shared_keys += SPECIAL_SHARED_CACHE_KEYS.get(model.__class__.__name__, ())
    return shared_keys


def init_cache(positional_inputs, inputs):
    """
    Initializes special model inputs by adding positional inputs if missing.

    Args:
        positional_inputs (list): List of positional inputs to add to inputs.
        inputs (dict): Dictionary of model inputs.

    Modifies:
        inputs (dict): Adds "positional_inputs" key if not present.
    """
    from auto_round.utils.model import to_device

    if "positional_inputs" not in inputs:  # for chatglm Series
        inputs["positional_inputs"] = []
    for idx, item in enumerate(positional_inputs):
        inputs["positional_inputs"] = to_device(positional_inputs)


def reset_params(inputs):
    """
    Resets specific input parameters to avoid saving the key-value cache during fine-tuning.

    Args:
        inputs (dict): Dictionary of model inputs.

    Modifies:
        inputs (dict): Sets "use_cache" to False if the key is present.
    """
    if "use_cache" in inputs.keys():  # Not storing kv cache
        inputs["use_cache"] = False


class IndexSampler:
    """A cyclic sampler that returns shuffled index batches.

    This sampler maintains internal state so that each call to `next_batch()`
    continues from where it left off. When the remaining number of samples is
    less than `batch_size`, the sampler reshuffles all indices and starts from
    the beginning, discarding the last incomplete batch.

    Attributes:
        nsamples (int): Total number of samples.
        batch_size (int): Number of indices to return in each batch.
        index (int): Current position in the index list.
        indices (List[int]): Shuffled list of indices.
    """

    def __init__(self, nsamples: int, batch_size: int) -> None:
        """Initializes the sampler.

        Args:
            nsamples (int): Total number of samples (must be >= batch_size).
            batch_size (int): Number of indices per batch.

        Raises:
            ValueError: If batch_size is not in the range (0, nsamples].
        """
        if batch_size <= 0 or batch_size > nsamples:
            raise ValueError("batch_size must be > 0 and <= nsamples")

        self.nsamples: int = nsamples
        self.batch_size: int = batch_size
        self.index: int = 0

        self.indices: list[int] = list(range(nsamples))
        random.shuffle(self.indices)

    def next_batch(self) -> list[int]:
        """Returns the next batch of shuffled indices.

        If the remaining indices are fewer than `batch_size`, the sampler
        reshuffles the entire list and starts from the beginning.

        Returns:
            list[int]: A list of size `batch_size` containing sample indices.
        """
        if self.index + self.batch_size > self.nsamples:
            random.shuffle(self.indices)
            self.index = 0

        batch = self.indices[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return batch


def _get_quantized_layer_names_outside_blocks(model, layer_config, supported_types, quant_block_list) -> list:
    """Gets the names of quantized layers outside blocks in the model.

    Returns:
        list: List of layer names outside blocks.
    """
    if layer_config is None or len(layer_config) == 0:
        return []

    layer_names = []
    all_layers_in_block = get_layer_names_in_block(model, supported_types, quant_block_list)

    for key in layer_config.keys():
        if key in all_layers_in_block:
            continue
        layer = get_module(model, key)
        if layer is None:
            raise ValueError(f"could not find layer {key} in the model")
        if type(layer) in supported_types and check_to_quantized(layer_config[key]):
            layer_names.append(key)

    return layer_names


def _get_save_folder_name(format, *args, **kwargs) -> str:
    """Generates the save folder name based on the provided format string.

    If there are multiple formats to handle, the function creates a subfolder
    named after the format string with special characters replaced. If there's
    only one format, it returns the original output directory directly.

    Args:
        format: The OutputFormat instance.

    Returns:
        str: The path to the folder where results should be saved.
    """
    from auto_round.context.compress import CompressContext
    from auto_round.context.model import ModelContext

    compress_context = CompressContext.get_context()
    model_context = ModelContext.get_context()
    # Replace special characters to make the folder name filesystem-safe
    sanitized_format = format.get_backend_name().replace(":", "-").replace("_", "-")

    # Use a subfolder only if there are multiple formats
    if len(compress_context.formats) > 1:
        return os.path.join(compress_context.output_dir, sanitized_format)

    return compress_context.output_dir


def immediate_pack(name: str, layer_config: dict):
    from auto_round.context.compress import CompressContext
    from auto_round.context.model import ModelContext

    compress_context = CompressContext.get_context()
    model_context = ModelContext.get_context()

    if not compress_context.is_immediate_packing:
        return
    compress_context.formats[0].immediate_pack(
        name=name,
        model=model_context.model,
        device=compress_context.device,
        output_dir=_get_save_folder_name(compress_context.formats[0]),
        layer_config=layer_config,
        tokenizer=model_context.tokenizer,
        mllm=model_context.is_mllm,
        processor=getattr(model_context, "processor", None),
        image_processor=getattr(model_context, "image_processor", None),
        quant_nontext_module=getattr(model_context, "quant_nontext_module", False),
    )
