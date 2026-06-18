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
"""Model utilities — loading, detection, and slicing."""

__all__ = [
    # utils.py
    "check_seqlen_compatible",
    "check_start_with_block_name",
    "check_to_quantized",
    "convert_dtype_str2torch",
    "convert_dtype_torch2str",
    "convert_dtype_torch2str_hf",
    "get_attr",
    "get_common_prefix",
    "get_layer_features",
    "get_module",
    "get_nested_attr",
    "mv_module_from_gpu",
    "safe_device_move_with_meta_handling",
    "set_attr",
    "set_module",
    "set_nested_attr",
    "to_device",
    "to_dtype",
    "unsupported_meta_device",
    # load.py
    "clean_module_parameter",
    "config_save_pretrained",
    "copy_python_files_from_model_cache",
    "download_hf_model",
    "get_model_dtype",
    "hook_ngram_embeddings_on_cpu",
    "llm_load_model",
    "mllm_load_model",
    # detect.py
    "ARCHITECTURE_MODEL_TYPE_MAP",
    "_get_reference_amax_from_experts",
    "_is_fused_experts_module",
    "_is_mllm_model_cache",
    "_LLM_ONLY_MODEL_TYPES",
    "_set_amax_for_moe_auxiliary_layers",
    "extract_block_names_to_str",
    "find_matching_blocks",
    "get_block_names",
    "get_expert_input_proj_names",
    "get_expert_linear_names",
    "get_layer_names_in_block",
    "get_lm_head_name",
    "get_model_name_or_path",
    "is_mllm_model",
    "is_moe_layer",
    "is_moe_model",
    "is_moe_model_via_config",
    "is_separate_lm_head",
    "is_separate_tensor",
    "is_pure_text_model",
    "merge_block_output_keys",
    "resolve_model_type",
    "set_amax_for_all_moe_layers",
    "set_amax_for_uncalibrated_experts",
    "wrap_block_forward_positional_to_kwargs",
    # slice.py
    "find_layers_from_config",
]

# Re-export from submodules for backward compatibility
from auto_round.utils.model.utils import (
    check_seqlen_compatible,
    check_start_with_block_name,
    check_to_quantized,
    convert_dtype_str2torch,
    convert_dtype_torch2str,
    convert_dtype_torch2str_hf,
    get_attr,
    get_common_prefix,
    get_layer_features,
    get_module,
    get_nested_attr,
    mv_module_from_gpu,
    safe_device_move_with_meta_handling,
    set_attr,
    set_module,
    set_nested_attr,
    to_device,
    to_dtype,
    unsupported_meta_device,
)

from auto_round.utils.model.load import (
    clean_module_parameter,
    config_save_pretrained,
    copy_python_files_from_model_cache,
    download_hf_model,
    get_model_dtype,
    hook_ngram_embeddings_on_cpu,
    llm_load_model,
    mllm_load_model,
)

from auto_round.utils.model.detect import (
    ARCHITECTURE_MODEL_TYPE_MAP,
    _is_mllm_model_cache,
    _LLM_ONLY_MODEL_TYPES,
    get_model_name_or_path,
    is_mllm_model,
    is_pure_text_model,
    _get_reference_amax_from_experts,
    _is_fused_experts_module,
    _set_amax_for_moe_auxiliary_layers,
    extract_block_names_to_str,
    find_matching_blocks,
    get_block_names,
    get_expert_input_proj_names,
    get_expert_linear_names,
    get_layer_names_in_block,
    get_lm_head_name,
    is_moe_layer,
    is_moe_model,
    is_moe_model_via_config,
    is_separate_lm_head,
    is_separate_tensor,
    merge_block_output_keys,
    resolve_model_type,
    set_amax_for_all_moe_layers,
    set_amax_for_uncalibrated_experts,
    wrap_block_forward_positional_to_kwargs,
)

from auto_round.utils.model.slice import (
    find_layers_from_config,
)

# Backward compatibility: these were previously in model.py directly
# handle_generation_config is defined in load.py and not re-exported from __init__
# to avoid confusion (it's called internally during model loading)
