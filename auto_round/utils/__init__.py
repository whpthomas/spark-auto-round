# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""auto_round.utils — utility functions and monkey patches.

WARNING: Importing this module has side effects:
- monkey_patch() modifies transformers internals (AutoModelForCausalLM, etc.)
- datasets.load_dataset is patched for transformers v5.0+ compatibility
See AGENTS.md for details.
"""

# Explicit imports — no wildcards. Each submodule defines __all__.
from auto_round.utils.common import (
    SHARED_CACHE_KEYS,
    SUPPORTED_DTYPES,
    SUPPORTED_FORMATS,
    SUPPORTED_LAYER_TYPES,
    GlobalState,
    LazyImport,
    SupportedFormats,
    VISION_MM_KEYS,
    AUDIO_MM_KEYS,
    INNER_SUPPORTED_LAYER_TYPES,
    MM_KEYS,
    MM_MODULE_KEYS,
    TORCH_VERSION_AT_LEAST_2_4,
    TORCH_VERSION_AT_LEAST_2_5,
    TORCH_VERSION_AT_LEAST_2_6,
    TORCH_VERSION_AT_LEAST_2_6_PRE_RELEASE,
    apply_checkpoint_conversion_mapping,
    auto_gptq,
    compress_layer_names,
    compare_versions,
    contain_any_mm_keys,
    deepspeed_exists,
    download_audiocaps_csv,
    flatten_list,
    get_checkpoint_conversion_mapping,
    get_library_version,
    get_reciprocal,
    get_reverse_checkpoint_conversion_mapping,
    global_state,
    htcore,
    infer_bits_by_data_type,
    is_debug_mode,
    is_local_path,
    is_transformers_version_greater_or_equal_5,
    is_transformers_version_greater_or_equal_5_4_0,
    json_serialize,
    logger,
    matches_any_regex,
    monkey_patch,
    monkey_patch_model,
    monkey_patch_transformers,
    normalize_no_split_modules,
    parse_layer_config_arg,
    preserve_original_visual_block_name,
    revert_checkpoint_conversion_mapping,
    str2bool,
    to_standard_regex,
    torch_version_at_least,
)
from auto_round.utils.device import (
    CpuInfo,
    ClearMemory,
    DEVICE_ENVIRON_VARIABLE_MAPPING,
    MemoryMonitor,
    _allocate_layers_to_devices,
    _bump_dynamo_cache_limit,
    _estimate_param_count_from_config,
    _force_trim_malloc,
    bytes_to_gigabytes,
    can_pack_with_numba,
    check_memory_availability,
    clear_memory,
    clear_memory_if_reached_threshold,
    compile_func,
    compile_func_on_cuda_or_cpu,
    detect_device,
    detect_device_count,
    dispatch_model_block_wise,
    dispatch_model_by_all_available_devices,
    dump_memory_usage_ctx,
    dump_mem_usage,
    estimate_memory_strategy,
    estimate_tuning_block_mem,
    get_device_and_parallelism,
    get_device_memory,
    get_first_available_attr,
    get_major_device,
    get_max_vram,
    get_moe_memory_ratio,
    get_packing_device,
    is_auto_device_mapping,
    is_autoround_exllamav2_available,
    is_gaudi2,
    is_hpex_available,
    is_numba_available,
    is_package_available,
    is_tbb_available,
    log_memory_analysis,
    memory_monitor,
    out_of_vram,
    override_cuda_device_capability,
    parse_available_devices,
    partition_dict_numbers,
    set_auto_device_map_for_block_with_tuning,
    set_avg_auto_device_map,
    set_cuda_visible_devices,
    set_non_auto_device_map,
    set_tuning_device_for_layer,
)
from auto_round.utils.model import (
    ARCHITECTURE_MODEL_TYPE_MAP,
    _get_reference_amax_from_experts,
    _is_fused_experts_module,
    _is_mllm_model_cache,
    _LLM_ONLY_MODEL_TYPES,
    _set_amax_for_moe_auxiliary_layers,
    check_seqlen_compatible,
    check_start_with_block_name,
    check_to_quantized,
    clean_module_parameter,
    config_save_pretrained,
    convert_dtype_str2torch,
    convert_dtype_torch2str,
    convert_dtype_torch2str_hf,
    copy_python_files_from_model_cache,
    download_hf_model,
    extract_block_names_to_str,
    find_layers_from_config,
    find_matching_blocks,
    get_attr,
    get_block_names,
    get_common_prefix,
    get_expert_input_proj_names,
    get_expert_linear_names,
    get_layer_features,
    get_layer_names_in_block,
    get_lm_head_name,
    get_model_dtype,
    get_model_name_or_path,
    get_module,
    get_nested_attr,
    hook_ngram_embeddings_on_cpu,
    is_mllm_model,
    is_moe_layer,
    is_moe_model,
    is_moe_model_via_config,
    is_separate_lm_head,
    is_separate_tensor,
    llm_load_model,
    merge_block_output_keys,
    mllm_load_model,
    mv_module_from_gpu,
    resolve_model_type,
    safe_device_move_with_meta_handling,
    set_amax_for_all_moe_layers,
    set_amax_for_uncalibrated_experts,
    set_attr,
    set_module,
    set_nested_attr,
    to_device,
    to_dtype,
    unsupported_meta_device,
    wrap_block_forward_positional_to_kwargs,
)
from auto_round.utils.weight_handler import (
    check_and_mark_quantized_module,
    convert_module_to_hp_if_necessary,
    detect_weight_type,
    is_quantized_input_module,
)
from auto_round.utils.missing_tensors import copy_missing_tensors_from_source

import transformers
from packaging.version import Version

DATASET_PATCHED = False
# tmp batch for transformers v5.0
if Version(transformers.__version__) >= Version("5.0.0") and not DATASET_PATCHED:
    import datasets

    datasets.original_load_dataset = datasets.load_dataset

    def patch_load_dataset(*args, **kwargs):
        for dataset_name, replace_name in [("openbookqa", "allenai/openbookqa")]:
            if len(args) > 0 and dataset_name in args[0]:
                args = (replace_name,) + args[1:]
            if "path" in kwargs and kwargs["path"] is not None:
                if dataset_name in kwargs["path"] and replace_name not in kwargs["path"]:
                    kwargs["path"] = kwargs["path"].replace(dataset_name, replace_name)
            if "name" in kwargs and kwargs["name"] is not None:
                if dataset_name in kwargs["name"] and replace_name not in kwargs["name"]:
                    kwargs["name"] = kwargs["name"].replace(dataset_name, replace_name)
        return datasets.original_load_dataset(*args, **kwargs)

    datasets.load_dataset = patch_load_dataset
    DATASET_PATCHED = True
