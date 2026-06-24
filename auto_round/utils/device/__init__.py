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
"""Device utilities — detection, memory management, and patches."""

__all__ = [
    # detect.py
    "CpuInfo",
    "DEVICE_ENVIRON_VARIABLE_MAPPING",
    "detect_device",
    "detect_device_count",
    "get_device_and_parallelism",
    "get_packing_device",
    "is_auto_device_mapping",
    "is_autoround_exllamav2_available",
    "is_gaudi2",
    "is_hpex_available",
    "is_package_available",
    "override_cuda_device_capability",
    "parse_available_devices",
    "set_cuda_visible_devices",
    # memory.py
    "ClearMemory",
    "MemoryMonitor",
    "_estimate_param_count_from_config",
    "_force_trim_malloc",
    "bytes_to_gigabytes",
    "check_memory_availability",
    "clear_memory",
    "clear_memory_if_reached_threshold",
    "dump_memory_usage_ctx",
    "dump_mem_usage",
    "estimate_memory_strategy",
    "get_device_memory",
    "get_max_vram",
    "log_memory_analysis",
    "memory_monitor",
    "out_of_vram",
    # memory_estimator.py
    "_get_block_params",
    "_get_hidden_dimensions",
    "estimate_peak_memory_per_block",
    # patches.py
    "_bump_dynamo_cache_limit",
    "_allocate_layers_to_devices",
    "can_pack_with_numba",
    "compile_func",
    "compile_func_on_cuda_or_cpu",
    "dispatch_model_block_wise",
    "dispatch_model_by_all_available_devices",
    "estimate_tuning_block_mem",
    "get_first_available_attr",
    "get_major_device",
    "get_moe_memory_ratio",
    "is_numba_available",
    "is_tbb_available",
    "partition_dict_numbers",
    "set_auto_device_map_for_block_with_tuning",
    "set_avg_auto_device_map",
    "set_non_auto_device_map",
    "set_tuning_device_for_layer",
]

# Re-export from detect.py
from auto_round.utils.device.detect import (
    CpuInfo,
    DEVICE_ENVIRON_VARIABLE_MAPPING,
    detect_device,
    detect_device_count,
    get_device_and_parallelism,
    get_packing_device,
    is_auto_device_mapping,
    is_autoround_exllamav2_available,
    is_gaudi2,
    is_hpex_available,
    is_package_available,
    override_cuda_device_capability,
    parse_available_devices,
    set_cuda_visible_devices,
)

# Re-export from memory.py
from auto_round.utils.device.memory import (
    ClearMemory,
    MemoryMonitor,
    _estimate_param_count_from_config,
    _force_trim_malloc,
    bytes_to_gigabytes,
    check_memory_availability,
    clear_memory,
    clear_memory_if_reached_threshold,
    dump_memory_usage_ctx,
    dump_mem_usage,
    estimate_memory_strategy,
    get_device_memory,
    get_max_vram,
    log_memory_analysis,
    memory_monitor,
    out_of_vram,
)

# Re-export from memory_estimator.py
from auto_round.utils.device.memory_estimator import (
    _get_block_params,
    _get_hidden_dimensions,
    estimate_peak_memory_per_block,
)

# Re-export from patches.py
from auto_round.utils.device.patches import (
    _bump_dynamo_cache_limit,
    _allocate_layers_to_devices,
    can_pack_with_numba,
    compile_func,
    compile_func_on_cuda_or_cpu,
    dispatch_model_block_wise,
    dispatch_model_by_all_available_devices,
    estimate_tuning_block_mem,
    get_first_available_attr,
    get_major_device,
    get_moe_memory_ratio,
    is_numba_available,
    is_tbb_available,
    partition_dict_numbers,
    set_auto_device_map_for_block_with_tuning,
    set_avg_auto_device_map,
    set_non_auto_device_map,
    set_tuning_device_for_layer,
)
