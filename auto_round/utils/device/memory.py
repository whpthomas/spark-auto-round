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
"""Memory management utilities for device operations."""
import ctypes
import gc
import os
from contextlib import contextmanager
from threading import Lock

import psutil
import torch

from auto_round.logger import logger
from auto_round.utils.device.detect import is_hpex_available


def bytes_to_gigabytes(bytes) -> int:
    """Converts bytes to gigabytes."""
    return bytes / 1024 / 1024 / 1024


def _clear_memory_for_cpu_and_cuda(
    tensor: torch.Tensor | list[torch.Tensor] | None = None,
    device_list: tuple | list | str | torch.device | None = None,
):
    """Clear memory for CPU and CUDA devices."""
    if isinstance(tensor, list):
        for i in range(len(tensor)):
            tensor[i] = None
    tensor = None
    gc.collect()
    _maybe_trim_malloc()

    if isinstance(device_list, (str, torch.device)):
        device_list = [device_list]

    if torch.cuda.is_available():
        if not device_list:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        else:
            devices = []
            for dev in device_list:
                dev = str(dev)
                if not dev.startswith("cuda"):
                    continue
                if ":" in dev:
                    devid = int(dev.split(":")[-1])
                else:
                    devid = 0
                devices.append(devid)

            for d in devices:
                torch.cuda.synchronize(d)

            torch.cuda.empty_cache()


_malloc_trim_counter = 0


def _force_trim_malloc() -> None:
    """Unconditionally release glibc heap pages back to the OS on Linux."""
    if os.name != "posix":
        return
    if os.environ.get("AR_ENABLE_MALLOC_TRIM", "1") != "1":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def _maybe_trim_malloc() -> None:
    """Optionally release glibc heap pages back to OS on Linux."""
    global _malloc_trim_counter

    if os.name != "posix":
        return
    if os.environ.get("AR_ENABLE_MALLOC_TRIM", "1") != "1":
        return

    try:
        every = int(os.environ.get("AR_MALLOC_TRIM_EVERY", "10"))
    except ValueError:
        every = 10
    every = max(1, every)

    _malloc_trim_counter += 1
    if _malloc_trim_counter % every != 0:
        return

    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


class ClearMemory:
    """Clear memory helper."""

    def __init__(self, device_list: list | tuple | None = None):
        self.device_list = device_list

    def __call__(
        self,
        tensor: torch.Tensor | None | list[torch.Tensor | dict] = None,
        device_list: list | tuple | None = None,
    ):
        if device_list is not None:
            self.device_list = device_list
        final_device_list = self.device_list
        memory_monitor.update(final_device_list)
        _clear_memory_for_cpu_and_cuda(tensor, final_device_list)


clear_memory = torch._dynamo.disable()(ClearMemory(device_list=[0]))


def clear_memory_if_reached_threshold(threshold=0.85, device_list=None):
    """Check all available devices and clear memory if any device is using close to the threshold."""
    if torch.cuda.is_available():
        name, device_api = "cuda", torch.cuda
    else:
        return False

    num_devices = device_api.device_count()
    for i in range(num_devices):
        try:
            total_memory = device_api.get_device_properties(i).total_memory
            reserved_memory = device_api.memory_reserved(i)
            memory_usage_ratio = reserved_memory / total_memory

            if memory_usage_ratio >= threshold:
                logger.warning_once(
                    f"Major device ({name}:{i}) has reached memory threshold. "
                    + "Memory clearing operation will be called during each iteration, which "
                    + "will result in more time consumption."
                )
                logger.warning_once(
                    "To alleviate high memory usage on the major device, consider reducing the `batch_size` "
                    + "(and correspondingly increasing `gradient_accumulation_steps) or shortening the seqlen."
                )
                clear_memory(device_list=device_list)
                return True
        except Exception as e:
            logger.warning_once(f"Failed to check memory for {name}:{i}: {e}")
    return False


def check_memory_availability(device, inputs, weight, org_seqlen, org_bs):
    """Checks the availability of memory on the specified device."""
    weight_memory = weight.numel() * weight.element_size()
    if "cuda" in device:
        current_gpu_index = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory
        used_memory = torch.cuda.memory_allocated(current_gpu_index)
        free_space = total_memory - used_memory
    else:
        return True, org_seqlen, org_bs

    free_space = free_space - weight_memory * 10
    seqlen = org_seqlen
    bs = org_bs
    in_feature = weight.shape[1]
    out_feature = weight.shape[0]
    while seqlen >= 128:
        input_size = bs * seqlen * in_feature
        output_size = bs * seqlen * out_feature
        input_output_memory = 2 * (input_size * inputs.element_size() + output_size * inputs.element_size())
        if input_output_memory < free_space:
            return True, seqlen, bs
        seqlen = seqlen // 2
        bs = 1

    return False, seqlen, bs


def out_of_vram(error_msg):
    """Check if error message indicates OOM."""
    error_msg = str(error_msg)
    if "CUDA out of memory" in error_msg:
        return True
    return False


def get_max_vram(ratio: float = 0.9) -> dict:
    """Get max VRAM for each device."""
    max_memory = {}
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            max_mem_gb = int(total_mem / 1024**3 * ratio)
            max_memory[i] = f"{max_mem_gb}GiB"
    else:
        raise RuntimeError("No CUDA devices found.")
    return max_memory


def get_device_memory(i: int = 0) -> int:
    """Gets the available memory on the specified device in gigabytes."""
    if torch.cuda.is_available():
        total_memory = bytes_to_gigabytes(torch.cuda.get_device_properties(i).total_memory)
    else:
        raise RuntimeError("No CUDA devices found.")
    return total_memory


def _estimate_param_count_from_config(config) -> int:
    """Estimate total parameter count from HuggingFace AutoConfig."""
    hidden_size = getattr(config, "hidden_size", None)
    num_layers = getattr(config, "num_hidden_layers", 0)
    vocab_size = getattr(config, "vocab_size", 0)
    intermediate_size = (
        getattr(config, "intermediate_size", None)
        or getattr(config, "ffn_dim", None)
        or getattr(config, "moe_intermediate_size", None)
    )

    if hidden_size is None or intermediate_size is None:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            hidden_size = hidden_size or getattr(text_cfg, "hidden_size", None)
            num_layers = num_layers or getattr(text_cfg, "num_hidden_layers", 0)
            vocab_size = vocab_size or getattr(text_cfg, "vocab_size", 0)
            intermediate_size = (
                intermediate_size
                or getattr(text_cfg, "intermediate_size", None)
                or getattr(text_cfg, "ffn_dim", None)
                or getattr(text_cfg, "moe_intermediate_size", None)
            )

    if hidden_size is None or intermediate_size is None:
        logger.warning(
            "Cannot estimate parameter count from config — "
            "hidden_size or intermediate_size missing. "
            "Defaulting to 0 (will use whole-model path)."
        )
        return 0

    block_params = 4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size

    num_experts = getattr(config, "num_local_experts", None) or getattr(config, "num_experts", None)
    if not num_experts:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            num_experts = getattr(text_cfg, "num_local_experts", None) or getattr(text_cfg, "num_experts", None)
    if num_experts and num_experts > 1:
        mlp_params = 3 * hidden_size * intermediate_size
        block_params = 4 * hidden_size * hidden_size + mlp_params * num_experts

    total = block_params * num_layers
    total += vocab_size * hidden_size
    total += hidden_size

    return total


def estimate_memory_strategy(
    model_path: str,
    memory_utilization: float = 0.75,
) -> tuple[bool, dict]:
    """Decide whole-model vs block-offload based on model size from config."""
    memory_utilization = max(0.5, min(0.95, memory_utilization))

    try:
        from transformers import AutoConfig
        # If model_path is a local directory, use local_files_only=True
        # to avoid huggingface_hub repo_id validation on absolute paths
        if os.path.isdir(model_path):
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
        else:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        raise ValueError(
            f"Cannot load config for {model_path}: {e}. "
            "Provide a valid model path or HuggingFace model ID."
        )

    num_params = _estimate_param_count_from_config(config)
    num_blocks = getattr(config, "num_hidden_layers", 0)
    if num_blocks == 0:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            num_blocks = getattr(text_cfg, "num_hidden_layers", 0)

    dtype_bytes = 2
    dtype_name = "bfloat16"
    model_bytes = num_params * dtype_bytes

    if torch.cuda.is_available():
        # Use total - allocated instead of mem_get_info to include reclaimable
        # cache.  mem_get_info only reports truly free memory and ignores the
        # CUDA allocator's cached blocks that can be reclaimed.
        total_mem = torch.cuda.get_device_properties(0).total_memory
        allocated_mem = torch.cuda.memory_allocated(0)
        available_bytes = total_mem - allocated_mem
    else:
        available_bytes = 0

    threshold_bytes = int(available_bytes * memory_utilization)
    use_offload = model_bytes > threshold_bytes
    strategy = "block-offload" if use_offload else "whole-model"

    block_size_bytes = model_bytes // max(num_blocks, 1)

    info = {
        "model_name": model_path,
        "num_params": num_params,
        "model_bytes": model_bytes,
        "dtype": dtype_name,
        "available_bytes": available_bytes,
        "threshold_bytes": threshold_bytes,
        "strategy": strategy,
        "num_blocks": num_blocks,
        "block_size_bytes": block_size_bytes,
    }

    return use_offload, info


def log_memory_analysis(info: dict, memory_utilization: float = 0.75, budget_gb: float = 96.0) -> None:
    """Log comprehensive memory analysis at INFO level."""
    model_gb = info["model_bytes"] / (1024 ** 3)
    avail_gb = info["available_bytes"] / (1024 ** 3)
    block_mb = info["block_size_bytes"] / (1024 ** 2)
    num_params_b = info["num_params"] / 1e9

    total_gpu_gb = 0.0
    if torch.cuda.is_available():
        total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    logger.info(
        "Memory analysis:\n"
        "  Model:          %s\n"
        "  Parameters:     %.1fB\n"
        "  Model size:     %.1f GB (%s)\n"
        "  GPU:            %.1f GB total, %.1f GB available\n"
        "  Budget:         %.1f GiB (auto-tuner per-block ceiling)\n"
        "  Strategy:       %s%s\n"
        "  Blocks:         %d layers\n"
        "  Block size:     ~%.0f MB each",
        info["model_name"],
        num_params_b,
        model_gb, info["dtype"],
        total_gpu_gb, avail_gb,
        budget_gb,
        info["strategy"],
        " (model exceeds threshold)" if info["strategy"] == "block-offload"
            else " (fits in memory)",
        info["num_blocks"],
        block_mb,
    )


class MemoryMonitor:
    """Global memory monitor for tracking peak RAM and VRAM usage."""

    _instance = None
    _lock = Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.peak_ram = 0.0
        self.peak_vram = {}
        self.enabled = True

    def update(self, device_list=None):
        """Update current memory usage and track peaks."""
        if not self.enabled:
            return
        process = psutil.Process()
        current_ram = process.memory_info().rss / 1024**3
        self.peak_ram = max(self.peak_ram, current_ram)
        if device_list is None:
            device_list = [0]
        if device_list is not None:
            if not isinstance(device_list, (list, tuple)):
                device_list = [device_list]
        else:
            if torch.cuda.is_available():
                device_list = list(range(torch.cuda.device_count()))

        for device in device_list:
            if str(device) == "cpu":
                continue
            if torch.cuda.is_available():
                try:
                    current_vram = torch.cuda.memory_reserved(device) / 1024**3
                except (RuntimeError, Exception):
                    continue
                if device == "cuda":
                    device = "0"
            elif torch.xpu.is_available():
                try:
                    current_vram = torch.xpu.memory_reserved(device) / 1024**3
                except (RuntimeError, Exception):
                    continue
                if device == "xpu":
                    device = "0"
            elif is_hpex_available():
                try:
                    current_vram = torch.hpu.memory_allocated(device) / 1024**3
                except Exception:
                    current_vram = 0.0
                if device == "hpu":
                    device = "0"
            else:
                return

            device = str(device).split(":")[-1]
            if current_vram > 0:
                if device not in self.peak_vram:
                    self.peak_vram[device] = 0.0

                self.peak_vram[device] = max(self.peak_vram[device], current_vram)

    def update_cpu(self):
        if not self.enabled:
            return
        process = psutil.Process()
        current_ram = process.memory_info().rss / 1024**3
        self.peak_ram = max(self.peak_ram, current_ram)

    def update_hpu(self, device_list=None):
        """Track HPU VRAM usage. No-op in this CUDA-only fork."""
        pass

    def reset(self):
        self.peak_ram = 0.0
        self.peak_vram = {}

    def get_summary(self):
        summary = f"'peak_ram': {round(self.peak_ram, 2)}GB"
        if len(self.peak_vram) > 0:
            sorted_items = sorted(self.peak_vram.items())
            if len(self.peak_vram) == 1:
                key, value = sorted_items[0]
                summary += f", 'peak_vram': {round(value, 2)}GB"
            else:
                items_str = ", ".join([f"'{k}': {round(v, 2)}GB" for k, v in sorted_items])
                summary += f", 'peak_vram': {{{items_str}}}"
        return summary

    def log_summary(self, msg: str = "", level: str = "info"):
        summary = self.get_summary()
        logger_method = getattr(logger, level.lower(), logger.info)
        if len(msg):
            logger_method(f"{msg} {summary}")
        else:
            logger_method(f"{summary}")

        return summary


memory_monitor = MemoryMonitor()


@contextmanager
def dump_memory_usage_ctx(msg: str = "", log_level: str = "info"):
    """Context manager to dump memory usage before and after a code block."""
    memory_monitor.update_cpu()
    logger_method = getattr(logger, log_level.lower(), logger.info)
    logger_method(f"[Memory Monitor] Before {msg}: {memory_monitor.get_summary()}")
    try:
        yield
    finally:
        memory_monitor.update_cpu()
        logger_method(f"[Memory Monitor] After {msg}: {memory_monitor.get_summary()}")


def dump_mem_usage(msg: str = "", log_level: str = "info"):
    """Decorator to dump memory usage before and after a function call."""
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            memory_monitor.update_cpu()
            logger_method = getattr(logger, log_level.lower(), logger.info)
            logger_method(f"[Memory Monitor] Before {msg}: {memory_monitor.get_summary()}")
            result = func(*args, **kwargs)
            memory_monitor.update_cpu()
            logger_method(f"[Memory Monitor] After {msg}: {memory_monitor.get_summary()}")
            return result
        return wrapper
    return decorator
