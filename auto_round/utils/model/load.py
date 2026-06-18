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
"""Model loading and initialization utilities."""
import json
import os
import re
from pathlib import Path

import torch
from packaging import version

from auto_round.logger import logger
from auto_round.utils.common import monkey_patch_model
from auto_round.utils.weight_handler import (
    check_and_mark_quantized_module,
)
from auto_round.utils.model.utils import (
    convert_dtype_str2torch,
    get_module,
)


def clean_module_parameter(submodule: torch.nn.Module, param_name: str) -> None:
    """Clean a module parameter by setting it to an empty tensor.

    This is recommended over module.weight = None for models with tied
    word embeddings, as setting to None causes lm_head to reallocate memory.
    """
    if submodule is None:
        return
    is_buffer = param_name in submodule._buffers
    with torch.no_grad():
        if is_buffer:
            buf = submodule._buffers[param_name]
            if buf is not None:
                buf.data = torch.empty(0, dtype=buf.dtype, device=buf.device)
                buf.requires_grad = False
        else:
            param = submodule._parameters[param_name]
            if param is not None:
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
                param.requires_grad = False


def download_hf_model(repo_id, cache_dir=None, repo_type=None, revision=None):
    """Download hugging face model from hf hub."""
    from huggingface_hub.constants import DEFAULT_REVISION, HUGGINGFACE_HUB_CACHE
    from huggingface_hub.file_download import REGEX_COMMIT_HASH, repo_folder_name

    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if revision is None:
        revision = DEFAULT_REVISION
    if repo_type is None:
        repo_type = "model"
    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))
    commit_hash = None
    if REGEX_COMMIT_HASH.match(revision):
        commit_hash = revision
    else:
        ref_path = os.path.join(storage_folder, "refs", revision)
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                commit_hash = f.read()
    if storage_folder and commit_hash:
        pointer_path = os.path.join(storage_folder, "snapshots", commit_hash)
        if os.path.isdir(pointer_path):
            return pointer_path
    else:  # pragma: no cover
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(repo_id)
        return model_path


def _check_accelerate_version():
    """Check accelerate version and warn if it may cause high RAM usage."""
    from auto_round.utils.common import get_library_version

    accelerate_version = get_library_version("accelerate")
    from packaging.version import Version

    if Version(accelerate_version) > Version("1.5.1") and Version(accelerate_version) < Version("1.10.0"):
        logger.warning(
            f"Detected accelerate version {accelerate_version}. "
            "Versions between 1.5.1 and 1.10.0 may cause high RAM usage during model loading. "
            "It is recommended to upgrade to version 1.10.0 or above."
        )


def _to_model_dtype(model, model_dtype):
    """Convert model to the specified dtype."""
    if model_dtype is not None:
        try:
            if (model_dtype == "float16" or model_dtype == "fp16") and model.dtype != torch.float16:
                model = model.to(torch.float16)
            elif (
                model_dtype == "bfloat16" or model_dtype == "bfp16" or model_dtype == "bf16"
            ) and model.dtype != torch.bfloat16:
                model = model.to(torch.bfloat16)
            elif model_dtype == "float32" or model_dtype == "fp32" and model.dtype != torch.bfloat32:
                model = model.to(torch.float32)
        except Exception:
            logger.error("please use more device to fit the device or just use one device")
            exit()
    return model


def get_model_dtype(model_dtype, default="auto"):
    """Normalize model dtype string to a standard value."""
    if model_dtype is None or model_dtype == "auto":
        model_dtype = default
    elif model_dtype in ["bf16", "bfloat16"]:
        model_dtype = "bfloat16"
    elif model_dtype in ["f16", "float16", "fp16"]:
        model_dtype = "float16"
    elif model_dtype in ["f32", "float32", "fp32"]:
        model_dtype = "float32"
    else:
        logger.warning(f"Unable to identify model_dtype {model_dtype}, reset to default model_dtype {default}")
        model_dtype = default
    return model_dtype


def handle_generation_config(model: torch.nn.Module):
    """Handle generation config settings for do_sample."""
    if hasattr(model, "generation_config"):
        generation_config = model.generation_config
        if hasattr(generation_config, "top_p") and generation_config.top_p != 1.0:
            model.generation_config.do_sample = True
        if hasattr(generation_config, "top_k") and generation_config.top_k != 0:
            model.generation_config.do_sample = True
        if hasattr(generation_config, "temperature") and generation_config.temperature != 1.0:
            model.generation_config.do_sample = True


# Extra non-weight files that some models require at load time but are not saved
# by model.save_pretrained().  These are copied from the source model cache to
# the quantized output directory so that from_pretrained() works out of the box.
_EXTRA_MODEL_FILES = {
    "spk_dict.pt",  # Qwen2.5-Omni speaker dictionary for audio output
    "llm_config.json",  # BAGEL sub-model config
    "vit_config.json",  # BAGEL vision transformer config
    "preprocessor_config.json",  # BAGEL / Qwen VL image preprocessor config
    "processor_config.json",  # Qwen VL processor config (image + video processors)
}


def _copy_extra_model_files(src_dir: str, dst_dir: str):
    """Copy known extra model files from *src_dir* to *dst_dir* if they exist."""
    import shutil

    for file in os.listdir(src_dir):
        if file in _EXTRA_MODEL_FILES:
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                logger.debug(f"Transferring extra model file {src_file} to {dst_dir}")
                shutil.copy(src_file, dst_dir)


def copy_python_files_from_model_cache(model, save_path: str, copy_folders: bool | list[str] | tuple[str, ...] = False):
    """Copy Python files (and optionally subdirectories) from the model cache to *save_path*."""
    import shutil

    from huggingface_hub import hf_hub_download

    config = model.config
    if not hasattr(config, "_name_or_path"):
        return

    if version.parse(torch.__version__) < version.parse("5.0.0") and version.parse("0.0.0") < version.parse("5.0.0"):
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_dir = os.environ.get("HF_HOME", HF_HUB_CACHE)
    else:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_dir = os.environ.get("HF_HOME", HF_HUB_CACHE)
    from transformers.utils import http_user_agent

    cache_path = config._name_or_path
    if not os.path.exists(cache_path):
        user_agent = http_user_agent()
        config_file_path = hf_hub_download(
            repo_id=cache_path,
            filename="config.json",
            cache_dir=cache_dir,
            force_download=False,
            user_agent=user_agent,
        )
        cache_path = os.path.sep.join(config_file_path.split(os.path.sep)[:-1])

    for file in os.listdir(cache_path):
        full_file_name = os.path.join(cache_path, file)
        if file.endswith(".py") and os.path.isfile(full_file_name):
            logger.debug(f"Transferring {full_file_name} to {save_path}")
            shutil.copy(full_file_name, save_path)

    _copy_extra_model_files(cache_path, save_path)

    if copy_folders is not False:
        for entry in os.listdir(cache_path):
            src_entry = os.path.join(cache_path, entry)
            dst_entry = os.path.join(save_path, entry)
            if not os.path.isdir(src_entry):
                continue
            if copy_folders is True or entry in copy_folders:
                if not os.path.exists(dst_entry):
                    logger.debug(f"Transferring folder {src_entry} to {save_path}")
                    shutil.copytree(src_entry, dst_entry)


def config_save_pretrained(config, file_name, save_directory, model=None):
    """Save model config to a directory."""
    import json

    if os.path.isfile(save_directory):
        raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
    os.makedirs(save_directory, exist_ok=True)
    output_config_file = os.path.join(save_directory, file_name)

    config_dict = dict(config)
    if model is not None:
        if file_name == "config.json" and hasattr(model.config, "quantization_config"):
            config_dict["quantization_config"] = model.config.quantization_config

    with open(output_config_file, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(config_dict, indent=2, sort_keys=True) + "\n")


def hook_ngram_embeddings_on_cpu(model):
    """Hook ngram embeddings to run on CPU to save GPU memory."""
    has_ngram_embeddings = hasattr(model, "model") and hasattr(model.model, "ngram_embeddings")
    if has_ngram_embeddings:
        raw_ngram_embeddings = model.model.ngram_embeddings

        def hook_input_output_device_for_cpu_module(module):
            from accelerate.hooks import AlignDevicesHook, add_hook_to_module

            hook = AlignDevicesHook(
                io_same_device=True,
                execution_device="cpu",
            )

            add_hook_to_module(module, hook)

        hook_input_output_device_for_cpu_module(raw_ngram_embeddings)
    return has_ngram_embeddings, raw_ngram_embeddings if has_ngram_embeddings else None


def llm_load_model(
    pretrained_model_name_or_path: str,
    platform: str = "hf",
    trust_remote_code: bool = True,
    model_dtype: str = None,
    device: str = "cpu",
    **kwargs,
):
    """Load a LLM model and tokenizer from HuggingFace.

    Args:
        pretrained_model_name_or_path: Model path or HuggingFace model ID.
        platform: Platform to download model ("hf" or "model_scope").
        trust_remote_code: Whether to trust remote code.
        model_dtype: Model dtype for loading.
        device: Device to load model on.

    Returns:
        tuple: (model, tokenizer)
    """
    assert platform.lower() == "hf", "Only hf platform is supported in this fork."

    _check_accelerate_version()

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from auto_round.utils.device import (
        get_device_and_parallelism,
        override_cuda_device_capability,
    )

    device_str, use_auto_mapping = get_device_and_parallelism(device)
    torch_dtype = "auto"

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "device_map": "auto" if use_auto_mapping else None,
    }

    # BAGEL requires a custom loader (Qwen2 + not extensions, not in transformers)
    _config_path = (
        os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.isdir(pretrained_model_name_or_path)
        else None
    )
    if _config_path and os.path.exists(_config_path):
        with open(_config_path) as _f:
            _mt = json.load(_f).get("model_type")
        if _mt == "bagel":
            from auto_round.utils.bagel_loader import load_bagel_model

            model, tokenizer = load_bagel_model(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
            )
            model = _to_model_dtype(model, model_dtype)
            model._autoround_to_quant_block_names = "language_model.model.layers"
            return model, tokenizer

    is_glm = bool(re.search("chatglm", pretrained_model_name_or_path.lower()))

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)

    model_cls = AutoModel if is_glm else AutoModelForCausalLM
    if "deepseek" in pretrained_model_name_or_path.lower() and trust_remote_code:
        logger.warning("trust_remote_code is enabled by default, please ensure its correctness.")

    # ── Meta device path: load architecture without weights ──────────────
    if device == "meta":
        from transformers import AutoConfig

        logger.info(
            "Loading model on meta device (architecture only, no weights). "
            "Blocks will be loaded on demand during quantization."
        )

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )

        model = model_cls.from_config(config, trust_remote_code=trust_remote_code)
        model = model.to(torch.device("meta"))

        model.path = pretrained_model_name_or_path
        model.config = config

        monkey_patch_model(model)
        check_and_mark_quantized_module(model)
        handle_generation_config(model)
        model = _to_model_dtype(model, model_dtype)

        return model, tokenizer
    # ── End meta device path ─────────────────────────────────────────────

    # ── Non-meta path ─────────────────────────────────────────────────
    # Load the raw config BEFORE model loading so we can preserve the
    # original structure (text_config, vision_config, etc.).
    # AutoModelForCausalLM.from_pretrained() flattens sub-configs and
    # rewrites model_type (e.g. qwen3_5 -> qwen3_5_text), which breaks
    # vLLM compatibility.
    from transformers import AutoConfig

    raw_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code
    )

    try:
        model = model_cls.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
    except ValueError as e:
        if "FP8 quantized" in str(e):
            with override_cuda_device_capability():
                model = model_cls.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
            logger.warning("the support for fp8 model as input is experimental, please use with caution.")
        else:
            raise
    except OSError as e:
        logger.warning(f"fail to load {pretrained_model_name_or_path}, set trust_remote_code to False and retry.")
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path, **{**load_kwargs, "trust_remote_code": False}
        )

    # Restore the raw config to preserve sub-configs (text_config, vision_config)
    # and the original model_type. The flattened config from transformers breaks
    # vLLM's config resolution (e.g. max_position_embeddings lookup).
    model.config = raw_config

    model = model.eval()
    monkey_patch_model(model)
    check_and_mark_quantized_module(model)
    handle_generation_config(model)
    model = _to_model_dtype(model, model_dtype)

    return model, tokenizer


def mllm_load_model(
    pretrained_model_name_or_path: str,
    platform: str = "hf",
    device: str = "cpu",
    torch_dtype: str = "auto",
    use_auto_mapping: bool = True,
    trust_remote_code: bool = True,
    model_dtype: str = None,
    **kwargs,
):
    """Load a multimodal (MLLM) model, processor, tokenizer, and image_processor.

    Ported from upstream auto-round auto_round/utils/model.py:500-798.
    Uses the correct architecture class (e.g. Qwen3_5ForConditionalGeneration)
    instead of AutoModelForCausalLM, preserving nested model structure.

    Returns:
        tuple: (model, processor, tokenizer, image_processor)
    """
    _check_accelerate_version()

    assert platform.lower() == "hf", "Only hf platform is supported in this fork."

    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    from auto_round.utils.device import get_device_and_parallelism, override_cuda_device_capability

    device_str, use_auto_mapping = get_device_and_parallelism(device)
    torch_dtype = "auto"

    # Read raw config BEFORE model loading to preserve nested structure
    # (text_config, vision_config, image_token_id, etc.).
    # cls.from_pretrained() flattens sub-configs and rewrites model_type
    # (e.g. qwen3_5 -> qwen3_5_text), which breaks vLLM compatibility.
    raw_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code
    )

    model_type = raw_config.model_type
    processor, image_processor = None, None

    # Use the correct architecture class
    architectures = raw_config.architectures[0]

    cls = None
    # 1. Try AutoModelForCausalLM (handles most models)
    if architectures.endswith("Model") and hasattr(
        AutoModelForCausalLM, n := architectures.replace("Model", "ForConditionalGeneration")
    ):
        cls = getattr(AutoModelForCausalLM, n)
    elif hasattr(AutoModelForCausalLM, architectures):
        cls = getattr(AutoModelForCausalLM, architectures)

    # 2. Try importing from transformers directly (needed for some MLLM models
    #    like Qwen3_5ForConditionalGeneration that aren't in AutoModelForCausalLM)
    if cls is None:
        import transformers as _tf
        if hasattr(_tf, architectures):
            cls = getattr(_tf, architectures)
            logger.info("Using architecture class %s from transformers", architectures)

    # 3. Fall back to AutoModelForCausalLM
    if cls is None:
        cls = AutoModelForCausalLM

    try:
        model = cls.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map="auto" if use_auto_mapping else None,
        )
    except ValueError as e:
        if "FP8 quantized" in str(e):
            with override_cuda_device_capability():
                model = cls.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype,
                    device_map="auto" if use_auto_mapping else None,
                )
            logger.warning("the support for fp8 model as input is experimental, please use with caution.")
        else:
            raise

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code
    )

    try:
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
    except Exception:
        pass

    try:
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
    except Exception:
        pass

    # Restore the raw config to preserve sub-configs (text_config, vision_config)
    # and the original model_type. The flattened config from transformers breaks
    # vLLM's config resolution (e.g. max_position_embeddings lookup).
    model.config = raw_config

    # Set block names for MLLM models if the config specifies them.
    # For MLLM models, to_quant_block_names should be a string pattern
    # (e.g. "model.language_model.layers") or None (auto-detect all blocks).
    # We do NOT set it to a list here — find_matching_blocks handles lists
    # by returning them directly, which bypasses block detection.
    config_block_names = getattr(raw_config, "block_name_to_quantize", None)
    if config_block_names and not isinstance(config_block_names, (list, tuple)):
        model._autoround_to_quant_block_names = config_block_names

    # Mark vision encoder for optional quantization
    # By default, vision encoder runs in full precision (bf16)
    # Set quant_nontext_module=True to quantize vision encoder as well
    if hasattr(model, 'visual'):
        model._autoround_quant_nontext_module = kwargs.get('quant_nontext_module', False)
        if model._autoround_quant_nontext_module:
            logger.info("Vision encoder will be quantized (quant_nontext_module=True)")
        else:
            logger.info("Vision encoder will run in full precision (quant_nontext_module=False)")

    model = model.eval()
    check_and_mark_quantized_module(model)
    handle_generation_config(model)
    model = _to_model_dtype(model, model_dtype)

    # Log multi-GPU and memory usage information
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            logger.info(f"Using multi-GPU setup: {gpu_count} GPUs detected")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                logger.debug(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        # Log memory usage for MLLM models
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info(
            f"MLLM model loaded. GPU memory: "
            f"{memory_allocated:.2f} GB allocated, "
            f"{memory_reserved:.2f} GB reserved"
        )

    return model, processor, tokenizer, image_processor
