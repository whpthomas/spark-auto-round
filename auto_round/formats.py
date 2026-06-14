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

from __future__ import annotations

import copy
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict
from enum import Enum
from typing import TYPE_CHECKING, Callable, Union

import torch
import transformers

from auto_round.compressors.utils import (
    is_standard_fp,
)
from auto_round.schemes import (
    PRESET_SCHEMES,
    QuantizationScheme,
)
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_FORMATS,
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    compress_layer_names,
    copy_python_files_from_model_cache,
    find_matching_blocks,
    get_block_names,
    get_module,
    logger,
    unsupported_meta_device,
)


class AutoRoundExportFormat(str, Enum):
    """Export format identifiers used internally by AutoRoundFormat.

    Only FP8 and WINT_A16 are used in the current W4A16-only build.
    Other members (MXFP, NVFP, etc.) were removed in the Phase 3 prune.
    """
    FP8 = "fp8"
    WINT_A16 = "wint_a16"


if TYPE_CHECKING:
    from auto_round.compressors.base import BaseCompressor


def _check_compatibility(formats: list[str], ar: BaseCompressor):
    """Validate format compatibility with the current quantization config."""
    return formats


def get_formats(
    format: str,
    ar: BaseCompressor,
) -> list[OutputFormat]:
    """Get the list of OutputFormat instances based on the provided name."""

    def remove_duplicates(lst):
        seen = set()
        return [x for x in lst if not (x in seen or seen.add(x))]

    formats = format.lower().replace("q*_", f"q{ar.bits}_").replace(" ", "").split(",")
    formats = remove_duplicates(formats)  # need the keep origin order

    formats = _check_compatibility(formats, ar)

    formats = remove_duplicates(formats)

    for fmt in formats:
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"{fmt} is not supported, we only support {SUPPORTED_FORMATS}")

    for i in range(len(formats)):
        if formats[i] not in OutputFormat._format_list:
            raise KeyError(f"Unsupported format {formats[i]}, please choose from {SUPPORTED_FORMATS}")
        else:
            formats[i] = OutputFormat._format_list[formats[i]](formats[i], ar)

        new_format = formats[i].check_and_reset_format(ar)
        if new_format is not None:
            if new_format not in format:
                formats[i] = OutputFormat._format_list[new_format](new_format, ar)
            else:
                formats[i] = None

    formats = [fmt for fmt in formats if fmt is not None]

    return formats


def _check_divisible_by_32(ar):
    from auto_round.schemes import preset_name_to_scheme

    if isinstance(ar.scheme, str):
        default_dict = asdict(preset_name_to_scheme(ar.scheme.upper()))
    else:
        default_dict = asdict(ar.scheme)
    skipped_layers = []
    if default_dict["data_type"] == "int" and default_dict["act_bits"] >= 16:
        for n, m in ar.model.named_modules():
            if type(m) in SUPPORTED_LAYER_TYPES or m.__class__.__name__ in INNER_SUPPORTED_LAYER_TYPES:
                if m.weight.shape[0] % 32 or m.weight.shape[1] % 32:
                    if ar.layer_config is None:
                        ar.layer_config = {}
                    if ar.layer_config.get(n) is not None and ar.layer_config[n]["bits"] >= 16:
                        continue
                    ar.layer_config.setdefault(n, copy.deepcopy(default_dict))
                    ar.layer_config[n].update({"bits": 16, "data_type": "fp", "fixed_by_user": True})
                    skipped_layers.append(n)
    compressed_skipped_layers = compress_layer_names(skipped_layers)
    logger.warning_once(
        f"some layers are skipped quantization (shape not divisible by 32): {compressed_skipped_layers}"
    )


class OutputFormat(ABC):
    """Base class for different output formats.

    format: determines which method from export module to use for exporting.
            For example, auto_round or fake.
    backend: determines the specific export process within the format.
            For example, auto_round:auto_gptq, auto_round:auto_awq, etc.
    """

    support_schemes: list = []
    _format_list: dict[str, OutputFormat] = {}
    format_name = "base"

    def __init__(self, format: str, ar: BaseCompressor):
        """Initialize the OutputFormat class."""
        self.output_format = format
        self.backend = None

        if not self.is_fake() and not self.is_support_scheme(ar.scheme):
            logger.error(
                f"Currently, the {self.format_name} format only supports {self.support_schemes}, "
                f"but got scheme {ar.scheme}, please change to fake or auto_round etc."
            )
            exit(-1)

    @classmethod
    def register(cls, *names: str) -> Callable[[OutputFormat], OutputFormat]:
        assert names

        def func(output_format: OutputFormat) -> OutputFormat:
            for name in names:
                cls._format_list[name] = output_format
            return output_format

        return func

    @classmethod
    def get_support_matrix(cls: OutputFormat) -> str:
        output_str = ""
        for k, v in sorted(cls._format_list.items()):
            if k == "fake":
                support_schemes = "All schemes"
            else:
                if ":" in k and k.split(":")[1] in cls._format_list:
                    support_schemes = cls._format_list[k.split(":")[1]].support_schemes
                else:
                    support_schemes = v.support_schemes
                support_schemes = ", ".join(support_schemes).rstrip(",")
            output_str += f"\x1b[31;1m{k}\x1b[0m support scheme:\n\t{support_schemes}\n"
        return output_str

    def get_backend_name(self) -> str:
        if self.backend is None:
            return self.output_format

        # auto_round:auto_gptq, auto_round:fp8, etc.
        if self.backend.backend is not None:
            return f"{self.output_format}:{self.backend.get_backend_name()}"
        # auto_round:fp8, auto_round:auto_awq
        else:
            return self.backend.get_backend_name()

    @classmethod
    def is_support_scheme(cls: OutputFormat, scheme: Union[str, QuantizationScheme]) -> bool:
        if isinstance(scheme, str) and scheme.upper() in cls.support_schemes:
            return True
        if isinstance(scheme, QuantizationScheme):
            return cls.check_scheme_args(scheme)
        return False

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        return True

    def check_and_reset_format(self, ar: BaseCompressor) -> str:
        if self.backend is not None:
            new_format = self.backend.check_and_reset_format(ar)
            self.backend = OutputFormat._format_list[new_format](new_format, ar) if new_format else self.backend

        w_fp8 = ar.data_type.startswith("fp") and ar.bits == 8
        act_fp8 = ar.act_data_type.startswith("fp") and ar.act_bits == 8

        if (w_fp8 or act_fp8):
            error_msg = (
                f"is only supported to export auto_round format,"
                f" but got {self.format_name}, please check."
            )
            error_msg = ("act_data_type<fp8> " + error_msg) if act_fp8 else error_msg
            error_msg = ("data_type<fp8> " + error_msg) if w_fp8 else error_msg
            logger.error(error_msg)
            sys.exit(-1)

        if ar.act_bits <= 8 and (not is_standard_fp(ar.act_data_type) or ar.act_dynamic):
            logger.warning(
                f"{self.format_name} format does not support the current activation quantization configuration,"
                " reset to fake format and save."
            )
            return "fake"

        return None

    @abstractmethod
    def pack_layer(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_quantized(self, *args, **kwargs):
        pass

    def immediate_pack(self, name: str, model: torch.nn.Module, device: torch.device, **kwargs):
        m = get_module(model, name)
        if not check_to_quantized(m):
            return

        self.pack_layer(name, model, device=device)

    def is_fake(self) -> bool:
        return self.output_format == "fake"

    def is_gguf(self) -> bool:
        """GGUF format is not supported in this build. Always returns False."""
        return False

    def is_gptq(self) -> bool:
        """GPTQ format is not a user-facing format in this build."""
        return "gptq" in self.output_format or (self.backend is not None and self.backend.is_gptq())

    def is_awq(self) -> bool:
        """AWQ format is not a user-facing format in this build."""
        return "awq" in self.output_format or (self.backend is not None and self.backend.is_awq())

    def is_llm_compressor(self) -> bool:
        """LLM Compressor format is not a user-facing format in this build."""
        return "llm_compressor" in self.output_format or (self.backend is not None and self.backend.is_llm_compressor())


@OutputFormat.register("fake")
class FakeFormat(OutputFormat):
    support_schemes = None
    format_name = "fake"

    def check_and_reset_format(self, ar: BaseCompressor) -> str:
        return None

    # fake format will not execute pack_layer.
    def pack_layer(self, *args, **kwargs):
        pass

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable = None,
        layer_config: dict = None,
        inplace: bool = True,
        device: Union[str, torch.device] = "cpu",
        serialization_dict: dict = None,
        **kwargs,
    ):
        if not unsupported_meta_device(model):
            model = model.to("cpu")
            model.save_pretrained(output_dir)
        elif hasattr(model, "config") and model.config is not None:
            model.config.save_pretrained(output_dir)

        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)
        processor = kwargs.get("processor", None)
        if processor is not None:
            processor.save_pretrained(output_dir)
        try:
            copy_python_files_from_model_cache(model, output_dir)
        except Exception as e:
            logger.warning("Skipping source model Python file copy due to error: %s", e)
        return model



class _GPTQBackend(OutputFormat):
    """Internal backend for symmetric INT quantization export (auto_round:auto_gptq)."""
    support_schemes = ["W4A16", "W2A16", "W3A16", "W8A16", "BF16", "W2A16G64", "W2A16G32"]
    format_name = "auto_gptq"

    def check_and_reset_format(self, ar):
        if not ar.sym:
            logger.warning(
                "the asymmetrical kernel of the GPTQ format may result in a noticeable accuracy drop,"
                " particularly for 2-bit quantization and smaller models."
                " We recommend exporting to the AutoRound format."
            )
        if self.backend is None:
            _check_divisible_by_32(ar)
        return super().check_and_reset_format(ar)

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        error_logs = []
        if scheme.bits not in [2, 3, 4, 8, 16]:
            error_logs.append(f"bits={scheme.bits}")
        if not re.search("int", scheme.data_type):
            error_logs.append(f"data_type={scheme.data_type}")
        if scheme.super_bits:
            error_logs.append(f"super_bits={scheme.super_bits}")
        if scheme.super_group_size:
            error_logs.append(f"super_group_size={scheme.super_group_size}")
        if error_logs:
            raise ValueError(
                f"{cls.format_name} format support quantization scheme with {','.join(cls.support_schemes)} "
                f"but got {', '.join(error_logs)}, please have a check."
            )
        return True

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        from auto_round.export.export_to_autoround.export import pack_layer

        pack_layer(layer_name, model, backend=self.output_format, device=device)

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable = None,
        layer_config: dict = None,
        inplace: bool = True,
        device: Union[str, torch.device] = "cpu",
        serialization_dict: dict = None,
        **kwargs,
    ) -> torch.nn.Module:
        backend = self.get_backend_name()
        from auto_round.export.export_to_autoround.export import save_quantized_as_autoround

        export_func = save_quantized_as_autoround
        return export_func(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            layer_config=layer_config,
            inplace=inplace,
            device=device,
            backend=backend,
            serialization_dict=serialization_dict,
            **kwargs,
        )


class _AWQBackend(OutputFormat):
    """Internal backend for asymmetric INT4 quantization export (auto_round:auto_awq)."""
    support_schemes = ["W4A16"]
    format_name = "auto_awq"

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        error_logs = []
        if scheme.bits != 4:
            error_logs.append(f"bits={scheme.bits}")
        if not re.search("int", scheme.data_type):
            error_logs.append(f"data_type={scheme.data_type}")
        if scheme.super_bits:
            error_logs.append(f"super_bits={scheme.super_bits}")
        if scheme.super_group_size:
            error_logs.append(f"super_group_size={scheme.super_group_size}")
        if error_logs:
            raise ValueError(
                f"{cls.format_name} format support quantization scheme with {','.join(cls.support_schemes)} "
                f"but got {', '.join(error_logs)}, please have a check."
            )
        return True

    @staticmethod
    def check_awq_gemm_compatibility(model, bits, group_size, sym, layer_configs=None):
        """Checks if a model is compatible with the AutoAWQ GEMM kernel.

        Args:
            model: The model object to evaluate, typically a PyTorch model.
            bits (int): The number of bits for quantization (must be 4 for compatibility).
            group_size (int): The group size for quantization.
            sym (bool): Whether symmetric quantization is used (not utilized in the current function logic).
            layer_configs (dict, optional): A dictionary mapping layer names to configurations, where each
                configuration can specify a custom number of bits for the layer.

        Returns:
            tuple: A tuple containing:
                - bool: `True` if the model is compatible, `False` otherwise.
                - str: An error message describing why the model is incompatible, or an empty string if compatible.
        """
        from auto_round.utils.model import get_layer_names_in_block, get_module

        if bits != 4:
            return False, "AutoAWQ GEMM kernel only supports 4 bits"
        for n, m in model.named_modules():
            if type(m) == transformers.pytorch_utils.Conv1D:
                return False, "AutoAWQ GEMM kernel does not support conv1d"

        layer_names = get_layer_names_in_block(model)
        for layer_name in layer_names:
            if (
                layer_configs is not None
                and layer_name in layer_configs.keys()
                and layer_configs[layer_name].get("bits", bits) > 8
            ):
                continue

            layer = get_module(model, layer_name)
            if layer.in_features % group_size != 0:
                return False, f"Layer {layer_name} in_features is not multiple of group_size {group_size}"
            if layer.out_features % (32 // bits) != 0:
                return False, f"Layer {layer_name} out_features is not multiple of 32 // bits"

        return True, ""

    def check_and_reset_format(self, ar):
        awq_supported, info = self.check_awq_gemm_compatibility(
            ar.model, ar.bits, ar.group_size, ar.sym, ar.layer_config
        )
        if not awq_supported:
            logger.warning(f"The AutoAWQ format may not be supported due to {info}")
        if ar.bits != 4:
            raise ValueError(f"auto_awq format support quantization scheme with W4A16 but got bits={ar.bits}")

        if self.backend is None:
            _check_divisible_by_32(ar)

        return super().check_and_reset_format(ar)

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        from auto_round.export.export_to_autoround.export import pack_layer

        pack_layer(layer_name, model, backend=self.output_format, device=device)

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable = None,
        layer_config: dict = None,
        inplace: bool = True,
        device: Union[str, torch.device] = "cpu",
        serialization_dict: dict = None,
        **kwargs,
    ) -> torch.nn.Module:
        backend = self.get_backend_name()
        from auto_round.export.export_to_autoround.export import save_quantized_as_autoround

        export_func = save_quantized_as_autoround

        return export_func(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            layer_config=layer_config,
            inplace=inplace,
            backend=backend,
            device=device,
            serialization_dict=serialization_dict,
            **kwargs,
        )






@OutputFormat.register("auto_round")
@OutputFormat.register("auto_round:auto_gptq")
@OutputFormat.register("auto_round:auto_awq")
@OutputFormat.register("auto_round:fp8")
class AutoRoundFormat(OutputFormat):
    support_schemes = [
        "W4A16",
        "W2A16",
        "W3A16",
        "W8A16",
        "FPW8A16",
        "W2A16G64",
        "W2A16G32",
        "BF16",
    ]
    format_name = "auto_round"

    def __init__(self, format: str, ar: BaseCompressor):
        self.output_format = "auto_round"
        self.backend = None

        if format == "auto_round":
            if ar.sym and "int" in ar.data_type:
                self.backend = _GPTQBackend("auto_round:auto_gptq", ar)
            elif ar.bits == 4 and not ar.sym and "int" in ar.data_type:
                if ar.layer_config is None:
                    enable_awq = True
                else:
                    enable_awq = all(
                        config["bits"] == ar.bits or config["bits"] >= 16 for config in ar.layer_config.values()
                    )
                if enable_awq:
                    self.backend = _AWQBackend("auto_round:auto_awq", ar)
            elif ar.data_type.startswith("fp") and ar.bits == 8 and ar.act_bits >= 16:  # woq fp8
                self.backend = AutoRoundFormat(AutoRoundExportFormat.FP8.value, ar)
            elif ar.data_type.startswith("fp") and ar.bits == 8 and isinstance(ar.group_size, tuple):
                self.backend = AutoRoundFormat("auto_round:fp8", ar)
            elif ar.act_bits < 16:
                raise ValueError(
                    "AutoRound format does not support exporting "
                    "for the current quantization configuration, "
                    "please change to `fake` format for research purpose"
                )
        # for auto_round:fp8 etc.
        elif not format.startswith("auto_round"):
            if format.upper() not in list(AutoRoundExportFormat.__members__.keys()):
                raise KeyError(f"Unsupported backend format auto_round:{format}, please check")
            else:
                self.output_format = f"auto_round:{format}"
                self.backend = None
        else:
            backend = format.split(":")[1] if ":" in format else None
            self.backend = self._format_list.get(backend)(format, ar) if backend else None

        if self.backend is not None:
            self.support_schemes = self.backend.support_schemes

    def check_and_reset_format(self, ar):
        if self.backend is not None:
            new_format = self.backend.check_and_reset_format(ar)
            self.backend = OutputFormat._format_list[new_format](new_format, ar) if new_format else self.backend

        if self.backend is None:
            _check_divisible_by_32(ar)
        return None

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        if self.backend is not None:
            return self.backend.pack_layer(layer_name, model, device=device, **kwargs)

        backend = self.get_backend_name()

        if self.output_format in [
            f"auto_round:{AutoRoundExportFormat.FP8.value}",
        ]:
            from auto_round.export.export_to_autoround.export_to_fp8 import pack_layer

            pack_func = pack_layer
        else:
            from auto_round.export.export_to_autoround.export import pack_layer

            pack_func = pack_layer
        return pack_func(layer_name, model, backend, device)

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable = None,
        layer_config: dict = None,
        inplace: bool = True,
        device: Union[str, torch.device] = "cpu",
        serialization_dict: dict = None,
        **kwargs,
    ) -> torch.nn.Module:
        if self.backend is not None:
            return self.backend.save_quantized(
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                layer_config=layer_config,
                inplace=inplace,
                device=device,
                serialization_dict=serialization_dict,
                **kwargs,
            )
        backend = self.get_backend_name()
        if serialization_dict.get("data_type", "int") == "fp" and serialization_dict.get("bits", 16) == 8:
            from auto_round.export.export_to_autoround.export_to_fp8 import save_quantized_as_autoround

            backend = "auto_round:fp8_static" if serialization_dict.get("act_bits", 16) == 8 else None
            export_func = save_quantized_as_autoround
        else:
            from auto_round.export.export_to_autoround.export import save_quantized_as_autoround

            export_func = save_quantized_as_autoround
        return export_func(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            layer_config=layer_config,
            inplace=inplace,
            device=device,
            backend=backend,
            serialization_dict=serialization_dict,
            **kwargs,
        )
