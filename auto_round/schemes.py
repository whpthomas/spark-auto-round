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
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from auto_round.logger import logger
from auto_round.utils import SUPPORTED_DTYPES, infer_bits_by_data_type

__all__ = ["QuantizationScheme", "preset_name_to_scheme"]




@dataclass
class QuantizationScheme:
    bits: int = 4
    group_size: int = 128
    sym: bool = True
    data_type: str = "int"
    act_bits: Optional[int] = None
    act_group_size: Optional[int] = None
    act_sym: Optional[bool] = None
    act_data_type: Optional[str] = None
    act_dynamic: Optional[bool] = None
    super_bits: Optional[int] = None
    super_group_size: Optional[int] = None
    rotation_config: Optional[dict] = None

    @classmethod
    def from_dict(cls, config: dict):
        field_names = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config.items() if k in field_names}
        return cls(**filtered_config)

    @classmethod
    def get_attributes(cls: "QuantizationScheme") -> list[str]:
        return [field.name for field in fields(cls)]

    def __getitem__(self, key: str):
        if key not in self.get_attributes():
            raise KeyError(f"{key} is not a valid attribute")
        return getattr(self, key)

    def __setitem__(self, key: str, value: None | int | str):
        if key not in self.get_attributes():
            raise KeyError(f"{key} is not a valid attribute")
        setattr(self, key, value)

    def items(self):
        return ((field, getattr(self, field)) for field in self.get_attributes())

    def keys(self):
        return self.get_attributes()

    def values(self):
        return (getattr(self, field) for field in self.get_attributes())

    def get(self, key: str, default=None):
        if key not in self.get_attributes():
            return default
        res = getattr(self, key)
        # In case the attribute is explicitly set to None, return default
        if res is None:
            return default
        return getattr(self, key)

    def __eq__(self, other: "QuantizationScheme") -> bool:
        if not isinstance(other, QuantizationScheme):
            return False
        skip_act_check = False
        self_act_bits = 16 if self.act_bits is None else self.act_bits
        other_act_bits = 16 if other.act_bits is None else other.act_bits
        if self_act_bits == other_act_bits and other_act_bits >= 16:
            skip_act_check = True

        for field in self.get_attributes():
            if skip_act_check and field.startswith("act_"):
                continue
            self_val = getattr(self, field)
            other_val = getattr(other, field)
            # Treat None and empty dict as equivalent for dict fields like rotation_config
            if self_val != other_val:
                if isinstance(self_val, dict) and not self_val and other_val is None:
                    continue
                if isinstance(other_val, dict) and not other_val and self_val is None:
                    continue
                return False
        return True


def preset_name_to_scheme(name: str) -> QuantizationScheme:
    """Get a QuantizationScheme instance from a preset scheme name."""
    name = name.upper()

    if name not in PRESET_SCHEMES:
        raise KeyError(f"Unknown preset scheme name {name}, " f"available names: {list(PRESET_SCHEMES.keys())}")

    scheme_args = deepcopy(PRESET_SCHEMES[name])
    return scheme_args


def scheme_to_preset_name(scheme: Union[str, QuantizationScheme]) -> str:
    """Get preset scheme name from a QuantizationScheme instance."""
    if isinstance(scheme, str):
        name = scheme.upper()
        return name if name in PRESET_SCHEMES else ""

    for key, val in PRESET_SCHEMES.items():
        if val == scheme:
            return key
    return ""


def is_preset_scheme(name: str) -> bool:
    """Check if the given name is a preset scheme name."""
    return name.upper() in PRESET_SCHEMES


def _reconcile_bits_and_dtype(config: dict, prefix: str = ""):
    """
    Harmonizes 'bits' and 'data_type' for weights or activations.
    Ensures internal consistency by prioritizing data_type inference.
    """

    dt_key = f"{prefix}data_type"
    bits_key = f"{prefix}bits"

    if config.get(dt_key) is None:
        return

    # Infer the correct bit-width based on the data_type string
    inferred_bits = infer_bits_by_data_type(config[dt_key])

    if inferred_bits is not None and inferred_bits < 16:
        # Check for conflict between user-specified bits and inferred bits
        if inferred_bits != config.get(bits_key):
            logger.warning(f"'{dt_key}' does not match '{bits_key}'. " f"Resetting '{bits_key}' to {inferred_bits}.")
            config[bits_key] = inferred_bits

        # Normalize data_type (e.g., 'mx_fp4' -> 'mx')
        for supported in SUPPORTED_DTYPES:
            if config[dt_key] == f"{supported}{inferred_bits}":
                config[dt_key] = supported
                break


def _override_scheme_with_user_specify(
    scheme: Union[str, dict, QuantizationScheme], user_scheme_overrides: dict[str, Any], return_str=True
) -> Union[str, QuantizationScheme]:
    """
    Updates a base quantization scheme with user-provided overrides.
    Synchronizes weight/activation parameters.
    """
    # 1. Convert input scheme to a dictionary for processing
    if isinstance(scheme, QuantizationScheme):
        scheme_dict = asdict(scheme)
    elif isinstance(scheme, str):
        normalized_name = scheme.strip("'\" ").upper()
        # If no overrides exist, return the normalized string immediately
        if not user_scheme_overrides and return_str:
            return normalized_name
        scheme_dict = asdict(preset_name_to_scheme(normalized_name))
    else:
        scheme_dict = scheme.copy()

    # 3. Apply overrides and define default behaviors
    scheme_dict.update(user_scheme_overrides)

    if scheme_dict.get("act_dynamic") is None:
        scheme_dict["act_dynamic"] = True

    # 4. Reconcile weight settings (bits vs data_type)
    _reconcile_bits_and_dtype(scheme_dict)

    # 5. Fallback logic: Inherit activation settings from weight settings
    scheme_dict["act_group_size"] = (
        scheme_dict.get("act_group_size")
        if scheme_dict.get("act_group_size") is not None
        else scheme_dict.get("group_size")
    )
    scheme_dict["act_bits"] = scheme_dict.get("act_bits") or 16
    scheme_dict["act_sym"] = (
        scheme_dict.get("act_sym") if scheme_dict.get("act_sym") is not None else scheme_dict.get("sym")
    )

    # 6. Activation data_type logic
    if scheme_dict.get("act_data_type") is None:
        is_supported = scheme_dict["data_type"] in SUPPORTED_DTYPES
        if is_supported and scheme_dict["act_bits"] < 16:
            scheme_dict["act_data_type"] = scheme_dict["data_type"]
            logger.info(f"Activation adopting weight data_type: {scheme_dict['data_type']}")
        else:
            scheme_dict["act_data_type"] = "float"

    # 7. Reconcile activation settings
    _reconcile_bits_and_dtype(scheme_dict, prefix="act_")

    return QuantizationScheme.from_dict(scheme_dict)


def _parse_scheme(
    scheme: Union[str, dict, QuantizationScheme], user_scheme_overrides: dict[str, Any]
) -> tuple[Union[str, QuantizationScheme], dict[str, Any]]:
    """Parses the final scheme.

    Returns:
        Tuple of (resolved_scheme, final_attrs_dict)
    """
    default_scheme = _override_scheme_with_user_specify(scheme, user_scheme_overrides)

    # Extract attributes from the chosen default_scheme
    if isinstance(default_scheme, str):
        final_attrs = _override_scheme_with_user_specify(default_scheme, user_scheme_overrides, return_str=False)
        final_attrs = asdict(final_attrs)
    else:
        final_attrs = asdict(default_scheme)
    return default_scheme, final_attrs


W4A16 = QuantizationScheme.from_dict(
    {
        "bits": 4,
        "sym": True,
        "group_size": 128,
        "data_type": "int",
        "act_bits": 16,
    }
)

W5A16 = QuantizationScheme.from_dict(
    {
        "bits": 5,
        "sym": True,
        "group_size": 128,
        "data_type": "int",
        "act_bits": 16,
    }
)

W6A16 = QuantizationScheme.from_dict(
    {
        "bits": 6,
        "sym": True,
        "group_size": 128,
        "data_type": "int",
        "act_bits": 16,
    }
)

W2A16 = QuantizationScheme.from_dict(
    {
        "bits": 2,
        "sym": True,
        "group_size": 128,
        "data_type": "int",
        "act_bits": 16,
    }
)

W2A16G64 = QuantizationScheme.from_dict(
    {
        "bits": 2,
        "sym": True,
        "group_size": 64,
        "data_type": "int",
        "act_bits": 16,
    }
)

W2A16G32 = QuantizationScheme.from_dict(
    {
        "bits": 2,
        "sym": True,
        "group_size": 32,
        "data_type": "int",
        "act_bits": 16,
    }
)

W3A16 = QuantizationScheme.from_dict(
    {
        "bits": 3,
        "sym": True,
        "group_size": 128,
        "data_type": "int",
        "act_bits": 16,
    }
)

W8A16 = QuantizationScheme.from_dict(
    {
        "bits": 8,
        "sym": True,
        "group_size": 128,
        "data_type": "int",
        "act_bits": 16,
    }
)


FPW8A16 = QuantizationScheme.from_dict(
    {
        "bits": 8,
        "group_size": 0,
        "data_type": "fp",
        "act_bits": 16,
        "act_data_type": "fp",
    }
)

# For AutoScheme 16 bits options
BF16 = QuantizationScheme.from_dict(
    {
        "bits": 16,
        "group_size": 128,
        "data_type": "fp",
        "act_bits": 16,
        "act_data_type": "fp",
    }
)

PRESET_SCHEMES = {
    "W4A16": W4A16,
    "W2A16": W2A16,
    "W3A16": W3A16,
    "W5A16": W5A16,
    "W6A16": W6A16,
    "W8A16": W8A16,
    "FPW8A16": FPW8A16,
    "W2A16G64": W2A16G64,
    "W2A16G32": W2A16G32,
    "BF16": BF16,
}

