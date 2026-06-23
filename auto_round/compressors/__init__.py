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

# Lazy imports to avoid circular dependencies
# Users should import from specific modules instead of this __init__.py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auto_round.compressors.base import BaseCompressor
    from auto_round.compressors.config import (
        ExtraConfig,
        SARConfig,
        SchemeExtraConfig,
        TuningExtraConfig,
    )
    from auto_round.compressors.data_driven import DataDrivenCompressor
    from auto_round.compressors.entry import AutoRoundCompatible, AutoRound
    from auto_round.compressors.memory_estimator import estimate_peak_memory_per_block
    from auto_round.compressors.auto_tune import (
        auto_tune,
        format_preflight_message,
        format_resume_message,
    )

__all__ = [
    "AutoRound",
    "BaseCompressor",
    "DataDrivenCompressor",
    "AutoRoundCompatible",
    "ExtraConfig",
    "SARConfig",
    "TuningExtraConfig",
    "SchemeExtraConfig",
    "estimate_peak_memory_per_block",
    "auto_tune",
    "format_preflight_message",
    "format_resume_message",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "AutoRound" or name == "AutoRoundCompatible":
        from auto_round.compressors.entry import AutoRound, AutoRoundCompatible

        if name == "AutoRound":
            return AutoRound
        return AutoRoundCompatible
    elif name == "BaseCompressor":
        from auto_round.compressors.base import BaseCompressor

        return BaseCompressor
    elif name == "DataDrivenCompressor":
        from auto_round.compressors.data_driven import DataDrivenCompressor

        return DataDrivenCompressor
    elif name in ("ExtraConfig", "SARConfig", "TuningExtraConfig", "SchemeExtraConfig"):
        from auto_round.compressors.config import (
            ExtraConfig,
            SARConfig,
            SchemeExtraConfig,
            TuningExtraConfig,
        )

        return {
            "ExtraConfig": ExtraConfig,
            "SARConfig": SARConfig,
            "TuningExtraConfig": TuningExtraConfig,
            "SchemeExtraConfig": SchemeExtraConfig,
        }[name]
    elif name == "estimate_peak_memory_per_block":
        from auto_round.compressors.memory_estimator import estimate_peak_memory_per_block

        return estimate_peak_memory_per_block
    elif name == "auto_tune":
        from auto_round.compressors.auto_tune import auto_tune

        return auto_tune
    elif name == "format_preflight_message":
        from auto_round.compressors.auto_tune import format_preflight_message

        return format_preflight_message
    elif name == "format_resume_message":
        from auto_round.compressors.auto_tune import format_resume_message

        return format_resume_message
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
