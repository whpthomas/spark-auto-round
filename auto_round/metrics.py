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
"""Per-layer sensitivity metrics for quantization quality assessment.

Provides two standalone functions for comparing reference (FP16) and
quantized block outputs:

- ``compute_psnr``: Peak Signal-to-Noise Ratio in dB
- ``compute_block_sensitivity``: Cosine similarity + PSNR combined

Both accept ``list[torch.Tensor]`` — one tensor per calibration sample —
which is the format returned by ``quantizer._get_block_outputs()``.
"""

from __future__ import annotations

import math

import torch


def compute_psnr(
    reference_output: list[torch.Tensor],
    quantized_output: list[torch.Tensor],
) -> float:
    """Compute average Peak Signal-to-Noise Ratio (PSNR) in dB.

    PSNR measures the magnitude of quantization error relative to signal
    strength. Unlike cosine similarity, it catches cases where the quantized
    output is a correctly-directed but scaled-down version of the reference.

    Args:
        reference_output: List of tensors from FP16/block output (one per sample).
        quantized_output: List of tensors from quantized block output (one per sample).

    Returns:
        Average PSNR across all samples in dB. Higher is better.
        Returns ``float('inf')`` when MSE is zero (perfect reconstruction).
        Returns ``float('inf')`` for empty inputs.
    """
    if not reference_output or not quantized_output:
        return float("inf")

    psnrs = []
    for ref, q in zip(reference_output, quantized_output):
        mse = torch.mean((ref.float() - q.float()) ** 2).item()
        if mse == 0:
            psnrs.append(float("inf"))
        else:
            max_val = ref.float().abs().max().item()
            psnrs.append(10 * math.log10(max_val**2 / mse))

    return sum(psnrs) / len(psnrs)


def compute_block_sensitivity(
    reference_output: list[torch.Tensor],
    quantized_output: list[torch.Tensor],
) -> tuple[float, float]:
    """Compute average cosine similarity and PSNR between block outputs.

    Args:
        reference_output: List of tensors from FP16/block output (one per sample).
        quantized_output: List of tensors from quantized block output (one per sample).

    Returns:
        Tuple of ``(cosine_sim, psnr_db)``. Both are averaged across all
        samples. Returns ``(1.0, float('inf'))`` for empty inputs.
    """
    if not reference_output or not quantized_output:
        return 1.0, float("inf")

    sims = []
    psnrs = []
    for ref, q in zip(reference_output, quantized_output):
        ref_flat = ref.float().flatten()
        q_flat = q.float().flatten()
        sim = torch.nn.functional.cosine_similarity(
            ref_flat.unsqueeze(0), q_flat.unsqueeze(0)
        )
        sims.append(sim.item())

        mse = torch.mean((ref.float() - q.float()) ** 2).item()
        if mse == 0:
            psnrs.append(float("inf"))
        else:
            max_val = ref.float().abs().max().item()
            psnrs.append(10 * math.log10(max_val**2 / mse))

    return sum(sims) / len(sims), sum(psnrs) / len(psnrs)
