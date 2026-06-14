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

from contextlib import nullcontext
from functools import partial
from typing import Callable, Union

import torch
import transformers
from torch import autocast

from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
from auto_round.algorithms.quantization.utils import register_imatrix_hooks

from auto_round.data_type.int import quant_tensor_asym, quant_tensor_sym, search_scales
from auto_round.data_type.utils import (
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
    round_ste,
)
from auto_round.logger import logger
from auto_round.utils import check_to_quantized, compile_func, get_reciprocal
from auto_round.wrapper import WrapperLinear, wrapper_block


class SignRoundOptimizedWrapperLinear(WrapperLinear):
    minmax_scale_bound = (0.0, 2.0)

    def _init_tuning_params_and_quant_func(self):
        super()._init_tuning_params_and_quant_func()

        orig_weight = getattr(self.orig_layer, "get_weight", lambda: self.orig_layer.weight)()
        if type(self.orig_layer) == transformers.pytorch_utils.Conv1D:
            orig_weight = orig_weight.t()
        weight_reshape, _, _ = reshape_pad_tensor_by_group_size(orig_weight.data, self.orig_layer.group_size)
        if hasattr(self.orig_layer, "imatrix"):
            imatrix = self.orig_layer.imatrix.reshape(1, -1)
            imatrix = reshape_pad_tensor_by_group_size(imatrix, self.orig_layer.group_size, val=1e-5)[0].view(1, -1)
            imatrix = imatrix.expand(weight_reshape.numel() // imatrix.numel(), -1)
            imatrix = imatrix.reshape(weight_reshape.shape).to(orig_weight.device)
        else:
            imatrix = 1.0

        if self.orig_layer.data_type.startswith("int"):
            self.init_scale = search_scales(weight_reshape, self.orig_layer.bits, imatrix)
            self.init_scale = torch.where(
                self.init_scale < 0,
                torch.clamp(self.init_scale, max=-self.q_scale_thresh),
                torch.clamp(self.init_scale, min=self.q_scale_thresh),
            )
            self.weight_quant_func = quant_tensor_sym
        else:
            raise ValueError(f"unsupported SignRound optimized data type: {self.orig_layer.data_type}")

        self.data_type = self.orig_layer.data_type
        if hasattr(self.orig_layer, "imatrix"):
            delattr(self.orig_layer, "imatrix")
        if self.enable_torch_compile:
            self.weight_quant_func = compile_func(self.weight_quant_func, self.device)


def _named_wrapper_block(wrapper_cls, name: str):
    wrapped = partial(wrapper_block, wrapper_cls=wrapper_cls)
    wrapped.__name__ = name
    return wrapped


class SignRoundV2Quantizer(SignRoundQuantizer):
    """SignRound variant using the open algorithm-extension path in the new architecture."""

    def __init__(self, config: SignRoundConfig):
        super().__init__(config)
        self._use_outlier_suppressed_loss = False
        logger.info("using algorithm extension for quantization.")

        if self.sym and self.super_group_size is None and self.data_type.startswith("int"):
            if self.bits > 2:
                logger.warning_once(
                    "algorithm extension has only undergone limited validation on "
                    "W2A16 and INT4; use with caution."
                )
            if self.act_bits <= 4 or self.bits < 4:
                self._use_outlier_suppressed_loss = True
            else:
                self._use_outlier_suppressed_loss = False
            self.wrapper_block = _named_wrapper_block(SignRoundOptimizedWrapperLinear, "wrapper_block")

    def _get_loss(
        self,
        output_q: torch.Tensor,
        current_output: torch.Tensor,
        indices: torch.Tensor,
        mse_loss: Callable,
        device: Union[str, torch.device] = "cpu",
    ):
        if self._use_outlier_suppressed_loss:
            loss_diff = torch.abs(output_q - current_output)
            flat_diff = loss_diff.view(-1)
            topk = max(1, int(flat_diff.numel() / 1000))
            _, top_indices = torch.topk(torch.abs(flat_diff), topk)
            mask = torch.zeros_like(flat_diff, dtype=torch.bool)
            mask[top_indices] = True
            mask = (~mask).view_as(loss_diff)

            autocast_ctx = (
                autocast(device_type=str(device).split(":")[0], dtype=self.amp_dtype) if self.amp else nullcontext()
            )
            if self.attention_mask:
                tmp_attention_mask = [self.attention_mask[i] for i in indices]
                tmp_attention_mask = torch.cat(tmp_attention_mask, dim=0).to(device)
                tmp_attention_mask.unsqueeze_(-1)
                with autocast_ctx:
                    return torch.mean(
                        (
                            torch.abs(output_q.to(torch.float32) - current_output.to(torch.float32))
                            * tmp_attention_mask
                            * mask
                        )
                        ** 2
                    )

            with autocast_ctx:
                return torch.mean(
                    (torch.abs(output_q.to(torch.float32) - current_output.to(torch.float32)) * mask) ** 2
                )
        return super()._get_loss(output_q, current_output, indices, mse_loss, device)

    def register_calibration_hooks(self, model, *, act_max: bool = True, imatrix: bool = True):
        hook_handles = super().register_calibration_hooks(model, act_max=act_max, imatrix=imatrix)
        if not imatrix:
            return hook_handles

        is_wint4aint4 = ("int4" in self.act_data_type or ("int" in self.act_data_type and self.act_bits == 4)) and (
            "int4" in self.data_type or ("int" in self.data_type and self.bits == 4)
        )
        if is_wint4aint4:
            return hook_handles
        hook_handles.extend(register_imatrix_hooks(self, model))
        return hook_handles
