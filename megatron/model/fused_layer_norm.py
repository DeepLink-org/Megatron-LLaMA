# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import importlib

from megatron.core.utils import make_viewless_tensor
from megatron import get_args
# from .normalization_fused_layer_norm import FusedRMSNormAffineFunction


HAVE_PERSIST_LAYER_NORM = False

global fused_layer_norm_cuda
fused_layer_norm_cuda = None

def manual_rms_norm(my_input, normalized_shape, weight, eps):
    # layer norm should always be calculated in float32
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = my_input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    my_input = my_input * torch.rsqrt(variance + eps)

    if weight is None:
        return my_input

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        my_input = my_input.to(weight.dtype)
    return weight * my_input

class MixedFusedLayerNorm(torch.nn.Module):

  def __init__(self, normalized_shape, eps=1e-5,
               no_persist_layer_norm=True,
               sequence_parallel=False,
               apply_layernorm_1p=False):
        super(MixedFusedLayerNorm, self).__init__()

        args = get_args()

        self.apply_layernorm_1p = apply_layernorm_1p

        global fused_layer_norm_cuda
        fused_layer_norm_cuda = None

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [1024, 1536, 2048, 2304, 3072, 3840, 4096,
            5120, 6144, 8192, 10240, 12288, 12800, 15360, 16384, 18432, 20480,
            24576, 25600, 30720, 32768, 40960, 49152, 65536]
        if normalized_shape not in persist_ln_hidden_sizes or \
                not HAVE_PERSIST_LAYER_NORM:
            no_persist_layer_norm = True

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        if not args.RMSNorm:
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
        self.no_persist_layer_norm = no_persist_layer_norm
        self.sequence_parallel = sequence_parallel

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
        if not args.RMSNorm:
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)


  def reset_parameters(self):
    args = get_args()

    if self.apply_layernorm_1p:
        init.zeros_(self.weight)
        if not args.RMSNorm:
            init.zeros_(self.bias)
    else:
        init.ones_(self.weight)
        if not args.RMSNorm:
            init.zeros_(self.bias)

  def forward(self, input):

    args = get_args()

    weight = self.weight + 1 if self.apply_layernorm_1p else self.weight

    if args.RMSNorm:
        # return FusedRMSNormAffineFunction.apply(input, weight, self.normalized_shape, self.eps)
        return manual_rms_norm(input, self.normalized_shape, weight, self.eps)

    if self.no_persist_layer_norm:
        return FusedLayerNormAffineFunction.apply(input, weight, self.bias, self.normalized_shape, self.eps)
    else:
        output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

        # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
        # a populated '_base' field). This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        output = make_viewless_tensor(inp = output,
                                      requires_grad = input.requires_grad,
                                      keep_graph = True)

        return output
