import torch
import deeplink_ext.cpp_extensions as ext

class FusedRMSNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps, memory_efficient = False):

        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.memory_efficient = memory_efficient
        input_ = input.contiguous()
        weight_ = weight.contiguous()

        output = torch.empty_like(input_)
        inv_rms_shape = list(input_.shape[:-1]) + [1]
        invvar = torch.empty(
            inv_rms_shape, dtype=torch.float32, device=input.device
        )
        ext.rms_norm(output, invvar, input_, ctx.normalized_shape, weight_, None, ctx.eps)
        if ctx.memory_efficient:
            ctx.save_for_backward(output, weight_, invvar)
        else:
            ctx.save_for_backward(input_, weight_, invvar)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input_or_output, weight_, invvar = ctx.saved_tensors

        grad_input = torch.empty_like(input_or_output)
        grad_weight = torch.empty_like(weight_)
        ext.rms_norm_backward(
            grad_input,
            grad_weight,
            None,
            grad_output.contiguous(),
            input_or_output,
            weight_,
            None,
            invvar,
            ctx.normalized_shape,
            ctx.eps,
        )

        return grad_input, grad_weight, None, None, None
