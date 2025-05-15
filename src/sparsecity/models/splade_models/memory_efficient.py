import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl


# Triton kernel
@triton.jit
def sparse_activation_kernel(
    logits_ptr,
    mask_ptr,
    output_ptr,
    indices_ptr,
    batch_size,
    seq_len,
    vocab_size,
    logit_batch_stride,
    logit_seq_stride,
    logit_vocab_stride,
    mask_batch_stride,
    mask_seq_stride,
    output_batch_stride,
    output_vocab_stride,
    indices_batch_stride,
    indices_vocab_stride,
    BLOCK_V: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    compute_dtype = (
        tl.float32 if output_ptr.dtype.element_ty == tl.float32 else tl.bfloat16
    )
    pid_batch = tl.program_id(0)
    pid_v_chunk = tl.program_id(1)

    v_offset = pid_v_chunk * BLOCK_V
    v_indices = v_offset + tl.arange(0, BLOCK_V)
    batch_idx = pid_batch

    max_accumulator = tl.full((BLOCK_V,), -float("inf"), dtype=compute_dtype)
    argmax_accumulator = tl.full((BLOCK_V,), -1, dtype=tl.int32)

    for s_offset in range(0, seq_len, BLOCK_S):
        s_indices = s_offset + tl.arange(0, BLOCK_S)
        mask_offsets = batch_idx * mask_batch_stride + s_indices * mask_seq_stride
        mask = tl.load(mask_ptr + mask_offsets, mask=s_indices < seq_len, other=0.0).to(
            compute_dtype
        )

        logit_offsets = (
            (batch_idx * logit_batch_stride)
            + (s_indices[:, None] * logit_seq_stride)
            + (v_indices[None, :] * logit_vocab_stride)
        )
        logits = tl.load(
            logits_ptr + logit_offsets,
            mask=(s_indices[:, None] < seq_len) & (v_indices[None, :] < vocab_size),
            other=-float("inf"),
        ).to(compute_dtype)

        activated = tl.math.log(1 + tl.maximum(logits, 0.0)) * mask[:, None]
        chunk_max = tl.max(activated, axis=0).to(compute_dtype)
        chunk_argmax = tl.argmax(activated, axis=0)
        update_mask = chunk_max > max_accumulator
        max_accumulator = tl.where(update_mask, chunk_max, max_accumulator)
        argmax_accumulator = tl.where(
            update_mask, s_offset + chunk_argmax, argmax_accumulator
        )

    output_offsets = batch_idx * output_batch_stride + v_indices * output_vocab_stride
    indices_offsets = (
        batch_idx * indices_batch_stride + v_indices * indices_vocab_stride
    )
    tl.store(output_ptr + output_offsets, max_accumulator, mask=v_indices < vocab_size)
    tl.store(
        indices_ptr + indices_offsets, argmax_accumulator, mask=v_indices < vocab_size
    )


# Define the forward pass as a Triton operation
@triton_op("custom::sparse_activation", mutates_args={})
@torch.compiler.disable()
def sparse_activation(
    logits: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device

    values = torch.empty(batch_size, vocab_size, device=device, dtype=logits.dtype)
    argmax_indices = torch.empty(
        batch_size, vocab_size, device=device, dtype=torch.int32
    )

    grid = lambda meta: (batch_size, triton.cdiv(vocab_size, meta["BLOCK_V"]))
    wrap_triton(sparse_activation_kernel)[grid](
        logits,
        attention_mask.to(logits.dtype),
        values,
        argmax_indices,
        batch_size,
        seq_len,
        vocab_size,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        attention_mask.stride(0),
        attention_mask.stride(1),
        values.stride(0),
        values.stride(1),
        argmax_indices.stride(0),
        argmax_indices.stride(1),
        BLOCK_V=1024,
        BLOCK_S=128,
    )

    sparse_activation._argmax_indices = argmax_indices  # Temporary storage
    return values


# Define setup_context and backward
def setup_context(ctx, inputs, output):
    logits, attention_mask = inputs
    argmax_indices = sparse_activation._argmax_indices
    ctx.save_for_backward(logits, attention_mask, argmax_indices)


def backward(ctx, grad_output):
    logits, attention_mask, argmax_indices = ctx.saved_tensors
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device

    grad_logits = torch.zeros_like(logits)
    b_indices = (
        torch.arange(batch_size, device=device).view(-1, 1).expand(-1, vocab_size)
    )
    v_indices = (
        torch.arange(vocab_size, device=device).view(1, -1).expand(batch_size, -1)
    )
    s_indices = argmax_indices

    logits_at_max = logits[b_indices, s_indices, v_indices]
    relu_logits_at_max = F.relu(logits_at_max)
    mask_at_max = attention_mask[b_indices, s_indices].to(logits.dtype)

    grad_logits_at_max = (
        grad_output
        * (1 / (1 + relu_logits_at_max))
        * (logits_at_max > 0).to(logits.dtype)
        * mask_at_max
    )

    grad_logits[b_indices, s_indices, v_indices] = grad_logits_at_max.to(logits.dtype)
    return grad_logits, None


sparse_activation.register_autograd(backward, setup_context=setup_context)


class MemoryEfficientSplade(nn.Module):
    """
    Memory-efficient SPLADE implementation using Triton kernels.

    This implementation provides the same functionality as SpladeModel but with improved
    memory efficiency and potentially better performance on GPUs.
    """

    def __init__(self, transformer_model: nn.Module):
        super().__init__()
        self.model = transformer_model

    def forward(self, input_ids, attention_mask, top_k=64):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        values = sparse_activation(logits, attention_mask)

        # Apply top-k filtering to ensure sparsity
        top_values, _ = torch.topk(values, k=top_k, dim=-1)
        threshold = top_values[..., -1, None]
        values = values * (values >= threshold)

        return values
