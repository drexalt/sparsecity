import torch
import triton
import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from dataclasses import dataclass
from omegaconf import DictConfig
from hydra import compose, initialize
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sparsecity.models.splade_models.model_registry import get_splade_model
from sparsecity.models.splade_models.splade import SpladeModel
from sparsecity.data.dataset import MultipleNegativesCollateFn
from torch.utils.data import DataLoader


# Your Triton kernel remains unchanged
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
    pid_batch = tl.program_id(0)
    pid_v_chunk = tl.program_id(1)

    v_offset = pid_v_chunk * BLOCK_V
    v_indices = v_offset + tl.arange(0, BLOCK_V)
    batch_idx = pid_batch

    max_accumulator = tl.full((BLOCK_V,), -float("inf"), dtype=tl.float32)
    argmax_accumulator = tl.full((BLOCK_V,), -1, dtype=tl.int32)

    for s_offset in range(0, seq_len, BLOCK_S):
        s_indices = s_offset + tl.arange(0, BLOCK_S)
        mask_offsets = batch_idx * mask_batch_stride + s_indices * mask_seq_stride
        mask = tl.load(mask_ptr + mask_offsets, mask=s_indices < seq_len, other=0.0)

        logit_offsets = (
            (batch_idx * logit_batch_stride)
            + (s_indices[:, None] * logit_seq_stride)
            + (v_indices[None, :] * logit_vocab_stride)
        )
        logits = tl.load(
            logits_ptr + logit_offsets,
            mask=(s_indices[:, None] < seq_len) & (v_indices[None, :] < vocab_size),
            other=-float("inf"),
        )

        activated = (
            tl.extra.cuda.libdevice.log1p(tl.maximum(logits, 0.0)) * mask[:, None]
        )
        chunk_max = tl.max(activated, axis=0)
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


# Your SparseActivation class remains unchanged
class SparseActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, attention_mask):
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        values = torch.empty(batch_size, vocab_size, device=device, dtype=torch.float32)
        argmax_indices = torch.empty(
            batch_size, vocab_size, device=device, dtype=torch.int32
        )

        grid = lambda meta: (batch_size, triton.cdiv(vocab_size, meta["BLOCK_V"]))
        sparse_activation_kernel[grid](
            logits,
            attention_mask.float(),
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

        ctx.save_for_backward(logits, attention_mask, argmax_indices)
        return values

    @staticmethod
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
        mask_at_max = attention_mask[b_indices, s_indices].float()

        grad_logits_at_max = (
            grad_output
            * (1 / (1 + relu_logits_at_max))
            * (logits_at_max > 0).float()
            * mask_at_max
        )

        grad_logits[b_indices, s_indices, v_indices] = grad_logits_at_max

        return grad_logits, None


@dataclass
class TrainingConfig:
    seed: int
    data: DictConfig
    model: DictConfig
    batch_size: int
    num_negatives: int
    lambda_d: float
    lambda_q: float
    T_d: float
    T_q: float
    top_k: int
    epochs: int
    log_every: int
    optimizer: DictConfig
    checkpoint: DictConfig
    wandb: bool
    wandb_project: str
    use_distillation: bool
    evaluation: DictConfig


class MemoryEfficientSplade(nn.Module):
    def __init__(self, transformer_model: nn.Module):
        super().__init__()
        self.model = transformer_model

    def forward(self, input_ids, attention_mask, top_k=64):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        logits = outputs.logits
        values = SparseActivation.apply(logits, attention_mask)
        top_values, _ = torch.topk(values, k=top_k, dim=-1)
        threshold = top_values[..., -1, None]
        values = values * (values >= threshold)
        return values, logits


if __name__ == "__main__":
    # Dataset and configuration setup
    text_only = load_dataset(
        "json",
        data_files={
            "train": "/root/data/msmarco_triplets/msmarco-triplets-stella.jsonl.gz"
        },
        split="train",
        encoding="utf-8",
    )

    initialize(config_path="conf", version_base=None)
    cfg = compose(config_name="modernbert_base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainingConfig(**cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForMaskedLM.from_pretrained(cfg.model.name)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    base_model = SpladeModel(model).to(device)
    custom_kernel_model = MemoryEfficientSplade(model).to(device)

    dataloader = DataLoader(
        text_only,
        collate_fn=MultipleNegativesCollateFn(tokenizer, num_negatives=4),
        batch_size=4,
        shuffle=False,
        pin_memory=True,
    )

    # Get a batch
    batch = next(iter(dataloader))
    batch_input_ids, batch_attention_mask = batch[0].to(device), batch[1].to(device)

    # Ensure deterministic behavior
    torch.manual_seed(1776)
    custom_kernel_model.train()
    model.train()

    # Forward pass
    custom_values, custom_logits = custom_kernel_model(
        batch_input_ids, batch_attention_mask
    )
    torch.manual_seed(1776)
    base_values, base_logits = base_model(batch_input_ids, batch_attention_mask)

    # Verify forward pass
    print("Logits identical:", torch.equal(custom_logits, base_logits))
    print("Values identical:", torch.equal(custom_values, base_values))
    print("Values max diff:", torch.max(torch.abs(custom_values - base_values)))

    # TEST BACKWARD WITH ACTUAL LOGITS AND ATTENTION MASK
    # Use actual logits, detached and requiring gradients
    logits = custom_logits.detach().requires_grad_(True)
    attention_mask = batch_attention_mask

    # Custom forward and backward
    output_custom = SparseActivation.apply(logits, attention_mask)
    loss = output_custom.sum()
    loss.backward()
    grad_logits_custom = logits.grad.clone()

    # Reset gradients
    logits.grad.zero_()

    # PyTorch forward and backward
    activations = torch.log1p(torch.relu(logits)) * attention_mask.unsqueeze(-1).float()
    output_torch = torch.amax(activations, dim=1)
    loss_torch = output_torch.sum()
    loss_torch.backward()
    grad_logits_torch = logits.grad.clone()

    # Compare gradients
    print("Gradients identical:", torch.equal(grad_logits_custom, grad_logits_torch))
    print("Max diff:", torch.max(torch.abs(grad_logits_custom - grad_logits_torch)))
    print("Custom grad mean:", grad_logits_custom.mean().item())
    print("Torch grad mean:", grad_logits_torch.mean().item())
