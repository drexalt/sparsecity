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
from sparsecity.models.splade_models.splade import SpladeModel_NoTopK
from sparsecity.data.dataset import KDProcessingCollateFn
from torch.utils.data import DataLoader
from torch.library import triton_op, wrap_triton
from sparsecity.data.dataset import KDProcessing


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_V": 256, "BLOCK_S": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_V": 256, "BLOCK_S": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_V": 512, "BLOCK_S": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_V": 512, "BLOCK_S": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_V": 512, "BLOCK_S": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_V": 1024, "BLOCK_S": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_V": 1024, "BLOCK_S": 128}, num_warps=8, num_stages=2),
]


# Triton kernel
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["vocab_size", "seq_len"])
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


@dataclass
class TrainingConfig:
    seed: int
    data: DictConfig
    model: DictConfig
    sparse_embed: bool
    custom_kernel: bool
    use_grad_cache: bool
    bf16: bool
    accum_steps: int
    batch_size: int
    mini_batch: int
    num_negatives: int
    sample_size: int  # Number of negatives to sample from total num_negatives
    n_ways: int  # How many negatives to throw into InfoNCE loss
    proximity_threshold: float
    mse_weight: float
    kl_weight: float
    max_length: int
    lambda_d: float
    lambda_q: float
    T_d: float
    T_q: float
    top_k: int
    schedule_top_k: bool
    initial_top_k: int
    top_k_warmup_steps: int
    epochs: int
    init_ce_temp: float
    init_kl_temp: float
    log_every: int
    optimizer: DictConfig
    checkpoint: DictConfig
    wandb: bool
    wandb_project: str
    use_distillation: bool
    evaluation: DictConfig


class MemoryEfficientSplade_noTopK(nn.Module):
    """
    Memory-efficient SPLADE implementation using Triton kernels.

    This implementation provides the same functionality as SpladeModel but with improved
    memory efficiency and potentially better performance on GPUs.
    """

    def __init__(self, transformer_model: nn.Module):
        super().__init__()
        self.model = transformer_model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        values = sparse_activation(logits, attention_mask)

        return values, logits


if __name__ == "__main__":
    # Dataset and configuration setup

    train_dataset = load_dataset(
        "lightonai/ms-marco-en-bge-gemma",
        "train",
        split="train",
    )

    queries = load_dataset(
        "lightonai/ms-marco-en-bge-gemma",
        "queries",
        split="train",
    )

    documents = load_dataset(
        "lightonai/ms-marco-en-bge-gemma",
        "documents",
        split="train",
    )

    train_dataset.set_transform(
        KDProcessing(queries=queries, documents=documents).transform
    )
    initialize(config_path="conf", version_base=None)
    cfg = compose(config_name="cocondenser_base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainingConfig(**cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForMaskedLM.from_pretrained(cfg.model.name)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    base_model = SpladeModel_NoTopK(model).to(device)
    custom_kernel_model = MemoryEfficientSplade_noTopK(model).to(device)

    dataloader = DataLoader(
        train_dataset,
        collate_fn=KDProcessingCollateFn(
            tokenizer,
            num_negatives=cfg.num_negatives,
            sample_size=cfg.sample_size,
            proximity_threshold=cfg.proximity_threshold,
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
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
    output_custom = sparse_activation(logits, attention_mask)
    loss = output_custom.sum()
    loss.backward()
    grad_logits_custom = logits.grad.clone()

    # Reset gradients
    logits.grad.zero_()

    # PyTorch forward and backward
    activations = torch.log1p(torch.relu(logits)) * attention_mask.unsqueeze(-1).to(
        logits.dtype
    )
    output_torch = torch.amax(activations, dim=1)
    loss_torch = output_torch.sum()
    loss_torch.backward()
    grad_logits_torch = logits.grad.clone()

    # Compare gradients
    print("Gradients identical:", torch.equal(grad_logits_custom, grad_logits_torch))
    print("Max diff:", torch.max(torch.abs(grad_logits_custom - grad_logits_torch)))
    print("Custom grad mean:", grad_logits_custom.mean().item())
    print("Torch grad mean:", grad_logits_torch.mean().item())

    # Test kernel values with dot product
    print(f"Custom values shape: {custom_values.shape}")
    scores = custom_values[0] @ custom_values[1:].T

    # -------------------------
    # Speed benchmarks (forward + backward)
    # -------------------------
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Reuse the real logits/mask you already have on device
    bench_logits = custom_logits.detach()  # same for both paths
    bench_mask = batch_attention_mask

    @torch.compile()
    def _torch_activation(x, mask):
        # Baseline path: log1p(relu(.)) * mask, then max over sequence
        act = torch.log1p(torch.relu(x)) * mask.unsqueeze(-1).float()
        return torch.amax(act, dim=1)

    def _triton_activation(x, mask):
        return sparse_activation(x, mask)

    def _time_forward(fn, x, mask, warmup=20, iters=200):
        with torch.inference_mode():
            # warmup
            for _ in range(warmup):
                _ = fn(x, mask)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # timed
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                _ = fn(x, mask)
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
        return total_ms / iters

    def _time_backward(fn, x, mask, warmup=10, iters=100):
        # We measure fwd+bwd of a simple sum loss of the activation output
        # to keep the graphs comparable.
        # Use a persistent tensor to avoid clone cost in-loop.
        x_req = x.clone().detach().requires_grad_(True)

        # warmup
        for _ in range(warmup):
            x_req.grad = None
            y = fn(x_req, mask)
            loss = y.sum()
            loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # timed
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            x_req.grad = None
            y = fn(x_req, mask)
            loss = y.sum()
            loss.backward()
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        return total_ms / iters

    # Trigger JIT/compilation explicitly first so the benchmark is clean
    _ = _triton_activation(bench_logits, bench_mask)
    _ = _torch_activation(bench_logits, bench_mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Forward (activation-only)
    triton_fwd_ms = _time_forward(_triton_activation, bench_logits, bench_mask)
    torch_fwd_ms = _time_forward(_torch_activation, bench_logits, bench_mask)

    # Fwd + Bwd
    triton_bwd_ms = _time_backward(_triton_activation, bench_logits, bench_mask)
    torch_bwd_ms = _time_backward(_torch_activation, bench_logits, bench_mask)

    print("\n=== Activation microbenchmarks (avg ms/iter) ===")
    print(f"Triton forward: {triton_fwd_ms:.4f} ms")
    print(f"Torch  forward: {torch_fwd_ms:.4f} ms")
    if triton_fwd_ms > 0:
        print(f"Forward speedup (Torch/Triton): {torch_fwd_ms / triton_fwd_ms:.2f}x")

    print(f"\nTriton fwd+bwd: {triton_bwd_ms:.4f} ms")
    print(f"Torch  fwd+bwd: {torch_bwd_ms:.4f} ms")
    if triton_bwd_ms > 0:
        print(f"Fwd+Bwd speedup (Torch/Triton): {torch_bwd_ms / triton_bwd_ms:.2f}x")

    # -------------------------
    # Peak memory benchmarks (GPU)
    # -------------------------
    def _bytes_to_mib(x):
        return x / (1024**2)

    def _reset_cuda_peaks():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def _report_peaks(tag, elapsed_ms_per_iter):
        alloc = torch.cuda.max_memory_allocated()
        reserv = torch.cuda.max_memory_reserved()
        print(f"\n[{tag}] avg time: {elapsed_ms_per_iter:.4f} ms/iter")
        print(f"[{tag}] peak allocated: {_bytes_to_mib(alloc):.2f} MiB")
        print(f"[{tag}] peak reserved : {_bytes_to_mib(reserv):.2f} MiB")

    def measure_peak_forward(fn, x, mask, warmup=20, iters=100):
        # warmup (compile + kernel autotune)
        with torch.inference_mode():
            for _ in range(warmup):
                _ = fn(x, mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # reset and measure
        _reset_cuda_peaks()
        with torch.inference_mode():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                _ = fn(x, mask)
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
        return total_ms / iters

    def measure_peak_fwd_bwd(fn, x, mask, warmup=10, iters=50):
        # Use a fresh leaf tensor with grad to avoid graph accumulation
        x_req = x.clone().detach().requires_grad_(True)

        # warmup
        for _ in range(warmup):
            x_req.grad = None
            y = fn(x_req, mask)
            (y.sum()).backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # reset and measure
        _reset_cuda_peaks()
        x_req = x.clone().detach().requires_grad_(True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            x_req.grad = None
            y = fn(x_req, mask)
            (y.sum()).backward()
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        return total_ms / iters

    # Make sure the activation fns from the speed test exist:
    # _triton_activation(x, mask) and _torch_activation(x, mask)

    if torch.cuda.is_available():
        # Forward-only peak memory
        triton_fwd_ms = measure_peak_forward(
            _triton_activation, bench_logits, bench_mask
        )
        _report_peaks("Triton forward", triton_fwd_ms)

        torch_fwd_ms = measure_peak_forward(_torch_activation, bench_logits, bench_mask)
        _report_peaks("Torch forward", torch_fwd_ms)

        # Forward+Backward peak memory
        triton_fb_ms = measure_peak_fwd_bwd(
            _triton_activation, bench_logits, bench_mask
        )
        _report_peaks("Triton fwd+bwd", triton_fb_ms)

        torch_fb_ms = measure_peak_fwd_bwd(_torch_activation, bench_logits, bench_mask)
        _report_peaks("Torch fwd+bwd", torch_fb_ms)
    else:
        print("CUDA not available; GPU peak memory benchmarks skipped.")
