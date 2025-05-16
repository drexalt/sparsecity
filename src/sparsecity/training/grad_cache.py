import torch
from typing import Sequence, Callable
from torch.utils.checkpoint import get_device_states, set_device_states


class RandContext:
    """Save & restore CPU + CUDA RNG state for one micro-batch."""

    def __init__(self, *tensors):
        self.cpu_state = torch.get_rng_state()
        self.gpu_devs, self.gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.gpu_devs, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.cpu_state)
        set_device_states(self.gpu_devs, self.gpu_states)

    def __exit__(self, et, ev, tb):
        self._fork.__exit__(et, ev, tb)
        self._fork = None


def gc_backward_and_zero_grad(
    loss: torch.Tensor,
    cached_tensors: Sequence[torch.Tensor],
    recompute_fn: Callable[[slice, RandContext], Sequence[torch.Tensor]],
    rng_states: list[RandContext],
    model: torch.nn.Module,
    mini_batch_size: int = 32,
):
    loss.backward()
    grads = [t.grad.detach().clone() for t in cached_tensors]
    for t in cached_tensors:
        t.grad = None

    for n, p in model.named_parameters():
        if n in ("log_t_ce", "log_t_kl"):
            continue  # keep their grads
        p.grad = None
    B = cached_tensors[0].size(0)
    num_expected_rng_states = (B + mini_batch_size - 1) // mini_batch_size
    if len(rng_states) != num_expected_rng_states:
        raise ValueError(
            f"Mismatch in RNG states. Expected {num_expected_rng_states}, got {len(rng_states)}"
        )

    for i, start in enumerate(range(0, B, mini_batch_size)):
        sl = slice(start, min(start + mini_batch_size, B))
        current_rng_context_from_pass1 = rng_states[i]

        with current_rng_context_from_pass1:  # Restore RNG state for this mini-batch
            # The second argument to recompute_fn might be optional if it doesn't need the context object itself
            replayed_outputs_mb = recompute_fn(sl, current_rng_context_from_pass1)

        current_mb_grads = [g_full[sl] for g_full in grads]
        torch.autograd.backward(
            tensors=replayed_outputs_mb,
            grad_tensors=current_mb_grads,
            retain_graph=False,
        )
