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
    model: torch.nn.Module,
    mini_batch_size: int = 32,
):
    loss.backward()
    grads = [t.grad.detach().clone() for t in cached_tensors]
    for t in cached_tensors:
        t.grad = None
    model.zero_grad(set_to_none=True)

    B = cached_tensors[0].size(0)
    rng_states: list[RandContext] = []

    for start in range(0, B, mini_batch_size):
        sl = slice(start, min(start + mini_batch_size, B))
        rng_states.append(RandContext(cached_tensors[0][sl]))

    for i, start in enumerate(range(0, B, mini_batch_size)):
        sl = slice(start, min(start + mini_batch_size, B))
        with rng_states[i]:
            replayed = recompute_fn(sl, rng_states[i])
        torch.autograd.backward(
            replayed,
            [g[sl] for g in grads],
            retain_graph=False,
        )
