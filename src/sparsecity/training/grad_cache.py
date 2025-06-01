import torch
from typing import Sequence, Callable, List, Union, Any, Tuple, Dict
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch import nn, Tensor
from contextlib import nullcontext
from collections import UserDict
from itertools import repeat
import wandb

import logging

logger = logging.getLogger(__name__)


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
    # loss.backward()
    # saved_grads = [t.grad.detach().clone() for t in cached_tensors]
    # for t in cached_tensors:
    #     t.grad = None

    cached_grads = torch.autograd.grad(
        loss, cached_tensors, retain_graph=False, create_graph=False, allow_unused=False
    )
    for n, p in model.named_parameters():
        if n in ("log_t_ce", "log_t_kl"):
            continue  # keep their grads
        p.grad = None

    for t in cached_tensors:
        t.grad = None

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

        current_mb_grads = [g_full[sl].detach() for g_full in cached_grads]

        torch.autograd.backward(
            tensors=replayed_outputs_mb,
            grad_tensors=current_mb_grads,
            retain_graph=False,
        )

        # with torch.no_grad():
        #     # grab the grads that have just been accumulated on q_emb and d_emb
        #     true_grads = []
        #     for t, g_full in zip(cached_tensors, saved_grads):
        #         if t.grad is None:  # not used in this micro-batch
        #             # just use the injected grad - we know it’s the correct shape
        #             true_grads.append(g_full[sl].clone())
        #         else:
        #             true_grads.append(t.grad[sl].clone())
        #     # the grads you fed in come from the first backward

        #     diff = sum(
        #         ((a - f) ** 2).sum() for a, f in zip(true_grads, current_mb_grads)
        #     ).sqrt()
        #     if diff > 1e-6:
        #         print(
        #             f"[BUG] micro-batch {i}: fed grads != true replay grads  |Δ|={diff.item():.4e}"
        #         )
        #         break


class GradCache:
    """
    Gradient Cache class. Implements input chunking, first graph-less forward pass, Gradient Cache creation, second
    forward & backward gradient computation. Optimizer step is not included. Native torch automatic mixed precision is
    supported. User needs to handle gradient unscaling and scaler update after a gradeitn cache step.
    """

    def __init__(
        self,
        models: List[nn.Module],
        chunk_sizes: Union[int, List[int]],
        loss_fn: Callable[..., Tensor],
        split_input_fn: Callable[[Any, int], Any] = None,
        get_rep_fn: Callable[..., Tensor] = None,
        mixed_precision: str = "fp32",
        scaler: GradScaler = None,
        rep_grad_clip: float | None = None,
        clip_start_step: int | None = 0,
    ):
        """
        Initialize the Gradient Cache class instance.
        :param models: A list of all encoder models to be updated by the current cache.
        :param chunk_sizes: An integer indicating chunk size. Or a list of integers of chunk size for each model.
        :param loss_fn: A loss function that takes arbitrary numbers of representation tensors and
        arbitrary numbers of keyword arguments as input. It should not in any case modify the input tensors' relations
        in the autograd graph, which are later relied upon to create the gradient cache.
        :param split_input_fn: An optional function that split generic model input into chunks. If not provided, this
        class will try its best to split the inputs of supported types. See `split_inputs` function.
        :param get_rep_fn: An optional function that takes generic model output and return representation tensors. If
        not provided, the generic output is assumed to be the representation tensor.
        :param fp16: If True, run mixed precision training, which requires scaler to also be set.
        :param scaler: A GradScaler object for automatic mixed precision training.
        """
        self.models = models

        if isinstance(chunk_sizes, int):
            self.chunk_sizes = [chunk_sizes for _ in range(len(models))]
        else:
            self.chunk_sizes = chunk_sizes

        self.split_input_fn = split_input_fn
        self.get_rep_fn = get_rep_fn
        self.loss_fn = loss_fn

        self.mixed_precision = mixed_precision
        if mixed_precision == "fp16":
            assert scaler is not None, "fp16 requires a gradient scaler"
        self.scaler = scaler if mixed_precision == "fp16" else None
        self._get_input_tensors_strict = False
        self.step = 0
        self.clip_start_step = clip_start_step
        self.rep_grad_clip = rep_grad_clip

    def __call__(self, *args, **kwargs):
        """
        Call the cache_step function.
        :return: Current step loss.
        """
        return self.cache_step(*args, **kwargs)

    def split_inputs(self, model_input, chunk_size: int) -> List:
        """
        Split input into chunks. Will call user provided `split_input_fn` if specified. Otherwise,
        it can handle input types of tensor, list of tensors and dictionary of tensors.
        :param model_input: Generic model input.
        :param chunk_size:  Size of each chunk.
        :return: A list of chunked model input.
        """
        # delegate splitting to user provided function
        if self.split_input_fn is not None:
            return self.split_input_fn(model_input, chunk_size)

        if isinstance(model_input, (dict, UserDict)) and all(
            isinstance(x, Tensor) for x in model_input.values()
        ):
            keys = list(model_input.keys())
            chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
            return [
                dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))
            ]

        elif isinstance(model_input, list) and all(
            isinstance(x, Tensor) for x in model_input
        ):
            chunked_x = [t.split(chunk_size, dim=0) for t in model_input]
            return [list(s) for s in zip(*chunked_x)]

        elif isinstance(model_input, Tensor):
            return list(model_input.split(chunk_size, dim=0))

        elif isinstance(model_input, tuple) and list(map(type, model_input)) == [
            list,
            dict,
        ]:
            args_chunks = self.split_inputs(model_input[0], chunk_size)
            kwargs_chunks = self.split_inputs(model_input[1], chunk_size)
            return list(zip(args_chunks, kwargs_chunks))

        else:
            raise NotImplementedError(
                f"Model input split not implemented for type {type(model_input)}"
            )

    def get_input_tensors(self, model_input) -> List[Tensor]:
        """
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, Tensor):
            return [model_input]

        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])

        elif self._get_input_tensors_strict:
            raise NotImplementedError(
                f"get_input_tensors not implemented for type {type(model_input)}"
            )

        else:
            return []

    def model_call(self, model: nn.Module, model_input):
        """
        Literally call the model's __call__ method.
        :param model: model to be called
        :param model_input: input to the model call
        :return: model output
        """
        dtype = (
            torch.float16
            if self.mixed_precision == "fp16"
            else torch.bfloat16
            if self.mixed_precision == "bf16"
            else None
        )
        with (
            autocast("cuda", dtype=dtype)
            if self.mixed_precision != "none"
            else nullcontext()
        ):
            if isinstance(model_input, Tensor):
                return model(model_input)
            elif isinstance(model_input, list):
                return model(*model_input)
            elif isinstance(model_input, (dict, UserDict)):
                return model(**model_input)
            elif isinstance(model_input, tuple) and list(map(type, model_input)) == [
                list,
                dict,
            ]:
                model_args, model_kwargs = model_input
                return model(*model_args, **model_kwargs)
            else:
                raise NotImplementedError

    def get_reps(self, model_out) -> Tensor:
        """
        Return representation tensor from generic model output
        :param model_out: generic model output
        :return: a single tensor corresponding to the model representation output
        """
        if self.get_rep_fn is not None:
            return self.get_rep_fn(model_out)
        else:
            return model_out

    def compute_loss(self, *reps: Tensor, **loss_kwargs) -> Tensor:
        """
        Compute the loss based on the representation tensors. The tensors should be ordered same as the list of models
        registered in this GradCache class instance.
        :param reps: Representations for computing the loss.
        :param loss_kwargs: Keyword arguments input to the loss function.
        :return: the loss tensor.
        """
        loss = self.loss_fn(*reps, **loss_kwargs)
        return loss

    def forward_no_grad(
        self,
        model: nn.Module,
        model_inputs,
    ) -> Tuple[Tensor, List[RandContext]]:
        """
        The first forward pass without gradient computation.
        :param model: Encoder model.
        :param model_inputs: Model input already broken into chunks.
        :return: A tuple of a) representations and b) recorded random states.
        """
        rnd_states = []
        model_reps = []

        with torch.no_grad():
            for x in model_inputs:
                rnd_states.append(RandContext(*self.get_input_tensors(x)))
                y = self.model_call(model, x)
                model_reps.append(self.get_reps(y))

        # concatenate all sub-batch representations
        model_reps = torch.cat(model_reps, dim=0)
        return model_reps, rnd_states

    def build_cache(
        self, *reps: Tensor, **loss_kwargs
    ) -> Tuple[List[Tensor], Tensor, Dict[str, Tensor]]:
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor
        """
        reps = [r.detach().requires_grad_() for r in reps]
        dtype = (
            torch.float16
            if self.mixed_precision == "fp16"
            else torch.bfloat16
            if self.mixed_precision == "bf16"
            else None
        )
        with (
            autocast("cuda", dtype=dtype)
            if self.mixed_precision != "none"
            else nullcontext()
        ):
            loss_out = self.compute_loss(
                *reps, **loss_kwargs
            )  # Returns tuple (total_loss, {parts})

        if isinstance(loss_out, tuple):
            loss, parts = loss_out
        else:
            loss, parts = loss_out, {}

        if self.mixed_precision == "fp16":
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_norms = []
        grad_tensors = []
        for r_tensor in reps:
            g = r_tensor.grad
            if g is not None:
                grad_norms.append(g.norm().item())
                grad_tensors.append(g)

        if grad_norms:
            grad_norms_tensor = torch.tensor(
                grad_norms, device=loss.device
            )  # Move to device for torch functions
            median_grad_norm = torch.median(grad_norms_tensor).item()
            mean_grad_norm = torch.mean(grad_norms_tensor).item()
            max_grad_norm = torch.max(grad_norms_tensor).item()
            logger.info(
                f"Step {self.step}: Median norm of grad for reps: {median_grad_norm}"
            )
            logger.info(
                f"Step {self.step}: Mean norm of grad for reps: {mean_grad_norm}"
            )
            logger.info(f"Step {self.step}: Max norm of grad for reps: {max_grad_norm}")
            if wandb.run and self.step % 20 == 0:
                wandb.log(
                    {
                        "grad_norm/median": median_grad_norm,
                        "grad_norm/mean": mean_grad_norm,
                        "grad_norm/max": max_grad_norm,
                    }
                )

            if self.rep_grad_clip is not None and self.step >= self.clip_start_step:
                torch.nn.utils.clip_grad_norm_(grad_tensors, self.rep_grad_clip)

            self.step += 1

        cache = [r.grad for r in reps]

        return cache, loss.detach(), {k: v.detach() for k, v in parts.items()}

    def forward_backward(
        self,
        model: nn.Module,
        model_inputs,
        cached_gradients: List[Tensor],
        random_states: List[RandContext],
        no_sync_except_last: bool = False,
    ):
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        if no_sync_except_last:
            sync_contexts = [model.no_sync for _ in range(len(model_inputs) - 1)] + [
                nullcontext
            ]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]

        for x, state, gradient, sync_context in zip(
            model_inputs, random_states, cached_gradients, sync_contexts
        ):
            with sync_context():
                with state:
                    y = self.model_call(model, x)
                reps = self.get_reps(y)

                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                surrogate.backward()

    def cache_step(
        self, *model_inputs, no_sync_except_last: bool = False, **loss_kwargs
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Run a cached step to compute gradient over the inputs.
        :param model_inputs: Input to each encoder model. Should be in similar order as the class's model.
        :param no_sync_except_last: If True, under distributed setup, for each model, only trigger gradient reduction
        across processes for the last sub-batch's forward-backward pass.
        :param loss_kwargs: Additional keyword arguments to the loss function.
        :return: The current's loss.
        """
        all_reps = []
        all_rnd_states = []

        if no_sync_except_last:
            assert all(
                map(
                    lambda m: isinstance(m, nn.parallel.DistributedDataParallel),
                    self.models,
                )
            ), (
                "Some of models are not wrapped in DistributedDataParallel. Make sure you are running DDP with "
                "proper initializations."
            )

        model_inputs = [
            self.split_inputs(x, chunk_size)
            for x, chunk_size in zip(model_inputs, self.chunk_sizes)
        ]

        for model, x in zip(self.models, model_inputs):
            model_reps, rnd_states = self.forward_no_grad(model, x)
            all_reps.append(model_reps)
            all_rnd_states.append(rnd_states)

        cache, loss, parts = self.build_cache(*all_reps, **loss_kwargs)
        cache = [c.split(chunk_size) for c, chunk_size in zip(cache, self.chunk_sizes)]

        for model, x, model_cache, rnd_states in zip(
            self.models, model_inputs, cache, all_rnd_states
        ):
            self.forward_backward(
                model,
                x,
                model_cache,
                rnd_states,
                no_sync_except_last=no_sync_except_last,
            )

        return loss, parts
