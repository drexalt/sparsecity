from omegaconf import OmegaConf, DictConfig
import os
import torch
from heapq import heappush, heappop


def flatten_dict(d, parent_key="", sep="/"):
    """
    Flatten a nested dictionary, using separator to join keys.

    Args:
        d: Dictionary to flatten
        parent_key: Key of parent (for recursion)
        sep: Separator to use between nested keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, DictConfig):
            v = OmegaConf.to_container(v, resolve=True)
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dump_debug_bundle(
    batch,
    model,
    optimizer,
    step: int,
    use_distillation: bool,
    path: str = "debug_dumps",
) -> None:
    """
    Save a self-contained snapshot that can be re-loaded for post-mortem.

    Args
    ----
    batch            Tuple[List/Tensor] that your KD collate fn produced.
    model            The model whose parameters just saw the bad gradient.
    optimizer        Same optimiser instance.
    step             Global training step.
    use_distillation Whether `batch` contains the teacher-score tensor.
    path             Directory that will receive the *.pt file.
    """
    os.makedirs(path, exist_ok=True)

    if use_distillation:
        (q_ids, q_mask, d_ids, d_mask, t_scores) = batch
        batch_dict = {
            "query_input_ids": q_ids.cpu(),
            "query_attention_mask": q_mask.cpu(),
            "doc_input_ids": d_ids.cpu(),
            "doc_attention_mask": d_mask.cpu(),
            "teacher_scores": t_scores.cpu(),
        }
    else:
        (q_ids, q_mask, d_ids, d_mask) = batch
        batch_dict = {
            "query_input_ids": q_ids.cpu(),
            "query_attention_mask": q_mask.cpu(),
            "doc_input_ids": d_ids.cpu(),
            "doc_attention_mask": d_mask.cpu(),
        }

    torch.save(
        {
            "step": step,
            "batch": batch_dict,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
        },
        os.path.join(path, f"debug_step_{step}.pt"),
    )


# Schedulers


def compute_top_k(
    initial_top_k: int,
    final_top_k: int,
    global_step: int,
    warmup_steps: int,
) -> int:
    """
    Computes the top_k value for the current training step using a decreasing quadratic schedule.
    """
    if global_step >= warmup_steps:
        return final_top_k

    step_ratio = global_step / warmup_steps
    # Quadratic schedule: k(t) = (k_initial - k_final) * (1 - t)^2 + k_final
    current_k = (initial_top_k - final_top_k) * ((1 - step_ratio) ** 2) + final_top_k
    return int(round(current_k))


def compute_lambda_t(lambda_val: float, step_ratio: float) -> float:
    return min(lambda_val, lambda_val * (step_ratio**2))


def compute_lambda_exact(
    lambda_max: float, global_step: int, warmup_steps: int, min_lambda: float = 0.0
) -> float:
    if global_step >= warmup_steps:
        return lambda_max
    ratio = global_step / warmup_steps
    return (lambda_max - min_lambda) * (ratio**2) + min_lambda


# Checkpointing


def save_checkpoint(
    step: int,
    score: float,
    splade_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
) -> str:
    checkpoint = {
        "splade_model": splade_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "ndcg@10": score,
    }
    filepath = os.path.join(
        checkpoint_path, f"checkpoint_step_{step}_msmarco_mrr@10_{score:.4f}.pt"
    )
    torch.save(checkpoint, filepath)
    return filepath


def update_checkpoint_tracking(
    step: int,
    score: float,
    checkpoint_scores: list,
    max_checkpoints: int,
    splade_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
) -> list:
    # Create a new list to maintain purity
    updated_checkpoint_scores = checkpoint_scores.copy()

    if len(updated_checkpoint_scores) < max_checkpoints:
        filepath = save_checkpoint(
            step, score, splade_model, optimizer, checkpoint_path
        )
        heappush(updated_checkpoint_scores, (score, step, filepath))
    elif score > updated_checkpoint_scores[0][0]:  # Compare with lowest score
        # Remove lowest scoring checkpoint
        _, old_step, old_filepath = heappop(updated_checkpoint_scores)
        if os.path.exists(old_filepath):
            os.remove(old_filepath)
        # Save new checkpoint
        filepath = save_checkpoint(
            step, score, splade_model, optimizer, checkpoint_path
        )
        heappush(updated_checkpoint_scores, (score, step, filepath))

    return updated_checkpoint_scores
