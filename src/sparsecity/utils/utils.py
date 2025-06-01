from omegaconf import OmegaConf, DictConfig
import os
import torch


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
