# from sparsecity.training.trainer import train_step
from sparsecity.training.trainer import (
    train_step_kldiv_ibn,
    train_step_mse,
    train_step_kldiv_gradcache,
    train_step_kldiv_NO_GC,
)
from datetime import datetime
from collections import deque

# from sparsecity.training.sparse_trainer import train_step
from sparsecity.data.dataset import (
    MultipleNegativesCollateFn,
    MultipleNegativesDistilCollateFn,
    KDProcessingCollateFn,
)
from sparsecity.models.splade_models.model_registry import get_splade_model
from sparsecity.training.grad_cache import GradCache
from sparsecity.training.losses import (
    contrastive_kd_loss,
    contrastive_kd_loss_with_hard_negatives,
)
from sparsecity.utils.utils import flatten_dict, dump_debug_bundle
from sparsecity.evaluation.validate import validate_model
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.similarity_functions import dot_score
from transformers import AutoTokenizer, AutoConfig
import os
import torch
from torch.utils.data import DataLoader
import wandb
from datasets import load_dataset
from dataclasses import dataclass
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import dataclasses
from schedulefree import AdamWScheduleFree
from sparsecity.data.dataset import KDProcessing
from transformers import get_linear_schedule_with_warmup

from heapq import heappush, heappop
import logging

torch.set_float32_matmul_precision("high")
torch._dynamo.reset()


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
    max_length: int
    lambda_d: float
    lambda_q: float
    T_d: float
    T_q: float
    T_d_start: int
    T_q_start: int
    top_k: int
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


logger = logging.getLogger(__name__)


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


def compute_lambda_t(lambda_val: float, step_ratio: float) -> float:
    return min(lambda_val, lambda_val * (step_ratio**2))


def compute_lambda_exact(
    lambda_max: float, global_step: int, warmup_steps: int, min_lambda: float = 0.0
) -> float:
    if global_step >= warmup_steps:
        return lambda_max
    ratio = global_step / warmup_steps
    return (lambda_max - min_lambda) * (ratio**2) + min_lambda


def train_model(splade_model, tokenizer, cfg, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to device
    splade_model = splade_model.to(device)

    evaluator = NanoBEIREvaluator(
        dataset_names=cfg.evaluation.datasets,
        score_functions={"dot": dot_score},
        batch_size=cfg.evaluation.batch_size,
        show_progress_bar=True,
    )

    # Create optimizer and scheduler

    # Separate learning rate for temperatures
    if cfg.init_ce_temp is not None and cfg.init_kl_temp is not None:
        temp_params = [splade_model.log_t_ce, splade_model.log_t_kl]
        other_params = [
            p
            for n, p in splade_model.named_parameters()
            if n not in {"log_t_ce", "log_t_kl"}
        ]

        optim_param_groups = [
            {"params": temp_params, "lr": cfg.optimizer.learning_rate},
            {"params": other_params, "lr": cfg.optimizer.learning_rate},
        ]

    warmup_steps = cfg.optimizer.warmup_steps

    def lambda_lr(step):
        if warmup_steps == 0:
            return 1.0
        return min(1, step / warmup_steps)

    optimizer = torch.optim.AdamW(
        optim_param_groups
        if cfg.init_ce_temp is not None
        else splade_model.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda_lr,
    )

    # optimizer = AdamWScheduleFree(
    #     optim_param_groups
    #     if cfg.init_ce_temp is not None
    #     else splade_model.parameters(),
    #     warmup_steps=cfg.optimizer.warmup_steps,
    #     weight_decay=cfg.optimizer.weight_decay,
    #     betas=(0.98, 0.999),
    # )

    if cfg.max_length is not None:
        tokenizer.model_max_length = cfg.max_length
    if cfg.use_distillation:
        dataloader = DataLoader(
            dataset,
            collate_fn=KDProcessingCollateFn(
                tokenizer,
                num_negatives=cfg.num_negatives,
                sample_size=cfg.sample_size,
                proximity_threshold=cfg.proximity_threshold,
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,  # Add multiple workers for better data loading
            persistent_workers=True,  # Keep workers alive between iterations
            prefetch_factor=2,
            drop_last=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            collate_fn=MultipleNegativesCollateFn(
                tokenizer, num_negatives=cfg.num_negatives
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,  # Add multiple workers for better data loading
            persistent_workers=True,  # Keep workers alive between iterations
            prefetch_factor=2,
            drop_last=True,
        )

    # Initialize wandb if enabled
    if cfg.wandb:
        wandb.init(
            project=cfg.wandb_project,
            config={
                "optimizer": optimizer.__class__.__name__,
                **(flatten_dict(dataclasses.asdict(cfg))),
            },
            config_exclude_keys=[
                "data",
                "seed",
                "checkpoint/checkpoint_path",
                "checkpoint/max_to_keep",
                "checkpoint/save_interval_steps",
                "evaluation/eval_every_steps",
                "evaluation/datasets",
                "evaluation/batch_size",
                "wandb",
                "wandb_project",
                "log_every",
            ],
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_directory = os.path.join(
        hydra.utils.to_absolute_path(cfg.checkpoint.checkpoint_path), timestamp
    )
    os.makedirs(checkpoint_directory, exist_ok=True)
    checkpoint_scores = []

    def optimized_step():
        optimizer.step()
        # scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    def maybe_optim_step(
        step_in_epoch: int,
        accum_steps: int,
        dataloader_len: int,
        safe_grad_window: deque,
    ) -> tuple[float | None, bool, bool]:
        """Perform optimizer.step() every `accum_steps` micro-batches."""

        is_last_mb = step_in_epoch + 1 == dataloader_len
        should_step = (step_in_epoch + 1) % accum_steps == 0 or is_last_mb
        if not should_step:
            return None, False, False

        # --- gradient clipping / explosion check happens **here** --------------
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            splade_model.parameters(), float("inf")
        )
        grad_norm_val = total_grad_norm.item()

        exploded = not torch.isfinite(total_grad_norm) or (
            len(safe_grad_window) == safe_grad_window.maxlen
            and grad_norm_val > 10 * max(safe_grad_window)
        )
        safe_grad_window.append(grad_norm_val)

        if exploded:
            optimizer.zero_grad(set_to_none=True)
            return grad_norm_val, True, False

        optimizer.step()
        scheduler.step()  # uncomment if you use one
        optimizer.zero_grad(set_to_none=True)

        return grad_norm_val, False, True

    global_step = 0
    accum_steps = cfg.accum_steps
    LOG_EVERY_MICRO = cfg.log_every * accum_steps
    EVAL_EVERY_MICRO = cfg.evaluation.eval_every_steps * accum_steps

    # Grad Cache
    gc = GradCache(
        models=[splade_model, splade_model],
        chunk_sizes=cfg.mini_batch,
        loss_fn=contrastive_kd_loss_with_hard_negatives,
        mixed_precision="bf16" if cfg.bf16 else "fp32",
        rep_grad_clip=cfg.optimizer.rep_grad_clip,
        clip_start_step=cfg.optimizer.grad_clip_warmup_steps,
    )

    safe_grad_window = deque(maxlen=200)

    # Training loop
    for epoch in range(cfg.epochs):
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            global_step += 1
            if cfg.use_distillation:
                query_ids, query_mask, doc_ids, doc_mask, teacher_scores = (
                    t.to(device) for t in batch
                )
            else:
                query_ids, query_mask, doc_ids, doc_mask = (t.to(device) for t in batch)

            lambda_t_d = compute_lambda_exact(
                cfg.lambda_d, global_step // accum_steps, cfg.T_d
            )
            lambda_t_q = compute_lambda_exact(
                cfg.lambda_q, global_step // accum_steps, cfg.T_q
            )

            # optimizer.train()

            mse_weight = torch.tensor(0.05, device=device)
            temperature_ce = torch.tensor(1.0, device=device)
            temperature_kl = torch.tensor(1.0, device=device)
            loss_scale = torch.tensor(1.0 / accum_steps, device=device)

            train_kwargs = dict(
                model=splade_model,
                query_input_ids=query_ids,
                query_attention_mask=query_mask,
                doc_input_ids=doc_ids,
                doc_attention_mask=doc_mask,
                lambda_t_d=torch.tensor(lambda_t_d, device=device),
                lambda_t_q=torch.tensor(lambda_t_q, device=device),
                temperature_ce=temperature_ce,
                temperature_kl=temperature_kl,
                n_ways=cfg.n_ways,
                teacher_scores=teacher_scores if cfg.use_distillation else None,
                mse_weight=mse_weight,
            )

            rep_grad_clip = (
                torch.tensor(cfg.optimizer.rep_grad_clip, device=device)
                if cfg.optimizer.rep_grad_clip
                else None
            )

            if cfg.use_grad_cache:
                metrics = train_step_kldiv_gradcache(
                    gc,
                    **train_kwargs,
                )
            else:
                metrics = train_step_kldiv_NO_GC(
                    **train_kwargs,
                    loss_scale=loss_scale,
                    rep_grad_clip=rep_grad_clip,
                    step=global_step,
                    clip_start_step=cfg.optimizer.grad_clip_warmup_steps,
                    bf16=cfg.bf16,
                )

            grad_norm_val, exploded, stepped = maybe_optim_step(
                step_in_epoch=step,
                accum_steps=accum_steps,
                dataloader_len=len(dataloader),
                safe_grad_window=safe_grad_window,
            )

            if exploded:
                dump_debug_bundle(
                    batch,
                    splade_model,
                    optimizer,
                    global_step,
                    use_distillation=cfg.use_distillation,
                    path=checkpoint_directory,
                )
                logger.error(
                    f"Gradient explosion at step {global_step}. Skipping batch. Batch logged to {checkpoint_directory}/debug_step_{global_step}.pt"
                )
                continue

            log_metrics = {
                "loss/total_loss": metrics["total_loss"].item(),
                "loss/triplet_loss": metrics["triplet_loss"].item(),
                "loss/kl_loss": metrics["kl_loss"].item(),
                "loss/flops": metrics["flops_loss"].item(),
                "loss/anti_zero": metrics["anti_zero_loss"].item(),
                "loss/mse": metrics["mse_loss"].item(),
                "metrics/query_min_non_zero": metrics["query_min_non_zero"].item(),
                "metrics/doc_min_non_zero": metrics["doc_min_non_zero"].item(),
                "metrics/avg_query_non_zero_count": metrics["avg_query_non_zero_count"],
                "metrics/avg_doc_non_zero_count": metrics["avg_doc_non_zero_count"],
                "metrics/query_median_non_zero": metrics[
                    "query_median_non_zero"
                ].item(),
                "metrics/doc_median_non_zero": metrics["doc_median_non_zero"].item(),
                "metrics/total_grad_norm": grad_norm_val,
            }

            # Non-Grad-Cache case
            if metrics["q_grad_norm"] is not None:
                log_metrics["metrics/q_grad_norm"] = metrics["q_grad_norm"]
            if metrics["d_grad_norm"] is not None:
                log_metrics["metrics/d_grad_norm"] = metrics["d_grad_norm"]

            if cfg.wandb and global_step % LOG_EVERY_MICRO == 0:
                wandb.log({**log_metrics}, step=global_step // accum_steps)

            if (global_step + 1) % EVAL_EVERY_MICRO == 0 or global_step == 50:
                splade_model.eval()
                # optimizer.eval()
                val_results = validate_model(
                    evaluator,
                    splade_model,
                    tokenizer,
                    device,
                    sparse_embed=cfg.sparse_embed,
                    top_k=cfg.top_k,
                )
                splade_model.train()

                if cfg.wandb:
                    # Flatten results for wandb logging
                    wandb.log(
                        {
                            "validation/ndcg@10": val_results["ndcg@10"],
                            "validation/mrr@10": val_results["mrr@10"],
                            "validation/map@100": val_results["map@100"],
                            **{
                                f"validation/supplementary/{k}": v
                                for k, v in val_results.items()
                                if k not in ["ndcg@10", "mrr@10", "map@100"]
                            },
                        },
                        step=global_step // accum_steps,
                    )

                # Save checkpoint
                checkpoint_scores = update_checkpoint_tracking(
                    step=global_step,
                    score=val_results["msmarco_mrr@10"],
                    checkpoint_scores=checkpoint_scores,
                    max_checkpoints=cfg.checkpoint.max_to_keep,
                    splade_model=splade_model,
                    optimizer=optimizer,
                    checkpoint_path=checkpoint_directory,
                )


@hydra.main(config_path="conf", config_name="cocondenser_base", version_base=None)
def main(cfg: DictConfig):
    cfg = TrainingConfig(**cfg)
    config = AutoConfig.from_pretrained(cfg.model.name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = get_splade_model(
        cfg.model.name,
        config=config,
        custom_kernel=cfg.custom_kernel,
        sparse_embed=cfg.sparse_embed,
        checkpoint_path=cfg.model.checkpoint_path
        if cfg.model.checkpoint_path
        else None,
        top_k=cfg.top_k if cfg.top_k else None,
        init_ce_temp=cfg.init_ce_temp,
        init_kl_temp=cfg.init_kl_temp,
    )
    # train_dataset = load_dataset(
    #     cfg.data.name,
    #     split=cfg.data.split,
    # )
    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

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
    # set_torch()
    train_model(
        splade_model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        dataset=train_dataset,
    )


if __name__ == "__main__":
    main()
