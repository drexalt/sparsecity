# from sparsecity.training.trainer import train_step
from sparsecity.training.trainer import train_step_kldiv_ibn, train_step_mse
from datetime import datetime

# from sparsecity.training.sparse_trainer import train_step
from sparsecity.data.dataset import (
    MultipleNegativesCollateFn,
    MultipleNegativesDistilCollateFn,
    KDProcessingCollateFn,
)
from sparsecity.models.splade_models.model_registry import get_splade_model
from sparsecity.evaluation.validate import validate_model
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.similarity_functions import dot_score
from transformers import AutoTokenizer, BertConfig
import os
import torch
from torch.utils.data import DataLoader
import wandb
from datasets import load_dataset
from dataclasses import dataclass
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from schedulefree import AdamWScheduleFree
from sparsecity.data.dataset import KDProcessing

# from heavyball.utils import trust_region_clip_, rmsnorm_clip_
# from heavyball.utils import set_torch
from heapq import heappush, heappop

torch.set_float32_matmul_precision("high")
torch._dynamo.reset()


@dataclass
class TrainingConfig:
    seed: int
    data: DictConfig
    model: DictConfig
    sparse_embed: bool
    custom_kernel: bool
    batch_size: int
    mini_batch: int
    num_negatives: int
    sample_size: int  # Number of negatives to sample from total num_negatives
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
        checkpoint_path, f"checkpoint_step_{step}_ndcg_{score:.4f}.pt"
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


def compute_lambda_t_delayed(
    lambda_val: float,
    global_step: int,
    start_step: int,
    end_step: int,  # T_d or T_q
) -> float:
    if global_step < start_step:
        return 0.0

    # progress ∈ (0, 1] during the ramp
    progress = (global_step - start_step + 1) / (end_step - start_step + 1)
    return min(lambda_val, lambda_val * (progress**2))


def train_model(splade_model, tokenizer, cfg, dataset):
    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

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

    optimizer = AdamWScheduleFree(
        optim_param_groups,
        warmup_steps=cfg.optimizer.warmup_steps,
        weight_decay=cfg.optimizer.weight_decay,
    )
    if cfg.max_length is not None:
        tokenizer.model_max_length = cfg.max_length
    if cfg.use_distillation:
        dataloader = DataLoader(
            dataset,
            collate_fn=KDProcessingCollateFn(
                tokenizer, num_negatives=cfg.num_negatives, sample_size=cfg.sample_size
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
                "batch_size": cfg.batch_size,
                "mini_batch": cfg.mini_batch,
                "learning_rate": cfg.optimizer.learning_rate,
                "warmup_steps": cfg.optimizer.warmup_steps,
                "optimizer": optimizer.__class__.__name__,
                "sparse_embed": cfg.sparse_embed,
            },
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_directory = os.path.join(
        hydra.utils.to_absolute_path(cfg.checkpoint.checkpoint_path), timestamp
    )
    os.makedirs(checkpoint_directory, exist_ok=True)
    checkpoint_scores = []

    def optimized_step():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    global_step = 0

    # Training loop
    for epoch in range(cfg.epochs):
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if cfg.use_distillation:
                query_ids, query_mask, doc_ids, doc_mask, teacher_scores = (
                    t.to(device) for t in batch
                )
            else:
                query_ids, query_mask, doc_ids, doc_mask = (t.to(device) for t in batch)

            lambda_t_d = compute_lambda_t_delayed(
                cfg.lambda_d, global_step, cfg.T_d_start, cfg.T_d
            )
            lambda_t_q = compute_lambda_t_delayed(
                cfg.lambda_q, global_step, cfg.T_q_start, cfg.T_q
            )
            mse_weight = torch.tensor(1.0, device=device)
            optimizer.train()
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            metrics = train_step_kldiv_ibn(
                splade_model,
                query_ids,
                query_mask,
                doc_ids,
                doc_mask,
                cfg.top_k,
                torch.tensor(lambda_t_d, device=device),
                torch.tensor(lambda_t_q, device=device),
                device,
                splade_model.temperature_ce,
                splade_model.temperature_kl,
                mini_batch=cfg.mini_batch,
                teacher_scores=teacher_scores if cfg.use_distillation else None,
            )

            optimized_step()
            metrics = {
                "loss/total_loss": metrics["loss"].item(),
                "loss/triplet_loss": metrics["triplet_loss"].item(),
                "loss/kl_loss": metrics["kl_loss"].item(),
                "loss/flops": metrics["flops_loss"].item(),
                "loss/anti_zero": metrics["anti_zero_loss"].item(),
                "metrics/query_min_non_zero": metrics["query_min_non_zero"].item(),
                "metrics/doc_min_non_zero": metrics["doc_min_non_zero"].item(),
                "metrics/avg_query_non_zero_count": metrics["avg_query_non_zero_count"],
                "metrics/avg_doc_non_zero_count": metrics["avg_doc_non_zero_count"],
                "metrics/query_median_non_zero": metrics[
                    "query_median_non_zero"
                ].item(),
                "metrics/doc_median_non_zero": metrics["doc_median_non_zero"].item(),
            }

            metrics["metrics/kl_temp"] = splade_model.temperature_kl.detach().item()
            metrics["metrics/ce_temp"] = splade_model.temperature_ce.detach().item()

            # For SparseEmbed models - will clean eventually when I adapt SparseEmbed for KLDiv
            # metrics = {
            #     "total_loss": metrics["total_loss"].item(),
            #     "triplet_loss": metrics["triplet_loss"].item(),
            #     "dense_loss": metrics["dense_loss"].item(),
            #     "sparse_mse_loss": metrics["sparse_mse_loss"].item(),
            #     "dense_mse_loss": metrics["dense_mse_loss"].item(),
            #     "flops_loss": metrics["flops_loss"].item(),
            #     "anti_zero_loss": metrics["anti_zero_loss"].item(),
            #     "query_sparsity": metrics["query_sparsity"].item(),
            #     "doc_sparsity": metrics["doc_sparsity"].item(),
            #     "avg_query_non_zero_count": metrics["avg_query_non_zero_count"],
            # }
            if cfg.wandb and step % cfg.log_every == 0:
                wandb.log({**metrics}, step=global_step)

            if (step + 1) % cfg.evaluation.eval_every_steps == 0 or global_step == 50:
                splade_model.eval()
                optimizer.eval()
                val_results = validate_model(
                    evaluator,
                    splade_model,
                    tokenizer,
                    device,
                    sparse_embed=cfg.sparse_embed,
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
                        step=global_step,
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
            global_step += 1


@hydra.main(config_path="conf", config_name="cocondenser_base", version_base=None)
def main(cfg: DictConfig):
    cfg = TrainingConfig(**cfg)
    config = BertConfig.from_pretrained(cfg.model.name)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = get_splade_model(
        cfg.model.name,
        config=config,
        custom_kernel=cfg.custom_kernel,
        sparse_embed=cfg.sparse_embed,
        checkpoint_path=cfg.model.checkpoint_path
        if cfg.model.checkpoint_path
        else None,
        init_ce_temp=cfg.init_ce_temp,
        init_kl_temp=cfg.init_kl_temp,
    )
    # dataset = load_dataset(
    #     cfg.data.name,
    #     split=cfg.data.split,
    # )

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
