from sparsecity.training.trainer import train_step
from sparsecity.data.dataset import (
    MultipleNegativesCollateFn,
    MultipleNegativesDistilCollateFn,
)
from sparsecity.models.splade_models.model_registry import get_splade_model
from sparsecity.evaluation.validate import validate_model
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.similarity_functions import dot_score
from transformers import AutoTokenizer
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
import heavyball
from heavyball.utils import trust_region_clip_, rmsnorm_clip_
from heavyball.utils import set_torch
from heapq import heappush, heappop, heapreplace

torch.set_float32_matmul_precision("high")
torch._dynamo.reset()


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
    optimizer = heavyball.ForeachPSGDKron(
        splade_model.parameters(),
        lr=cfg.optimizer.learning_rate,
        warmup_steps=cfg.optimizer.warmup_steps,
        weight_decay=cfg.optimizer.weight_decay,
        caution=True,
        foreach=True,
        delayed=True,
        gradient_clipping=trust_region_clip_,
        update_clipping=rmsnorm_clip_,
    )
    # optimizer = AdamWScheduleFree(
    #     splade_model.parameters(),
    #     lr=cfg.optimizer.learning_rate,
    #     warmup_steps=cfg.optimizer.warmup_steps,
    # )
    # optimizer.train()
    if cfg.use_distillation:
        dataloader = DataLoader(
            dataset,
            collate_fn=MultipleNegativesDistilCollateFn(
                tokenizer, num_negatives=cfg.num_negatives
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,  # Add multiple workers for better data loading
            persistent_workers=True,  # Keep workers alive between iterations
            prefetch_factor=2,
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
        )

    # Initialize wandb if enabled
    if cfg.wandb:
        wandb.init(
            project=cfg.wandb_project,
            config={
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.optimizer.learning_rate,
                "warmup_steps": cfg.optimizer.warmup_steps,
                "optimizer": optimizer.__class__.__name__,
            },
        )
    os.makedirs(cfg.checkpoint.checkpoint_path, exist_ok=True)
    checkpoint_scores = []

    def optimized_step():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Training loop
    for epoch in range(cfg.epochs):
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if cfg.use_distillation:
                query_ids, query_mask, doc_ids, doc_mask, teacher_scores = (
                    t.to(device) for t in batch
                )
            else:
                query_ids, query_mask, doc_ids, doc_mask = (t.to(device) for t in batch)

            step_ratio_d = (step + 1) / (cfg.T_d + 1)
            step_ratio_q = (step + 1) / (cfg.T_q + 1)

            lambda_t_d = compute_lambda_t(cfg.lambda_d, step_ratio_d)
            lambda_t_q = compute_lambda_t(cfg.lambda_q, step_ratio_q)
            temperature = torch.tensor(10.0, device=device)
            mse_weight = torch.tensor(0.1, device=device)
            # optimizer.train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                metrics = train_step(
                    splade_model,
                    query_ids,
                    query_mask,
                    doc_ids,
                    doc_mask,
                    cfg.top_k,
                    torch.tensor(lambda_t_d, device=device),
                    torch.tensor(lambda_t_q, device=device),
                    device,
                    temperature,
                    mse_weight,
                    teacher_scores=teacher_scores if cfg.use_distillation else None,
                )
            optimized_step()
            metrics = {
                "total_loss": metrics["loss"].item(),
                "triplet_loss": metrics["triplet_loss"].item(),
                "margin_mse_loss": metrics["margin_mse_loss"].item(),
                "flops": metrics["flops_loss"].item(),
                "anti_zero": metrics["anti_zero_loss"].item(),
                "query_sparsity": metrics["query_sparsity"].item(),
                "doc_sparsity": metrics["doc_sparsity"].item(),
                "query_min_non_zero": metrics["query_min_non_zero"].item(),
                "doc_min_non_zero": metrics["doc_min_non_zero"].item(),
                "avg_query_non_zero_count": metrics["avg_query_non_zero_count"],
                "avg_doc_non_zero_count": metrics["avg_doc_non_zero_count"],
                "query_median_non_zero": metrics["query_median_non_zero"].item(),
                "doc_median_non_zero": metrics["doc_median_non_zero"].item(),
            }
            if cfg.wandb and step % cfg.log_every == 0:
                wandb.log({**metrics}, step=(epoch * len(dataloader)) + step)

            if (step + 1) % cfg.evaluation.eval_every_steps == 0 or step == 5:
                splade_model.eval()
                val_results = validate_model(evaluator, splade_model, tokenizer, device)
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
                        step=(epoch * len(dataloader)) + step,
                    )

                # Save checkpoint
                checkpoint_scores = update_checkpoint_tracking(
                    step=(epoch * len(dataloader)) + step,
                    score=val_results["ndcg@10"],
                    checkpoint_scores=checkpoint_scores,
                    max_checkpoints=cfg.checkpoint.max_to_keep,
                    splade_model=splade_model,
                    optimizer=optimizer,
                    checkpoint_path=cfg.checkpoint.checkpoint_path,
                )


@hydra.main(config_path="conf", config_name="distilbert_base", version_base=None)
def main(cfg: DictConfig):
    cfg = TrainingConfig(**cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = get_splade_model(cfg.model.name)
    dataset = load_dataset(
        "jturner116/msmarco-hard-negatives-scored-stella",
        split="train",
    )
    set_torch()
    train_model(
        splade_model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        dataset=dataset,
    )


if __name__ == "__main__":
    main()
