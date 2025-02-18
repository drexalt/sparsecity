from sparsecity.training.trainer import train_step
from sparsecity.data.dataset import (
    MultipleNegativesCollateFn,
    MultipleNegativesDistilCollateFn,
)
from sparsecity.models.splade_models.model_registry import get_splade_model
from sparsecity.evaluation.validate import validate_model
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
from heavyball.utils import set_torch

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

    # Create optimizer and scheduler
    optimizer = heavyball.ForeachPSGDKron(
        splade_model.parameters(),
        lr=cfg.optimizer.learning_rate,
        warmup_steps=cfg.optimizer.warmup_steps,
        caution=True,
        foreach=True,
        delayed=True,
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
        )

    # Initialize wandb if enabled
    if cfg.wandb:
        wandb.init(
            project=cfg.wandb_project,
            config={
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.optimizer.learning_rate,
                "warmup_steps": cfg.optimizer.warmup_steps,
            },
        )
    os.makedirs(cfg.checkpoint.checkpoint_path, exist_ok=True)

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
            epsilon = torch.tensor(1e-8, device=device)
            temperature = torch.tensor(10.0, device=device)
            # optimizer.train()
            total_loss, triplet_loss, margin_mse_loss, flops, anti_zero = train_step(
                splade_model,
                query_ids,
                query_mask,
                doc_ids,
                doc_mask,
                cfg.top_k,
                torch.tensor(lambda_t_d, device=device),
                torch.tensor(lambda_t_q, device=device),
                device,
                epsilon,
                temperature,
                teacher_scores=teacher_scores if cfg.use_distillation else None,
            )
            optimized_step()
            metrics = {
                "total_loss": total_loss.item(),
                "triplet_loss": triplet_loss.item(),
                "margin_mse_loss": margin_mse_loss.item(),
                "flops": flops.item(),
                "anti_zero": anti_zero.item(),
            }
            if cfg.wandb and step % cfg.log_every == 0:
                wandb.log({**metrics}, step=step)

            if (step + 1) % cfg.evaluation.eval_every_steps == 0 or step == 5:
                splade_model.eval()
                val_results = validate_model(splade_model, tokenizer, cfg, device)
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
                        step=step,
                    )

            # Save checkpoint
            if (
                cfg.checkpoint.save_interval_steps > 0
                and step % cfg.checkpoint.save_interval_steps == 0
            ):
                checkpoint = {
                    "splade_model": splade_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        cfg.checkpoint.checkpoint_path, f"checkpoint_{step}.pt"
                    ),
                )


@hydra.main(config_path="conf", config_name="distilbert_base", version_base=None)
def main(cfg: DictConfig):
    cfg = TrainingConfig(**cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = get_splade_model(cfg.model.name)
    dataset = load_dataset(
        "json",
        data_files={"train": cfg.data.train_path},
        split="train",
        encoding="utf-8",
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
