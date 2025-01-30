from sparsecity.training.trainer import train_step
from sparsecity.data.dataset import MultipleNegativesCollateFn
from sparsecity.models.splade_models.model_registry import get_splade_model
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

    # optimizer.train()
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
            query_ids, query_mask, doc_ids, doc_mask = (t.to(device) for t in batch)

            step_ratio_d = (step + 1) / (cfg.T_d + 1)
            step_ratio_q = (step + 1) / (cfg.T_q + 1)

            lambda_t_d = compute_lambda_t(cfg.lambda_d, step_ratio_d)
            lambda_t_q = compute_lambda_t(cfg.lambda_q, step_ratio_q)
            epsilon = torch.tensor(1e-8, device=device)

            total_loss, triplet_loss, flops, anti_zero = train_step(
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
            )
            optimized_step()
            metrics = {
                "total_loss": total_loss.item(),
                "triplet_loss": triplet_loss.item(),
                "flops": flops.item(),
                "anti_zero": anti_zero.item(),
            }

            if cfg.wandb and step % cfg.log_every == 0:
                wandb.log({**metrics}, step=step)

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

    train_model(
        splade_model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        dataset=dataset,
    )


if __name__ == "__main__":
    main()
