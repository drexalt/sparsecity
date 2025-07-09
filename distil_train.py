# from sparsecity.training.trainer import train_step
from sparsecity.training.trainer import train_step_straight_distil
from datetime import datetime
from collections import deque

# from sparsecity.training.sparse_trainer import train_step
from sparsecity.data.dataset import (
    MultipleNegativesCollateFn,
    MsmarcoDocumentCollateFn,
    QueryDocStream,
)
from sparsecity.models.splade_models.model_registry import get_splade_model
from sparsecity.utils.utils import (
    flatten_dict,
    update_checkpoint_tracking,
)
from sparsecity.evaluation.validate import validate_model
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.similarity_functions import dot_score
from transformers import AutoTokenizer, AutoConfig, BertConfig
import os
import torch
from torch.utils.data import DataLoader
import wandb
from datasets import load_dataset, Dataset, Features, Value
from dataclasses import dataclass
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import dataclasses
from transformers import get_wsd_schedule
import heavyball
from heavyball.utils import trust_region_clip_, rmsnorm_clip_
import ir_datasets

import logging

torch.set_float32_matmul_precision("high")
torch._dynamo.reset()


@dataclass
class TrainingConfig:
    seed: int
    data: DictConfig
    model: DictConfig
    teacher_model: str
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
    top_k: int
    max_length: int
    epochs: int
    log_every: int
    optimizer: DictConfig
    checkpoint: DictConfig
    wandb: bool
    wandb_project: str
    use_distillation: bool
    evaluation: DictConfig


logger = logging.getLogger(__name__)


def train_model(
    splade_model, teacher_model, tokenizer, cfg, docs_dataset, queries_dataset
):
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

    warmup_steps = cfg.optimizer.warmup_steps

    # optimizer = torch.optim.AdamW(
    #     splade_model.parameters(),
    #     lr=cfg.optimizer.learning_rate,
    #     weight_decay=cfg.optimizer.weight_decay,
    # )

    # optimizer = heavyball.ForeachAdamW(
    #     splade_model.parameters(),
    #     lr=cfg.optimizer.learning_rate,
    #     weight_decay=cfg.optimizer.weight_decay,
    #     caution=True,
    # )
    optimizer = heavyball.ForeachPSGDKron(
        splade_model.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        delayed=True,
        cached=True,
        gradient_clipping=trust_region_clip_,
        update_clipping=rmsnorm_clip_,
    )

    if cfg.max_length is not None:
        tokenizer.model_max_length = cfg.max_length
    if cfg.use_distillation:
        dataloader = DataLoader(
            QueryDocStream(docs_dataset, queries_dataset),
            collate_fn=MsmarcoDocumentCollateFn(tokenizer, max_length=cfg.max_length),
            batch_size=cfg.batch_size,
            # pin_memory=True,
            num_workers=0,  # Add multiple workers for better data loading
            # persistent_workers=True,  # Keep workers alive between iterations
            # prefetch_factor=2,
            drop_last=True,
        )
    else:
        dataloader = DataLoader(
            QueryDocStream(docs_dataset, queries_dataset),
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

    scheduler = get_wsd_schedule(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_decay_steps=cfg.optimizer.decay_steps,
        min_lr_ratio=0.1,
        num_stable_steps=cfg.optimizer.stable_steps,
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

    def optim_step():
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    global_step = 0
    accum_steps = cfg.accum_steps
    LOG_EVERY_MICRO = cfg.log_every * accum_steps
    EVAL_EVERY_MICRO = cfg.evaluation.eval_every_steps * accum_steps

    # Training loop
    for epoch in range(cfg.epochs):
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            global_step += 1

            if cfg.use_distillation:
                query_ids, query_mask, doc_ids, doc_mask = (t.to(device) for t in batch)
            else:
                query_ids, query_mask, doc_ids, doc_mask = (t.to(device) for t in batch)

            # optimizer.train()
            query_weight = torch.tensor(1.0, device=device)
            doc_weight = torch.tensor(1.0, device=device)

            loss_scale = torch.tensor(1.0 / accum_steps, device=device)

            train_kwargs = dict(
                model=splade_model,
                teacher_model=teacher_model,
                query_input_ids=query_ids,
                query_attention_mask=query_mask,
                doc_input_ids=doc_ids,
                doc_attention_mask=doc_mask,
                query_weight=query_weight,
                doc_weight=doc_weight,
                loss_scale=loss_scale,
            )

            metrics = train_step_straight_distil(
                **train_kwargs,
                bf16=cfg.bf16,
            )

            optim_step()

            log_metrics = {
                "loss/mse": metrics["mse_loss"].item(),
                "metrics/query_min_non_zero": metrics["query_min_non_zero"].item(),
                "metrics/doc_min_non_zero": metrics["doc_min_non_zero"].item(),
                "metrics/avg_query_non_zero_count": metrics["avg_query_non_zero_count"],
                "metrics/avg_doc_non_zero_count": metrics["avg_doc_non_zero_count"],
                "metrics/query_median_non_zero": metrics[
                    "query_median_non_zero"
                ].item(),
                "metrics/doc_median_non_zero": metrics["doc_median_non_zero"].item(),
            }

            if cfg.wandb and global_step % LOG_EVERY_MICRO == 0:
                wandb.log({**log_metrics}, step=global_step // accum_steps)

            if (global_step + 1) % EVAL_EVERY_MICRO == 0:
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
                    score=val_results["mrr@10"],
                    checkpoint_scores=checkpoint_scores,
                    max_checkpoints=cfg.checkpoint.max_to_keep,
                    splade_model=splade_model,
                    optimizer=optimizer,
                    checkpoint_path=checkpoint_directory,
                )


@hydra.main(config_path="conf", config_name="mosaic_distil", version_base=None)
def main(cfg: DictConfig):
    cfg = TrainingConfig(**cfg)
    config = AutoConfig.from_pretrained(cfg.model.name, trust_remote_code=True)
    teacher_config = AutoConfig.from_pretrained(cfg.teacher_model)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load teacher model first to avoid config errors
    teacher_model = get_splade_model(
        cfg.teacher_model,
        config=teacher_config,
        custom_kernel=cfg.custom_kernel,
        trust_remote_code=False,
    )

    model = get_splade_model(
        cfg.model.name,
        config=config,
        custom_kernel=cfg.custom_kernel,
        sparse_embed=cfg.sparse_embed,
        checkpoint_path=cfg.model.checkpoint_path
        if cfg.model.checkpoint_path
        else None,
    )

    # train_dataset = load_dataset(
    #     cfg.data.name,
    #     split=cfg.data.split,
    # )
    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    # train_dataset = load_dataset(
    #     "lightonai/ms-marco-en-bge-gemma",
    #     "train",
    #     split="train",
    # )

    # queries = load_dataset(
    #     "lightonai/ms-marco-en-bge-gemma",
    #     "queries",
    #     split="train",
    # )

    # documents = load_dataset(
    #     "lightonai/ms-marco-en-bge-gemma",
    #     "documents",
    #     split="train",
    # )

    # train_dataset.set_transform(
    #     KDProcessing(queries=queries, documents=documents).transform
    # )
    # set_torch()
    train_docs = load_dataset(
        "irds/msmarco-document-v2", "docs", trust_remote_code=True
    )

    def gen_queries():
        for q in queries_iter:
            yield {"query_id": q.query_id, "text": q.text}

    queries_iter = ir_datasets.load("msmarco-document-v2/train").queries_iter()
    train_queries = Dataset.from_generator(
        gen_queries,
        features=Features(
            {
                "query_id": Value("string"),
                "text": Value("string"),
            }
        ),
    )
    train_model(
        splade_model=model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        cfg=cfg,
        docs_dataset=train_docs,
        queries_dataset=train_queries,
    )  # Assumes same tokenizer for teacher and student, obviously


if __name__ == "__main__":
    main()
