#!/usr/bin/env python3
"""
Train a SPLADE-style SparseEncoder using SentenceTransformers on the same dataset
pipeline used in splade_train.py (LightOn KDProcessing), but with the ST training
framework and the NeoBERT HF model from conf/model/neo.yaml.

- Uses SpladeLoss + SparseMultipleNegativesRankingLoss (in-batch negatives, plus optional explicit negatives)
- Converts the KDProcessing-transformed dataset to (query, positive, negative_1..negative_k)
- Evaluates with SparseNanoBEIREvaluator on standard NanoBEIR subsets

Note on collators & negatives:
- SentenceTransformers' SparseEncoderTrainer uses its own data collator and expects text columns.
- To preserve explicit negatives, we expand negatives into separate text columns (negative_1, ..., negative_k).
  MultipleNegativesRankingLoss will use these as additional candidates alongside in-batch negatives.
"""

import logging
import os
import traceback
import random
from typing import List

import yaml
from datasets import load_dataset, Dataset

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.sparse_encoder import evaluation, losses
from sentence_transformers.sparse_encoder.callbacks import (
    SpladeRegularizerWeightSchedulerCallback,
)
from sentence_transformers.sparse_encoder.data_collator import (
    SparseEncoderDataCollator,
)
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling
from sentence_transformers.training_args import BatchSamplers
from transformers import get_wsd_schedule
from torch import Tensor
from torch import nn
import torch
import math
from datetime import datetime
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

# Reuse your KDProcessing transform to create 'query', 'documents', 'scores'
from src.sparsecity.data.dataset import KDProcessing


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def load_model_name_from_yaml(yaml_path: str) -> str:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    model_name = cfg.get("name")
    if not model_name:
        raise ValueError(f"No 'name' found in {yaml_path}")
    return model_name


def _safe_score(scores: list[float], idx: int, default: float = 0.0) -> float:
    if isinstance(scores, list) and 0 <= idx < len(scores):
        value = scores[idx]
        if value is not None:
            return float(value)
    return default


def _select_negatives_per_row(
    docs: list[str], scores: list[float], num_negs: int, positive_idx: int
) -> tuple[list[str], list[float]]:
    if not docs:
        return [""] * num_negs, [0.0] * num_negs

    score_aligned = isinstance(scores, list) and len(scores) == len(docs)
    candidate_indices = [i for i in range(len(docs)) if i != positive_idx]

    if score_aligned:
        candidate_indices.sort(key=lambda i: scores[i], reverse=True)

    if not candidate_indices:
        candidate_indices = [positive_idx]

    if len(candidate_indices) < num_negs:
        last_idx = candidate_indices[-1]
        candidate_indices.extend([last_idx] * (num_negs - len(candidate_indices)))
    else:
        candidate_indices = candidate_indices[:num_negs]

    neg_docs = [docs[i] for i in candidate_indices]
    neg_scores = [
        float(scores[i]) if score_aligned and scores[i] is not None else 0.0
        for i in candidate_indices
    ]
    return neg_docs, neg_scores


def build_train_eval_datasets(
    num_explicit_negatives: int = 8,
    max_train_samples: int | None = 100_000,
    eval_size: int | None = 10_000,
    seed: int = 42,
) -> tuple[Dataset, Dataset | None]:
    """
    - Loads LightOn MS MARCO datasets and applies KDProcessing to get 'query', 'documents', 'scores'.
    - Converts to SentenceTransformers format: ['query', 'positive', 'negative_1', ..., 'negative_k']
    - Splits into train/eval.
    """
    logging.info("Loading LightOn MS MARCO datasets...")
    train_raw = load_dataset(
        "lightonai/ms-marco-en-bge-gemma", "train_unormalized", split="train"
    )
    queries = load_dataset("lightonai/ms-marco-en-bge-gemma", "queries", split="train")
    documents = load_dataset(
        "lightonai/ms-marco-en-bge-gemma", "documents", split="train"
    )

    logging.info(
        "Materializing KDProcessing (adds 'query', 'documents', 'scores') via map()..."
    )
    processor = KDProcessing(queries=queries, documents=documents)
    # Apply the non-lazy map so downstream map/filter see the new columns reliably
    train_proc = train_raw.map(
        processor.map,
        batched=False,
        remove_columns=train_raw.column_names,
    )

    logging.info(
        "Converting to (query, positive, negatives[1..%d]) columns...",
        num_explicit_negatives,
    )

    negative_cols = [f"negative_{i + 1}" for i in range(num_explicit_negatives)]

    def to_triplets(batch):
        qs = batch["query"]
        docs_list = batch["documents"]
        scores_list = batch["scores"]

        positives = []
        negatives_matrix = [[] for _ in range(num_explicit_negatives)]
        labels = []
        # Store full pool for random negative selection at collator time
        all_negatives = []
        all_neg_scores = []
        pos_scores = []

        for docs, scores in zip(docs_list, scores_list):
            if not isinstance(docs, list) or not docs:
                positives.append("")
                for negs in negatives_matrix:
                    negs.append("")
                labels.append([0.0] * (num_explicit_negatives + 1))
                continue

            score_list = scores if isinstance(scores, list) else []
            if score_list and len(score_list) == len(docs):
                pos_idx = max(range(len(score_list)), key=lambda i: score_list[i])
            else:
                pos_idx = 0

            positives.append(docs[pos_idx])
            neg_docs, neg_scores = _select_negatives_per_row(
                docs, score_list, num_explicit_negatives, pos_idx
            )
            for i, neg in enumerate(neg_docs):
                negatives_matrix[i].append(neg)

            positive_score = _safe_score(score_list, pos_idx)
            labels.append([positive_score, *neg_scores])

            # Full negative pool (excluding the positive) for dynamic sampling
            pool_docs = [d for i, d in enumerate(docs) if i != pos_idx]
            if score_list and len(score_list) == len(docs):
                pool_scores = [
                    float(score_list[i]) for i in range(len(docs)) if i != pos_idx
                ]
            else:
                pool_scores = [0.0] * len(pool_docs)
            all_negatives.append(pool_docs)
            all_neg_scores.append(pool_scores)
            pos_scores.append(positive_score)

        output = {
            "query": qs,
            "positive": positives,
            "label": labels,
            "all_negatives": all_negatives,
            "all_neg_scores": all_neg_scores,
            "pos_score": pos_scores,
        }
        for i, col in enumerate(negative_cols):
            output[col] = negatives_matrix[i]
        return output

    mapped = train_proc.map(to_triplets, batched=True)

    # Keep only the columns in the expected order so the collator/loss interpret correctly
    ordered_cols = [
        "query",
        "positive",
        *negative_cols,
        "label",
        "all_negatives",
        "all_neg_scores",
        "pos_score",
    ]

    def _row_ok(ex):
        if not ex.get("query") or not ex.get("positive"):
            return False
        for col in negative_cols:
            neg = ex.get(col)
            if not neg:
                return False
        label = ex.get("label")
        if label is None or len(label) != num_explicit_negatives + 1:
            return False
        return True

    mapped = mapped.filter(_row_ok)

    keep_cols = [c for c in ordered_cols if c in mapped.column_names]
    mapped = mapped.select_columns(keep_cols)

    # Optionally subsample for quick test runs
    if max_train_samples is not None and len(mapped) > max_train_samples:
        logging.info(
            "Subsampling to first %d training samples for quick test.",
            max_train_samples,
        )
        mapped = mapped.select(range(max_train_samples))

    # Split into train/eval
    if eval_size:
        eval_size = min(eval_size, max(1000, int(0.05 * len(mapped))))
        dataset_dict = mapped.train_test_split(
            test_size=eval_size, seed=seed, shuffle=True
        )
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]
    else:
        train_dataset = mapped
        eval_dataset = None

    logging.info(train_dataset)
    if eval_dataset is not None:
        logging.info(eval_dataset)

    return train_dataset, eval_dataset


class RandomNegativesCollator(SparseEncoderDataCollator):
    """
    Collator that randomly samples k negatives per example from the stored full pool and
    rebuilds the (query, positive, negative_i..., label) batch before tokenization.
    """

    def __init__(
        self,
        *args,
        negatives_per_sample: int = 8,
        seed: int = 42,
        allow_replacement: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.negatives_per_sample = negatives_per_sample
        self.allow_replacement = allow_replacement
        self._base_seed = seed
        self._rng = random.Random(seed)

    def set_epoch_seed(self, epoch_seed: int) -> None:
        self._rng = random.Random(epoch_seed)

    def _sample_indices(self, n: int, k: int) -> list[int]:
        if n <= 0:
            return []
        if k <= n and not self.allow_replacement:
            return self._rng.sample(range(n), k)
        # Not enough items or replacement allowed → sample with replacement
        return [self._rng.randrange(n) for _ in range(k)]

    def __call__(self, features: list[dict]) -> dict:
        k = self.negatives_per_sample
        processed: list[dict] = []

        for ex in features:
            negs = ex.get("all_negatives") or []
            neg_scores = ex.get("all_neg_scores") or []
            pos_score = ex.get("pos_score", 0.0)

            idxs = self._sample_indices(len(negs), k)
            if idxs:
                chosen_negs = [negs[i] for i in idxs]
                chosen_scores = (
                    [neg_scores[i] for i in idxs] if neg_scores else [0.0] * k
                )
            else:
                chosen_negs = [""] * k
                chosen_scores = [0.0] * k

            row = {
                "query": ex["query"],
                "positive": ex["positive"],
                "label": [pos_score, *chosen_scores],
            }
            for i in range(k):
                row[f"negative_{i + 1}"] = chosen_negs[i]
            processed.append(row)

        # Let the base collator tokenize and pack
        return super().__call__(processed)


class RandomNegativesReseedCallback(TrainerCallback):
    def __init__(self, data_collator: RandomNegativesCollator, base_seed: int = 42):
        super().__init__()
        self.data_collator = data_collator
        self.base_seed = base_seed

    def on_epoch_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        # Change RNG each epoch but keep runs reproducible
        epoch_idx = int(state.epoch or 0)
        self.data_collator.set_epoch_seed(self.base_seed + epoch_idx)


class SparseAntiZeroLoss(nn.Module):
    """
    Penalises batches whose SPLADE activations collapse to zero.
    Computes 1/(sum(query)^2) + 1/(sum(docs)^2) and leaves weighting to the caller.
    """

    def __init__(self, model: SparseEncoder, epsilon: float = 1e-12) -> None:
        super().__init__()
        self.model = model
        self.epsilon = epsilon

    def forward(
        self, sentence_features: List[dict[str, Tensor]], labels: Tensor | None = None
    ) -> Tensor:
        raise AttributeError(
            "SparseAntiZeroLoss should not be used alone. Call compute_loss_from_embeddings within SpladeLoss/CSRLoss."
        )

    def compute_loss_from_embeddings(
        self, embeddings: List[Tensor], labels: Tensor | None = None
    ) -> Tensor:
        if not embeddings:
            raise ValueError("SparseAntiZeroLoss received no embeddings.")

        query_embeddings = embeddings[0]
        document_embeddings = (
            torch.cat(embeddings[1:], dim=0)
            if len(embeddings) > 1
            else query_embeddings
        )

        eps = self.epsilon
        query_sum = torch.sum(query_embeddings) + eps
        doc_sum = torch.sum(document_embeddings) + eps

        loss = (1.0 / (query_sum**2)) + (1.0 / (doc_sum**2))
        return loss


class SparseDistillMarginCombinedLoss(nn.Module):
    def __init__(
        self,
        model: SparseEncoder,
        distill_temperature: float = 2.0,
        distill_weight: float = 1.0,
        margin_weight: float = 0.05,
        anti_zero_weight: float = 0.0,
        anti_zero_epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        self.model = model
        self.distill = losses.SparseDistillKLDivLoss(
            model, temperature=distill_temperature
        )
        self.margin = losses.SparseMarginMSELoss(model)
        self.anti_zero = SparseAntiZeroLoss(model, epsilon=anti_zero_epsilon)
        self.distill_weight = distill_weight
        self.margin_weight = margin_weight
        self.anti_zero_weight = anti_zero_weight

    def compute_loss_from_embeddings(
        self, embeddings: List[Tensor], labels: Tensor
    ) -> dict[str, Tensor]:
        losses_dict = {}
        losses_dict["distill_kl_loss"] = (
            self.distill.compute_loss_from_embeddings(embeddings, labels)
            * self.distill_weight
        )
        losses_dict["margin_mse_loss"] = (
            self.margin.compute_loss_from_embeddings(embeddings, labels)
            * self.margin_weight
        )
        if self.anti_zero_weight > 0:
            anti_zero = (
                self.anti_zero.compute_loss_from_embeddings(embeddings, labels)
                * self.anti_zero_weight
            )
            losses_dict["anti_zero_loss"] = anti_zero
        return losses_dict


def main():
    # Hyperparameters for a quick trial
    train_batch_size = 4
    num_epochs = 2
    learning_rate = 2e-5
    weight_decay = 0.05
    query_regularizer_weight = 0.0000004
    document_regularizer_weight = 0.000001
    max_seq_length = 256
    num_explicit_negatives = 8
    regularizer_scheduler_type = "quadratic"
    regularizer_warmup_ratio = 0.6
    anti_zero_weight = 0.5

    # Load HF model name from your conf/model/neo.yaml
    model_yaml_path = os.path.join("conf", "model", "neo.yaml")
    model_name = load_model_name_from_yaml(model_yaml_path)
    short_model_name = model_name.split("/")[-1]

    # 1) Define SparseEncoder model
    # Force SPLADE path by explicitly building MLMTransformer + SpladePooling
    # and override tokenizer to use BERT (NeoBERT uses BERT tokenizer but may not register it).
    mlm = MLMTransformer(
        model_name_or_path=model_name,
        tokenizer_name_or_path="bert-base-uncased",  # Fallback tokenizer for NeoBERT
        max_seq_length=max_seq_length,
        model_args={"trust_remote_code": True},
        config_args={"trust_remote_code": True},
    )
    splade_pool = SpladePooling(pooling_strategy="max")
    model = SparseEncoder(
        modules=[mlm, splade_pool],
        model_card_data=SparseEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"splade-{short_model_name} trained on LightOn MS MARCO (triplets)",
        ),
    )
    # model = SparseEncoder(
    #     "models/splade-NeoBERT-RetroMAE-pretrain-lighton-msmarco-triplets/checkpoint-10200/",
    #     trust_remote_code=True,
    # )
    logging.info(
        "Using explicit MLMTransformer+SpladePooling. Model max length: %s",
        model.max_seq_length,
    )

    # 2) Build datasets: keep your KDProcessing setup; expand to explicit negatives columns
    train_dataset, eval_dataset = build_train_eval_datasets(
        num_explicit_negatives=num_explicit_negatives,
        max_train_samples=400_000,
        eval_size=10_000,
        seed=42,
    )

    # 3) Define SPLADE loss with in-batch negatives ranking (plus explicit negatives)
    loss = losses.SpladeLoss(
        model=model,
        loss=SparseDistillMarginCombinedLoss(
            model=model, anti_zero_weight=anti_zero_weight
        ),
        query_regularizer_weight=query_regularizer_weight,
        document_regularizer_weight=document_regularizer_weight,
    )

    # 4) Evaluator: NanoBEIR (lightweight retrieval metrics)
    evaluator = evaluation.SparseNanoBEIREvaluator(
        dataset_names=["scifact", "msmarco", "touche2020", "scidocs", "nfcorpus"],
        batch_size=train_batch_size,
    )

    # 5) Training arguments
    run_name_base = f"splade-{short_model_name}-lighton-msmarco-triplets-distill"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{run_name_base}-{timestamp}"
    args = SparseEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        gradient_accumulation_steps=16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # In-batch negatives benefit from no duplicates
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_mean_dot_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=300,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=3,
        logging_steps=10,
        run_name=run_name,  # Used by W&B if installed
        seed=42,
    )

    # Optimizer
    #
    #
    steps_per_epoch = math.ceil(len(train_dataset) / train_batch_size)
    updates_per_epoch = math.ceil(steps_per_epoch / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * num_epochs

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = get_wsd_schedule(
        optimizer,
        num_training_steps=total_updates,
        num_warmup_steps=round(total_updates * 0.1),
        num_decay_steps=round(total_updates * 0.5),
    )
    callbacks = []
    callbacks.append(
        SpladeRegularizerWeightSchedulerCallback(
            loss=loss,
            scheduler_type=regularizer_scheduler_type,
            warmup_ratio=regularizer_warmup_ratio,
        )
    )

    # Random negatives collator & reseeding per epoch
    data_collator = RandomNegativesCollator(
        tokenize_fn=model.tokenize,
        negatives_per_sample=num_explicit_negatives,
        seed=args.seed,
        router_mapping=args.router_mapping,
        prompts=args.prompts,
        all_special_ids=set(model.tokenizer.all_special_ids)
        if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "all_special_ids")
        else set(),
    )
    callbacks.append(RandomNegativesReseedCallback(data_collator, base_seed=args.seed))

    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
        data_collator=data_collator,
        callbacks=callbacks or None,
        optimizers=(optimizer, scheduler),
    )
    trainer.train()

    # 7) Evaluate the final model over full NanoBEIR test suite
    test_evaluator = evaluation.SparseNanoBEIREvaluator(
        show_progress_bar=True, batch_size=train_batch_size
    )
    test_evaluator(model)

    # 8) Save final model
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)

    # # 9) Optional: push to hub
    # try:
    #     model.push_to_hub(run_name)
    # except Exception:
    #     logging.error(
    #         "Error uploading model to the Hugging Face Hub:\n%sTo upload it manually, run `huggingface-cli login`, then:\n"
    #         "  model = SparseEncoder(%r)\n  model.push_to_hub('%s')",
    #         traceback.format_exc(),
    #         final_output_dir,
    #         run_name,
    #     )


if __name__ == "__main__":
    main()
