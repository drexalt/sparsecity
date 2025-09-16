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
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling
from sentence_transformers.training_args import BatchSamplers
from transformers import get_wsd_schedule

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


def _select_negatives_per_row(
    docs: List[str], scores: List[float], num_negs: int
) -> List[str]:
    if not docs:
        return [""] * num_negs
    if not scores:
        # If scores missing, take all except first, pad if needed
        candidates = docs[1:]
    else:
        # Exclude the positive (max score), then sort remaining by score desc
        pos_idx = max(range(len(scores)), key=lambda i: scores[i])
        idxs = [i for i in range(len(docs)) if i != pos_idx]
        # Guard length mismatch
        if len(scores) != len(docs):
            # fall back to using docs order if mismatch
            sorted_negs = [docs[i] for i in idxs]
        else:
            sorted_negs = [
                doc
                for _, doc in sorted(
                    ((scores[i], docs[i]) for i in idxs),
                    key=lambda x: x[0],
                    reverse=True,
                )
            ]
        candidates = sorted_negs

    # Pad/truncate to desired length
    if len(candidates) < num_negs:
        pad_val = candidates[-1] if candidates else docs[0]
        candidates = candidates + [pad_val] * (num_negs - len(candidates))
    else:
        candidates = candidates[:num_negs]
    return candidates


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
    train_raw = load_dataset("lightonai/ms-marco-en-bge-gemma", "train", split="train")
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

        for docs, scores in zip(docs_list, scores_list):
            if not isinstance(docs, list) or not docs:
                positives.append("")
                for i in range(num_explicit_negatives):
                    negatives_matrix[i].append("")
                continue

            # Positive = highest score (fallback to first doc if scores missing)
            if (
                isinstance(scores, list)
                and len(scores) == len(docs)
                and len(scores) > 0
            ):
                pos_idx = max(range(len(scores)), key=lambda i: scores[i])
            else:
                pos_idx = 0

            positives.append(docs[pos_idx])
            negs = _select_negatives_per_row(
                docs, scores if isinstance(scores, list) else [], num_explicit_negatives
            )
            for i, neg in enumerate(negs):
                negatives_matrix[i].append(neg)

        output = {"query": qs, "positive": positives}
        for i, col in enumerate(negative_cols):
            output[col] = negatives_matrix[i]
        return output

    mapped = train_proc.map(to_triplets, batched=True)

    # Keep only the columns in the expected order so the collator/loss interpret correctly
    ordered_cols = ["query", "positive", *negative_cols]
    keep_cols = [c for c in ordered_cols if c in mapped.column_names]
    mapped = mapped.select_columns(keep_cols)

    # Filter empty rows
    def _row_ok(ex):
        if not ex["query"] or not ex["positive"]:
            return False
        for col in negative_cols:
            if col in ex and not ex[col]:
                return False
        return True

    mapped = mapped.filter(_row_ok)

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


def main():
    # Hyperparameters for a quick trial
    train_batch_size = 4
    num_epochs = 3
    learning_rate = 4e-5
    query_regularizer_weight = 2e-4
    document_regularizer_weight = 9e-4
    max_seq_length = 256
    num_explicit_negatives = 8

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
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=query_regularizer_weight,
        document_regularizer_weight=document_regularizer_weight,
    )

    # 4) Evaluator: NanoBEIR (lightweight retrieval metrics)
    evaluator = evaluation.SparseNanoBEIREvaluator(
        dataset_names=["scifact", "msmarco", "touche2020", "scidocs", "nfcorpus"],
        batch_size=train_batch_size,
    )

    # 5) Training arguments
    run_name = f"splade-{short_model_name}-lighton-msmarco-triplets"
    args = SparseEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.2,
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
        save_total_limit=2,
        logging_steps=10,
        run_name=run_name,  # Used by W&B if installed
        seed=42,
    )

    args.set_lr_scheduler("cosine")

    # 6) Create the trainer & train
    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
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
