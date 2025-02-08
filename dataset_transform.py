import argparse
import logging
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch
from plsfix import fix_text

# Configure logging to output to both console and a log file.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("processing.log", mode="a")],
)

model = SentenceTransformer(
    "dunzhang/stella_en_1.5B_v5",
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.float16},
).cuda()

query_prompt_name = "s2p_query"


def add_similarity_scores(batch):
    try:
        # Retrieve original texts
        queries = batch["query"]

        # Flatten the passages from pos and neg lists for all samples in the batch.
        passages_list = [
            text
            for pos_list, neg_list in zip(batch["pos"], batch["neg"])
            for text in (pos_list + neg_list)
        ]
        # Fix the texts for both queries and passages.
        fixed_passages = [fix_text(text) for text in passages_list]
        fixed_queries = [fix_text(text) for text in queries]

        # Precompute the total number of passages per sample and cumulative offsets.
        sample_lengths = [
            len(pos) + len(neg) for pos, neg in zip(batch["pos"], batch["neg"])
        ]
        offsets = [0]
        for length in sample_lengths:
            offsets.append(offsets[-1] + length)

        # Encode the fixed texts with the model.
        with torch.inference_mode():
            query_embeddings = model.encode(
                fixed_queries,
                prompt_name=query_prompt_name,
                convert_to_tensor=True,
                batch_size=64,  # Adjust if needed.
                show_progress_bar=False,
            )
            if fixed_passages:
                passage_embeddings = model.encode(
                    fixed_passages,
                    convert_to_tensor=True,
                    batch_size=64,  # Adjust if needed.
                    show_progress_bar=False,
                )
            else:
                passage_embeddings = None

        new_pos = []
        new_neg = []
        # Process each sample in the batch using the precomputed offsets.
        for i, (pos_list, neg_list) in enumerate(zip(batch["pos"], batch["neg"])):
            start = offsets[i]
            end = offsets[i + 1]
            n_total = sample_lengths[i]

            if n_total > 0:
                # Slice the passage embeddings corresponding to this sample.
                sample_passage_emb = passage_embeddings[start:end]
                # Compute cosine similarity between the query and its passages.
                sims = util.cos_sim(
                    query_embeddings[i].unsqueeze(0), sample_passage_emb
                )[0]
                sims = sims.tolist()  # Convert tensor to list.
                n_pos = len(pos_list)
                pos_sims = sims[:n_pos]
                neg_sims = sims[n_pos:]
            else:
                pos_sims, neg_sims = [], []

            # Slice the fixed texts for this sample.
            sample_fixed_texts = fixed_passages[start:end]
            fixed_pos = sample_fixed_texts[: len(pos_list)]
            fixed_neg = sample_fixed_texts[len(pos_list) :]

            # Build the new "pos" entries.
            new_pos.append(
                [
                    {"text": text, "score": score}
                    for text, score in zip(fixed_pos, pos_sims)
                ]
            )
            # Build and sort the "neg" entries.
            neg_items = [
                {"text": text, "score": score}
                for text, score in zip(fixed_neg, neg_sims)
            ]
            neg_items.sort(key=lambda x: x["score"], reverse=True)
            new_neg.append(neg_items)

        return {"pos": new_pos, "neg": new_neg}

    except Exception as e:
        logging.exception("Exception in add_similarity_scores: %s", e)
        raise


def main(args):
    try:
        logging.info("Loading dataset from %s", args.input)
        dataset = load_dataset(
            "json",
            data_files={"train": args.input},
            split="train",
            encoding="utf-8",
        )

        logging.info(
            "Mapping dataset with batch_size=%d and num_proc=%d",
            args.batch_size,
            args.num_proc,
        )
        dataset = dataset.map(
            add_similarity_scores,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
        )

        logging.info("Saving processed dataset to %s", args.output)
        dataset.to_json(args.output)
        logging.info("Processed dataset saved to %s", args.output)

    except Exception as e:
        logging.exception("Exception in main: %s", e)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add similarity scores to a multiple negatives dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to input jsonl.gz file",
        default="/root/data/msmarco_triplets/msmarco-triplets-stella.jsonl.gz",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output jsonl.gz file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for processing"
    )
    parser.add_argument(
        "--num_proc", type=int, default=1, help="Number of processes for dataset.map()"
    )
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logging.exception("Fatal error encountered: %s", e)
        exit(1)
