import torch
from typing import List, Dict, Any, Tuple
import ast
import datasets
import logging

logger = logging.getLogger(__name__)


class KDProcessing:
    """Dataset processing class for knowledge distillation training.
    FROM PyLate: https://github.com/lightonai/pylate/tree/main
    Parameters
    ----------
    queries
        Queries dataset.
    documents
        Documents dataset.
    split
        Split to use for the queries and documents datasets. Used only if the queries and documents are of type `datasets.DatasetDict`.
    n_ways
        Number of scores to keep for the distillation.

    Examples
    --------
    >>> from datasets import load_dataset
    >>> from pylate import utils

    >>> train = load_dataset(
    ...    path="lightonai/lighton-ms-marco-mini",
    ...    name="train",
    ...    split="train",
    ... )

    >>> queries = load_dataset(
    ...    path="lightonai/lighton-ms-marco-mini",
    ...    name="queries",
    ...    split="train",
    ... )

    >>> documents = load_dataset(
    ...    path="lightonai/lighton-ms-marco-mini",
    ...    name="documents",
    ...    split="train",
    ... )

    >>> train.set_transform(
    ...    utils.KDProcessing(
    ...        queries=queries, documents=documents
    ...    ).transform,
    ... )

    >>> for sample in train:
    ...     assert "documents" in sample and isinstance(sample["documents"], list)
    ...     assert "query" in sample and isinstance(sample["query"], str)
    ...     assert "scores" in sample and isinstance(sample["scores"], list)

    """

    def __init__(
        self,
        queries: datasets.Dataset | datasets.DatasetDict,
        documents: datasets.Dataset | datasets.DatasetDict,
        split: str = "train",
        n_ways: int = 32,
    ) -> None:
        if isinstance(queries, datasets.DatasetDict):
            self.queries = queries[split]
        else:
            self.queries = queries

        if isinstance(documents, datasets.DatasetDict):
            self.documents = documents[split]
        else:
            self.documents = documents

        self.n_ways = n_ways

        self.queries_index = {
            query_id: i for i, query_id in enumerate(iterable=self.queries["query_id"])
        }

        self.documents_index = {
            document_id: i
            for i, document_id in enumerate(iterable=self.documents["document_id"])
        }

    def transform(self, examples: dict) -> dict:
        """Update the input dataset with the queries and documents."""
        if isinstance(examples["scores"][0], str):
            examples["scores"] = [
                ast.literal_eval(node_or_string=score) for score in examples["scores"]
            ]

        examples["scores"] = [score[: self.n_ways] for score in examples["scores"]]

        if isinstance(examples["document_ids"][0], str):
            examples["document_ids"] = [
                ast.literal_eval(node_or_string=document_ids)
                for document_ids in examples["document_ids"]
            ]

        examples["document_ids"] = [
            document_ids[: self.n_ways] for document_ids in examples["document_ids"]
        ]

        examples["query"] = [
            self.queries[self.queries_index[query_id]]["text"]
            for query_id in examples["query_id"]
        ]

        examples["documents"] = []
        for document_ids in examples["document_ids"]:
            documents = []
            for document_id in document_ids:
                try:
                    documents.append(
                        self.documents[self.documents_index[document_id]]["text"]
                    )
                except KeyError:
                    documents.append("")
                    logger.warning(f"Unable to find document: {document_id}")

            examples["documents"].append(documents)

        return examples

    def map(self, example: dict) -> dict:
        """Process a single example.

        Parameters
        ----------
        example
            Example to process.

        Examples
        --------
        >>> from datasets import load_dataset
        >>> from pylate import utils

        >>> train = load_dataset(
        ...    path="lightonai/lighton-ms-marco-mini",
        ...    name="train",
        ...    split="train",
        ... )

        >>> queries = load_dataset(
        ...    path="lightonai/lighton-ms-marco-mini",
        ...    name="queries",
        ...    split="train",
        ... )

        >>> documents = load_dataset(
        ...    path="lightonai/lighton-ms-marco-mini",
        ...    name="documents",
        ...    split="train",
        ... )

        >>> train = train.map(
        ...    utils.KDProcessing(
        ...        queries=queries, documents=documents
        ...    ).map,
        ... )

        >>> for sample in train:
        ...     assert "documents" in sample and isinstance(sample["documents"], list)
        ...     assert "query" in sample and isinstance(sample["query"], str)
        ...     assert "scores" in sample and isinstance(sample["scores"], list)


        """
        if isinstance(example["scores"], str):
            example["scores"] = ast.literal_eval(node_or_string=example["scores"])

        example["scores"] = example["scores"][: self.n_ways]

        if isinstance(example["document_ids"], str):
            example["document_ids"] = ast.literal_eval(
                node_or_string=example["document_ids"]
            )

        example["document_ids"] = example["document_ids"][: self.n_ways]

        processed_example = {
            "scores": example["scores"],
            "query": self.queries[self.queries_index[example["query_id"]]]["text"],
        }

        documents = []
        for document_id in example["document_ids"]:
            try:
                documents.append(
                    self.documents[self.documents_index[document_id]]["text"]
                )
            except KeyError:
                documents.append("")
                logger.warning(f"Unable to find document: {document_id}")

        processed_example["documents"] = documents

        return processed_example


class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        queries = [item["query"] for item in batch]
        positives = [item["pos"][0] for item in batch]
        negatives = [item["neg"][0] for item in batch]

        # Tokenize queries
        query_encodings = self.tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            return_tensors="pt",  # Return PyTorch tensors
        )

        # Tokenize positive and negative documents
        all_docs = positives + negatives
        doc_encodings = self.tokenizer(
            all_docs,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Split doc encodings back into positives and negatives
        pos_doc_encodings = {
            key: val[: len(batch)] for key, val in doc_encodings.items()
        }
        neg_doc_encodings = {
            key: val[len(batch) :] for key, val in doc_encodings.items()
        }

        # Combine positive and negative documents
        # Shape: [batch_size, 2, seq_length]
        doc_input_ids = torch.stack(
            [
                pos_doc_encodings["input_ids"],
                neg_doc_encodings["input_ids"],
            ],
            dim=1,
        )
        doc_attention_mask = torch.stack(
            [
                pos_doc_encodings["attention_mask"],
                neg_doc_encodings["attention_mask"],
            ],
            dim=1,
        )

        return {
            "query_input_ids": query_encodings["input_ids"],
            "query_attention_mask": query_encodings["attention_mask"],
            "doc_input_ids": doc_input_ids,
            "doc_attention_mask": doc_attention_mask,
        }


class MultipleNegativesCollateFn:
    def __init__(self, tokenizer, num_negatives: int = 2):
        self.tokenizer = tokenizer
        self.num_negatives = num_negatives
        self.max_length = self.tokenizer.model_max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        queries = [item["query"] for item in batch]
        positives = [item["pos"][0] for item in batch]

        negatives = []
        for item in batch:
            item_negs = item["neg"][: self.num_negatives]
            # If not enough negatives, pad with copies of the first negative
            if len(item_negs) < self.num_negatives:
                padding_needed = self.num_negatives - len(item_negs)
                item_negs = item_negs + [item_negs[0]] * padding_needed
            negatives.append(item_negs)

        # Tokenize queries
        query_encodings = self.tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize positive and negative documents
        pos_encodings = self.tokenizer(
            positives,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        neg_encodings = self.tokenizer(
            [neg for neg_list in negatives for neg in neg_list],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        pos_doc_encodings = {
            key: val.unsqueeze(1) for key, val in pos_encodings.items()
        }

        neg_doc_encodings = {
            key: val.reshape(len(batch), self.num_negatives, -1)
            for key, val in neg_encodings.items()
        }

        # Combine positive and negative documents
        # Shape: [batch_size, 1 + num_negatives, seq_length]
        doc_input_ids = torch.cat(
            [
                pos_doc_encodings["input_ids"],
                neg_doc_encodings["input_ids"],
            ],
            dim=1,
        )
        doc_attention_mask = torch.cat(
            [
                pos_doc_encodings["attention_mask"],
                neg_doc_encodings["attention_mask"],
            ],
            dim=1,
        )

        return (
            query_encodings["input_ids"],
            query_encodings["attention_mask"],
            doc_input_ids,
            doc_attention_mask,
        )


class MultipleNegativesDistilCollateFn:
    def __init__(self, tokenizer, num_negatives: int = 2):
        self.tokenizer = tokenizer
        self.num_negatives = num_negatives
        self.max_length = self.tokenizer.model_max_length

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Extract queries
        queries = [item["query"] for item in batch]

        # Extract positive texts and teacher scores
        positives = [item["pos"][0]["text"] for item in batch]
        pos_scores = [item["pos"][0]["score"] for item in batch]

        negatives = []
        neg_scores = []
        for item in batch:
            item_negs = item["neg"][: self.num_negatives]
            if len(item_negs) < self.num_negatives:
                padding_needed = self.num_negatives - len(item_negs)
                item_negs = item_negs + [item_negs[0]] * padding_needed
            negatives.append([neg["text"] for neg in item_negs])
            neg_scores.append([neg["score"] for neg in item_negs])

        query_encodings = self.tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        pos_encodings = self.tokenizer(
            positives,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        neg_texts = [neg for neg_list in negatives for neg in neg_list]
        neg_encodings = self.tokenizer(
            neg_texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Reshape positive encodings: (batch_size, 1, seq_length)
        pos_doc_encodings = {
            key: val.unsqueeze(1) for key, val in pos_encodings.items()
        }

        # Reshape negative encodings: (batch_size, num_negatives, seq_length)
        neg_doc_encodings = {
            key: val.reshape(len(batch), self.num_negatives, -1)
            for key, val in neg_encodings.items()
        }

        # Combine positive and negative documents along dimension 1
        # Final shape: [batch_size, 1 + num_negatives, seq_length]
        doc_input_ids = torch.cat(
            [pos_doc_encodings["input_ids"], neg_doc_encodings["input_ids"]],
            dim=1,
        )
        doc_attention_mask = torch.cat(
            [pos_doc_encodings["attention_mask"], neg_doc_encodings["attention_mask"]],
            dim=1,
        )

        teacher_scores_list = []
        for pos_score, neg_score_list in zip(pos_scores, neg_scores):
            if not isinstance(pos_score, (int, float)):
                raise ValueError(f"Invalid pos_score: {pos_score}")
            if any(not isinstance(s, (int, float)) for s in neg_score_list):
                raise ValueError(f"Invalid neg_score in {neg_score_list}")

            teacher_scores_list.append([pos_score] + neg_score_list)
        teacher_scores = torch.tensor(teacher_scores_list, dtype=torch.float)

        return (
            query_encodings["input_ids"],
            query_encodings["attention_mask"],
            doc_input_ids,
            doc_attention_mask,
            teacher_scores,
        )


class KDProcessingCollateFn:
    def __init__(self, tokenizer, num_negatives: int = 32):
        """
        Initialize the collator with a tokenizer and number of negatives.
        Also modified from PyLate collate function: https://github.com/lightonai/pylate/tree/main/pylate/utils
        Args:
            tokenizer: The tokenizer to use for encoding texts.
            num_negatives (int): Number of negative documents per example.
        """
        self.tokenizer = tokenizer
        self.num_negatives = num_negatives
        self.max_length = self.tokenizer.model_max_length

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Constructed to work in conjunction with LightOn datasets and PyLate KDProcessing.

        See KDProcessing docstring for information on dataset transformation.

        Args:
            batch: List of dictionaries, each containing 'query', 'documents', and 'scores'.

        Returns:
            Tuple of tensors: query_input_ids, query_attention_mask, doc_input_ids,
            doc_attention_mask, and teacher_scores.
            (Matches output format of MultipleNegativesDistillCollateFn)
        """
        queries = [item["query"] for item in batch]

        positives = []
        pos_scores = []

        negatives = []
        neg_scores = []
        for item in batch:
            docs = item["documents"]
            scores = item["scores"]
            if len(docs) == 0:
                raise ValueError("Each example must have at least one document.")
            if len(docs) != len(scores):
                raise ValueError("Documents and scores must have the same length.")

            if len(scores) == 1:
                pos_idx = 0
            else:
                pos_idx = scores.index(max(scores))  # Index of highest score
            positive = docs[pos_idx]
            pos_score = scores[pos_idx]
            positives.append(positive)
            pos_scores.append(pos_score)

            # Collect negatives: all documents except the positive, up to num_negatives
            neg_indices = [i for i in range(len(docs)) if i != pos_idx]
            item_negs = [docs[i] for i in neg_indices][: self.num_negatives]
            item_neg_scores = [scores[i] for i in neg_indices][: self.num_negatives]

            # Pad negatives if needed
            if len(item_negs) < self.num_negatives:
                padding_needed = self.num_negatives - len(item_negs)
                # Use last negative if available, else positive
                pad_doc = item_negs[-1] if item_negs else positive
                pad_score = item_neg_scores[-1] if item_neg_scores else pos_score
                item_negs += [pad_doc] * padding_needed
                item_neg_scores += [pad_score] * padding_needed

            negatives.append(item_negs)
            neg_scores.append(item_neg_scores)

        query_encodings = self.tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        pos_encodings = self.tokenizer(
            positives,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        neg_texts = [neg for neg_list in negatives for neg in neg_list]
        neg_encodings = self.tokenizer(
            neg_texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        pos_doc_encodings = {
            key: val.unsqueeze(1) for key, val in pos_encodings.items()
        }

        neg_doc_encodings = {
            key: val.reshape(len(batch), self.num_negatives, -1)
            for key, val in neg_encodings.items()
        }

        doc_input_ids = torch.cat(
            [pos_doc_encodings["input_ids"], neg_doc_encodings["input_ids"]],
            dim=1,
        )
        doc_attention_mask = torch.cat(
            [pos_doc_encodings["attention_mask"], neg_doc_encodings["attention_mask"]],
            dim=1,
        )

        teacher_scores_list = []
        for pos_score, neg_score_list in zip(pos_scores, neg_scores):
            if not isinstance(pos_score, (int, float)):
                raise ValueError(f"Invalid pos_score: {pos_score}")
            if any(not isinstance(s, (int, float)) for s in neg_score_list):
                raise ValueError(f"Invalid neg_score in {neg_score_list}")
            teacher_scores_list.append([pos_score] + neg_score_list)
        teacher_scores = torch.tensor(teacher_scores_list, dtype=torch.float)

        return (
            query_encodings["input_ids"],
            query_encodings["attention_mask"],
            doc_input_ids,
            doc_attention_mask,
            teacher_scores,
        )
