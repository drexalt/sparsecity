import torch
from typing import List, Dict, Any, Tuple


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
