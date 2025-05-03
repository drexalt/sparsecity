from typing import Dict, Optional
from jaxtyping import Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F


# @torch.compile(mode="default")
# def train_step(
#     model: nn.Module,
#     query_input_ids: Int[torch.Tensor, "batch_size seq_len"],
#     query_attention_mask: Int[torch.Tensor, "batch_size seq_len"],
#     doc_input_ids: Int[torch.Tensor, "batch_size num_docs seq_len"],
#     doc_attention_mask: Int[torch.Tensor, "batch_size num_docs seq_len"],
#     top_k: int,
#     lambda_t_d: Float[torch.Tensor, ""],  # Scalar tensor
#     lambda_t_q: Float[torch.Tensor, ""],  # Scalar tensor
#     device: torch.device,
#     temperature: Float[torch.Tensor, ""],  # Scalar tensor
#     mse_weight: Float[torch.Tensor, ""],  # Scalar tensor
#     teacher_scores: Optional[Float[torch.Tensor, "batch_size num_docs"]] = None,
# ) -> Dict[str, Float[torch.Tensor, ""]]:
#     torch.compiler.cudagraph_mark_step_begin()
#     model.train()

#     # optimizer.zero_grad()

#     batch_size = query_input_ids.shape[0]
#     num_docs = doc_input_ids.shape[1]

#     # Combine queries and documents into a single batch
#     doc_input_ids_flat = doc_input_ids.reshape(-1, doc_input_ids.shape[-1])
#     doc_attention_mask_flat = doc_attention_mask.reshape(
#         -1, doc_attention_mask.shape[-1]
#     )

#     # Concatenate query and document inputs
#     combined_input_ids = torch.cat([query_input_ids, doc_input_ids_flat])
#     combined_attention_mask = torch.cat([query_attention_mask, doc_attention_mask_flat])

#     # Single forward pass for both queries and documents
#     combined_embeddings: Float[
#         torch.Tensor, "batch_size+batch_size*num_docs vocab_size"
#     ] = model(
#         input_ids=combined_input_ids,
#         attention_mask=combined_attention_mask,
#         top_k=top_k,
#     )
#     # Split the embeddings back into queries and documents
#     query_embeddings = combined_embeddings[:batch_size]
#     doc_embeddings = combined_embeddings[batch_size:].reshape(batch_size, num_docs, -1)

#     scores = torch.sum(query_embeddings.unsqueeze(1) * doc_embeddings, dim=-1)
#     scores = scores / temperature
#     # Create labels (assuming first document is positive)
#     labels = torch.zeros(batch_size, dtype=torch.long, device=device)

#     # Compute losses
#     triplet_loss = F.cross_entropy(scores, labels)

#     # Compute regularization terms
#     doc_flops = torch.sum(
#         torch.abs(doc_embeddings.reshape(-1, doc_embeddings.shape[-1])), dim=-1
#     ).mean()
#     query_l1 = torch.sum(torch.abs(query_embeddings), dim=-1).mean()
#     flops = lambda_t_d * doc_flops + lambda_t_q * query_l1

#     # Compute anti-zero loss
#     anti_zero = torch.clamp(
#         torch.reciprocal(torch.sum(query_embeddings) ** 2 + 1e-8)
#         + torch.reciprocal(torch.sum(doc_embeddings) ** 2 + 1e-8),
#         max=1.0,
#     )
#     # query_sum = torch.sum(torch.abs(query_embeddings))  # L1 norm to avoid cancellation
#     # doc_sum = torch.sum(torch.abs(doc_embeddings))
#     # anti_zero = torch.log1p(1.0 / (query_sum + 1e-4)) + torch.log1p(
#     #     1.0 / (doc_sum + 1e-4)
#     # )
#     teacher_pos = teacher_scores[:, 0]  # Positive teacher score
#     teacher_neg = teacher_scores[:, 1:]  # Negative teacher scores
#     student_pos = scores[:, 0]  # Positive student score
#     student_neg = scores[:, 1:]  # Negative student scores

#     teacher_margins = (
#         teacher_pos.unsqueeze(1) - teacher_neg
#     )  # shape: (batch_size, num_negatives)
#     student_margins = (
#         student_pos.unsqueeze(1) - student_neg
#     )  # shape: (batch_size, num_negatives)
#     margin_mse_loss = F.mse_loss(student_margins, teacher_margins)

#     # Total loss
#     total_loss = triplet_loss + flops + anti_zero + (mse_weight * margin_mse_loss)

#     # Backward pass
#     total_loss.backward()
#     # optimizer.step()

#     metrics = {}

#     metrics["loss"] = total_loss
#     metrics["triplet_loss"] = triplet_loss
#     metrics["margin_mse_loss"] = margin_mse_loss
#     metrics["flops_loss"] = flops
#     metrics["anti_zero_loss"] = anti_zero

#     metrics["query_sparsity"] = (query_embeddings == 0).float().mean()
#     metrics["doc_sparsity"] = (doc_embeddings == 0).float().mean()

#     query_non_zero_vals = query_embeddings[query_embeddings != 0]
#     doc_non_zero_vals = doc_embeddings[doc_embeddings != 0]

#     metrics["query_min_non_zero"] = (
#         query_non_zero_vals.abs().min()
#         if query_non_zero_vals.numel() > 0
#         else torch.tensor(0.0, device=device)
#     )
#     metrics["doc_min_non_zero"] = (
#         doc_non_zero_vals.abs().min()
#         if doc_non_zero_vals.numel() > 0
#         else torch.tensor(0.0, device=device)
#     )
#     metrics["query_median_non_zero"] = (
#         torch.median(query_non_zero_vals.abs())
#         if query_non_zero_vals.numel() > 0
#         else torch.tensor(0.0, device=device)
#     )
#     metrics["doc_median_non_zero"] = (
#         torch.median(doc_non_zero_vals.abs())
#         if doc_non_zero_vals.numel() > 0
#         else torch.tensor(0.0, device=device)
#     )

#     metrics["avg_query_non_zero_count"] = query_non_zero_vals.numel() / batch_size
#     metrics["avg_doc_non_zero_count"] = doc_non_zero_vals.numel() / (
#         batch_size * num_docs
#     )

#     # metrics["query_non_zero_count"] = query_non_zero_vals.numel()
#     # metrics["doc_non_zero_count"] = doc_non_zero_vals.numel()

#     return metrics


# @torch.compile(mode="default")
def train_step(
    model: nn.Module,
    query_input_ids: torch.Tensor,
    query_attention_mask: torch.Tensor,
    doc_input_ids: torch.Tensor,
    doc_attention_mask: torch.Tensor,
    top_k: int,
    lambda_t_d: torch.Tensor,
    lambda_t_q: torch.Tensor,
    device: torch.device,
    temperature: torch.Tensor,
    mse_weight: torch.Tensor,
    teacher_scores: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    torch.compiler.cudagraph_mark_step_begin()
    model.train()

    batch_size = query_input_ids.shape[0]
    num_docs = doc_input_ids.shape[1]

    # Combine queries and documents into a single batch
    doc_input_ids_flat = doc_input_ids.reshape(-1, doc_input_ids.shape[-1])
    doc_attention_mask_flat = doc_attention_mask.reshape(
        -1, doc_attention_mask.shape[-1]
    )
    combined_input_ids = torch.cat([query_input_ids, doc_input_ids_flat])
    combined_attention_mask = torch.cat([query_attention_mask, doc_attention_mask_flat])

    # Single forward pass for both queries and documents
    combined_outputs = model(
        input_ids=combined_input_ids,
        attention_mask=combined_attention_mask,
        top_k=top_k,
    )

    # Split outputs back into queries and documents
    query_outputs = {k: v[:batch_size] for k, v in combined_outputs.items()}
    doc_outputs_flat = {k: v[batch_size:] for k, v in combined_outputs.items()}
    doc_outputs = {
        "sparse_activations": doc_outputs_flat["sparse_activations"].reshape(
            batch_size, num_docs, -1
        ),
        "activations": doc_outputs_flat["activations"].reshape(
            batch_size, num_docs, -1
        ),
        "embeddings": doc_outputs_flat["embeddings"].reshape(
            batch_size, num_docs, top_k, -1
        ),
    }

    # Compute sparse scores (like original SPLADE)
    sparse_scores = (
        torch.sum(
            query_outputs["sparse_activations"].unsqueeze(1)
            * doc_outputs["sparse_activations"],
            dim=-1,
        )
        / temperature
    )
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    sparse_loss = F.cross_entropy(sparse_scores, labels)

    # Compute dense scores using contextual embeddings
    def compute_dense_scores(q_emb, q_idx, d_emb, d_idx):
        # Find matching terms between query and document
        matches = (
            q_idx.unsqueeze(2) == d_idx.unsqueeze(1)
        ).float()  # [batch_size, top_k_q, top_k_d]
        # Compute dot products for all pairs
        dot_products = torch.bmm(
            q_emb, d_emb.transpose(-1, -2)
        )  # [batch_size, top_k_q, top_k_d]
        # Sum dot products for matching terms
        dense_scores = (dot_products * matches).sum(dim=[1, 2])  # [batch_size]
        return dense_scores

    dense_scores = (
        torch.stack(
            [
                compute_dense_scores(
                    query_outputs["embeddings"],
                    query_outputs["activations"],
                    doc_outputs["embeddings"][:, i],
                    doc_outputs["activations"][:, i],
                )
                for i in range(num_docs)
            ],
            dim=1,
        )
        / temperature
    )
    dense_loss = F.cross_entropy(dense_scores, labels)

    anti_zero_loss = torch.clamp(
        torch.reciprocal(torch.sum(query_outputs["sparse_activations"]) ** 2 + 1e-8)
        + torch.reciprocal(torch.sum(doc_outputs["sparse_activations"]) ** 2 + 1e-8),
        max=1.0,
    )

    # Regularization (FLOPS-like loss on sparse activations)
    doc_flops = torch.sum(torch.abs(doc_outputs["sparse_activations"]), dim=-1).mean()
    query_flops = torch.sum(
        torch.abs(query_outputs["sparse_activations"]), dim=-1
    ).mean()
    flops_loss = lambda_t_d * doc_flops + lambda_t_q * query_flops

    # Distillation loss using both sparse and dense scores
    teacher_pos = teacher_scores[:, 0]
    teacher_neg = teacher_scores[:, 1:]
    sparse_pos, sparse_neg = sparse_scores[:, 0], sparse_scores[:, 1:]
    dense_pos, dense_neg = dense_scores[:, 0], dense_scores[:, 1:]

    teacher_margins = teacher_pos.unsqueeze(1) - teacher_neg
    sparse_margins = sparse_pos.unsqueeze(1) - sparse_neg
    dense_margins = dense_pos.unsqueeze(1) - dense_neg

    sparse_mse_loss = F.mse_loss(sparse_margins, teacher_margins)
    dense_mse_loss = F.mse_loss(dense_margins, teacher_margins)

    # Total loss: Ascending weights for sparse and dense components
    total_loss = (
        sparse_loss  # Ranking loss on sparse scores
        + dense_loss  # Ranking loss on dense scores
        + flops_loss  # Sparsity regularization
        + mse_weight * (sparse_mse_loss + dense_mse_loss)  # Distillation loss
        + anti_zero_loss
    )

    total_loss.backward()

    # Metrics
    metrics = {
        "total_loss": total_loss,
        "triplet_loss": sparse_loss,
        "dense_loss": dense_loss,
        "sparse_mse_loss": sparse_mse_loss,
        "dense_mse_loss": dense_mse_loss,
        "anti_zero_loss": anti_zero_loss,
        "flops_loss": flops_loss,
        "query_sparsity": (query_outputs["sparse_activations"] == 0).float().mean(),
        "doc_sparsity": (doc_outputs["sparse_activations"] == 0).float().mean(),
        "avg_query_non_zero_count": (query_outputs["sparse_activations"] != 0)
        .float()
        .sum(dim=-1)
        .mean(),
        "avg_doc_non_zero_count": (doc_outputs["sparse_activations"] != 0)
        .float()
        .sum(dim=-1)
        .mean(),
    }

    return metrics
