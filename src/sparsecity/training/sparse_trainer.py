from typing import Dict, Optional
from jaxtyping import Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.compile(mode="max-autotune")
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
    epsilon: torch.Tensor,
    teacher_scores: Optional[torch.Tensor] = None,
):
    torch.compiler.cudagraph_mark_step_begin()
    model.train()
    # optimizer.zero_grad()

    batch_size = query_input_ids.shape[0]
    num_docs = doc_input_ids.shape[1]

    # Combine queries and documents into a single batch
    doc_input_ids_flat = doc_input_ids.reshape(-1, doc_input_ids.shape[-1])
    doc_attention_mask_flat = doc_attention_mask.reshape(
        -1, doc_attention_mask.shape[-1]
    )

    # Concatenate query and document inputs
    combined_input_ids = torch.cat([query_input_ids, doc_input_ids_flat])
    combined_attention_mask = torch.cat([query_attention_mask, doc_attention_mask_flat])

    # Single forward pass for both queries and documents
    combined_embeddings = model(
        input_ids=combined_input_ids,
        attention_mask=combined_attention_mask,
        top_k=top_k,
    )
    # Split the embeddings back into queries and documents
    query_embeddings = combined_embeddings[:batch_size]
    doc_embeddings = combined_embeddings[batch_size:].reshape(batch_size, num_docs, -1)
    scores = torch.sum(query_embeddings.unsqueeze(1) * doc_embeddings, dim=-1)

    # Create labels (assuming first document is positive)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Compute losses
    triplet_loss = F.cross_entropy(scores, labels)

    # Compute regularization terms
    doc_flops = torch.sum(
        torch.abs(doc_embeddings.reshape(-1, doc_embeddings.shape[-1])), dim=-1
    ).mean()
    query_l1 = torch.sum(torch.abs(query_embeddings), dim=-1).mean()
    flops = lambda_t_d * doc_flops + lambda_t_q * query_l1

    # Compute anti-zero loss
    # anti_zero = 1 / (torch.sum(query_embeddings) ** 2 + 1e-8) + 1 / (
    #     torch.sum(doc_embeddings) ** 2 + 1e-8
    # )
    query_sum_squared = torch.sum(query_embeddings).square()
    doc_sum_squared = torch.sum(doc_embeddings).square()

    anti_zero = torch.reciprocal(query_sum_squared + epsilon) + torch.reciprocal(
        doc_sum_squared + epsilon
    )

    teacher_pos = teacher_scores[:, 0]  # Positive teacher score
    teacher_neg = teacher_scores[:, 1:]  # Negative teacher scores
    student_pos = scores[:, 0]  # Positive student score
    student_neg = scores[:, 1:]  # Negative student scores

    teacher_margins = (
        teacher_pos.unsqueeze(1) - teacher_neg
    )  # shape: (batch_size, num_negatives)
    student_margins = (
        student_pos.unsqueeze(1) - student_neg
    )  # shape: (batch_size, num_negatives)
    margin_mse_loss = F.mse_loss(student_margins, teacher_margins)

    # Total loss
    total_loss = triplet_loss + flops + anti_zero + margin_mse_loss

    # Backward pass
    total_loss.backward()
    # optimizer.step()

    return total_loss, triplet_loss, margin_mse_loss, flops, anti_zero
