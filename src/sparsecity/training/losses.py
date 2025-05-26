import torch
from typing import Tuple, Optional, Dict
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


# Losses
def compute_flops(inputs: Tensor) -> Tensor:
    """
    Compute FLOPS regularization term.

    Args:
        inputs: Tensor of shape [batch_size, vocab_size]
    """
    return torch.sum(torch.square(torch.mean(torch.abs(inputs), dim=0)))


def compute_L1(inputs: Tensor) -> Tensor:
    """
    Compute L1 regularization term.

    Args:
        inputs: Tensor of shape [batch_size, vocab_size]
    """
    return torch.sum(torch.abs(inputs), dim=-1).mean()


# def compute_flops_sparse(inputs: Union[Tensor, torch.sparse.Tensor]) -> Tensor:
#     """
#     Compute FLOPS regularization term for sparse inputs.
#     Will work with both dense and sparse tensors.

#     Args:
#         inputs: Tensor or sparse tensor of shape [batch_size, vocab_size]
#     """
#     if isinstance(inputs, torch.sparse.Tensor):
#         inputs = inputs.to_dense()
#     return compute_flops(inputs)


# def compute_L1_sparse(inputs: Union[Tensor, torch.sparse.Tensor]) -> Tensor:
#     """
#     Compute L1 regularization term for sparse inputs.
#     Will work with both dense and sparse tensors.

#     Args:
#         inputs: Tensor or sparse tensor of shape [batch_size, vocab_size]
#     """
#     if isinstance(inputs, torch.sparse.Tensor):
#         inputs = inputs.to_dense()
#     return compute_L1(inputs)


def create_batch(batch: dict) -> dict:
    """Convert batch dictionary values to PyTorch tensors."""
    return {k: torch.tensor(v) for k, v in batch.items()}


def calculate_contrastive_loss(
    scores: torch.Tensor,  # Query-document scores [B, NumChoices]
    labels: torch.Tensor,  # Target labels [B]
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Computes CrossEntropy-based contrastive loss."""
    return F.cross_entropy(scores / temperature, labels)


def calculate_flops_regularization(
    query_embeddings: torch.Tensor,  # [B, V]
    doc_embeddings: torch.Tensor,  # [B, N, V] or [B*N, V]
    lambda_q: torch.Tensor,
    lambda_d: torch.Tensor,
) -> torch.Tensor:
    """Computes L1 regularization loss on embeddings (proxy for FLOPs)."""
    if doc_embeddings.dim() == 3:  # [B, N, V]
        doc_embeddings_flat = doc_embeddings.reshape(-1, doc_embeddings.shape[-1])
    else:  # Assumed [B*N, V]
        doc_embeddings_flat = doc_embeddings

    doc_l1 = torch.sum(doc_embeddings_flat.abs(), dim=-1).mean()
    query_l1 = torch.sum(query_embeddings.abs(), dim=-1).mean()
    return lambda_d * doc_l1 + lambda_q * query_l1


def calculate_anti_zero_loss(
    query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor
) -> torch.Tensor:
    """Computes anti-zero loss to prevent embedding collapse."""
    q_sum_sq_inv = torch.reciprocal(query_embeddings.sum().pow(2) + 1e-8)
    d_sum_sq_inv = torch.reciprocal(doc_embeddings.sum().pow(2) + 1e-8)
    return torch.clamp(q_sum_sq_inv + d_sum_sq_inv, max=1.0)


def calculate_kl_divergence_distillation(
    student_query_embeddings: torch.Tensor,  # [B, V]
    student_doc_embeddings: torch.Tensor,  # [B, N, V] (docs corresponding to each query)
    teacher_scores: torch.Tensor,  # [B, N] (teacher's scores for q_i vs d_i_j)
    temperature_kl: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Computes KL divergence loss between student and teacher score distributions."""
    if teacher_scores is None:
        return torch.tensor(0.0, device=device)

    # Student scores for q_i vs its own N documents
    student_row_logits = torch.einsum(
        "bv,bnv->bn", student_query_embeddings, student_doc_embeddings
    )

    student_log_softmax = F.log_softmax(student_row_logits / temperature_kl, dim=-1)
    teacher_log_softmax = F.log_softmax(teacher_scores / temperature_kl, dim=-1)

    return F.kl_div(
        student_log_softmax, teacher_log_softmax, reduction="batchmean", log_target=True
    )


def calculate_margin_mse_distillation(
    student_scores_per_item: torch.Tensor,  # [B,N] student scores (e.g. q_i vs d_i_j)
    teacher_scores_per_item: torch.Tensor,  # [B,N] teacher scores
    temperature: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Computes Margin MSE loss between student and teacher margins."""
    if teacher_scores_per_item is None:
        return torch.tensor(0.0, device=device)

    # Assumes first score in N is positive, rest are negative
    teacher_pos_scores = teacher_scores_per_item[:, 0]
    teacher_neg_scores = teacher_scores_per_item[:, 1:]
    student_pos_scores = student_scores_per_item[:, 0]
    student_neg_scores = student_scores_per_item[:, 1:]

    teacher_margins = (
        teacher_pos_scores.unsqueeze(1) - teacher_neg_scores
    ) / temperature
    student_margins = (
        student_pos_scores.unsqueeze(1) - student_neg_scores
    ) / temperature
    return F.mse_loss(student_margins, teacher_margins)


def contrastive_kd_loss(
    q_rep: Tensor,  # [B, D]
    d_rep_flat: Tensor,  # [(B*n_docs), D]
    n_docs_per_query: int,
    lambda_t_d: Tensor,
    lambda_t_q: Tensor,
    temperature_ce: Tensor,
    temperature_kl: Tensor,
    teacher_scores: Optional[Tensor] = None,  # [B, n_docs]
    mse_weight: Optional[Tensor] = None,
) -> Tensor:
    """Combined CE‑based contrastive loss + regularisation + KD."""

    device = q_rep.device
    B, D = q_rep.shape

    # --------------- reshape documents --------------------------------
    d_rep = d_rep_flat.view(B, n_docs_per_query, D)  # [B, n_docs, D]

    # --------------- build in‑batch negatives score matrix -------------
    # Full matrix of q_i · d_j  (d_j are all documents in the *batch*)
    scores_full = q_rep.float() @ d_rep_flat.float().T  # [B, B*n_docs]

    # Gather indices: first col = positive, rest = negatives -------------
    pos_idx_flat = torch.arange(
        0, B * n_docs_per_query, n_docs_per_query, device=device
    )  # [B]
    pos_cols = pos_idx_flat.unsqueeze(1)  # [B,1]

    owner = (
        torch.arange(B * n_docs_per_query, device=device) // n_docs_per_query
    )  # [B*n_docs]
    neg_mask = owner.unsqueeze(0) != torch.arange(B, device=device).unsqueeze(
        1
    )  # [B, B*n_docs]

    all_cols = torch.arange(B * n_docs_per_query, device=device).expand(
        B, -1
    )  # [B, B*n_docs]
    neg_cols = all_cols[neg_mask].view(B, -1)  # [B, (B-1)*n_docs]

    gather_cols = torch.cat([pos_cols, neg_cols], dim=1)  # [B, 1+(B-1)*n_docs]

    # Gather per‑query logits (still in fp32 for numerical stability)
    logits_ce = scores_full.gather(1, gather_cols)

    # Cross‑entropy labels: positive always in column 0
    labels_ce = torch.zeros(B, dtype=torch.long, device=device)

    # --- 1. Triplet / contrastive CE loss ------------------------------------
    triplet_loss = calculate_contrastive_loss(logits_ce, labels_ce, temperature_ce)

    # --- 2. FLOPs regularisation ---------------------------------------------
    flops_loss = calculate_flops_regularization(
        query_embeddings=q_rep,
        doc_embeddings=d_rep,  # helper handles 3‑D tensors transparently
        lambda_q=lambda_t_q,
        lambda_d=lambda_t_d,
    )

    # --- 3. Anti‑zero ---------------------------------------------------------
    anti_zero_loss = calculate_anti_zero_loss(q_rep, d_rep)

    # --- 4. Knowledge‑distillation (optional) ---------------------------------
    kl_loss = q_rep.new_tensor(0.0)
    mse_loss = q_rep.new_tensor(0.0)

    if teacher_scores is not None:
        kl_loss = calculate_kl_divergence_distillation(
            student_query_embeddings=q_rep,
            student_doc_embeddings=d_rep,
            teacher_scores=teacher_scores,
            temperature_kl=temperature_kl,
            device=device,
        )

        # Student logits against its own docs (shape [B, n_docs])
        student_logits = (q_rep.float().unsqueeze(1) * d_rep.float()).sum(-1)

        mse_raw = calculate_margin_mse_distillation(
            student_scores_per_item=student_logits,
            teacher_scores_per_item=teacher_scores,
            temperature=temperature_kl,
            device=device,
        )
        if mse_weight is not None:
            mse_loss = mse_raw * mse_weight
        else:
            mse_loss = mse_raw

    # --- Diagnostic sparsity / magnitude metrics -----------------------------
    q_abs = q_rep.abs()
    d_abs = d_rep.abs()

    is_q_nonzero = q_abs > 1e-9
    is_d_nonzero = d_abs > 1e-9

    q_nonzero_vals = q_abs[is_q_nonzero]
    d_nonzero_vals = d_abs[is_d_nonzero]

    query_sparsity = (~is_q_nonzero).float().mean()
    doc_sparsity = (~is_d_nonzero).float().mean()

    query_min_non_zero = (
        q_nonzero_vals.min() if q_nonzero_vals.numel() > 0 else q_rep.new_tensor(0.0)
    )
    doc_min_non_zero = (
        d_nonzero_vals.min() if d_nonzero_vals.numel() > 0 else q_rep.new_tensor(0.0)
    )

    query_median_non_zero = (
        torch.median(q_nonzero_vals)
        if q_nonzero_vals.numel() > 0
        else q_rep.new_tensor(0.0)
    )
    doc_median_non_zero = (
        torch.median(d_nonzero_vals)
        if d_nonzero_vals.numel() > 0
        else q_rep.new_tensor(0.0)
    )

    avg_query_non_zero_count = is_q_nonzero.sum() / B
    avg_doc_non_zero_count = is_d_nonzero.sum() / (B * n_docs_per_query)

    # --- Package everything ---------------------------------------------------
    parts: Dict[str, Tensor] = {
        # Loss constituents
        "triplet_loss": triplet_loss,
        "flops_loss": flops_loss,
        "anti_zero_loss": anti_zero_loss,
        "kl_loss": kl_loss,
        "mse_loss": mse_loss,
        # Extra metrics
        "query_sparsity": query_sparsity,
        "doc_sparsity": doc_sparsity,
        "query_min_non_zero": query_min_non_zero,
        "doc_min_non_zero": doc_min_non_zero,
        "query_median_non_zero": query_median_non_zero,
        "doc_median_non_zero": doc_median_non_zero,
        "avg_query_non_zero_count": avg_query_non_zero_count,
        "avg_doc_non_zero_count": avg_doc_non_zero_count,
    }

    total_loss: Tensor = triplet_loss + flops_loss + anti_zero_loss + kl_loss + mse_loss
    return total_loss, parts
