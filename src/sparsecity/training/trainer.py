from typing import Dict, Optional, List, Tuple
from jaxtyping import Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F
from .grad_cache import gc_backward_and_zero_grad, RandContext


# @torch.compile(mode="default")
def train_step_mse(
    model: nn.Module,
    query_input_ids: torch.Tensor,
    query_attention_mask: torch.Tensor,
    doc_input_ids: torch.Tensor,
    doc_attention_mask: torch.Tensor,
    lambda_t_d: torch.Tensor,
    lambda_t_q: torch.Tensor,
    device: torch.device,
    temperature: torch.Tensor,
    mse_weight: Optional[torch.Tensor] = None,
    teacher_scores: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
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
    )
    # Split the embeddings back into queries and documents
    query_embeddings = combined_embeddings[:batch_size]
    doc_embeddings = combined_embeddings[batch_size:].reshape(batch_size, num_docs, -1)

    scores = torch.sum(query_embeddings.unsqueeze(1) * doc_embeddings, dim=-1)
    scores = scores / temperature
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
    anti_zero = torch.clamp(
        torch.reciprocal(torch.sum(query_embeddings) ** 2 + 1e-8)
        + torch.reciprocal(torch.sum(doc_embeddings) ** 2 + 1e-8),
        max=1.0,
    )
    # query_sum = torch.sum(torch.abs(query_embeddings))  # L1 norm to avoid cancellation
    # doc_sum = torch.sum(torch.abs(doc_embeddings))
    # anti_zero = torch.log1p(1.0 / (query_sum + 1e-4)) + torch.log1p(
    #     1.0 / (doc_sum + 1e-4)
    # )
    teacher_pos = teacher_scores[:, 0]  # Positive teacher score
    teacher_neg = teacher_scores[:, 1:]  # Negative teacher scores
    student_pos = scores[:, 0] / temperature  # Positive student score
    student_neg = scores[:, 1:] / temperature  # Negative student scores

    teacher_margins = (
        teacher_pos.unsqueeze(1) - teacher_neg
    )  # shape: (batch_size, num_negatives)
    student_margins = (
        student_pos.unsqueeze(1) - student_neg
    )  # shape: (batch_size, num_negatives)
    margin_mse_loss = F.mse_loss(student_margins, teacher_margins)

    # teacher_logits = teacher_scores / temperature
    # student_logits = scores / temperature

    # log_p_s = F.log_softmax(student_logits, dim=-1)
    # log_p_t = F.log_softmax(teacher_logits, dim=-1)

    # kl_loss = F.kl_div(log_p_s, log_p_t, reduction="batchmean", log_target=True)

    # Total loss
    total_loss = triplet_loss + flops + anti_zero + margin_mse_loss

    # Backward pass
    total_loss.backward()
    # optimizer.step()

    metrics = {}

    metrics["loss"] = total_loss
    metrics["triplet_loss"] = triplet_loss
    metrics["margin_mse_loss"] = margin_mse_loss
    metrics["flops_loss"] = flops
    metrics["anti_zero_loss"] = anti_zero

    metrics["query_sparsity"] = (query_embeddings == 0).float().mean()
    metrics["doc_sparsity"] = (doc_embeddings == 0).float().mean()

    query_non_zero_vals = query_embeddings[query_embeddings != 0]
    doc_non_zero_vals = doc_embeddings[doc_embeddings != 0]

    metrics["query_min_non_zero"] = (
        query_non_zero_vals.abs().min()
        if query_non_zero_vals.numel() > 0
        else torch.tensor(0.0, device=device)
    )
    metrics["doc_min_non_zero"] = (
        doc_non_zero_vals.abs().min()
        if doc_non_zero_vals.numel() > 0
        else torch.tensor(0.0, device=device)
    )
    metrics["query_median_non_zero"] = (
        torch.median(query_non_zero_vals.abs())
        if query_non_zero_vals.numel() > 0
        else torch.tensor(0.0, device=device)
    )
    metrics["doc_median_non_zero"] = (
        torch.median(doc_non_zero_vals.abs())
        if doc_non_zero_vals.numel() > 0
        else torch.tensor(0.0, device=device)
    )

    metrics["avg_query_non_zero_count"] = query_non_zero_vals.numel() / batch_size
    metrics["avg_doc_non_zero_count"] = doc_non_zero_vals.numel() / (
        batch_size * num_docs
    )

    # metrics["query_non_zero_count"] = query_non_zero_vals.numel()
    # metrics["doc_non_zero_count"] = doc_non_zero_vals.numel()

    return metrics


# @torch.compile(mode="default")
def train_step_kldiv(
    model: nn.Module,
    query_input_ids: torch.Tensor,
    query_attention_mask: torch.Tensor,
    doc_input_ids: torch.Tensor,
    doc_attention_mask: torch.Tensor,
    lambda_t_d: torch.Tensor,
    lambda_t_q: torch.Tensor,
    device: torch.device,
    temperature: torch.Tensor,
    mse_weight: Optional[torch.Tensor] = None,
    teacher_scores: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
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
    )
    # Split the embeddings back into queries and documents
    query_embeddings = combined_embeddings[:batch_size]
    doc_embeddings = combined_embeddings[batch_size:].reshape(batch_size, num_docs, -1)

    scores = torch.sum(query_embeddings.unsqueeze(1) * doc_embeddings, dim=-1)
    scores = scores / temperature
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
    anti_zero = torch.clamp(
        torch.reciprocal(torch.sum(query_embeddings) ** 2 + 1e-8)
        + torch.reciprocal(torch.sum(doc_embeddings) ** 2 + 1e-8),
        max=1.0,
    )
    teacher_logits = teacher_scores / temperature
    student_logits = scores / temperature

    log_p_s = F.log_softmax(student_logits, dim=-1)
    log_p_t = F.log_softmax(teacher_logits, dim=-1)

    kl_loss = F.kl_div(log_p_s, log_p_t, reduction="batchmean", log_target=True)

    # Total loss
    total_loss = triplet_loss + flops + anti_zero + kl_loss

    # Backward pass
    total_loss.backward()
    # optimizer.step()

    metrics = {}

    metrics["loss"] = total_loss
    metrics["triplet_loss"] = triplet_loss
    metrics["margin_mse_loss"] = kl_loss
    metrics["flops_loss"] = flops
    metrics["anti_zero_loss"] = anti_zero

    metrics["query_sparsity"] = (query_embeddings == 0).float().mean()
    metrics["doc_sparsity"] = (doc_embeddings == 0).float().mean()

    query_non_zero_vals = query_embeddings[query_embeddings != 0]
    doc_non_zero_vals = doc_embeddings[doc_embeddings != 0]

    metrics["query_min_non_zero"] = (
        query_non_zero_vals.abs().min()
        if query_non_zero_vals.numel() > 0
        else torch.tensor(0.0, device=device)
    )
    metrics["doc_min_non_zero"] = (
        doc_non_zero_vals.abs().min()
        if doc_non_zero_vals.numel() > 0
        else torch.tensor(0.0, device=device)
    )
    metrics["query_median_non_zero"] = (
        torch.median(query_non_zero_vals.abs())
        if query_non_zero_vals.numel() > 0
        else torch.tensor(0.0, device=device)
    )
    metrics["doc_median_non_zero"] = (
        torch.median(doc_non_zero_vals.abs())
        if doc_non_zero_vals.numel() > 0
        else torch.tensor(0.0, device=device)
    )

    metrics["avg_query_non_zero_count"] = query_non_zero_vals.numel() / batch_size
    metrics["avg_doc_non_zero_count"] = doc_non_zero_vals.numel() / (
        batch_size * num_docs
    )

    # metrics["query_non_zero_count"] = query_non_zero_vals.numel()
    # metrics["doc_non_zero_count"] = doc_non_zero_vals.numel()

    return metrics


# @torch.compile(mode="max-autotune") # Kept commented as in source _vectorized version
def train_step_kldiv_ibn(  # Refactored from train_step_kldiv_ibn_vectorized
    model: nn.Module,
    query_input_ids: torch.Tensor,  # Shape: [B, Lq]
    query_attention_mask: torch.Tensor,  # Shape: [B, Lq]
    doc_input_ids: torch.Tensor,  # Shape: [B, n_docs_per_query, Ld]
    doc_attention_mask: torch.Tensor,  # Shape: [B, n_docs_per_query, Ld]
    lambda_t_d: torch.Tensor,
    lambda_t_q: torch.Tensor,
    device: torch.device,
    temperature_ce: torch.Tensor,  # Temperature for CrossEntropy loss
    temperature_kl: torch.Tensor,  # Temperature for KL divergence loss
    mini_batch: int,  # Mini-batch size for GradCache
    teacher_scores: Optional[
        torch.Tensor
    ] = None,  # Shape: [B, n_docs_per_query], aligns with doc_input_ids
    mse_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    # if torch.cuda.is_available(): torch.compiler.cudagraph_mark_step_begin()
    model.train()

    B, n_docs_per_query, Ld = doc_input_ids.shape
    embedding_dim: Optional[int] = None  # Will be inferred after the first model pass

    # --- Pass 1: Embeddings with no_grad (for GradCache) ---
    # This pass computes query and document embeddings in mini-batches without storing
    # the computation graph for activations, saving memory. RNG states are saved
    # to ensure consistency (e.g., for dropout) during the recomputation pass.
    q_emb_chunks, d_emb_chunks = [], []
    pass_1_rng_states: List[RandContext] = []

    for start_idx in range(0, B, mini_batch):
        sl = slice(start_idx, min(start_idx + mini_batch, B))
        current_mini_batch_size = sl.stop - sl.start

        ids_q_mb = query_input_ids[sl]
        mask_q_mb = query_attention_mask[sl]
        ids_d_mb = doc_input_ids[sl].reshape(
            current_mini_batch_size * n_docs_per_query, Ld
        )
        mask_d_mb = doc_attention_mask[sl].reshape(
            current_mini_batch_size * n_docs_per_query, Ld
        )

        ids_comb_mb = torch.cat([ids_q_mb, ids_d_mb], dim=0)
        mask_comb_mb = torch.cat([mask_q_mb, mask_d_mb], dim=0)

        # RandContext captures and restores RNG state for operations like dropout
        # ensuring consistency between this no_grad pass and the later recomputation pass.
        rng_ctx_for_mb = RandContext(
            ids_comb_mb
        )  # Needs a tensor on the correct device
        pass_1_rng_states.append(rng_ctx_for_mb)

        with (
            torch.no_grad(),
            rng_ctx_for_mb,
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            emb_mb = model(input_ids=ids_comb_mb, attention_mask=mask_comb_mb)
            if embedding_dim is None:
                embedding_dim = emb_mb.size(-1)

        q_emb_mb = emb_mb[:current_mini_batch_size]
        d_emb_mb = emb_mb[current_mini_batch_size:].reshape(
            current_mini_batch_size, n_docs_per_query, embedding_dim
        )

        q_emb_chunks.append(q_emb_mb)  # Detached due to torch.no_grad()
        d_emb_chunks.append(d_emb_mb)

    # Concatenate chunks and enable gradient tracking for these intermediate embeddings
    q_emb = (
        torch.cat(q_emb_chunks, 0).to(torch.float32).requires_grad_()
    )  # Shape: [B, embedding_dim]
    d_emb = (
        torch.cat(d_emb_chunks, 0).to(torch.float32).requires_grad_()
    )  # Shape: [B, n_docs, embedding_dim]

    # --- Score Calculation (In-Batch Negatives) ---
    # For each query q_i, the positive is d_emb[i,0,:] (the first document associated with q_i).
    # Negatives are all documents from *other* queries in the batch, i.e., d_emb[j,k,:] for all j != i and all k.

    # Flatten all document embeddings: [B * n_docs_per_query, embedding_dim]
    d_all_docs_flat = d_emb.reshape(B * n_docs_per_query, embedding_dim)

    # Full score matrix: query_i vs all_doc_j
    # scores_q_vs_all_docs[i,j] = score(q_emb[i], d_all_docs_flat[j])
    scores_q_vs_all_docs = torch.matmul(
        q_emb, d_all_docs_flat.T
    )  # Shape: [B, B * n_docs_per_query]

    # Labels for CrossEntropyLoss: positive is always at index 0 after gathering
    labels = torch.zeros(B, dtype=torch.long, device=device)

    # Indices of positive documents for each query in d_all_docs_flat
    # Positive for query `i` is `d_emb[i,0,:]`, located at `d_all_docs_flat[i * n_docs_per_query, :]`
    pos_doc_indices_flat = torch.arange(
        start=0, end=B * n_docs_per_query, step=n_docs_per_query, device=device
    )
    pos_item_cols = pos_doc_indices_flat.unsqueeze(1)  # Shape: [B, 1]

    # Indices of batch negative documents
    # Identify owner query for each doc in d_all_docs_flat
    doc_query_owner_idx = (
        torch.arange(B * n_docs_per_query, device=device) // n_docs_per_query
    )  # Shape: [B*n_docs_per_query]

    # Expand current query indices for broadcasting: [B, 1]
    current_query_indices_expanded = torch.arange(B, device=device).unsqueeze(1)

    # Mask: is_batch_negative_mask[i,j] is True if d_all_docs_flat[j] is a negative for query_i
    # This means doc_query_owner_idx[j] should not be equal to i.
    is_batch_negative_mask = (
        doc_query_owner_idx.unsqueeze(0) != current_query_indices_expanded
    )  # Shape: [B, B*n_docs_per_query]

    # Get indices of all documents, replicated for each query row
    all_doc_indices_replicated = (
        torch.arange(B * n_docs_per_query, device=device).unsqueeze(0).expand(B, -1)
    )

    num_batch_negs_per_query = (B - 1) * n_docs_per_query
    batch_neg_item_cols = all_doc_indices_replicated[is_batch_negative_mask].reshape(
        B, num_batch_negs_per_query
    )

    # Combine positive and negative indices for gathering
    # Final scores tensor will have shape: [B, 1 (positive) + (B-1)*n_docs_per_query (batch negatives)]
    final_score_indices = torch.cat([pos_item_cols, batch_neg_item_cols], dim=1)

    # Gather the scores based on the constructed indices
    scores_for_ce = torch.gather(scores_q_vs_all_docs, 1, final_score_indices)
    scores_for_ce = scores_for_ce / temperature_ce

    # --- Loss Computations ---
    # 1. Triplet Loss (CrossEntropy with in-batch negatives)
    triplet_loss = F.cross_entropy(scores_for_ce, labels)

    # 2. FLOPs Regularization (L1 norm on embeddings)
    doc_l1_sum_abs_mean = torch.sum(
        d_all_docs_flat.abs(), dim=-1
    ).mean()  # L1 norm per doc, then mean
    query_l1_sum_abs_mean = torch.sum(
        q_emb.abs(), dim=-1
    ).mean()  # L1 norm per query, then mean
    flops_loss = lambda_t_d * doc_l1_sum_abs_mean + lambda_t_q * query_l1_sum_abs_mean

    # 3. Anti-Zero Loss
    q_total_sum_sq_inv = torch.reciprocal(
        q_emb.sum().pow(2) + 1e-8
    )  # Sum over all elements
    d_total_sum_sq_inv = torch.reciprocal(
        d_emb.sum().pow(2) + 1e-8
    )  # Sum over all elements
    anti_zero_loss = torch.clamp(q_total_sum_sq_inv + d_total_sum_sq_inv, max=1.0)

    # 4. KL Divergence Distillation Loss
    kl_loss = torch.tensor(0.0, device=device)
    if teacher_scores is not None:
        # teacher_scores shape: [B, n_docs_per_query]
        # Student scores for q_i vs its *own* n_docs_per_query documents: d_emb[i, :, :]
        student_row_logits = torch.einsum(
            "bv,bnv->bn", q_emb, d_emb
        )  # Shape: [B, n_docs_per_query]

        teacher_log_softmax = F.log_softmax(teacher_scores / temperature_kl, dim=-1)
        student_log_softmax = F.log_softmax(student_row_logits / temperature_kl, dim=-1)

        kl_loss = F.kl_div(
            student_log_softmax,
            teacher_log_softmax,
            reduction="batchmean",  # Averages over the batch dimension (B)
            log_target=True,
        )

        if teacher_scores.size(1) > 1:
            teacher_margins = teacher_scores[:, 0].unsqueeze(1) - teacher_scores[:, 1:]
            student_margins = (
                student_row_logits[:, 0].unsqueeze(1) - student_row_logits[:, 1:]
            )

            mse_loss = F.mse_loss(student_margins, teacher_margins) * mse_weight

    total_loss = triplet_loss + flops_loss + kl_loss + anti_zero_loss + mse_loss

    # --- Backward Pass using GradCache ---
    # The _recompute function is called by gc_backward_and_zero_grad for each mini-batch.
    # It re-runs the model for that mini-batch, this time with autograd enabled,
    # to reconstruct the activations needed for gradient calculation.
    # The `rng_ctx_for_mb_recompute` ensures that dropout (and other RNG-based ops)
    # behave identically to the first pass.
    def _recompute_for_gradcache(
        sl_mb: slice,
        rng_ctx_for_mb_recompute: RandContext,  # Provided by gc_backward_and_zero_grad from pass_1_rng_states
    ):
        current_mini_batch_size_recompute = sl_mb.stop - sl_mb.start

        # Prepare inputs for the current mini-batch slice
        ids_q_recompute = query_input_ids[sl_mb]
        mask_q_recompute = query_attention_mask[sl_mb]

        if embedding_dim is None:  # Should be set from pass-1
            raise ValueError(
                "Embedding dimension (embedding_dim) not set from pass-1 during recomputation."
            )

        ids_d_recompute = doc_input_ids[sl_mb].reshape(
            current_mini_batch_size_recompute * n_docs_per_query, Ld
        )
        mask_d_recompute = doc_attention_mask[sl_mb].reshape(
            current_mini_batch_size_recompute * n_docs_per_query, Ld
        )

        ids_comb_recompute = torch.cat([ids_q_recompute, ids_d_recompute], dim=0)
        mask_comb_recompute = torch.cat([mask_q_recompute, mask_d_recompute], dim=0)

        # Re-execute model forward pass for this mini-batch with gradient tracking enabled.
        # The provided rng_ctx_for_mb_recompute ensures consistent RNG.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            emb_recompute = model(
                input_ids=ids_comb_recompute,
                attention_mask=mask_comb_recompute,
            )

        q_recomputed = emb_recompute[:current_mini_batch_size_recompute]
        d_recomputed = emb_recompute[current_mini_batch_size_recompute:].reshape(
            current_mini_batch_size_recompute, n_docs_per_query, embedding_dim
        )
        return [
            q_recomputed,
            d_recomputed,
        ]  # Must return in same order as tensors in second argument to gc_backward_and_zero_grad

    # gc_backward_and_zero_grad handles the chunked backward pass.
    # It recomputes activations for each chunk via _recompute_for_gradcache,
    # then backpropagates gradients for that chunk, and finally zeros model gradients.
    gc_backward_and_zero_grad(
        loss=total_loss,
        cached_tensors=[
            q_emb,
            d_emb,
        ],  # Tensors from pass-1 for which grads are needed
        rng_states=pass_1_rng_states,  # List of RNG contexts from pass-1
        recompute_fn=_recompute_for_gradcache,
        model=model,
        mini_batch_size=mini_batch,
    )

    # --- Metrics ---
    metrics_dict: Dict[str, torch.Tensor] = {}
    with torch.no_grad():  # Metrics calculation should not contribute to graph
        metrics_dict["loss"] = total_loss.detach()
        metrics_dict["triplet_loss"] = triplet_loss.detach()
        metrics_dict["kl_loss"] = kl_loss.detach()
        metrics_dict["flops_loss"] = flops_loss.detach()
        metrics_dict["anti_zero_loss"] = anti_zero_loss.detach()
        metrics_dict["mse_loss"] = mse_loss.detach()
        q_abs = q_emb.abs()
        d_abs = d_emb.abs()
        is_q_nonzero = q_abs > 1e-9
        is_d_nonzero = d_abs > 1e-9

        q_nonzero_vals = q_abs[is_q_nonzero]
        d_nonzero_vals = d_abs[is_d_nonzero]

        metrics_dict["query_sparsity"] = (~is_q_nonzero).float().mean()
        metrics_dict["doc_sparsity"] = (~is_d_nonzero).float().mean()

        metrics_dict["query_min_non_zero"] = (
            q_nonzero_vals.min()
            if q_nonzero_vals.numel() > 0
            else torch.tensor(0.0, device=device)
        )
        metrics_dict["doc_min_non_zero"] = (
            d_nonzero_vals.min()
            if d_nonzero_vals.numel() > 0
            else torch.tensor(0.0, device=device)
        )
        metrics_dict["query_median_non_zero"] = (
            torch.median(q_nonzero_vals)
            if q_nonzero_vals.numel() > 0
            else torch.tensor(0.0, device=device)
        )
        metrics_dict["doc_median_non_zero"] = (
            torch.median(d_nonzero_vals)
            if d_nonzero_vals.numel() > 0
            else torch.tensor(0.0, device=device)
        )

        metrics_dict["avg_query_non_zero_count"] = (
            is_q_nonzero.sum() / B if B > 0 else torch.tensor(0.0, device=device)
        )
        metrics_dict["avg_doc_non_zero_count"] = (
            is_d_nonzero.sum() / (B * n_docs_per_query)
            if B > 0 and n_docs_per_query > 0
            else torch.tensor(0.0, device=device)
        )

    # if torch.cuda.is_available(): torch.compiler.cudagraph_mark_step_end()
    return metrics_dict
