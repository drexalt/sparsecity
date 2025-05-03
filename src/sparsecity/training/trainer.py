from typing import Dict, Optional
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
    top_k: int,
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
        top_k=top_k,
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
    # teacher_pos = teacher_scores[:, 0]  # Positive teacher score
    # teacher_neg = teacher_scores[:, 1:]  # Negative teacher scores
    # student_pos = scores[:, 0]  # Positive student score
    # student_neg = scores[:, 1:]  # Negative student scores

    # teacher_margins = (
    #     teacher_pos.unsqueeze(1) - teacher_neg
    # )  # shape: (batch_size, num_negatives)
    # student_margins = (
    #     student_pos.unsqueeze(1) - student_neg
    # )  # shape: (batch_size, num_negatives)
    # margin_mse_loss = F.mse_loss(student_margins, teacher_margins)

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


# @torch.compile(mode="default")
def train_step_kldiv(
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
        top_k=top_k,
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


# @torch.compile(mode="default")
def train_step_kldiv_ibn(
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
    neg_mode: str = "row",
    mini_batch: int = 16,
    teacher_scores: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    torch.compiler.cudagraph_mark_step_begin()
    model.train()
    # optimizer.zero_grad()
    B, n, Ld = doc_input_ids.shape
    V = None  # (will know after fwd)
    world = (
        torch.distributed.get_world_size()
        if (neg_mode == "global" and torch.distributed.is_initialized())
        else 1
    )
    rank = torch.distributed.get_rank() if world > 1 else 0

    # ---------- pass-1: embed without graph --------------------------
    q_emb_chunks, d_emb_chunks = [], []
    rng_chunks = []

    for start in range(0, B, mini_batch):
        sl = slice(start, min(start + mini_batch, B))

        ids_q = query_input_ids[sl]
        mask_q = query_attention_mask[sl]

        ids_d = doc_input_ids[sl].reshape(-1, Ld)
        mask_d = doc_attention_mask[sl].reshape(-1, Ld)

        ids_comb = torch.cat([ids_q, ids_d], dim=0)
        mask_comb = torch.cat([mask_q, mask_d], dim=0)

        with torch.no_grad(), RandContext(ids_comb):
            emb = model(
                input_ids=ids_comb, attention_mask=mask_comb, top_k=top_k
            )  # [(mb + mb·n), V]
            V = emb.size(-1)

        q_emb = emb[: len(ids_q)]  # [mb, V]
        d_emb = emb[len(ids_q) :].reshape(len(ids_q), n, V)  # [mb, n, V]

        q_emb_chunks.append(q_emb.detach().requires_grad_())
        d_emb_chunks.append(d_emb.detach().requires_grad_())
        rng_chunks.append(RandContext(ids_comb))  # save RNG

    q_emb = torch.cat(q_emb_chunks, 0).detach().requires_grad_()  # [B, V]
    d_emb = torch.cat(d_emb_chunks, 0).detach().requires_grad_()  # [B, n, V]

    # ---------- build scores matrix & loss --------------------------
    if neg_mode == "row":
        # (B, n) …   positives at [:,0]
        scores = torch.einsum("bd,bnd->bn", q_emb, d_emb)

        labels = torch.zeros(B, dtype=torch.long, device=device)

    else:
        d_flat = d_emb.reshape(B * n, V)

        if neg_mode == "global" and world > 1:
            gathered = [torch.zeros_like(d_flat) for _ in range(world)]
            torch.distributed.all_gather(gathered, d_flat)
            d_flat = torch.cat(gathered, 0)  # [B*n*W, V]

        # dot-product against the big doc table
        scores_full = torch.matmul(q_emb, d_flat.T)  # [B, D*]

        if neg_mode == "batch":
            # keep 1 positive  +  all docs from *other* rows
            keep_cols = []
            for i in range(B):
                # col idx of pos for query i in local d_flat
                pos = i * n + (rank * B * n if neg_mode == "global" else 0)
                others = torch.arange(0, B * n * world, device=device)
                mask = (others // n) != i  # drop own row
                keep = torch.cat([others[pos : pos + 1], others[mask]])
                keep_cols.append(keep)
            scores = torch.stack([scores_full[i, keep_cols[i]] for i in range(B)])
            labels = torch.zeros(B, dtype=torch.long, device=device)

        else:  # neg_mode == "global"  (= full table)
            labels = torch.arange(B, device=device) + rank * B
            scores = scores_full

    scores = scores / temperature

    # ------------------ primary CE loss ----------------------------
    triplet_loss = F.cross_entropy(scores, labels)

    # ------------------ regularisers -------------------------------
    doc_vecs = d_emb.reshape(-1, d_emb.size(-1))  # [(B·n), V]
    doc_flops = torch.sum(doc_vecs.abs(), dim=-1).mean()  # scalar
    query_l1 = torch.sum(q_emb.abs(), dim=-1).mean()  # scalar
    flops = lambda_t_d * doc_flops + lambda_t_q * query_l1

    anti_zero = torch.clamp(
        torch.reciprocal(q_emb.sum() ** 2 + 1e-8)
        + torch.reciprocal(d_emb.sum() ** 2 + 1e-8),
        max=1.0,
    )

    kl_loss = torch.tensor(0.0, device=device)
    if teacher_scores is not None:
        t_logits = teacher_scores / temperature
        student_row_logits = torch.einsum("bd,bnd->bn", q_emb, d_emb)
        student_row_logits = student_row_logits / temperature
        kl_loss = F.kl_div(
            F.log_softmax(student_row_logits, dim=-1),
            F.log_softmax(t_logits, dim=-1),
            reduction="batchmean",
            log_target=True,
        )

    total_loss = triplet_loss + flops + anti_zero + kl_loss

    def _recompute(sl: slice, _rng: RandContext):
        # identical forward as in pass-1 but with grad enabled
        ids_q = query_input_ids[sl]
        mask_q = query_attention_mask[sl]
        ids_d = doc_input_ids[sl].reshape(-1, Ld)
        mask_d = doc_attention_mask[sl].reshape(-1, Ld)
        ids_comb = torch.cat([ids_q, ids_d], 0)
        mask_comb = torch.cat([mask_q, mask_d], 0)

        emb = model(
            input_ids=ids_comb, attention_mask=mask_comb, top_k=top_k
        )  # grad on
        q = emb[: len(ids_q)]
        d = emb[len(ids_q) :].reshape(len(ids_q), n, V)
        return [q, d]

    gc_backward_and_zero_grad(total_loss, [q_emb, d_emb], _recompute, model, mini_batch)

    # ------------------ metrics dict -------------------------------

    metrics = {}
    query_non_zero_vals = q_emb[q_emb != 0]
    doc_non_zero_vals = d_emb[d_emb != 0]

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

    metrics["avg_query_non_zero_count"] = query_non_zero_vals.numel() / B
    metrics["avg_doc_non_zero_count"] = doc_non_zero_vals.numel() / (B * n)

    metrics = {
        "loss": total_loss,
        "triplet_loss": triplet_loss,
        "kl_loss": kl_loss,
        "flops_loss": flops,
        "anti_zero_loss": anti_zero,
        "query_sparsity": (q_emb == 0).float().mean(),
        "doc_sparsity": (d_emb == 0).float().mean(),
        **metrics,
    }
    return metrics
