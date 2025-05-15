from typing import Dict, Optional, List
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


# @torch.compile(mode="max-autotune")
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
    neg_mode: str = "batch",
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
    pass_1_rng_states: List[RandContext] = []  # Stores RNG states from pass-1

    for start in range(0, B, mini_batch):
        sl = slice(start, min(start + mini_batch, B))

        ids_q = query_input_ids[sl]
        mask_q = query_attention_mask[sl]

        ids_d = doc_input_ids[sl].reshape(-1, Ld)
        mask_d = doc_attention_mask[sl].reshape(-1, Ld)

        ids_comb = torch.cat([ids_q, ids_d], dim=0)
        mask_comb = torch.cat([mask_q, mask_d], dim=0)

        rng_ctx_for_mb = RandContext(ids_comb)  # Or any tensor on the correct device(s)
        pass_1_rng_states.append(rng_ctx_for_mb)

        with torch.no_grad(), rng_ctx_for_mb:
            emb = model(
                input_ids=ids_comb, attention_mask=mask_comb, top_k=top_k
            )  # [(mb + mb·n), V]
            V = emb.size(-1)

        q_emb = emb[: len(ids_q)]  # [mb, V]
        d_emb = emb[len(ids_q) :].reshape(len(ids_q), n, V)  # [mb, n, V]

        q_emb_chunks.append(q_emb.detach().requires_grad_())
        d_emb_chunks.append(d_emb.detach().requires_grad_())

    q_emb = torch.cat(q_emb_chunks, 0).detach().requires_grad_()  # [B, V]
    d_emb = torch.cat(d_emb_chunks, 0).detach().requires_grad_()  # [B, n, V]

    # ---------- build scores matrix & loss --------------------------
    if neg_mode == "row":
        # We want           [pos | row-hard-negs | batch-negs]
        # shape per query :  1   +   (n-1)        +  (B-1)*n
        # ----------------------------------------------------------------
        d_flat = d_emb.reshape(B * n, V)  # [(B·n), V]
        scores_full = torch.matmul(q_emb, d_flat.T)  # [B, B·n]

        # build a (B, B·n) index tensor that selects the columns
        # in the order:   pos  →  hard-negs  →  other-rows
        keep_cols = []
        all_cols = torch.arange(B * n, device=device)
        for i in range(B):
            row_cols = all_cols[i * n : (i + 1) * n]  # this query’s row
            pos_col = row_cols[:1]  # first is positive
            hard_negs = row_cols[1:]  # rest in the same row
            batch_negs = all_cols[(all_cols // n) != i]  # every other row
            keep_cols.append(torch.cat([pos_col, hard_negs, batch_negs], dim=0))
        keep_cols = torch.stack(keep_cols, 0)  # [B, B·n]

        # fancy-index the full score table
        scores = torch.gather(scores_full, 1, keep_cols)  # [B, 1+(n-1)+(B-1)·n]
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

    def _recompute(sl: slice, _rng_ctx: RandContext):
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

    gc_backward_and_zero_grad(
        total_loss, [q_emb, d_emb], _recompute, pass_1_rng_states, model, mini_batch
    )

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


def train_step_kldiv_ibn_vectorized(
    model: nn.Module,
    query_input_ids: torch.Tensor,  # [B, Lq]
    query_attention_mask: torch.Tensor,  # [B, Lq]
    doc_input_ids: torch.Tensor,  # [B, n, Ld] (n = 1 pos + num_hard_neg from collate)
    doc_attention_mask: torch.Tensor,  # [B, n, Ld]
    top_k: int,
    lambda_t_d: torch.Tensor,
    lambda_t_q: torch.Tensor,
    device: torch.device,
    temperature_ce: torch.Tensor,
    temperature_kl: torch.Tensor,  # Added separate temperature for KL
    neg_mode: str = "batch",
    mini_batch: int = 16,
    teacher_scores: Optional[torch.Tensor] = None,  # [B, n], aligns with doc_input_ids
) -> Dict[str, torch.Tensor]:
    # if torch.cuda.is_available(): torch.compiler.cudagraph_mark_step_begin() # For torch.compile with CUDA graphs
    model.train()

    B, n_docs_per_query, Ld = (
        doc_input_ids.shape
    )  # n_docs_per_query is `n` from previous code
    V: Optional[int] = (
        None  # Vocabulary size, will be known after first model forward pass
    )

    # ---------- pass-1: embed without graph (for memory saving) --------------------------
    q_emb_chunks, d_emb_chunks = [], []
    pass_1_rng_states = []  # If your RandContext needs to save states from pass-1

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

        rng_context_for_this_mb = RandContext(
            ids_comb_mb
        )  # Or any tensor on the correct device(s)
        pass_1_rng_states.append(rng_context_for_this_mb)

        with (
            torch.no_grad(),
            rng_context_for_this_mb,
        ):  # Manage RNG for dropout consistency if needed
            emb_mb = model(
                input_ids=ids_comb_mb, attention_mask=mask_comb_mb, top_k=top_k
            )
            if V is None:
                V = emb_mb.size(-1)

        q_emb_mb = emb_mb[:current_mini_batch_size]
        d_emb_mb = emb_mb[current_mini_batch_size:].reshape(
            current_mini_batch_size, n_docs_per_query, V
        )

        q_emb_chunks.append(q_emb_mb)  # Already detached due to torch.no_grad()
        d_emb_chunks.append(d_emb_mb)

    q_emb = torch.cat(q_emb_chunks, 0).requires_grad_()  # [B, V]
    d_emb = torch.cat(d_emb_chunks, 0).requires_grad_()  # [B, n_docs_per_query, V]

    # ---------- build scores matrix & loss (vectorized negative sampling) --------------------------
    # d_flat contains all B * n_docs_per_query document embeddings sequentially
    d_flat = d_emb.reshape(B * n_docs_per_query, V)  # [B * n_docs_per_query, V]

    # scores_full[i, j] = score(q_emb[i], d_flat[j])
    scores_full = torch.matmul(q_emb, d_flat.T)  # [B, B * n_docs_per_query]

    if neg_mode == "row":
        # For query i: scores are [pos_i | hard_negs_i | batch_negs_i (all docs from other queries)]
        # Positive is always the first item for the CrossEntropyLoss.
        labels = torch.zeros(B, dtype=torch.long, device=device)

        # Indices of positive documents for each query within d_flat
        # The positive for query `i` is `d_emb[i,0,:]`, which is at `d_flat[i * n_docs_per_query, :]`
        pos_col_indices_in_d_flat = torch.arange(
            start=0, end=B * n_docs_per_query, step=n_docs_per_query, device=device
        )
        pos_item_cols = pos_col_indices_in_d_flat.unsqueeze(1)  # [B, 1]

        # Hard negative columns for each query (from d_emb[i, 1:, :])
        if n_docs_per_query > 1:  # If there are any hard negatives
            hard_neg_offsets = torch.arange(
                1, n_docs_per_query, device=device
            ).unsqueeze(0)  # [1, n_docs_per_query-1]
            # Broadcasting: [B,1] (pos_item_cols) + [1, n_docs_per_query-1] (offsets)
            hard_neg_item_cols = (
                pos_item_cols + hard_neg_offsets
            )  # [B, n_docs_per_query-1]
        else:  # No hard negatives if n_docs_per_query=1 (only positive)
            hard_neg_item_cols = torch.empty((B, 0), dtype=torch.long, device=device)

        if B == 1:  # Only one query, no batch negatives from "other" queries
            batch_neg_item_cols = torch.empty((B, 0), dtype=torch.long, device=device)
        else:
            # Batch negative columns: all (B-1)*n_docs_per_query documents from other queries
            all_doc_indices_global = torch.arange(
                B * n_docs_per_query, device=device
            )  # [B*n_docs_per_query]
            # doc_query_owner[k] = query_idx (0 to B-1) that document k belongs to
            doc_query_owner = all_doc_indices_global // n_docs_per_query

            current_query_idx_expanded = torch.arange(B, device=device).unsqueeze(
                1
            )  # [B, 1]
            # batch_neg_mask[i,j] is True if doc j is a batch_neg for query i (i.e., doc_query_owner[j] != i)
            batch_neg_mask = (
                doc_query_owner.unsqueeze(0) != current_query_idx_expanded
            )  # [B, B*n_docs_per_query]

            replicated_all_doc_indices = all_doc_indices_global.unsqueeze(0).expand(
                B, -1
            )  # [B, B*n_docs_per_query]
            # Select based on mask and reshape to [B, num_batch_negs]
            num_batch_negs = (B - 1) * n_docs_per_query
            batch_neg_item_cols = replicated_all_doc_indices[batch_neg_mask].reshape(
                B, num_batch_negs
            )

        keep_cols = torch.cat(
            [pos_item_cols, hard_neg_item_cols, batch_neg_item_cols], dim=1
        )
        # Shape: [B, 1 + (n_docs_per_query-1) + (B-1)*n_docs_per_query] = [B, B * n_docs_per_query]
        scores = torch.gather(scores_full, 1, keep_cols)

    elif neg_mode == "batch":
        # For query i: scores are [pos_i | batch_negs_i (all n_docs_per_query docs from each of (B-1) other queries)]
        labels = torch.zeros(B, dtype=torch.long, device=device)

        pos_col_indices_in_d_flat = torch.arange(
            start=0, end=B * n_docs_per_query, step=n_docs_per_query, device=device
        )
        pos_item_cols = pos_col_indices_in_d_flat.unsqueeze(1)  # [B, 1]

        if B == 1:  # Only one query, no "other" queries for batch negatives
            batch_neg_item_cols = torch.empty((B, 0), dtype=torch.long, device=device)
        else:
            # Batch negative columns logic is identical to that in "row" mode
            all_doc_indices_global = torch.arange(B * n_docs_per_query, device=device)
            doc_query_owner = all_doc_indices_global // n_docs_per_query
            current_query_idx_expanded = torch.arange(B, device=device).unsqueeze(1)
            batch_neg_mask = doc_query_owner.unsqueeze(0) != current_query_idx_expanded
            replicated_all_doc_indices = all_doc_indices_global.unsqueeze(0).expand(
                B, -1
            )
            num_batch_negs = (B - 1) * n_docs_per_query
            batch_neg_item_cols = replicated_all_doc_indices[batch_neg_mask].reshape(
                B, num_batch_negs
            )

        keep_cols = torch.cat([pos_item_cols, batch_neg_item_cols], dim=1)
        # Shape: [B, 1 + (B-1)*n_docs_per_query]
        scores = torch.gather(scores_full, 1, keep_cols)

    scores = scores / temperature_ce

    # ------------------ primary CE loss ----------------------------
    triplet_loss = F.cross_entropy(scores, labels)

    doc_vecs = d_emb.reshape(-1, d_emb.size(-1))  # [(B·n), V]
    doc_flops = torch.sum(doc_vecs.abs(), dim=-1).mean()  # scalar
    query_l1 = torch.sum(q_emb.abs(), dim=-1).mean()  # scalar
    flops_loss = lambda_t_d * doc_flops + lambda_t_q * query_l1

    # ------------------ Anti-zero loss (optional) -------------------
    # Sums over all elements in all query/document embeddings respectively
    q_sum_sq_inv = torch.reciprocal(q_emb.sum().pow(2) + 1e-8)
    d_sum_sq_inv = torch.reciprocal(d_emb.sum().pow(2) + 1e-8)
    anti_zero = torch.clamp(q_sum_sq_inv + d_sum_sq_inv, max=1.0)

    # ------------------ KL divergence distillation loss -------------
    kl_loss = torch.tensor(0.0, device=device)
    if teacher_scores is not None:
        # teacher_scores has shape [B, n_docs_per_query], aligning with d_emb[b, :, :]
        # Student scores for q_i vs its own n_docs_per_query documents:
        student_row_logits = torch.einsum(
            "bv,bnv->bn", q_emb, d_emb
        )  # [B, n_docs_per_query]

        teacher_log_softmax = F.log_softmax(teacher_scores / temperature_kl, dim=-1)
        student_log_softmax = F.log_softmax(student_row_logits / temperature_kl, dim=-1)

        kl_loss = F.kl_div(
            student_log_softmax,
            teacher_log_softmax,
            reduction="batchmean",  # Averages over the batch dimension (B)
            log_target=True,
        )

    total_loss = triplet_loss + flops_loss + kl_loss + anti_zero

    # ---------- Backward pass using gradient checkpointing strategy --------------------
    def _recompute(sl_mb: slice, _rng_ctx_dummy: RandContext):
        current_mini_batch_size_recompute = sl_mb.stop - sl_mb.start

        ids_q_recompute = query_input_ids[sl_mb]
        mask_q_recompute = query_attention_mask[sl_mb]
        # Ensure V is defined from pass-1; it should be.
        if V is None:
            raise ValueError("V (vocab_size/embedding_dim) not set from pass-1.")

        ids_d_recompute = doc_input_ids[sl_mb].reshape(
            current_mini_batch_size_recompute * n_docs_per_query, Ld
        )
        mask_d_recompute = doc_attention_mask[sl_mb].reshape(
            current_mini_batch_size_recompute * n_docs_per_query, Ld
        )

        ids_comb_recompute = torch.cat([ids_q_recompute, ids_d_recompute], dim=0)
        mask_comb_recompute = torch.cat([mask_q_recompute, mask_d_recompute], dim=0)

        with _rng_ctx_dummy:  # If RandContext is used for RNG restoration
            emb_recompute = model(
                input_ids=ids_comb_recompute,
                attention_mask=mask_comb_recompute,
                top_k=top_k,
            )

        q_recomputed = emb_recompute[:current_mini_batch_size_recompute]
        d_recomputed = emb_recompute[current_mini_batch_size_recompute:].reshape(
            current_mini_batch_size_recompute, n_docs_per_query, V
        )
        return [q_recomputed, d_recomputed]

    gc_backward_and_zero_grad(
        total_loss, [q_emb, d_emb], pass_1_rng_states, _recompute, model, mini_batch
    )

    # ------------------ metrics dict -------------------------------
    metrics_dict: Dict[str, torch.Tensor] = {}
    with torch.no_grad():  # Metrics calculation should not contribute to graph
        q_abs = q_emb.abs()
        d_abs = d_emb.abs()

        # Using a small threshold for "non-zero" for robustness with float precision
        is_q_nonzero = q_abs > 1e-9
        is_d_nonzero = d_abs > 1e-9

        q_nonzero_vals = q_abs[is_q_nonzero]
        d_nonzero_vals = d_abs[is_d_nonzero]

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
            is_q_nonzero.sum() / B
        )  # Sum of True values
        metrics_dict["avg_doc_non_zero_count"] = is_d_nonzero.sum() / (
            B * n_docs_per_query
        )

        metrics_dict["query_sparsity"] = (
            (~is_q_nonzero).float().mean()
        )  # Proportion of zero/near-zero values
        metrics_dict["doc_sparsity"] = (~is_d_nonzero).float().mean()

        metrics_dict["loss"] = total_loss.detach()
        metrics_dict["triplet_loss"] = triplet_loss.detach()
        metrics_dict["kl_loss"] = kl_loss.detach()
        metrics_dict["flops_loss"] = flops_loss.detach()

        metrics_dict["anti_zero_loss"] = anti_zero.detach()

    # if torch.cuda.is_available(): torch.compiler.cudagraph_mark_step_end()
    return metrics_dict
