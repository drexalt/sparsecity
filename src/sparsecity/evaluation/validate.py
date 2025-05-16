from .st_wrapper import ST_SPLADEModule, ST_SparseEmbedModule, ST_SPLADEModule_addTopK
from sentence_transformers.similarity_functions import dot_score
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator
import torch
from typing import Optional


def validate_model(
    evaluator,
    model,
    tokenizer,
    device,
    sparse_embed: bool = False,
    top_k: Optional[int] = None,
):
    """
    Run NanoBEIR evaluation on the current model state
    If top_k is none, we need to add top_k for evaluation to better mimic inference
    Args:
        evaluator: The evaluator to use
        model: The model to evaluate
        tokenizer: The tokenizer to use
        device: The device to use
        sparse_embed: Whether to use sparse embeddings
        top_k: The top-k to use

    """
    # Create temporary wrapper objects
    if sparse_embed:
        st_module = ST_SparseEmbedModule(
            model, tokenizer, max_length=tokenizer.model_max_length
        )
    else:
        if top_k is None:
            st_module = ST_SPLADEModule_addTopK(
                model,
                tokenizer,
                max_length=tokenizer.model_max_length,
                top_k=128,  # 128 to help comparison
            )
        else:
            st_module = ST_SPLADEModule(
                model, tokenizer, max_length=tokenizer.model_max_length
            )
    st_model = SentenceTransformer(modules=[st_module]).to(device)

    with torch.inference_mode():
        results = evaluator(st_model)

    # Clean up resources
    del st_module, st_model, evaluator
    torch.cuda.empty_cache()

    primary_metrics = {
        "ndcg@10": results["NanoBEIR_mean_dot_ndcg@10"],
        "mrr@10": results["NanoBEIR_mean_dot_mrr@10"],
        "map@100": results["NanoBEIR_mean_dot_map@100"],
    }

    # For WandB logging (more detailed)
    supplementary_metrics = {
        "msmarco_mrr@10": results["NanoMSMARCO_dot_mrr@10"],
        "msmarco_ndcg@10": results["NanoMSMARCO_dot_ndcg@10"],
        "msmarco_map@100": results["NanoMSMARCO_dot_map@100"],
    }

    return {**primary_metrics, **supplementary_metrics}
