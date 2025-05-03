from .st_wrapper import ST_SPLADEModule, ST_SparseEmbedModule
from sentence_transformers.similarity_functions import dot_score
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator
import torch


def validate_model(evaluator, model, tokenizer, device, sparse_embed: bool = False):
    """Run NanoBEIR evaluation on the current model state"""
    # Create temporary wrapper objects
    if sparse_embed:
        st_module = ST_SparseEmbedModule(
            model, tokenizer, max_length=tokenizer.model_max_length
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
        "recall@10": results["NanoBEIR_mean_dot_recall@10"],
        "precision@1": results["NanoBEIR_mean_dot_precision@1"],
    }

    return {**primary_metrics, **supplementary_metrics}
