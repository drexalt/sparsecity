from transformers import AutoModelForMaskedLM, AutoConfig
from .splade import SpladeModel, SparseEmbedModel, SpladeModel_LearnableTemp
from .memory_efficient import MemoryEfficientSplade, MemoryEfficientSplade_LearnableTemp
import torch
from typing import Optional


def get_splade_model(
    model_name: str = "bert-base-uncased",
    config: AutoConfig = None,
    device: str = "cuda",
    sparse_embed: bool = False,
    custom_kernel: bool = False,
    checkpoint_path: str = None,
    init_ce_temp: Optional[float] = 1.0,
    init_kl_temp: Optional[float] = 5.0,
) -> SpladeModel:
    """
    Get a SPLADE model based on a pretrained transformer model.

    Args:
        model_name: Name of the pretrained model (e.g., "bert-base-uncased", "distilbert-base-uncased")
        device: Device to load the model on ("cuda" or "cpu")
        custom_kernel: Custom activation kernel
        sparse_embed: Use contextual embeddings from SparseEmbed paper (not compatible with kl)
        checkpoint_path: Path to a checkpoint to load the model from
        init_ce_temp: Initial temperature for the CE loss
        init_kl_temp: Initial temperature for the KL loss
    Returns:
        SpladeModel instance
    """

    transformer_model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
    )
    if checkpoint_path:
        temp = torch.load(checkpoint_path, weights_only=False)
        state_dict = temp["model"]
        model_keys = set(transformer_model.state_dict().keys())

        encoder_state_dict = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.") and k.replace("encoder.", "") in model_keys
        }
        try:
            transformer_model.load_state_dict(encoder_state_dict, strict=False)
        except Exception as e:
            print(f"Error loading state_dict: {e}")
            print(f"Model keys: {model_keys}")
            print(f"State dict keys: {state_dict.keys()}")

    if sparse_embed:
        splade_model = SparseEmbedModel(transformer_model).to(device)
    elif custom_kernel:
        if init_kl_temp is None:
            splade_model = MemoryEfficientSplade(transformer_model).to(device)
        else:
            splade_model = MemoryEfficientSplade_LearnableTemp(
                transformer_model, init_ce_temp, init_kl_temp
            ).to(device)
    else:
        if init_kl_temp is None:
            splade_model = SpladeModel(transformer_model).to(device)
        else:
            splade_model = SpladeModel_LearnableTemp(
                transformer_model, init_ce_temp, init_kl_temp
            ).to(device)
    return splade_model
