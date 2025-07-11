from transformers import AutoModelForMaskedLM, AutoConfig
from .splade import (
    SpladeModel,
    SparseEmbedModel,
    SpladeModel_LearnableTemp,
    SpladeModel_LearnableTemp_noTopK,
    SpladeModel_NoTopK,
)
from .memory_efficient import (
    MemoryEfficientSplade,
    MemoryEfficientSplade_noTopK,
    MemoryEfficientSplade_LearnableTemp,
    MemoryEfficientSplade_LearnableTemp_noTopK,
)
import torch
from typing import Optional


def get_splade_model(
    model_name: str = "bert-base-uncased",
    config: AutoConfig = None,
    device: str = "cuda",
    sparse_embed: bool = False,
    custom_kernel: bool = False,
    checkpoint_path: str = None,
    init_ce_temp: Optional[float] = None,
    init_kl_temp: Optional[float] = None,
    top_k: Optional[int] = 128,
    trust_remote_code: bool = True,
) -> SpladeModel:
    """
    Get a SPLADE model based on a pretrained transformer model.
    There are many "redundant" model classes because conditionals are moved outside of the model.
    This sometimes makes torch compile easier.

    Args:
        model_name: Name of the pretrained model (e.g., "bert-base-uncased", "distilbert-base-uncased")
        device: Device to load the model on ("cuda" or "cpu")
        custom_kernel: Custom activation kernel
        sparse_embed: Use contextual embeddings from SparseEmbed paper (not compatible with kl)
        checkpoint_path: Path to a checkpoint to load the model from
        init_ce_temp: Initial temperature for the CE loss
        init_kl_temp: Initial temperature for the KL loss
        top_k: Top-k for the SPLADE activation
    Returns:
        SpladeModel instance
    """

    assert (init_kl_temp is not None) == (init_ce_temp is not None), (
        "init_kl_temp and init_ce_temp must be provided together"
    )

    transformer_model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=trust_remote_code,
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

    # Annoying logic here. Honi soit qui mal y pense.
    if sparse_embed:
        splade_model = SparseEmbedModel(transformer_model).to(device)
        print("Sparse embed model loaded")
    elif custom_kernel:
        if init_kl_temp is None:
            splade_model = (
                MemoryEfficientSplade_noTopK(transformer_model).to(device)
                if top_k is None
                else MemoryEfficientSplade(transformer_model, top_k).to(device)
            )
            if top_k is None:
                print("Memory efficient no top k model loaded")
            else:
                print("Memory efficient top k model loaded")
        else:
            splade_model = (
                MemoryEfficientSplade_LearnableTemp_noTopK(
                    transformer_model, init_ce_temp, init_kl_temp
                ).to(device)
                if top_k is None
                else MemoryEfficientSplade_LearnableTemp(
                    transformer_model, init_ce_temp, init_kl_temp, top_k
                ).to(device)
            )
            if top_k is None:
                print("Memory efficient learnable temp no top k model loaded")
            else:
                print("Memory efficient learnable temp top k model loaded")
    else:  # No custom kernel
        if init_kl_temp is None:
            splade_model = (
                SpladeModel_NoTopK(transformer_model).to(device)
                if top_k is None
                else SpladeModel(transformer_model, top_k).to(device)
            )
            if top_k is None:
                print("Splade no top k model loaded")
            else:
                print("Splade top k model loaded")
        else:
            splade_model = (
                SpladeModel_LearnableTemp_noTopK(
                    transformer_model, init_ce_temp, init_kl_temp
                ).to(device)
                if top_k is None
                else SpladeModel_LearnableTemp(
                    transformer_model, init_ce_temp, init_kl_temp, top_k
                ).to(device)
            )
            if top_k is None:
                print("Splade learnable temp no top k model loaded")
            else:
                print("Splade learnable temp top k model loaded")

    return splade_model
