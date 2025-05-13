from transformers import AutoModelForMaskedLM, AutoConfig
from .splade import SpladeModel, SparseEmbedModel
from .memory_efficient import MemoryEfficientSplade
import torch


def get_splade_model(
    model_name: str = "bert-base-uncased",
    config: AutoConfig = None,
    device: str = "cuda",
    sparse_embed: bool = False,
    custom_kernel: bool = False,
    checkpoint_path: str = None,
) -> SpladeModel:
    """
    Get a SPLADE model based on a pretrained transformer model.

    Args:
        model_name: Name of the pretrained model (e.g., "bert-base-uncased", "distilbert-base-uncased")
        device: Device to load the model on ("cuda" or "cpu")
        custom_kernel: Custom activation kernel
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
        splade_model = MemoryEfficientSplade(transformer_model).to(device)
    else:
        splade_model = SpladeModel(transformer_model).to(device)
    return splade_model
