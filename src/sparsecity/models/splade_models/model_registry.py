from transformers import AutoModelForMaskedLM
from .splade import SpladeModel
from .memory_efficient import MemoryEfficientSplade


def get_splade_model(
    model_name: str = "bert-base-uncased",
    device: str = "cuda",
    custom_kernel: bool = False,
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
    transformer_model = AutoModelForMaskedLM.from_pretrained(model_name)
    if custom_kernel:
        splade_model = MemoryEfficientSplade(transformer_model).to(device)
    else:
        splade_model = SpladeModel(transformer_model).to(device)
    return splade_model
