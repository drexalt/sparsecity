from transformers import AutoModelForMaskedLM
from .splade import SpladeModel


MODEL_REGISTRY = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
}


def get_splade_model(
    model_name: str = "bert-base-uncased", device: str = "cuda"
) -> SpladeModel:
    """
    Get a SPLADE model based on a pretrained transformer model.

    Args:
        model_name: Name of the pretrained model (e.g., "bert-base-uncased", "distilbert-base-uncased")
        device: Device to load the model on ("cuda" or "cpu")

    Returns:
        SpladeModel instance
    """
    transformer_model = AutoModelForMaskedLM.from_pretrained(model_name)
    splade_model = SpladeModel(transformer_model).to(device)
    return splade_model
