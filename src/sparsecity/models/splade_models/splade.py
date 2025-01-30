import torch
import torch.nn as nn
import torch.nn.functional as F


class SpladeModel(nn.Module):
    """
    SPLADE model that works with any transformer-based masked language model. Must provide top_k.
    """

    def __init__(self, transformer_model: nn.Module):
        super().__init__()
        self.model = transformer_model

    def forward(self, input_ids, attention_mask, top_k=64):
        # Get MLM logits from transformer
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        logits = outputs.logits

        # SPLADE activation
        activations = torch.log1p(F.relu(logits)) * attention_mask.unsqueeze(-1)
        values = torch.amax(activations, dim=1)

        top_values, _ = torch.topk(values, k=top_k, dim=-1)
        threshold = top_values[..., -1, None]
        values = values * (values >= threshold)

        return values


class SpladeModel_NoTopK(nn.Module):
    """
    SPLADE model that works with any transformer-based masked language model. Does not use top-k masking.
    """

    def __init__(self, transformer_model: nn.Module):
        super().__init__()
        self.model = transformer_model

    def forward(self, input_ids, attention_mask):
        # Get MLM logits from transformer
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        logits = outputs.logits

        # SPLADE activation
        activations = torch.log1p(F.relu(logits)) * attention_mask.unsqueeze(-1)
        values = torch.amax(activations, dim=1)

        return values
