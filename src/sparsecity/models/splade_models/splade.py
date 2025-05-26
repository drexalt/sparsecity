import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseEmbedModel(nn.Module):
    def __init__(self, transformer_model: nn.Module, embedding_dim: int = 128):
        super().__init__()
        self.model = transformer_model
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(transformer_model.config.hidden_size, embedding_dim)

    def forward(self, input_ids, attention_mask, top_k=64):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        sequence_encodings = outputs.hidden_states[
            -1
        ]  # [batch_size, seq_len, hidden_size]

        activations = torch.log1p(F.relu(logits)) * attention_mask.unsqueeze(-1)
        sparse_vector = torch.amax(activations, dim=1)  # [batch_size, vocab_size]

        top_values, top_indices = torch.topk(sparse_vector, k=top_k, dim=-1)
        threshold = top_values[..., -1, None]
        sparse_vector = sparse_vector * (sparse_vector >= threshold)

        index = top_indices.unsqueeze(1).expand(-1, logits.shape[1], -1)  # [b, seq, k]

        selected_logits = logits.gather(dim=2, index=index)  # [b, seq, k]
        attention_weights = F.softmax(selected_logits, dim=1)  # [b, seq, k]

        print(f"Shape of sequence_encodings: {sequence_encodings.shape}")
        assert len(sequence_encodings.shape) == 3, (
            f"sequence_encodings is not 3D! Shape: {sequence_encodings.shape}"
        )

        contextual_embeddings = torch.bmm(
            attention_weights.transpose(1, 2), sequence_encodings
        )

        contextual_embeddings = F.relu(self.projection(contextual_embeddings))
        # [batch_size, top_k, embedding_dim]

        return {
            "sparse_activations": sparse_vector,  # [batch_size, vocab_size]
            "activations": top_indices,  # [batch_size, top_k]
            "embeddings": contextual_embeddings,  # [batch_size, top_k, embedding_dim]
        }


class SpladeModel_LearnableTemp(nn.Module):
    """
    SPLADE model that works with any transformer-based masked language model. Must provide top_k.
    """

    def __init__(
        self,
        transformer_model: nn.Module,
        init_ce_temp: float = 5.0,
        init_kl_temp: float = 5.0,
    ):
        super().__init__()
        self.model = transformer_model
        self.log_t_ce = nn.Parameter(torch.log(torch.tensor(init_ce_temp)))
        self.log_t_kl = nn.Parameter(torch.log(torch.tensor(init_kl_temp)))

    @property
    def temperature_ce(self):
        return self.log_t_ce.exp()

    @property
    def temperature_kl(self):
        return self.log_t_kl.exp()

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


class SpladeModel(nn.Module):
    """
    SPLADE model that works with any transformer-based masked language model. Must provide top_k.
    """

    def __init__(self, transformer_model: nn.Module, top_k: int = 128):
        super().__init__()
        self.model = transformer_model
        self.top_k = top_k

    def forward(self, input_ids, attention_mask):
        # Get MLM logits from transformer
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        logits = outputs.logits

        # SPLADE activation
        activations = torch.log1p(F.relu(logits)) * attention_mask.unsqueeze(-1)
        values = torch.amax(activations, dim=1)

        top_values, _ = torch.topk(values, k=self.top_k, dim=-1)
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


class SpladeModel_LearnableTemp_noTopK(nn.Module):
    """
    SPLADE model that works with any transformer-based masked language model. Must provide top_k.
    """

    def __init__(
        self,
        transformer_model: nn.Module,
        init_ce_temp: float = 5.0,
        init_kl_temp: float = 5.0,
    ):
        super().__init__()
        self.model = transformer_model
        self.log_t_ce = nn.Parameter(torch.log(torch.tensor(init_ce_temp)))
        self.log_t_kl = nn.Parameter(torch.log(torch.tensor(init_kl_temp)))

    @property
    def temperature_ce(self):
        return self.log_t_ce.exp()

    @property
    def temperature_kl(self):
        return self.log_t_kl.exp()

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


class SpladeModel(nn.Module):
    """
    SPLADE model that works with any transformer-based masked language model. Must provide top_k.
    """

    def __init__(self, transformer_model: nn.Module, top_k: int = 128):
        super().__init__()
        self.model = transformer_model
        self.top_k = top_k

    def forward(self, input_ids, attention_mask):
        # Get MLM logits from transformer
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        logits = outputs.logits

        # SPLADE activation
        activations = torch.log1p(F.relu(logits)) * attention_mask.unsqueeze(-1)
        values = torch.amax(activations, dim=1)

        top_values, _ = torch.topk(values, k=self.top_k, dim=-1)
        threshold = top_values[..., -1, None]
        values = values * (values >= threshold)

        return values
