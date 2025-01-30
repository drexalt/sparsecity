import torch
from typing import Union
from torch import Tensor


# Losses
def compute_flops(inputs: Tensor) -> Tensor:
    """
    Compute FLOPS regularization term.

    Args:
        inputs: Tensor of shape [batch_size, vocab_size]
    """
    return torch.sum(torch.square(torch.mean(torch.abs(inputs), dim=0)))


def compute_L1(inputs: Tensor) -> Tensor:
    """
    Compute L1 regularization term.

    Args:
        inputs: Tensor of shape [batch_size, vocab_size]
    """
    return torch.sum(torch.abs(inputs), dim=-1).mean()


# def compute_flops_sparse(inputs: Union[Tensor, torch.sparse.Tensor]) -> Tensor:
#     """
#     Compute FLOPS regularization term for sparse inputs.
#     Will work with both dense and sparse tensors.

#     Args:
#         inputs: Tensor or sparse tensor of shape [batch_size, vocab_size]
#     """
#     if isinstance(inputs, torch.sparse.Tensor):
#         inputs = inputs.to_dense()
#     return compute_flops(inputs)


# def compute_L1_sparse(inputs: Union[Tensor, torch.sparse.Tensor]) -> Tensor:
#     """
#     Compute L1 regularization term for sparse inputs.
#     Will work with both dense and sparse tensors.

#     Args:
#         inputs: Tensor or sparse tensor of shape [batch_size, vocab_size]
#     """
#     if isinstance(inputs, torch.sparse.Tensor):
#         inputs = inputs.to_dense()
#     return compute_L1(inputs)


def create_batch(batch: dict) -> dict:
    """Convert batch dictionary values to PyTorch tensors."""
    return {k: torch.tensor(v) for k, v in batch.items()}
