import torch
import numpy as np


def generate_sequence_data(
    seq_length: int, feature_dim: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sequential data for testing memory retention.

    Args:
        seq_length: Length of sequences
        feature_dim: Dimension of features
        batch_size: Number of sequences

    Returns:
        tuple: (input_sequences, target_sequences)
    """
    # Generate random input sequences
    input_sequences = torch.rand(batch_size, seq_length, feature_dim)

    target_sequences = input_sequences + 0.1 * torch.rand_like(input_sequences)
    return input_sequences, target_sequences


def generate_memory_test_data(
    num_items: int, feature_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate test data for memory storage and retrieval.

    Args:
        num_items: Number of items to generate
        feature_dim: Dimension of features

    Returns:
        tuple: (store_data, query_data)
    """
    # Generate random memory storage data
    store_data = torch.rand(num_items, feature_dim)

    # Generate random query data, can be similar or different
    query_data = torch.rand(num_items, feature_dim)

    return store_data, query_data
