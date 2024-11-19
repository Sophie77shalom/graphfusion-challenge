import torch
import pytest
from model.attention import AttentionMechanism


# Example test for the attention mechanism
def test_attention_mechanism():
    # Create dummy data for the test
    batch_size = 2
    seq_length = 5
    feature_dim = 3

    # Initialize the AttentionMechanism
    attention = AttentionMechanism(feature_dim)

    # Create dummy input
    inputs = torch.randn(batch_size, seq_length, feature_dim)

    # Create dummy query (batch_size, feature_dim)
    query = torch.randn(batch_size, feature_dim)

    # Call the attention mechanism
    output, attention_weights = attention(inputs, query)

    # Check the output dimensions
    assert output.size() == (batch_size, feature_dim), "Output dimensions mismatch"

    # Check that attention weights sum to 1
    assert torch.allclose(
        attention_weights.sum(dim=1), torch.ones(batch_size)
    ), "Attention weights do not sum to 1"
