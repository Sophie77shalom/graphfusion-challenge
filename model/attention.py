import torch
import torch.nn as nn


class AttentionMechanism(nn.Module):
    def __init__(self, feature_dim: int):
        """
        Initialize the attention mechanism.

        Args:
            feature_dim: Dimension of feature vectors
        """
        super().__init__()
        self.feature_dim = feature_dim
        # Linear layer to project the query
        self.query_layer = nn.Linear(feature_dim, feature_dim)
        # Linear layer to project the keys (inputs)
        self.key_layer = nn.Linear(feature_dim, feature_dim)

    def forward(
        self, inputs: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform attention mechanism.

        Args:
            inputs: Input features (batch_size, seq_length, feature_dim)
            query: Query vector (batch_size, feature_dim)

        Returns:
            tuple: (output, attention_weights)
        """
        # Project inputs and query to the same dimension
        keys = self.key_layer(inputs)
        query = self.query_layer(query)

        # Calculate attention scores (dot product)
        scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=1)

        # Compute weighted sum of inputs based on attention weights
        output = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)

        return output, attention_weights
