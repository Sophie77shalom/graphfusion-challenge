import torch
import torch.nn as nn


class ConfidenceScoring(nn.Module):
    def __init__(self, feature_dim: int):
        """
        Initialize confidence scoring mechanism.

        Args:
            feature_dim: Dimension of feature vectors
        """
        super().__init__()
        self.linear = nn.Linear(
            feature_dim * 2, 1
        )  # Linear layer for combining query and retrieved memory

    def calculate_confidence(
        self, query: torch.Tensor, retrieved: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate confidence score for retrieved memories.

        Args:
            query: Original query vector (batch_size, feature_dim)
            retrieved: Retrieved memory vector (batch_size, feature_dim)

        Returns:
            Confidence score between 0 and 1 (batch_size, 1)
        """
        # Concatenate query and retrieved features
        combined = torch.cat((query, retrieved), dim=1)
        confidence_score = torch.sigmoid(self.linear(combined))

        return confidence_score
