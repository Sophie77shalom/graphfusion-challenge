import torch
import torch.nn as nn


class MemoryBank(nn.Module):
    def __init__(self, memory_size: int, feature_dim: int):
        """
        Initialize a differentiable memory bank.

        Args:
            memory_size: Number of memory slots
            feature_dim: Dimension of each memory slot
        """
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim

        # Initialize memory bank as a trainable parameter
        self.memory = nn.Parameter(torch.zeros(memory_size, feature_dim))

    def write(self, input_data: torch.Tensor, write_weights: torch.Tensor) -> None:
        """
        Write data to memory using attention weights.

        Args:
            input_data: Data to write (batch_size, feature_dim)
            write_weights: Where to write (batch_size, memory_size)
        """
        assert (
            input_data.size(1) == self.feature_dim
        ), "Input data feature dimension mismatch"
        assert (
            write_weights.size(1) == self.memory_size
        ), "Write weights memory size mismatch"

        # Normalize write_weights to ensure they sum to 1
        write_weights = write_weights / write_weights.sum(dim=1, keepdim=True)

        # Create a new tensor to accumulate updates
        update = torch.zeros_like(self.memory)

        # Write to memory by accumulating updates
        for i in range(input_data.size(0)):
            for j in range(self.memory_size):
                update[j] += write_weights[i, j] * input_data[i]

        # Update memory in a non-in-place manner
        self.memory.data += update

    def read(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory using attention mechanism.

        Args:
            query: Query vector (batch_size, query_dim)

        Returns:
            tuple: (retrieved_memory, confidence_scores)
        """
        scores = torch.matmul(query, self.memory.t())
        attention_weights = torch.softmax(scores, dim=1)

        retrieved_memory = torch.matmul(attention_weights, self.memory)
        confidence_scores = attention_weights.max(dim=1)[0]

        return retrieved_memory, confidence_scores


class ConfidenceScoring(nn.Module):
    def __init__(self, feature_dim: int):
        """
        Initialize confidence scoring mechanism.

        Args:
            feature_dim: Dimension of feature vectors
        """
        super().__init__()
        self.fc = nn.Linear(2 * feature_dim, 1)

    def calculate_confidence(
        self, query: torch.Tensor, retrieved: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat((query, retrieved), dim=1)
        score = self.fc(combined)
        confidence = torch.sigmoid(score).squeeze()

        return confidence
