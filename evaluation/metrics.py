import torch


def memory_retention_accuracy(retrieved: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate the accuracy of memory retention.

    Args:
        retrieved: Retrieved memory representations (batch_size, feature_dim)
        target: Target memory representations (batch_size, feature_dim)

    Returns:
        float: Memory retention accuracy as a percentage.
    """
    # Compute the number of correct predictions (using cosine similarity)
    cos = torch.nn.CosineSimilarity(dim=1)
    similarities = cos(retrieved, target)

    # Define a threshold for considering a retrieval correct
    threshold = 0.95
    correct_predictions = (similarities > threshold).float()

    return correct_predictions.mean().item() * 100


def retrieval_precision(retrieved: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Calculate the precision of retrievals.

    Args:
        retrieved: Retrieved items (batch_size, feature_dim)
        ground_truth: Ground truth items (batch_size, feature_dim)

    Returns:
        float: Precision score as a percentage.
    """
    # Define a simple threshold for precision calculation
    correct_retrievals = torch.all(
        torch.isclose(retrieved, ground_truth, atol=1e-2), dim=1
    )
    return correct_retrievals.float().mean().item() * 100


def confidence_score_correlation(
    confidence_scores: torch.Tensor, ground_truth: torch.Tensor
) -> float:
    """
    Calculate the correlation between confidence scores and ground truth.

    Args:
        confidence_scores: Confidence scores output by the model (batch_size,)
        ground_truth: Ground truth (batch_size,)

    Returns:
        float: Correlation coefficient between confidence scores and ground truth.
    """
    # Use Pearson correlation
    return torch.corrcoef(confidence_scores, ground_truth).item()


def training_stability(losses: list) -> float:
    """
    Evaluate training stability by calculating the standard deviation of the loss over epochs.

    Args:
        losses: List of loss values over training epochs.

    Returns:
        float: Standard deviation of losses.
    """
    return torch.std(torch.tensor(losses)).item()
