import torch
from model.memory import ConfidenceScoring


def test_confidence_scoring():
    feature_dim = 5
    confidence_scorer = ConfidenceScoring(feature_dim)

    # Create dummy query and retrieved memory
    query = torch.rand(2, feature_dim)
    retrieved = torch.rand(2, feature_dim)

    # Calculate confidence
    confidence_scores = confidence_scorer.calculate_confidence(query, retrieved)

    # Check shapes of outputs
    assert confidence_scores.shape == (2,), "Confidence scores shape is incorrect"
    assert (0 <= confidence_scores).all() and (
        confidence_scores <= 1
    ).all(), "Confidence scores should be between 0 and 1"
