import sys
import os
import json
from pathlib import Path
import torch
import numpy as np
from model.memory import MemoryBank, ConfidenceScoring
from data.data_generator import generate_sequence_data, generate_memory_test_data


def evaluate_model_performance():
    """
    Run comprehensive evaluation of the model implementation.
    Tests memory retention, retrieval accuracy, and confidence scoring.
    """
    results = {
        "memory_retention": {},
        "retrieval_accuracy": {},
        "confidence_correlation": {},
        "training_stability": {},
        "inference_speed": {},
    }

    try:
        # Memory Retention Test
        memory_bank = MemoryBank(memory_size=100, feature_dim=64)
        store_data, query_data = generate_memory_test_data(num_items=50, feature_dim=64)

        # Test memory writing
        write_weights = torch.ones(50, 100) / 100
        memory_bank.write(store_data, write_weights)

        # Test memory reading
        retrieved, confidence = memory_bank.read(query_data)

        # Calculate metrics
        retention_accuracy = torch.mean((retrieved - store_data).abs()).item()
        results["memory_retention"] = {
            "accuracy": retention_accuracy,
            "passed": retention_accuracy < 0.1,
        }

        # Retrieval Test
        sequence_data = generate_sequence_data(
            seq_length=20, feature_dim=64, batch_size=32
        )
        retrieval_accuracy = test_retrieval_accuracy(memory_bank, sequence_data)
        results["retrieval_accuracy"] = {
            "accuracy": retrieval_accuracy,
            "passed": retrieval_accuracy > 0.8,
        }

        # Performance Metrics
        inference_time = measure_inference_speed(memory_bank, query_data)
        results["inference_speed"] = {
            "time_ms": inference_time,
            "passed": inference_time < 100,  # 100ms threshold
        }

        # Save Results
        save_path = Path("evaluation/results")
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Exit based on overall performance
        overall_passed = all(v.get("passed", False) for v in results.values())
        sys.exit(0 if overall_passed else 1)

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        sys.exit(1)


def test_retrieval_accuracy(memory_bank, sequence_data):
    """Test accuracy of memory retrieval"""
    input_seq, target_seq = sequence_data
    retrieved, _ = memory_bank.read(input_seq)
    return torch.mean((retrieved - target_seq).abs()).item()


def measure_inference_speed(memory_bank, query_data):
    """Measure inference speed in milliseconds"""
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(100):  # Average over 100 runs
        memory_bank.read(query_data)
    end_time.record()

    torch.cuda.synchronize()
    return start_time.elapsed_time(end_time) / 100  # Average time per inference


if __name__ == "__main__":
    evaluate_model_performance()
