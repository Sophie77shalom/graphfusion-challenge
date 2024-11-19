import torch
from model.memory import MemoryBank


def test_memory_initialization():
    memory = MemoryBank(memory_size=10, feature_dim=5)
    assert memory.memory.size() == (10, 5)


def test_memory_write():
    memory = MemoryBank(memory_size=10, feature_dim=5)
    input_data = torch.rand(3, 5)  # Batch size 3, feature_dim 5
    write_weights = torch.ones(3, 10) / 10  # Uniform weights
    memory.write(input_data, write_weights)


def test_memory_read():
    memory = MemoryBank(memory_size=10, feature_dim=5)
    query = torch.rand(3, 5)  # Batch size 3, query_dim 5
    retrieved_memory, confidence_scores = memory.read(query)
    assert retrieved_memory.size() == (3, 5)
    assert confidence_scores.size() == (3,)
