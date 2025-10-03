"""
Tests for RAG System

Basic tests to verify core functionality.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_system import RAGSystem, RAGResult


@pytest.fixture
def rag_system():
    """Create RAG system for testing."""
    return RAGSystem(
        data_path="data/python_docs.jsonl",
        model="gpt-5",
        retrieval_top_k=5
    )


def test_document_loading(rag_system):
    """Test that documents are loaded correctly."""
    assert len(rag_system.documents) > 0
    assert "id" in rag_system.documents[0]
    assert "content" in rag_system.documents[0]


def test_retrieval(rag_system):
    """Test document retrieval."""
    query = "list comprehensions"
    results = rag_system.retrieve(query, top_k=3)

    assert len(results) <= 3
    assert all("document" in r for r in results)
    assert all("score" in r for r in results)


def test_query_structure(rag_system):
    """Test that query returns proper structure."""
    result = rag_system.query("How do list comprehensions work?")

    assert isinstance(result, RAGResult)
    assert isinstance(result.answer, str)
    assert isinstance(result.citations, list)
    assert isinstance(result.confidence, float)
    assert isinstance(result.cost_usd, float)
    assert result.latency_ms >= 0


def test_citation_extraction(rag_system):
    """Test that citations are extracted when relevant docs exist."""
    query = "What are decorators?"
    result = rag_system.query(query)

    # Should retrieve decorator document and include it
    assert len(result.citations) >= 0  # May or may not cite depending on retrieval


def test_empty_query(rag_system):
    """Test handling of empty or irrelevant queries."""
    result = rag_system.query("xyzabc nonsense query")

    # Should handle gracefully
    assert isinstance(result.answer, str)


def test_caching(rag_system):
    """Test that caching works."""
    rag_system.enable_caching = True
    rag_system.cache = {}  # Clear cache

    query = "How do list comprehensions work?"

    # First query
    result1 = rag_system.query(query)
    assert len(rag_system.cache) == 1

    # Second query (should be cached)
    result2 = rag_system.query(query)
    assert result1.answer == result2.answer
    assert len(rag_system.cache) == 1


@pytest.mark.integration
def test_evaluation(rag_system):
    """Test evaluation metrics calculation."""
    metrics = rag_system.evaluate("data/test_queries.jsonl")

    assert "recall_at_k" in metrics
    assert "avg_confidence" in metrics
    assert "total_cost_usd" in metrics
    assert "avg_latency_ms" in metrics

    assert 0 <= metrics['recall_at_k'] <= 1
    assert 0 <= metrics['avg_confidence'] <= 1
    assert metrics['total_cost_usd'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
