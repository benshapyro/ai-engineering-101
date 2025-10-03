"""
Pytest configuration and shared fixtures.

This file contains fixtures and utilities available to all tests.
"""

import os
import pytest
from typing import Dict, Any
from unittest.mock import Mock


@pytest.fixture
def mock_api_key(monkeypatch):
    """Provide a mock API key for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-123")


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "The quick brown fox jumps over the lazy dog."


@pytest.fixture
def sample_messages():
    """Sample message list for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"}
    ]


@pytest.fixture
def sample_json_schema():
    """Sample JSON schema for testing structured outputs."""
    return {
        "name": "TestSchema",
        "description": "A test schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name field"},
                "age": {"type": "integer", "description": "Age field"}
            },
            "required": ["name", "age"],
            "additionalProperties": False
        }
    }


@pytest.fixture
def sample_documents():
    """Sample documents for RAG testing."""
    return [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing deals with text analysis.",
        "Deep learning uses neural networks with many layers.",
        "Data science combines statistics and programming."
    ]


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Ensure PYTHONPATH includes project root
    import sys
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Auto-mark tests based on filename patterns
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark tests requiring API keys
        if "requires_api" in item.keywords or "api" in item.nodeid:
            item.add_marker(pytest.mark.requires_api)
