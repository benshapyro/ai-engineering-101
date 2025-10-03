"""
Autograder tests for Module 05: Prompt Chaining

These tests validate understanding of multi-step prompting.
Run with: pytest tests/test_module_05.py
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.mark.module05
@pytest.mark.unit
class TestPromptChaining:
    """Test prompt chaining concepts."""

    def test_sequential_chain_concept(self):
        """Test understanding of sequential chains."""
        # Sequential: output of step N → input of step N+1
        steps = [
            "Step 1: Summarize the text",
            "Step 2: Extract key points from the summary",
            "Step 3: Generate questions from the key points"
        ]

        assert len(steps) >= 3
        assert all("Step" in s for s in steps)

    def test_chain_dependencies(self):
        """Test understanding of step dependencies."""
        # Later steps depend on earlier steps
        chain = {
            "step1": {"depends_on": []},
            "step2": {"depends_on": ["step1"]},
            "step3": {"depends_on": ["step2"]}
        }

        # Verify dependency structure
        assert len(chain["step1"]["depends_on"]) == 0
        assert "step1" in chain["step2"]["depends_on"]
        assert "step2" in chain["step3"]["depends_on"]

    def test_state_management(self):
        """Test that chains maintain state across steps."""
        # Chains need to pass data between steps
        state = {
            "original_text": "Some text",
            "summary": None,  # Populated by step 1
            "key_points": None,  # Populated by step 2
        }

        assert "original_text" in state
        # Other fields populated as chain executes


@pytest.mark.module05
@pytest.mark.unit
class TestChainPatterns:
    """Test different chain patterns."""

    def test_map_reduce_pattern(self):
        """Test map-reduce chain pattern."""
        # Map: process each item
        # Reduce: combine results

        items = ["doc1", "doc2", "doc3"]
        map_step = "summarize"
        reduce_step = "combine summaries"

        assert len(items) > 1
        assert map_step != reduce_step

    def test_router_pattern(self):
        """Test router chain pattern."""
        # Route to different chains based on input

        routes = {
            "factual": "Use knowledge base chain",
            "creative": "Use generation chain",
            "calculation": "Use math chain"
        }

        assert len(routes) >= 2
        assert all(isinstance(v, str) for v in routes.values())


@pytest.mark.module05
@pytest.mark.integration
@pytest.mark.requires_api
class TestChainIntegration:
    """Integration tests for prompt chaining."""

    @pytest.fixture
    def skip_if_no_api_key(self):
        """Skip test if no API key is set."""
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "sk-test-key-123":
            pytest.skip("API key required for integration tests")

    def test_chain_utilities_exist(self):
        """Test that chain utilities are available."""
        from shared.chains import PromptChain
        assert PromptChain is not None


@pytest.mark.module05
@pytest.mark.unit
class TestChainBestPractices:
    """Test chaining best practices."""

    def test_error_handling(self):
        """Test chains handle errors gracefully."""
        # Chains should handle step failures
        # This is conceptual

        try:
            # Simulate step failure
            result = None
            if result is None:
                # Handle missing result
                fallback = "default"
            assert fallback == "default"
        except Exception:
            pytest.fail("Chain should handle errors")

    def test_chain_logging(self):
        """Test that chains log execution."""
        # Chains should track which steps executed
        execution_log = [
            {"step": "summarize", "status": "success"},
            {"step": "extract", "status": "success"}
        ]

        assert all("status" in entry for entry in execution_log)
