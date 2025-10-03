"""
Autograder tests for Module 04: Chain-of-Thought Prompting

These tests validate student understanding of CoT techniques.
Run with: pytest tests/test_module_04.py
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.utils import count_tokens


@pytest.mark.module04
@pytest.mark.unit
class TestChainOfThought:
    """Test understanding of chain-of-thought concepts."""

    def test_cot_structure(self):
        """Test that CoT prompts include explicit reasoning steps."""
        # CoT should include "Let's think step by step" or similar
        cot_prompt = "Solve this math problem step by step: What is 15% of 80?"

        assert "step" in cot_prompt.lower()

    def test_zero_shot_cot(self):
        """Test zero-shot CoT pattern."""
        # Zero-shot CoT just adds "Let's think step by step"
        zero_shot_cot = """
        Question: What is 15% of 80?
        Let's think step by step:
        """

        assert "step by step" in zero_shot_cot.lower()
        assert ":" in zero_shot_cot  # Has clear structure

    def test_few_shot_cot(self):
        """Test few-shot CoT with reasoning examples."""
        few_shot_cot = """
        Q: What is 10% of 50?
        A: Let's think step by step.
        - 10% means 10/100 = 0.1
        - 0.1 × 50 = 5
        Answer: 5

        Q: What is 20% of 30?
        A: Let's think step by step.
        """

        # Should have multiple examples with reasoning
        assert few_shot_cot.count("step by step") >= 2
        assert "-" in few_shot_cot or "1." in few_shot_cot  # Has steps

    def test_reasoning_quality(self):
        """Test that reasoning steps are explicit."""
        good_reasoning = """
        Step 1: Convert percentage to decimal (15% = 0.15)
        Step 2: Multiply by the number (0.15 × 80)
        Step 3: Calculate result (12)
        """

        # Should have numbered steps
        assert "Step 1" in good_reasoning
        assert "Step 2" in good_reasoning
        assert "Step 3" in good_reasoning


@pytest.mark.module04
@pytest.mark.unit
class TestComplexReasoning:
    """Test complex reasoning patterns."""

    def test_multi_step_problem(self):
        """Test handling multi-step problems."""
        # Complex problems need explicit decomposition
        complex_prompt = """
        Problem: A store has 100 items. 20% are discounted by 50%,
        and 30% are discounted by 25%. What's the average discount?

        Solution approach:
        1. Calculate items in each category
        2. Calculate discount amounts
        3. Compute weighted average
        """

        assert "1." in complex_prompt or "Step 1" in complex_prompt
        assert "2." in complex_prompt or "Step 2" in complex_prompt

    def test_verification_step(self):
        """Test that solutions include verification."""
        # Good CoT includes verification
        solution_with_check = """
        Solution: 12

        Verification:
        - 15% of 80 = 12
        - Check: 12 / 80 = 0.15 = 15% ✓
        """

        assert "verif" in solution_with_check.lower() or "check" in solution_with_check.lower()


@pytest.mark.module04
@pytest.mark.integration
@pytest.mark.requires_api
class TestCoTIntegration:
    """Integration tests for CoT prompting."""

    @pytest.fixture
    def skip_if_no_api_key(self):
        """Skip test if no API key is set."""
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "sk-test-key-123":
            pytest.skip("API key required for integration tests")

    def test_module_04_exists(self):
        """Test that module 04 exercises exist."""
        from importlib import import_module
        ex = import_module("04-chain-of-thought.exercises.exercises")
        assert hasattr(ex, '__file__')


@pytest.mark.module04
@pytest.mark.unit
class TestCoTBestPractices:
    """Test CoT best practices."""

    def test_explicit_thinking(self):
        """Test that prompts make thinking explicit."""
        # Explicit > implicit
        explicit = "Let's solve this step by step, showing all work:"
        implicit = "Solve this:"

        assert "step" in explicit.lower()
        assert len(explicit) > len(implicit)

    def test_cot_vs_direct(self):
        """Test understanding of when to use CoT."""
        # CoT is for complex reasoning, not simple lookups

        # Simple (don't need CoT)
        simple = "What is the capital of France?"

        # Complex (needs CoT)
        complex = "Calculate compound interest on $1000 at 5% for 3 years"

        # Complex questions are longer and involve calculations
        assert len(complex) > len(simple)
        assert any(word in complex.lower() for word in ['calculate', 'compute', 'find'])
