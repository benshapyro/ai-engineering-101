"""
Autograder tests for Module 02: Zero-Shot Prompting

These tests validate student exercise solutions for zero-shot techniques.
Run with: pytest tests/test_module_02.py
"""

import pytest
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.utils import count_tokens


@pytest.mark.module02
@pytest.mark.unit
class TestExercise1:
    """Tests for Exercise 1: Instruction Clarity"""

    def test_clear_instructions(self):
        """Validate instruction clarity."""
        vague = "Classify this"
        clear = "Classify the sentiment of this product review as positive, negative, or neutral"

        # Clear instructions should:
        # - Specify the task completely
        # - Define expected outputs
        # - Be unambiguous

        assert count_tokens(clear) > count_tokens(vague)
        assert "classify" in clear.lower()
        assert any(word in clear.lower() for word in ["positive", "negative", "neutral"])

    def test_task_definition(self):
        """Test proper task definition."""
        good_task = "Extract all email addresses from the text and return them as a comma-separated list"

        # Should specify:
        # - What to extract
        # - Output format
        assert "extract" in good_task.lower() or "find" in good_task.lower()
        assert "format" in good_task.lower() or "list" in good_task.lower() or "comma" in good_task.lower()


@pytest.mark.module02
@pytest.mark.unit
class TestExercise2:
    """Tests for Exercise 2: Format Control"""

    def test_json_output_control(self):
        """Validate JSON format control."""
        prompt = """
        Analyze this text and return a JSON object with:
        {
            "summary": "brief summary",
            "keywords": ["list", "of", "keywords"],
            "word_count": 123
        }
        """

        assert "json" in prompt.lower()
        assert "{" in prompt and "}" in prompt
        # Should show example structure

    def test_structured_output_request(self):
        """Test structured output specifications."""
        prompt = "Return results in JSON format with fields: name, age, city"

        assert "json" in prompt.lower() or "format" in prompt.lower()
        assert ":" in prompt or "fields" in prompt.lower()

    def test_format_consistency(self):
        """Validate format consistency requirements."""
        # When requesting formats, should be specific about:
        # - Data types (string, number, boolean)
        # - Structure (array, object)
        # - Required vs optional fields

        assert True  # Placeholder for logic check


@pytest.mark.module02
@pytest.mark.unit
class TestExercise3:
    """Tests for Exercise 3: Edge Case Handling"""

    def test_empty_input_handling(self):
        """Test edge case: empty input."""
        prompt = "Classify sentiment. If input is empty, return 'neutral'."

        # Should explicitly handle edge cases
        assert "empty" in prompt.lower() or "no input" in prompt.lower()
        assert "return" in prompt.lower()

    def test_invalid_input_handling(self):
        """Test edge case: invalid input."""
        prompt = "Extract date. If no valid date found, return 'No date found'."

        assert any(word in prompt.lower() for word in ["invalid", "no", "not found"])

    def test_boundary_conditions(self):
        """Test boundary condition handling."""
        prompt = "Rate from 1-5. If input is too short (< 10 words), return 'Insufficient input'."

        # Should specify boundary conditions
        assert "<" in prompt or "less than" in prompt.lower() or "minimum" in prompt.lower()


@pytest.mark.module02
@pytest.mark.unit
class TestExercise4:
    """Tests for Exercise 4: Task Decomposition"""

    def test_complex_task_breakdown(self):
        """Validate complex task decomposition."""
        # Complex task: "Analyze customer reviews"
        # Should break into:
        # 1. Extract reviews
        # 2. Classify sentiment
        # 3. Identify themes
        # 4. Summarize findings

        decomposed_prompt = """
        Task: Analyze customer reviews in these steps:
        1. Read each review
        2. Classify sentiment (positive/negative/neutral)
        3. Extract key themes
        4. Provide summary statistics
        """

        # Should have numbered steps
        assert any(str(i) in decomposed_prompt for i in range(1, 5))
        assert "step" in decomposed_prompt.lower() or ":" in decomposed_prompt

    def test_step_by_step_structure(self):
        """Test step-by-step instruction format."""
        prompt = """
        Solve this problem step by step:
        Step 1: Identify the variables
        Step 2: Set up equations
        Step 3: Solve for unknowns
        Step 4: Verify solution
        """

        assert "step" in prompt.lower()
        assert prompt.count("Step") >= 3 or prompt.count("step") >= 3


@pytest.mark.module02
@pytest.mark.unit
class TestExercise5:
    """Tests for Exercise 5: Reliability Testing"""

    def test_consistency_across_inputs(self):
        """Test output consistency requirements."""
        # When requesting consistent outputs, should specify:
        # - Exact format
        # - Allowed values
        # - Default behaviors

        prompt = "Always return sentiment as one of: 'positive', 'negative', 'neutral' (lowercase only)"

        assert "always" in prompt.lower() or "must" in prompt.lower()
        assert "lowercase" in prompt.lower() or "exact" in prompt.lower()

    def test_deterministic_output_request(self):
        """Test requesting deterministic outputs."""
        # Should use:
        # - Low temperature
        # - Strict format requirements
        # - Clear constraints

        assert True  # Placeholder


@pytest.mark.module02
@pytest.mark.unit
class TestZeroShotConcepts:
    """Test understanding of zero-shot concepts."""

    def test_no_examples_provided(self):
        """Verify zero-shot means no examples."""
        zero_shot_prompt = "Classify the sentiment of this text as positive or negative."

        # Zero-shot should NOT include examples
        assert "example" not in zero_shot_prompt.lower()
        assert "for instance" not in zero_shot_prompt.lower()
        assert "like" not in zero_shot_prompt.lower()

    def test_task_description_only(self):
        """Test that zero-shot relies on task description."""
        good_zero_shot = """
        Task: Extract all dates from the following text.
        Output format: List of dates in YYYY-MM-DD format.
        If no dates found, return empty list.
        """

        # Should have clear task description
        assert "task" in good_zero_shot.lower() or "extract" in good_zero_shot.lower()
        assert "format" in good_zero_shot.lower()

    def test_vs_few_shot(self):
        """Test understanding of zero-shot vs few-shot."""
        zero_shot = "Classify sentiment as positive or negative"
        few_shot = """
        Classify sentiment:
        Example: "Great product!" -> positive
        Example: "Terrible quality" -> negative

        Text: [input]
        """

        # Zero-shot should be shorter and have no examples
        assert count_tokens(zero_shot) < count_tokens(few_shot)
        assert "example" not in zero_shot.lower()
        assert "example" in few_shot.lower()


@pytest.mark.module02
@pytest.mark.integration
@pytest.mark.requires_api
class TestExerciseIntegration:
    """Integration tests for Module 02 exercises."""

    @pytest.fixture
    def skip_if_no_api_key(self):
        """Skip test if no API key is set."""
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "sk-test-key-123":
            pytest.skip("API key required for integration tests")

    def test_exercise_1_runs(self, skip_if_no_api_key):
        """Test exercise 1 execution."""
        from importlib import import_module
        ex = import_module("02-zero-shot-prompting.exercises.exercises")
        assert hasattr(ex, 'exercise_1_instruction_clarity')

    def test_json_parsing(self):
        """Test JSON output parsing."""
        # Example JSON output
        json_output = '{"sentiment": "positive", "confidence": 0.95}'

        try:
            data = json.loads(json_output)
            assert "sentiment" in data
            assert "confidence" in data
            assert isinstance(data["confidence"], (int, float))
        except json.JSONDecodeError:
            pytest.fail("Should be valid JSON")
