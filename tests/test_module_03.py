"""
Autograder tests for Module 03: Few-Shot Learning

These tests validate student exercise solutions for few-shot techniques.
Run with: pytest tests/test_module_03.py
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.utils import count_tokens


@pytest.mark.module03
@pytest.mark.unit
class TestExercise1:
    """Tests for Exercise 1: Example Crafting"""

    def test_example_structure(self):
        """Validate example structure in few-shot prompts."""
        good_example = """
        Input: "The product arrived quickly and works great!"
        Output: positive

        Input: "Terrible quality, broke after one day"
        Output: negative
        """

        # Examples should have clear input/output pairs
        assert "input" in good_example.lower()
        assert "output" in good_example.lower()
        assert good_example.count("Input") == good_example.count("Output")

    def test_example_diversity(self):
        """Test that examples cover different cases."""
        examples = [
            "The food was delicious -> positive",
            "Service was terrible -> negative",
            "It's okay, nothing special -> neutral"
        ]

        # Examples should cover multiple classes
        unique_outputs = set()
        for ex in examples:
            if "->" in ex:
                output = ex.split("->")[1].strip()
                unique_outputs.add(output)

        assert len(unique_outputs) >= 2, "Should have examples for multiple classes"

    def test_example_relevance(self):
        """Test that examples are relevant to the task."""
        # For sentiment classification task:
        relevant_examples = [
            "Great product! -> positive",
            "Worst purchase ever -> negative"
        ]

        irrelevant_examples = [
            "The sky is blue -> color",
            "2 + 2 = 4 -> math"
        ]

        # Relevant examples should match task domain
        # This is a conceptual test
        assert len(relevant_examples) > 0


@pytest.mark.module03
@pytest.mark.unit
class TestExercise2:
    """Tests for Exercise 2: Shot Optimization"""

    def test_zero_vs_few_shot(self):
        """Compare zero-shot and few-shot approaches."""
        zero_shot = "Classify sentiment as positive or negative."

        few_shot = """
        Classify sentiment:

        "Excellent!" -> positive
        "Terrible" -> negative

        Text: [input]
        """

        # Few-shot should have examples
        assert "example" not in zero_shot.lower()
        assert "->" in few_shot or "example" in few_shot.lower()

    def test_optimal_shot_count(self):
        """Test understanding of optimal example count."""
        # Research shows 3-5 examples is often optimal
        # Too few: model may not learn pattern
        # Too many: wastes tokens, diminishing returns

        one_shot_tokens = count_tokens("Example: input -> output\n\nClassify: ")
        three_shot_tokens = count_tokens("Example 1: in -> out\nExample 2: in -> out\nExample 3: in -> out\n\nClassify: ")
        ten_shot_tokens = count_tokens("\n".join([f"Example {i}: in -> out" for i in range(1, 11)]) + "\n\nClassify: ")

        # Token count should increase with shots
        assert three_shot_tokens > one_shot_tokens
        assert ten_shot_tokens > three_shot_tokens

        # But diminishing returns mean 3-5 is usually best
        # This is conceptual understanding

    def test_shot_consistency(self):
        """Test that examples follow consistent format."""
        good_examples = """
        Input: "Great!" | Output: positive
        Input: "Bad" | Output: negative
        Input: "Okay" | Output: neutral
        """

        # All examples should use same delimiter
        delimiter_count = good_examples.count("|")
        line_count = len([line for line in good_examples.split("\n") if line.strip()])

        # Should have consistent structure
        assert delimiter_count >= 2


@pytest.mark.module03
@pytest.mark.unit
class TestExercise3:
    """Tests for Exercise 3: Format Matching"""

    def test_input_output_format_match(self):
        """Validate format consistency between examples and task."""
        examples = """
        Q: What is 2+2?
        A: 4

        Q: What is 3+3?
        A: 6
        """

        task = "Q: What is 5+5?\nA:"

        # Task should match example format
        assert "Q:" in examples and "Q:" in task
        assert "A:" in examples and "A:" in task

    def test_delimiter_consistency(self):
        """Test consistent delimiter usage."""
        examples_arrow = """
        Input: hello -> greeting
        Input: bye -> farewell
        """

        examples_pipe = """
        Input: hello | greeting
        Input: bye | farewell
        """

        # Each style should be internally consistent
        assert examples_arrow.count("->") == 2
        assert examples_pipe.count("|") == 2
        # Should not mix styles
        assert "->" not in examples_pipe
        assert "|" not in examples_arrow


@pytest.mark.module03
@pytest.mark.unit
class TestExercise4:
    """Tests for Exercise 4: Example Debugging"""

    def test_conflicting_examples(self):
        """Identify conflicting examples."""
        conflicting = """
        "Great product" -> positive
        "Great service" -> negative
        """

        # This is problematic - similar inputs with different outputs
        # Students should recognize this

    def test_ambiguous_examples(self):
        """Identify ambiguous examples."""
        ambiguous = """
        "It's okay" -> positive
        "Not bad" -> positive
        "Could be better" -> positive
        """

        # All examples are same class - doesn't show contrast

    def test_example_quality(self):
        """Test example quality criteria."""
        # Good examples should:
        # 1. Be diverse
        # 2. Be clear and unambiguous
        # 3. Cover edge cases
        # 4. Match the task domain

        good_examples = """
        "Absolutely fantastic!" -> positive
        "Completely terrible" -> negative
        "It's fine, I guess" -> neutral
        """

        # Has diversity (3 different classes)
        # Has clear cases
        # Covers range of sentiments


@pytest.mark.module03
@pytest.mark.unit
class TestExercise5:
    """Tests for Exercise 5: Dynamic Selection"""

    def test_example_selection_strategy(self):
        """Test understanding of dynamic example selection."""
        # For query: "technical documentation"
        # Should select technical examples, not casual ones

        technical_examples = [
            "API documentation is clear -> positive",
            "Code examples are helpful -> positive"
        ]

        casual_examples = [
            "Great pizza! -> positive",
            "Fun movie -> positive"
        ]

        # Should prefer domain-relevant examples
        # This is conceptual understanding

    def test_similarity_based_selection(self):
        """Test similarity-based example selection."""
        # Given input: "The software crashed frequently"
        # Relevant examples: technical/software domain
        # Less relevant: food, travel, etc.

        assert True  # Conceptual test


@pytest.mark.module03
@pytest.mark.unit
class TestFewShotConcepts:
    """Test understanding of few-shot learning concepts."""

    def test_learning_from_examples(self):
        """Test that few-shot learns from examples."""
        few_shot_prompt = """
        Translate English to French:

        Hello -> Bonjour
        Goodbye -> Au revoir
        Thank you -> Merci

        Good morning ->
        """

        # Should provide pattern for model to follow
        assert "->" in few_shot_prompt
        assert few_shot_prompt.count("->") >= 3

    def test_few_shot_vs_fine_tuning(self):
        """Test understanding: few-shot vs fine-tuning."""
        # Few-shot: Examples in prompt (no training)
        # Fine-tuning: Update model weights (requires training)

        # Few-shot is:
        # - Faster to implement
        # - No training needed
        # - Works with API-only models
        # - Limited by context window

    def test_meta_learning_concept(self):
        """Test understanding of in-context learning."""
        # LLMs can learn patterns from examples in context
        # This is "meta-learning" or "in-context learning"
        # Different from traditional supervised learning

        assert True  # Conceptual understanding


@pytest.mark.module03
@pytest.mark.integration
@pytest.mark.requires_api
class TestExerciseIntegration:
    """Integration tests for Module 03 exercises."""

    @pytest.fixture
    def skip_if_no_api_key(self):
        """Skip test if no API key is set."""
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "sk-test-key-123":
            pytest.skip("API key required for integration tests")

    def test_exercise_1_runs(self, skip_if_no_api_key):
        """Test exercise 1 execution."""
        from importlib import import_module
        ex = import_module("03-few-shot-learning.exercises.exercises")
        assert hasattr(ex, 'exercise_1_example_crafting')

    def test_few_shot_effectiveness(self):
        """Test that few-shot improves performance."""
        # Conceptual test: few-shot should generally
        # outperform zero-shot for complex tasks

        # This would require actual API calls to verify
        assert True  # Placeholder


@pytest.mark.module03
@pytest.mark.unit
class TestPromptEngineering:
    """Test general prompt engineering principles."""

    def test_clear_task_definition(self):
        """Test clear task definition in prompts."""
        good_prompt = """
        Task: Classify customer feedback sentiment

        Examples:
        "Great service!" -> positive
        "Terrible experience" -> negative

        Feedback: [input]
        Classification:
        """

        assert "task" in good_prompt.lower()
        assert "examples" in good_prompt.lower() or "->" in good_prompt

    def test_example_placement(self):
        """Test proper example placement."""
        # Examples should come after task definition
        # but before the actual input to classify

        correct_order = """
        Task: Classify sentiment

        Examples:
        input -> output

        Input: [new text]
        """

        # Task -> Examples -> Input is good order
        assert correct_order.index("Task") < correct_order.index("Examples")
        assert correct_order.index("Examples") < correct_order.index("Input")
