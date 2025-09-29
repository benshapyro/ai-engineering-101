"""
Module 03: Few-Shot Learning - Exercises

Practice exercises for mastering few-shot prompting techniques.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from typing import List, Dict, Tuple


# ===== Exercise 1: Example Crafting =====

def exercise_1_example_crafting():
    """
    Exercise 1: Create effective examples for various tasks.

    TODO:
    1. Create 3 high-quality examples for each task type
    2. Ensure format consistency
    3. Cover edge cases
    4. Test with new inputs
    """
    client = LLMClient("openai")

    print("Exercise 1: Example Crafting")
    print("=" * 50)

    # Task 1: Create examples for data extraction
    print("\nTask 1: Create examples for extracting dates from text")

    # TODO: Create 3 examples that teach date extraction
    date_examples = [
        ("The meeting is on January 15, 2024", "..."),  # TODO: Add extraction
        ("...", "..."),  # TODO: Add example
        ("...", "...")   # TODO: Add example
    ]

    date_prompt = """Extract dates from text:

"""
    # TODO: Build prompt with your examples
    for text, date in date_examples:
        if text != "..." and date != "...":
            date_prompt += f'Text: "{text}"\nDate: {date}\n\n'

    # Test with new input
    test_text = "The project deadline is March 31st, 2024"
    date_prompt += f'Text: "{test_text}"\nDate:'

    if "..." not in date_prompt:
        response = client.complete(date_prompt, temperature=0.2, max_tokens=30)
        print(f"Test extraction: {response.strip()}")

    # Task 2: Create examples for style transformation
    print("\n" + "-" * 40)
    print("\nTask 2: Create examples for making text more professional")

    # TODO: Create 3 examples showing casual → professional transformation
    style_examples = [
        ("hey whats up", "..."),  # TODO: Add professional version
        ("...", "..."),  # TODO: Add example
        ("...", "...")   # TODO: Add example
    ]

    # Task 3: Create examples for classification
    print("\n" + "-" * 40)
    print("\nTask 3: Create examples for urgency classification")

    # TODO: Create examples for High/Medium/Low urgency
    urgency_examples = [
        ("Server is down!", "..."),  # TODO: Add classification
        ("...", "..."),  # TODO: Add example
        ("...", "...")   # TODO: Add example
    ]

    print("\nTODO: Complete the example sets above")


# ===== Exercise 2: Shot Optimization =====

def exercise_2_shot_optimization():
    """
    Exercise 2: Find the optimal number of examples for different tasks.

    TODO:
    1. Test with 0, 1, 3, 5, and 7 examples
    2. Measure consistency and accuracy
    3. Identify the sweet spot for each task
    """
    client = LLMClient("openai")

    print("Exercise 2: Shot Optimization")
    print("=" * 50)

    # Example pool for sentiment classification
    example_pool = [
        ("Amazing product!", "Positive"),
        ("Terrible experience", "Negative"),
        ("It's okay", "Neutral"),
        ("Love it!", "Positive"),
        ("Waste of money", "Negative"),
        ("Not bad", "Neutral"),
        ("Excellent service", "Positive"),
        ("Very disappointed", "Negative"),
        ("Average quality", "Neutral"),
    ]

    test_inputs = [
        "This is fantastic!",
        "Not worth the price",
        "It works as expected"
    ]

    shot_counts = [0, 1, 3, 5, 7]
    results = {}

    for shot_count in shot_counts:
        if shot_count == 0:
            # Zero-shot
            prompt_template = "Classify sentiment as Positive, Negative, or Neutral:\nText: {text}\nSentiment:"
        else:
            # Few-shot
            examples = example_pool[:shot_count]
            prompt_template = "Classify sentiment:\n\n"
            for ex_text, ex_label in examples:
                prompt_template += f'Text: "{ex_text}"\nSentiment: {ex_label}\n\n'
            prompt_template += 'Text: "{text}"\nSentiment:'

        # TODO: Test each shot count with all test inputs
        # Record accuracy and consistency

        print(f"\nTesting with {shot_count} examples...")
        # TODO: Implement testing logic

    # TODO: Analyze results and find optimal shot count
    print("\nTODO: Complete shot optimization analysis")


# ===== Exercise 3: Format Matching =====

def exercise_3_format_matching():
    """
    Exercise 3: Ensure outputs consistently match example formats.

    TODO:
    1. Create examples with strict format requirements
    2. Test format consistency across multiple runs
    3. Fix any format drift issues
    """
    client = LLMClient("openai")

    print("Exercise 3: Format Matching")
    print("=" * 50)

    # TODO: Create examples with specific JSON format
    json_format_examples = """Extract product information to JSON:

Text: "The iPhone 15 costs $999 and has 128GB storage"
JSON: {{"product": "iPhone 15", "price": 999, "storage": "128GB"}}

Text: "Samsung Galaxy S24 is priced at $899 with 256GB"
JSON: {{"product": "Samsung Galaxy S24", "price": 899, "storage": "256GB"}}

TODO: Add one more example here

Text: "Google Pixel 8 available for $699 with 128GB memory"
JSON:"""

    # TODO: Test format consistency
    test_runs = 5
    responses = []

    print("Testing format consistency...")
    # TODO: Run multiple times and check if format stays consistent

    # TODO: Implement format validation
    def validate_json_format(response: str) -> bool:
        """Check if response matches expected JSON format."""
        try:
            data = json.loads(response)
            required_keys = {"product", "price", "storage"}
            return required_keys.issubset(data.keys())
        except:
            return False

    print("\nTODO: Complete format matching exercise")


# ===== Exercise 4: Example Debugging =====

def exercise_4_example_debugging():
    """
    Exercise 4: Fix problematic few-shot prompts.

    TODO:
    1. Identify issues in the given prompts
    2. Fix the problems
    3. Test the improvements
    """
    client = LLMClient("openai")

    print("Exercise 4: Example Debugging")
    print("=" * 50)

    # Problematic Prompt 1: Inconsistent format
    bad_prompt_1 = """Translate to French:

hello → bonjour
How are you? = Comment allez-vous?
goodbye: au revoir

translate: Thank you
answer:"""

    print("PROBLEM 1: Inconsistent format")
    print("Current prompt has mixed separators (→, =, :)")

    # TODO: Fix the format consistency
    fixed_prompt_1 = """Translate to French:

TODO: Create consistent format

translate: Thank you
French:"""

    # Problematic Prompt 2: Poor example quality
    bad_prompt_2 = """Extract key information:

Text: "x"
Info: x

Text: "The meeting is at 3pm"
Info: 3pm

Text: "Call John Smith at 555-1234 about the project"
Info:"""

    print("\n" + "-" * 40)
    print("\nPROBLEM 2: Poor example quality")
    print("Examples are too simple and inconsistent")

    # TODO: Improve example quality
    fixed_prompt_2 = """Extract key information:

TODO: Create better examples

Text: "Call John Smith at 555-1234 about the project"
Info:"""

    # Problematic Prompt 3: Missing edge cases
    bad_prompt_3 = """Format phone numbers:

5551234567 → (555) 123-4567
9876543210 → (987) 654-3210

Format: 555-123-4567
Result:"""

    print("\n" + "-" * 40)
    print("\nPROBLEM 3: Missing edge cases")
    print("No examples for already-formatted or invalid numbers")

    # TODO: Add edge case examples
    fixed_prompt_3 = """Format phone numbers:

TODO: Add comprehensive examples including edge cases

Format: 555-123-4567
Result:"""

    print("\nTODO: Complete debugging and test fixes")


# ===== Exercise 5: Dynamic Selection =====

def exercise_5_dynamic_selection():
    """
    Exercise 5: Build a system that selects examples based on input.

    TODO:
    1. Create an example library
    2. Implement selection logic based on input characteristics
    3. Test with various inputs
    """
    client = LLMClient("openai")

    print("Exercise 5: Dynamic Selection")
    print("=" * 50)

    # TODO: Build example library
    example_library = {
        "technical": [
            # TODO: Add technical examples
        ],
        "business": [
            # TODO: Add business examples
        ],
        "casual": [
            # TODO: Add casual examples
        ]
    }

    def classify_input_type(text: str) -> str:
        """
        TODO: Implement logic to classify input type

        Returns: "technical", "business", or "casual"
        """
        # TODO: Add classification logic
        return "casual"  # Placeholder

    def select_relevant_examples(input_text: str, library: Dict, n: int = 3) -> List:
        """
        TODO: Select relevant examples based on input
        """
        input_type = classify_input_type(input_text)
        # TODO: Select and return relevant examples
        return []

    # Test inputs
    test_inputs = [
        "API endpoint returns 500 error",
        "Q3 revenue exceeded projections",
        "Hey, want to grab lunch?"
    ]

    for test_input in test_inputs:
        print(f"\nInput: '{test_input}'")
        selected = select_relevant_examples(test_input, example_library)
        print(f"Selected examples: {len(selected)} from {classify_input_type(test_input)} category")

    print("\nTODO: Implement dynamic selection system")


# ===== Challenge: Automatic Example Generation =====

def challenge_automatic_example_generation():
    """
    Challenge: Build a system that automatically generates good examples.

    Requirements:
    1. Given a task description, generate appropriate examples
    2. Ensure examples are diverse and cover edge cases
    3. Maintain format consistency
    4. Validate example quality

    TODO: Complete the implementation
    """
    client = LLMClient("openai")

    print("Challenge: Automatic Example Generation")
    print("=" * 50)

    def generate_examples(task_description: str, num_examples: int = 3) -> List[Tuple[str, str]]:
        """
        TODO: Generate examples automatically for a given task.

        Args:
            task_description: Description of the task
            num_examples: Number of examples to generate

        Returns:
            List of (input, output) tuples
        """
        # TODO: Implement example generation
        examples = []

        # Step 1: Understand the task
        # Step 2: Generate diverse inputs
        # Step 3: Generate corresponding outputs
        # Step 4: Validate quality

        return examples

    def validate_examples(examples: List[Tuple[str, str]], task_description: str) -> Dict:
        """
        TODO: Validate the quality of generated examples.

        Returns:
            Dictionary with validation metrics
        """
        validation_results = {
            "format_consistency": False,
            "coverage": False,
            "clarity": False,
            "diversity": False
        }

        # TODO: Implement validation logic

        return validation_results

    # Test the system
    tasks = [
        "Extract email addresses from text",
        "Convert informal text to formal business language",
        "Classify customer support tickets by priority"
    ]

    for task in tasks:
        print(f"\nTask: {task}")
        examples = generate_examples(task, num_examples=3)

        if examples:
            print(f"Generated {len(examples)} examples")
            for i, (inp, out) in enumerate(examples, 1):
                print(f"  Example {i}: {inp[:50]}... → {out[:50]}...")

            validation = validate_examples(examples, task)
            print(f"Validation: {validation}")
        else:
            print("TODO: Implement example generation")

    print("\nTODO: Complete automatic example generation system")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 03: Few-Shot Exercises")
    parser.add_argument("--exercise", type=int, help="Run specific exercise (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge")
    parser.add_argument("--all", action="store_true", help="Run all exercises")

    args = parser.parse_args()

    exercises = {
        1: exercise_1_example_crafting,
        2: exercise_2_shot_optimization,
        3: exercise_3_format_matching,
        4: exercise_4_example_debugging,
        5: exercise_5_dynamic_selection
    }

    if args.all:
        for ex_num, ex_func in exercises.items():
            print(f"\n{'='*60}")
            ex_func()
            print(f"\n{'='*60}\n")
        challenge_automatic_example_generation()
    elif args.challenge:
        challenge_automatic_example_generation()
    elif args.exercise and args.exercise in exercises:
        exercises[args.exercise]()
    else:
        print("Module 03: Few-Shot Learning - Exercises")
        print("\nUsage:")
        print("  python exercises.py --exercise N  # Run exercise N")
        print("  python exercises.py --challenge    # Run challenge")
        print("  python exercises.py --all          # Run all exercises")
        print("\nAvailable exercises:")
        print("  1: Example Crafting")
        print("  2: Shot Optimization")
        print("  3: Format Matching")
        print("  4: Example Debugging")
        print("  5: Dynamic Selection")
        print("  Challenge: Automatic Example Generation")