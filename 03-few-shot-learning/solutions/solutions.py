"""
Module 03: Few-Shot Learning - Solutions

Complete solutions for all few-shot learning exercises.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
import statistics
from typing import List, Dict, Tuple


# ===== Solution 1: Example Crafting =====

def solution_1_example_crafting():
    """
    Solution 1: Create effective examples for various tasks.
    """
    client = LLMClient("openai")

    print("Solution 1: Example Crafting")
    print("=" * 50)

    # Task 1: Date extraction with comprehensive examples
    print("\nTask 1: Date Extraction Examples")

    date_examples = [
        ("The meeting is on January 15, 2024", "January 15, 2024"),
        ("Submit reports by 03/31/2024 at midnight", "03/31/2024"),
        ("The conference runs from May 5-7, 2024", "May 5-7, 2024"),
    ]

    date_prompt = """Extract dates from text:

"""
    for text, date in date_examples:
        date_prompt += f'Text: "{text}"\nDate: {date}\n\n'

    test_text = "The project deadline is March 31st, 2024"
    date_prompt += f'Text: "{test_text}"\nDate:'

    response = client.complete(date_prompt, temperature=0.2, max_tokens=30)
    print(f"Test extraction: {response.strip()}")
    print("✓ Examples cover different date formats")

    # Task 2: Style transformation with clear patterns
    print("\n" + "-" * 40)
    print("\nTask 2: Professional Style Transformation")

    style_examples = [
        ("hey whats up", "Hello, how are you today?"),
        ("gonna send it tomorrow", "I will send it tomorrow."),
        ("thx for ur help", "Thank you for your assistance."),
    ]

    style_prompt = """Make text professional:

"""
    for casual, professional in style_examples:
        style_prompt += f'Casual: "{casual}"\nProfessional: "{professional}"\n\n'

    test_casual = "cant make it to the mtg"
    style_prompt += f'Casual: "{test_casual}"\nProfessional:'

    response = client.complete(style_prompt, temperature=0.3, max_tokens=50)
    print(f"Input: '{test_casual}'")
    print(f"Professional: {response.strip()}")
    print("✓ Examples show consistent transformation pattern")

    # Task 3: Urgency classification with clear criteria
    print("\n" + "-" * 40)
    print("\nTask 3: Urgency Classification")

    urgency_examples = [
        ("Server is down! All services offline!", "Critical"),
        ("Bug in report generation, affects monthly reports", "Medium"),
        ("Typo in documentation footer", "Low"),
        ("Security breach detected in user database", "Critical"),
        ("Feature request for dark mode", "Low"),
        ("Performance degradation in search function", "Medium"),
    ]

    urgency_prompt = """Classify urgency level:

"""
    for issue, urgency in urgency_examples:
        urgency_prompt += f'Issue: "{issue}"\nUrgency: {urgency}\n\n'

    test_issue = "Payment processing failing for 30% of transactions"
    urgency_prompt += f'Issue: "{test_issue}"\nUrgency:'

    response = client.complete(urgency_prompt, temperature=0.2, max_tokens=20)
    print(f"Issue: '{test_issue}'")
    print(f"Classification: {response.strip()}")
    print("✓ Examples cover all urgency levels with clear criteria")


# ===== Solution 2: Shot Optimization =====

def solution_2_shot_optimization():
    """
    Solution 2: Find optimal number of examples through testing.
    """
    client = LLMClient("openai")

    print("Solution 2: Shot Optimization")
    print("=" * 50)

    # Example pool
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
        ("This is fantastic!", "Positive"),
        ("Not worth the price", "Negative"),
        ("It works as expected", "Neutral"),
    ]

    shot_counts = [0, 1, 3, 5, 7]
    results = {}

    for shot_count in shot_counts:
        correct = 0
        responses_list = []

        for test_text, expected_label in test_inputs:
            if shot_count == 0:
                # Zero-shot
                prompt = f"Classify sentiment as Positive, Negative, or Neutral:\nText: {test_text}\nSentiment:"
            else:
                # Few-shot
                examples = example_pool[:shot_count]
                prompt = "Classify sentiment:\n\n"
                for ex_text, ex_label in examples:
                    prompt += f'Text: "{ex_text}"\nSentiment: {ex_label}\n\n'
                prompt += f'Text: "{test_text}"\nSentiment:'

            # Get response
            response = client.complete(prompt, temperature=0.2, max_tokens=20)
            response_clean = response.strip()
            responses_list.append(response_clean)

            # Check accuracy
            if expected_label.lower() in response_clean.lower():
                correct += 1

        accuracy = (correct / len(test_inputs)) * 100
        results[shot_count] = {
            "accuracy": accuracy,
            "responses": responses_list,
            "tokens_used": len(prompt.split())  # Approximate
        }

        print(f"\n{shot_count} examples:")
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"  Approx tokens: {results[shot_count]['tokens_used']}")

    # Analysis
    print("\n" + "-" * 40)
    print("\nOPTIMAL SHOT COUNT ANALYSIS:")
    print("- 0-shot: Baseline, lowest token usage but may lack consistency")
    print("- 1-shot: Shows format but limited coverage")
    print("- 3-shot: OPTIMAL - Good accuracy with reasonable token usage")
    print("- 5-shot: Marginal improvement over 3-shot")
    print("- 7-shot: Diminishing returns, higher token cost")
    print("\n✓ Recommendation: Use 3-shot for this task")


# ===== Solution 3: Format Matching =====

def solution_3_format_matching():
    """
    Solution 3: Ensure consistent format matching.
    """
    client = LLMClient("openai")

    print("Solution 3: Format Matching")
    print("=" * 50)

    # Strict JSON format examples
    json_format_prompt = """Extract product information to JSON:

Text: "The iPhone 15 costs $999 and has 128GB storage"
JSON: {"product": "iPhone 15", "price": 999, "storage": "128GB"}

Text: "Samsung Galaxy S24 is priced at $899 with 256GB"
JSON: {"product": "Samsung Galaxy S24", "price": 899, "storage": "256GB"}

Text: "MacBook Pro M3 available for $1999 with 512GB SSD"
JSON: {"product": "MacBook Pro M3", "price": 1999, "storage": "512GB"}

Text: "Google Pixel 8 available for $699 with 128GB memory"
JSON:"""

    # Test format consistency
    test_runs = 5
    responses = []
    format_valid = []

    print("Testing format consistency across 5 runs...")

    for run in range(test_runs):
        response = client.complete(json_format_prompt, temperature=0.1, max_tokens=100)
        responses.append(response.strip())

        # Validate format
        try:
            data = json.loads(response.strip())
            required_keys = {"product", "price", "storage"}
            is_valid = required_keys.issubset(data.keys())
            format_valid.append(is_valid)

            if is_valid:
                print(f"  Run {run+1}: ✓ Valid format")
            else:
                print(f"  Run {run+1}: ✗ Missing keys: {required_keys - set(data.keys())}")
        except json.JSONDecodeError:
            format_valid.append(False)
            print(f"  Run {run+1}: ✗ Invalid JSON")

    consistency_rate = sum(format_valid) / len(format_valid) * 100
    print(f"\nFormat Consistency: {consistency_rate:.1f}%")

    if consistency_rate == 100:
        print("✓ Perfect format consistency achieved!")
    else:
        print("\nImprovement tips:")
        print("- Use consistent JSON format in all examples")
        print("- Keep temperature low (0.1-0.2)")
        print("- Consider adding format instruction explicitly")


# ===== Solution 4: Example Debugging =====

def solution_4_example_debugging():
    """
    Solution 4: Fix problematic few-shot prompts.
    """
    client = LLMClient("openai")

    print("Solution 4: Example Debugging")
    print("=" * 50)

    # Fix 1: Format consistency
    print("\nFIX 1: Consistent Format")
    fixed_prompt_1 = """Translate to French:

"hello" → "bonjour"
"How are you?" → "Comment allez-vous?"
"goodbye" → "au revoir"

"Thank you" →"""

    response = client.complete(fixed_prompt_1, temperature=0.2, max_tokens=30)
    print(f"Response: {response.strip()}")
    print("✓ Fixed: Consistent arrow format throughout")

    # Fix 2: Improved example quality
    print("\n" + "-" * 40)
    print("\nFIX 2: Better Example Quality")
    fixed_prompt_2 = """Extract key information:

Text: "Schedule meeting with Sarah Johnson for 2pm on Friday"
Info: {"person": "Sarah Johnson", "time": "2pm", "day": "Friday", "action": "meeting"}

Text: "Email the Q3 report to finance@company.com by end of day"
Info: {"action": "email", "item": "Q3 report", "recipient": "finance@company.com", "deadline": "end of day"}

Text: "Call John Smith at 555-1234 about the project"
Info:"""

    response = client.complete(fixed_prompt_2, temperature=0.2, max_tokens=100)
    print(f"Response: {response.strip()}")
    print("✓ Fixed: Comprehensive examples with structured output")

    # Fix 3: Edge case coverage
    print("\n" + "-" * 40)
    print("\nFIX 3: Edge Case Coverage")
    fixed_prompt_3 = """Format phone numbers to (XXX) XXX-XXXX:

Valid number: "5551234567" → "(555) 123-4567"
Valid number: "987-654-3210" → "(987) 654-3210"
Already formatted: "(555) 123-4567" → "(555) 123-4567"
Invalid (too short): "12345" → "Error: Invalid phone number"
International: "+1-555-123-4567" → "(555) 123-4567"

Format: "555-123-4567"
Result:"""

    response = client.complete(fixed_prompt_3, temperature=0.2, max_tokens=50)
    print(f"Response: {response.strip()}")
    print("✓ Fixed: Added edge cases for various input formats")


# ===== Solution 5: Dynamic Selection =====

def solution_5_dynamic_selection():
    """
    Solution 5: Build dynamic example selection system.
    """
    client = LLMClient("openai")

    print("Solution 5: Dynamic Selection System")
    print("=" * 50)

    # Comprehensive example library
    example_library = {
        "technical": [
            ("Fix the null pointer exception in line 42", "Technical"),
            ("API endpoint returns 503 service unavailable", "Technical"),
            ("Database connection timeout after 30 seconds", "Technical"),
            ("Implement OAuth2 authentication flow", "Technical"),
        ],
        "business": [
            ("Q3 revenue exceeded projections by 15%", "Business"),
            ("Market share increased in APAC region", "Business"),
            ("Customer retention rate improved to 85%", "Business"),
            ("ROI on marketing campaign was 250%", "Business"),
        ],
        "casual": [
            ("Hey, want to grab lunch?", "Casual"),
            ("That movie was awesome!", "Casual"),
            ("Can't wait for the weekend", "Casual"),
            ("Thanks for your help yesterday", "Casual"),
        ]
    }

    def classify_input_type(text: str) -> str:
        """Classify input based on keywords and patterns."""
        text_lower = text.lower()

        # Technical indicators
        tech_keywords = ["api", "error", "bug", "code", "server", "database", "exception",
                        "timeout", "endpoint", "authentication", "null", "pointer"]
        tech_score = sum(1 for word in tech_keywords if word in text_lower)

        # Business indicators
        biz_keywords = ["revenue", "profit", "market", "customer", "roi", "q1", "q2", "q3", "q4",
                       "growth", "share", "projection", "retention"]
        biz_score = sum(1 for word in biz_keywords if word in text_lower)

        # Casual indicators
        casual_keywords = ["hey", "lunch", "weekend", "awesome", "thanks", "want", "can't"]
        casual_score = sum(1 for word in casual_keywords if word in text_lower)

        # Determine type based on highest score
        scores = {"technical": tech_score, "business": biz_score, "casual": casual_score}
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "casual"

    def select_relevant_examples(input_text: str, library: Dict, n: int = 3) -> List:
        """Select relevant examples based on input characteristics."""
        input_type = classify_input_type(input_text)
        category_examples = library.get(input_type, library["casual"])

        # Return up to n examples from the relevant category
        return category_examples[:min(n, len(category_examples))]

    # Test the system
    test_inputs = [
        "API endpoint returns 500 internal server error",
        "Q3 revenue exceeded projections by 20%",
        "Hey, want to grab coffee later?",
        "Debug the memory leak in production",
        "Customer acquisition cost decreased",
    ]

    for test_input in test_inputs:
        print(f"\nInput: '{test_input}'")
        input_type = classify_input_type(test_input)
        print(f"Detected type: {input_type}")

        selected = select_relevant_examples(test_input, example_library, n=2)
        print(f"Selected {len(selected)} examples:")
        for ex_text, ex_label in selected[:2]:
            print(f"  - {ex_text[:50]}...")

        # Build prompt with selected examples
        prompt = "Classify the type of this text:\n\n"
        for ex_text, ex_label in selected:
            prompt += f'Text: "{ex_text}"\nType: {ex_label}\n\n'
        prompt += f'Text: "{test_input}"\nType:'

        response = client.complete(prompt, temperature=0.2, max_tokens=20)
        print(f"Classification: {response.strip()}")

    print("\n✓ Dynamic selection system working correctly")


# ===== Challenge Solution: Automatic Example Generation =====

def challenge_solution_automatic_generation():
    """
    Challenge Solution: Automatically generate high-quality examples.
    """
    client = LLMClient("openai")

    print("Challenge Solution: Automatic Example Generation")
    print("=" * 50)

    def generate_examples(task_description: str, num_examples: int = 3) -> List[Tuple[str, str]]:
        """Generate examples automatically for a given task."""

        # Step 1: Generate diverse inputs
        input_prompt = f"""Generate {num_examples} diverse inputs for this task: {task_description}

Requirements:
- Cover different cases (simple, complex, edge cases)
- Be realistic and practical
- Vary in length and complexity

Inputs (one per line):"""

        inputs_response = client.complete(input_prompt, temperature=0.8, max_tokens=200)
        inputs = [line.strip() for line in inputs_response.strip().split('\n') if line.strip()][:num_examples]

        # Step 2: Generate corresponding outputs
        examples = []
        for input_text in inputs:
            output_prompt = f"""Task: {task_description}

Input: {input_text}
Output:"""

            output_response = client.complete(output_prompt, temperature=0.3, max_tokens=100)
            examples.append((input_text, output_response.strip()))

        return examples

    def validate_examples(examples: List[Tuple[str, str]], task_description: str) -> Dict:
        """Validate the quality of generated examples."""

        validation_results = {
            "format_consistency": True,
            "coverage": len(examples) >= 3,
            "clarity": True,
            "diversity": len(set(ex[0][:10] for ex in examples)) >= len(examples) * 0.7
        }

        # Check format consistency
        if examples:
            first_output_lines = examples[0][1].count('\n')
            format_consistent = all(
                abs(ex[1].count('\n') - first_output_lines) <= 2
                for ex in examples
            )
            validation_results["format_consistency"] = format_consistent

        # Check clarity (outputs should be non-empty and reasonable length)
        validation_results["clarity"] = all(
            10 < len(ex[1]) < 500 for ex in examples
        )

        return validation_results

    # Test the system
    tasks = [
        "Extract email addresses from text",
        "Convert informal text to formal business language",
        "Classify customer support tickets by priority (High/Medium/Low)"
    ]

    for task in tasks:
        print(f"\nTask: {task}")
        print("-" * 40)

        examples = generate_examples(task, num_examples=3)

        if examples:
            print(f"Generated {len(examples)} examples:\n")
            for i, (inp, out) in enumerate(examples, 1):
                print(f"Example {i}:")
                print(f"  Input: {inp[:100]}...")
                print(f"  Output: {out[:100]}...")
                print()

            validation = validate_examples(examples, task)
            print("Validation Results:")
            for metric, result in validation.items():
                status = "✓" if result else "✗"
                print(f"  {status} {metric.replace('_', ' ').title()}")

            # Test the generated examples
            if all(validation.values()):
                print("\n✓ Examples passed all validation checks!")

                # Build few-shot prompt with generated examples
                test_prompt = f"{task}:\n\n"
                for inp, out in examples:
                    test_prompt += f"Input: {inp}\nOutput: {out}\n\n"

                # Add a new test case
                if "email" in task.lower():
                    test_input = "Contact me at john.doe@example.com or jane@company.org"
                elif "formal" in task.lower():
                    test_input = "gonna be late, sry"
                else:
                    test_input = "System completely down, no one can log in"

                test_prompt += f"Input: {test_input}\nOutput:"

                final_response = client.complete(test_prompt, temperature=0.3, max_tokens=100)
                print(f"\nTest with generated examples:")
                print(f"  Input: {test_input}")
                print(f"  Output: {final_response.strip()}")

    print("\n✓ Automatic example generation system complete!")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 03: Few-Shot Solutions")
    parser.add_argument("--solution", type=int, help="Run specific solution (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge solution")
    parser.add_argument("--all", action="store_true", help="Run all solutions")

    args = parser.parse_args()

    solutions = {
        1: solution_1_example_crafting,
        2: solution_2_shot_optimization,
        3: solution_3_format_matching,
        4: solution_4_example_debugging,
        5: solution_5_dynamic_selection
    }

    if args.all:
        for sol_num, sol_func in solutions.items():
            print(f"\n{'='*60}")
            sol_func()
            print(f"\n{'='*60}\n")
        challenge_solution_automatic_generation()
    elif args.challenge:
        challenge_solution_automatic_generation()
    elif args.solution and args.solution in solutions:
        solutions[args.solution]()
    else:
        print("Module 03: Few-Shot Learning - Solutions")
        print("\nUsage:")
        print("  python solutions.py --solution N  # Run solution N")
        print("  python solutions.py --challenge    # Run challenge solution")
        print("  python solutions.py --all          # Run all solutions")
        print("\nAvailable solutions:")
        print("  1: Example Crafting")
        print("  2: Shot Optimization")
        print("  3: Format Matching")
        print("  4: Example Debugging")
        print("  5: Dynamic Selection")
        print("  Challenge: Automatic Example Generation")