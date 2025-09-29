"""
Module 03: Few-Shot Learning Basics Examples

Demonstrates fundamental few-shot prompting techniques with examples.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient, count_tokens
import json


def example_1_zero_one_few_comparison():
    """Compare zero-shot, one-shot, and few-shot for the same task."""
    print("=" * 60)
    print("Example 1: Zero vs One vs Few-Shot Comparison")
    print("=" * 60)

    client = LLMClient("openai")

    # Test input
    test_input = "The product arrived damaged and customer service was unhelpful"

    # Zero-shot
    zero_shot_prompt = """Classify the sentiment of this text as Positive, Negative, or Neutral.

Text: {input}
Sentiment:"""

    print("\nZERO-SHOT:")
    zero_response = client.complete(
        zero_shot_prompt.format(input=test_input),
        temperature=0.3,
        max_tokens=20
    )
    print(f"Prompt tokens: {count_tokens(zero_shot_prompt)}")
    print(f"Response: {zero_response.strip()}")

    # One-shot
    one_shot_prompt = """Classify the sentiment of these texts as Positive, Negative, or Neutral.

Text: "The service was excellent and the food was delicious!"
Sentiment: Positive

Text: "{input}"
Sentiment:"""

    print("\n" + "-" * 40)
    print("\nONE-SHOT:")
    one_response = client.complete(
        one_shot_prompt.format(input=test_input),
        temperature=0.3,
        max_tokens=20
    )
    print(f"Prompt tokens: {count_tokens(one_shot_prompt)}")
    print(f"Response: {one_response.strip()}")

    # Few-shot (3 examples)
    few_shot_prompt = """Classify the sentiment of these texts as Positive, Negative, or Neutral.

Text: "The service was excellent and the food was delicious!"
Sentiment: Positive

Text: "The product broke after one day of use. Very disappointed."
Sentiment: Negative

Text: "The hotel was okay. Nothing special but clean and functional."
Sentiment: Neutral

Text: "{input}"
Sentiment:"""

    print("\n" + "-" * 40)
    print("\nFEW-SHOT (3 examples):")
    few_response = client.complete(
        few_shot_prompt.format(input=test_input),
        temperature=0.3,
        max_tokens=20
    )
    print(f"Prompt tokens: {count_tokens(few_shot_prompt)}")
    print(f"Response: {few_response.strip()}")

    print("\n" + "-" * 40)
    print("\nANALYSIS:")
    print("- Zero-shot: Relies on model's understanding")
    print("- One-shot: Provides format but limited coverage")
    print("- Few-shot: Best consistency and accuracy")


def example_2_example_quality_impact():
    """Show how example quality affects output quality."""
    print("\n" + "=" * 60)
    print("Example 2: Example Quality Impact")
    print("=" * 60)

    client = LLMClient("openai")

    test_text = "Machine learning models require large amounts of data for training"

    # Poor quality examples
    poor_examples_prompt = """Extract key concepts from text:

Text: "The cat sat"
Concepts: cat

Text: "It rained"
Concepts: rain

Text: "{input}"
Concepts:"""

    print("POOR QUALITY EXAMPLES:")
    print("(Too simple, inconsistent extraction)")
    poor_response = client.complete(
        poor_examples_prompt.format(input=test_text),
        temperature=0.3,
        max_tokens=50
    )
    print(f"Response: {poor_response.strip()}")

    # High quality examples
    good_examples_prompt = """Extract key concepts from text:

Text: "Artificial intelligence is transforming healthcare through diagnostic imaging analysis"
Concepts: ["artificial intelligence", "healthcare", "diagnostic imaging", "analysis"]

Text: "Climate change affects global weather patterns and sea levels"
Concepts: ["climate change", "global weather patterns", "sea levels"]

Text: "Quantum computers use qubits to perform complex calculations exponentially faster"
Concepts: ["quantum computers", "qubits", "complex calculations", "exponential speed"]

Text: "{input}"
Concepts:"""

    print("\n" + "-" * 40)
    print("\nHIGH QUALITY EXAMPLES:")
    print("(Comprehensive, consistent extraction)")
    good_response = client.complete(
        good_examples_prompt.format(input=test_text),
        temperature=0.3,
        max_tokens=50
    )
    print(f"Response: {good_response.strip()}")

    print("\nDifference: High-quality examples lead to more thorough extraction")


def example_3_format_consistency():
    """Demonstrate the importance of consistent formatting in examples."""
    print("\n" + "=" * 60)
    print("Example 3: Format Consistency Importance")
    print("=" * 60)

    client = LLMClient("openai")

    test_input = "Sarah Johnson, CEO of TechCorp, announced $50M funding"

    # Inconsistent format
    inconsistent_prompt = """Extract person information:

Text: "John Smith is 30 years old"
Output: Name - John Smith, Age = 30

Text: "Maria Garcia works at Google"
→ Maria Garcia (Google)

Text: "Bob Wilson lives in Seattle"
Result: {{"name": "Bob Wilson", "location": "Seattle"}}

Text: "{input}"
Output:"""

    print("INCONSISTENT FORMAT:")
    inconsistent_response = client.complete(
        inconsistent_prompt.format(input=test_input),
        temperature=0.2,
        max_tokens=100
    )
    print(f"Response: {inconsistent_response.strip()}")
    print("Issue: Model unsure which format to follow")

    # Consistent format
    consistent_prompt = """Extract person information:

Text: "John Smith is 30 years old"
JSON: {{"name": "John Smith", "age": 30}}

Text: "Maria Garcia works at Google"
JSON: {{"name": "Maria Garcia", "company": "Google"}}

Text: "Bob Wilson lives in Seattle"
JSON: {{"name": "Bob Wilson", "location": "Seattle"}}

Text: "{input}"
JSON:"""

    print("\n" + "-" * 40)
    print("\nCONSISTENT FORMAT:")
    consistent_response = client.complete(
        consistent_prompt.format(input=test_input),
        temperature=0.2,
        max_tokens=100
    )
    print(f"Response: {consistent_response.strip()}")
    print("Success: Clear, consistent JSON format")


def example_4_dynamic_shot_selection():
    """Demonstrate selecting examples based on input characteristics."""
    print("\n" + "=" * 60)
    print("Example 4: Dynamic Shot Selection")
    print("=" * 60)

    client = LLMClient("openai")

    # Example library
    example_library = {
        "technical": [
            ("API endpoints should follow RESTful conventions", "Technical"),
            ("Database indexing improves query performance", "Technical")
        ],
        "business": [
            ("Q3 revenue increased by 15% year-over-year", "Business"),
            ("Market expansion strategy targets Asia-Pacific", "Business")
        ],
        "general": [
            ("The weather today is sunny and warm", "General"),
            ("The museum opens at 9 AM on weekdays", "General")
        ]
    }

    def classify_input_domain(text):
        """Simple domain classifier."""
        technical_words = ["API", "database", "code", "algorithm", "server"]
        business_words = ["revenue", "market", "strategy", "profit", "customer"]

        text_lower = text.lower()
        tech_score = sum(1 for word in technical_words if word.lower() in text_lower)
        biz_score = sum(1 for word in business_words if word.lower() in text_lower)

        if tech_score > biz_score:
            return "technical"
        elif biz_score > tech_score:
            return "business"
        else:
            return "general"

    def create_dynamic_prompt(input_text):
        """Create prompt with dynamically selected examples."""
        domain = classify_input_domain(input_text)
        examples = example_library[domain]

        prompt = "Classify the domain of these texts:\n\n"
        for ex_text, ex_label in examples:
            prompt += f'Text: "{ex_text}"\nDomain: {ex_label}\n\n'
        prompt += f'Text: "{input_text}"\nDomain:'

        return prompt, domain

    # Test with different inputs
    test_inputs = [
        "The API response time exceeds 500ms under load",
        "Our customer acquisition cost decreased by 20%",
        "The library closes at 6 PM on Sundays"
    ]

    for test_input in test_inputs:
        print(f"\nInput: '{test_input}'")
        prompt, detected_domain = create_dynamic_prompt(test_input)
        print(f"Detected domain: {detected_domain}")
        print(f"Selected examples: {detected_domain} domain examples")

        response = client.complete(prompt, temperature=0.2, max_tokens=20)
        print(f"Classification: {response.strip()}")


def example_5_negative_examples():
    """Show how to use negative examples effectively."""
    print("\n" + "=" * 60)
    print("Example 5: Using Negative Examples")
    print("=" * 60)

    client = LLMClient("openai")

    # Without negative examples
    without_negative = """Format email addresses correctly:

john@example.com → john@example.com
SARAH@COMPANY.COM → sarah@company.com
Mike.Jones@email.co.uk → mike.jones@email.co.uk

Format: ALICE@WORKPLACE.ORG
Result:"""

    print("WITHOUT NEGATIVE EXAMPLES:")
    response = client.complete(without_negative, temperature=0.1, max_tokens=50)
    print(f"Response: {response.strip()}")

    # With negative examples
    with_negative = """Format email addresses correctly:

✓ Correct: john@example.com
✗ Wrong: John@Example.com (don't capitalize)

✓ Correct: sarah@company.com
✗ Wrong: sarah@@company.com (no double @)

✓ Correct: mike.jones@email.co.uk
✗ Wrong: mike jones@email.co.uk (no spaces allowed)

Format: ALICE@WORKPLACE.ORG
Result:"""

    print("\n" + "-" * 40)
    print("\nWITH NEGATIVE EXAMPLES:")
    response = client.complete(with_negative, temperature=0.1, max_tokens=50)
    print(f"Response: {response.strip()}")
    print("\nBenefit: Negative examples prevent common mistakes")


def example_6_progressive_complexity():
    """Demonstrate building from simple to complex examples."""
    print("\n" + "=" * 60)
    print("Example 6: Progressive Complexity")
    print("=" * 60)

    client = LLMClient("openai")

    progressive_prompt = """Parse and evaluate these Python expressions:

Simple:
Expression: 5 + 3
Steps: Add 5 and 3
Result: 8

Moderate:
Expression: (10 - 2) * 3
Steps: 1) Evaluate parentheses: 10 - 2 = 8
       2) Multiply: 8 * 3 = 24
Result: 24

Complex:
Expression: ((15 / 3) + 2) * (10 - 6)
Steps: 1) Evaluate first parentheses: 15 / 3 = 5
       2) Add: 5 + 2 = 7
       3) Evaluate second parentheses: 10 - 6 = 4
       4) Multiply: 7 * 4 = 28
Result: 28

Now parse:
Expression: ((20 - 5) * 2) + (18 / 3)
Steps:"""

    print("PROGRESSIVE COMPLEXITY EXAMPLES:")
    print("Building from simple → moderate → complex")
    response = client.complete(progressive_prompt, temperature=0.2, max_tokens=200)
    print(f"\nResponse: {response.strip()}")
    print("\nBenefit: Model learns to handle complexity gradually")


def example_7_domain_adaptation():
    """Show how to adapt examples for specific domains."""
    print("\n" + "=" * 60)
    print("Example 7: Domain-Specific Examples")
    print("=" * 60)

    client = LLMClient("openai")

    # Generic examples
    generic_prompt = """Identify the main point:

Text: "Dogs are loyal pets that require regular exercise."
Main point: Dogs need exercise and are loyal

Text: "Pizza is a popular food originating from Italy."
Main point: Pizza is Italian and popular

Text: "Machine learning models can overfit when trained on limited data, leading to poor generalization on unseen examples."
Main point:"""

    print("GENERIC EXAMPLES:")
    generic_response = client.complete(generic_prompt, temperature=0.3, max_tokens=50)
    print(f"Response: {generic_response.strip()}")

    # Domain-specific examples
    domain_prompt = """Identify the main technical insight:

Text: "Neural networks with too many parameters relative to training data exhibit overfitting, reducing test accuracy."
Main insight: Overfitting occurs when model complexity exceeds data complexity

Text: "Gradient descent converges faster with momentum by accumulating velocity in consistent directions."
Main insight: Momentum accelerates convergence in consistent gradient directions

Text: "Batch normalization stabilizes training by normalizing inputs to each layer, allowing higher learning rates."
Main insight: Batch norm enables faster training through layer input normalization

Text: "Machine learning models can overfit when trained on limited data, leading to poor generalization on unseen examples."
Main insight:"""

    print("\n" + "-" * 40)
    print("\nDOMAIN-SPECIFIC EXAMPLES:")
    domain_response = client.complete(domain_prompt, temperature=0.3, max_tokens=50)
    print(f"Response: {domain_response.strip()}")
    print("\nDifference: Domain examples lead to more technical, precise extraction")


def example_8_error_recovery():
    """Use examples to prevent and recover from common errors."""
    print("\n" + "=" * 60)
    print("Example 8: Error Recovery with Examples")
    print("=" * 60)

    client = LLMClient("openai")

    # Without error handling examples
    basic_prompt = """Convert text to SQL query:

"Find all users named John"
SELECT * FROM users WHERE name = 'John'

"Get products with price under 50"
SELECT * FROM products WHERE price < 50

"Show orders from last month"
SELECT * FROM orders WHERE date > DATE_SUB(NOW(), INTERVAL 1 MONTH)

"Delete everything"
"""

    print("WITHOUT ERROR HANDLING:")
    print("Query: 'Delete everything'")
    basic_response = client.complete(basic_prompt, temperature=0.2, max_tokens=100)
    print(f"Response: {basic_response.strip()}")
    print("⚠️  Dangerous query might be generated!")

    # With error handling examples
    safe_prompt = """Convert text to SQL query (read-only):

"Find all users named John"
SELECT * FROM users WHERE name = 'John'

"Delete all records"
ERROR: Destructive operations not allowed. Use SELECT to view data instead.

"Get products with price under 50"
SELECT * FROM products WHERE price < 50

"Drop table users"
ERROR: DDL operations not allowed. Only SELECT queries permitted.

"Show orders from last month"
SELECT * FROM orders WHERE date > DATE_SUB(NOW(), INTERVAL 1 MONTH)

"Update all prices to 0"
ERROR: Modification operations not allowed. Use SELECT for data retrieval.

"Delete everything"
"""

    print("\n" + "-" * 40)
    print("\nWITH ERROR HANDLING EXAMPLES:")
    print("Query: 'Delete everything'")
    safe_response = client.complete(safe_prompt, temperature=0.2, max_tokens=100)
    print(f"Response: {safe_response.strip()}")
    print("✓ Safe: Examples teach error handling patterns")


def run_all_examples():
    """Run all few-shot basics examples."""
    examples = [
        example_1_zero_one_few_comparison,
        example_2_example_quality_impact,
        example_3_format_consistency,
        example_4_dynamic_shot_selection,
        example_5_negative_examples,
        example_6_progressive_complexity,
        example_7_domain_adaptation,
        example_8_error_recovery
    ]

    for example in examples:
        try:
            example()
            print("\n" + "=" * 60 + "\n")
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 03: Few-Shot Basics")
    parser.add_argument("--example", type=int, help="Run specific example (1-8)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_zero_one_few_comparison,
            2: example_2_example_quality_impact,
            3: example_3_format_consistency,
            4: example_4_dynamic_shot_selection,
            5: example_5_negative_examples,
            6: example_6_progressive_complexity,
            7: example_7_domain_adaptation,
            8: example_8_error_recovery
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Run with --all for all examples or --example N for a specific example")
        print("\nAvailable examples:")
        print("1. Zero vs One vs Few-Shot Comparison")
        print("2. Example Quality Impact")
        print("3. Format Consistency")
        print("4. Dynamic Shot Selection")
        print("5. Negative Examples")
        print("6. Progressive Complexity")
        print("7. Domain Adaptation")
        print("8. Error Recovery")