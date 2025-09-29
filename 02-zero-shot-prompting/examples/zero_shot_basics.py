"""
Module 02: Zero-Shot Basics Examples

Demonstrates fundamental zero-shot prompting techniques without providing examples.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient, count_tokens, estimate_cost


def example_1_task_clarity():
    """Compare vague vs clear zero-shot instructions."""
    print("=" * 60)
    print("Example 1: Task Clarity in Zero-Shot")
    print("=" * 60)

    client = LLMClient("openai")

    text = """The new smartphone has a great camera with 48MP resolution,
    but the battery life is disappointing at only 6 hours of screen time.
    The price of $999 seems high for these specs."""

    # Vague zero-shot prompt
    vague_prompt = f"Analyze this: {text}"

    print("\nVAGUE PROMPT:")
    print(vague_prompt)
    vague_response = client.complete(vague_prompt, temperature=0.3, max_tokens=150)
    print(f"\nResponse: {vague_response}")

    # Clear zero-shot prompt
    clear_prompt = f"""Analyze the following product review and provide:
1. Product category
2. Positive aspects (list format)
3. Negative aspects (list format)
4. Overall sentiment (Positive/Negative/Mixed)
5. Value assessment (Good value/Poor value/Uncertain)

Review: {text}

Analysis:"""

    print("\n" + "-" * 40)
    print("\nCLEAR PROMPT:")
    print(clear_prompt)
    clear_response = client.complete(clear_prompt, temperature=0.3)
    print(f"\nResponse: {clear_response}")


def example_2_format_specification():
    """Demonstrate precise output format control without examples."""
    print("\n" + "=" * 60)
    print("Example 2: Format Specification")
    print("=" * 60)

    client = LLMClient("openai")

    data = "Python is versatile and easy. Java is robust and enterprise-ready. JavaScript runs everywhere."

    # Without format specification
    no_format_prompt = f"Compare these programming languages: {data}"

    print("WITHOUT FORMAT SPECIFICATION:")
    print(no_format_prompt)
    response = client.complete(no_format_prompt, temperature=0.3, max_tokens=200)
    print(f"\nResponse: {response}")

    # With precise format specification
    formatted_prompt = f"""Compare the programming languages mentioned in the text below.

Text: {data}

Output format (use exactly this structure):
{{
    "languages": [
        {{
            "name": "language name",
            "strength": "main advantage",
            "use_case": "primary application"
        }}
    ],
    "comparison_summary": "one sentence comparing all three"
}}

JSON output:"""

    print("\n" + "-" * 40)
    print("\nWITH FORMAT SPECIFICATION:")
    print(formatted_prompt[:200] + "...")
    response = client.complete(formatted_prompt, temperature=0.2)
    print(f"\nResponse: {response}")


def example_3_role_based_zero_shot():
    """Show how role assignment affects zero-shot responses."""
    print("\n" + "=" * 60)
    print("Example 3: Role-Based Zero-Shot")
    print("=" * 60)

    client = LLMClient("openai")

    question = "What are the implications of quantum computing for cybersecurity?"

    roles = [
        {
            "role": "a security expert focused on practical risks",
            "perspective": "security"
        },
        {
            "role": "a physics professor explaining to students",
            "perspective": "educational"
        },
        {
            "role": "a business executive evaluating investments",
            "perspective": "business"
        }
    ]

    for role_info in roles:
        prompt = f"""You are {role_info['role']}.

Question: {question}

Provide a concise answer from your perspective:"""

        print(f"\n{role_info['perspective'].upper()} PERSPECTIVE:")
        print(f"Role: {role_info['role']}")
        response = client.complete(prompt, temperature=0.5, max_tokens=150)
        print(f"Response: {response}")
        print("-" * 40)


def example_4_chain_of_thought_zero_shot():
    """Demonstrate zero-shot chain-of-thought reasoning."""
    print("\n" + "=" * 60)
    print("Example 4: Chain-of-Thought Zero-Shot")
    print("=" * 60)

    client = LLMClient("openai")

    # Without chain-of-thought
    direct_prompt = """A bakery sells cookies for $2 each. On Monday they sold 45 cookies,
    on Tuesday 60% more than Monday, and on Wednesday half of Tuesday's amount.
    What was their total revenue for the three days?

    Answer:"""

    print("WITHOUT CHAIN-OF-THOUGHT:")
    print(direct_prompt)
    response = client.complete(direct_prompt, temperature=0.2, max_tokens=50)
    print(f"Response: {response}")

    # With chain-of-thought
    cot_prompt = """A bakery sells cookies for $2 each. On Monday they sold 45 cookies,
    on Tuesday 60% more than Monday, and on Wednesday half of Tuesday's amount.
    What was their total revenue for the three days?

    Let's solve this step by step:
    Step 1: Calculate Monday's sales
    Step 2: Calculate Tuesday's sales (60% more than Monday)
    Step 3: Calculate Wednesday's sales (half of Tuesday)
    Step 4: Sum up all cookies sold
    Step 5: Calculate total revenue

    Solution:"""

    print("\n" + "-" * 40)
    print("\nWITH CHAIN-OF-THOUGHT:")
    print(cot_prompt)
    response = client.complete(cot_prompt, temperature=0.2)
    print(f"Response: {response}")


def example_5_negative_instructions():
    """Show the power of negative instructions in zero-shot."""
    print("\n" + "=" * 60)
    print("Example 5: Negative Instructions")
    print("=" * 60)

    client = LLMClient("openai")

    text = "The Quick Brown Fox Jumps Over The Lazy Dog"

    # Without negative instructions
    basic_prompt = f"""Convert this text to a different case: {text}

    Result:"""

    print("WITHOUT NEGATIVE INSTRUCTIONS:")
    print(basic_prompt)
    response = client.complete(basic_prompt, temperature=0.3, max_tokens=50)
    print(f"Response: {response}")

    # With negative instructions
    negative_prompt = f"""Convert this text to lowercase: {text}

    DO NOT:
    - Add any explanations
    - Include the original text
    - Add punctuation that wasn't there
    - Change the spacing
    - Add quotes around the result

    Result:"""

    print("\n" + "-" * 40)
    print("\nWITH NEGATIVE INSTRUCTIONS:")
    print(negative_prompt)
    response = client.complete(negative_prompt, temperature=0.1)
    print(f"Response: {response}")


def example_6_constraint_setting():
    """Demonstrate the importance of constraints in zero-shot."""
    print("\n" + "=" * 60)
    print("Example 6: Constraint Setting")
    print("=" * 60)

    client = LLMClient("openai")

    topic = "artificial intelligence in healthcare"

    # Without constraints
    unconstrained_prompt = f"Write about {topic}"

    print("WITHOUT CONSTRAINTS:")
    print(unconstrained_prompt)
    response = client.complete(unconstrained_prompt, temperature=0.7, max_tokens=100)
    print(f"Response preview: {response[:200]}...")
    print(f"Token count: {count_tokens(response)}")

    # With specific constraints
    constrained_prompt = f"""Write about {topic}

    Constraints:
    - Exactly 3 sentences
    - Focus on diagnostic applications only
    - Include one specific percentage or statistic
    - Use present tense throughout
    - Target audience: hospital administrators

    Text:"""

    print("\n" + "-" * 40)
    print("\nWITH CONSTRAINTS:")
    print(constrained_prompt)
    response = client.complete(constrained_prompt, temperature=0.5)
    print(f"Response: {response}")
    print(f"Token count: {count_tokens(response)}")


def example_7_error_handling():
    """Show how to handle edge cases in zero-shot prompts."""
    print("\n" + "=" * 60)
    print("Example 7: Error Handling in Zero-Shot")
    print("=" * 60)

    client = LLMClient("openai")

    # Test with various edge cases
    test_inputs = [
        "Apple Inc. was founded in 1976 by Steve Jobs and Steve Wozniak.",
        "",  # Empty input
        "こんにちは世界",  # Non-English
        "123 456 789",  # Just numbers
    ]

    robust_prompt_template = """Extract company information from the text below.

    Text: {input_text}

    Instructions:
    - If no company is mentioned, return {{"company": "N/A", "year": "N/A", "founders": []}}
    - If text is empty, return {{"error": "Empty input"}}
    - If text is not in English, return {{"error": "Non-English text detected"}}
    - If text contains only numbers, return {{"error": "Invalid input format"}}

    Output (JSON):"""

    for test_input in test_inputs:
        print(f"\nInput: '{test_input[:50]}...' if '{test_input}' else '[EMPTY]'")
        prompt = robust_prompt_template.format(input_text=test_input if test_input else "[EMPTY STRING]")
        response = client.complete(prompt, temperature=0.2, max_tokens=100)
        print(f"Response: {response}")


def example_8_task_decomposition():
    """Break complex tasks into zero-shot subtasks."""
    print("\n" + "=" * 60)
    print("Example 8: Task Decomposition")
    print("=" * 60)

    client = LLMClient("openai")

    complex_text = """
    Sarah started her business in 2019 with $50,000 in savings. By 2021,
    her revenue reached $500,000 with 8 employees. She expanded to 3 locations
    by 2023, and now employs 25 people with projected revenue of $2 million.
    """

    # Single complex prompt (often fails or gives incomplete results)
    complex_prompt = f"""Analyze this business story and extract all information: {complex_text}"""

    print("SINGLE COMPLEX PROMPT:")
    print(complex_prompt)
    response = client.complete(complex_prompt, temperature=0.3, max_tokens=200)
    print(f"Response: {response[:300]}...")

    # Decomposed into focused subtasks
    print("\n" + "-" * 40)
    print("\nDECOMPOSED SUBTASKS:")

    subtasks = [
        ("Timeline", "List all years mentioned and what happened in each year:"),
        ("Financial", "Extract all financial figures mentioned (amounts and what they represent):"),
        ("Growth", "Identify growth metrics (employees, locations, revenue) and show the progression:"),
        ("Founder", "Extract information about the founder and initial conditions:")
    ]

    results = {}
    for task_name, task_prompt in subtasks:
        prompt = f"""{task_prompt}

Text: {complex_text}

{task_name} Information:"""

        print(f"\nSubtask: {task_name}")
        response = client.complete(prompt, temperature=0.2, max_tokens=150)
        results[task_name] = response
        print(f"Response: {response}")

    # Combine results
    print("\n" + "-" * 40)
    print("\nCOMBINED ANALYSIS:")
    for task_name, result in results.items():
        print(f"\n{task_name}:\n{result}")


def run_all_examples():
    """Run all zero-shot basics examples."""
    examples = [
        example_1_task_clarity,
        example_2_format_specification,
        example_3_role_based_zero_shot,
        example_4_chain_of_thought_zero_shot,
        example_5_negative_instructions,
        example_6_constraint_setting,
        example_7_error_handling,
        example_8_task_decomposition
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

    parser = argparse.ArgumentParser(description="Module 02: Zero-Shot Basics Examples")
    parser.add_argument("--example", type=int, help="Run specific example (1-8)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_task_clarity,
            2: example_2_format_specification,
            3: example_3_role_based_zero_shot,
            4: example_4_chain_of_thought_zero_shot,
            5: example_5_negative_instructions,
            6: example_6_constraint_setting,
            7: example_7_error_handling,
            8: example_8_task_decomposition
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Run with --all for all examples or --example N for a specific example")
        print("\nAvailable examples:")
        print("1. Task Clarity")
        print("2. Format Specification")
        print("3. Role-Based Zero-Shot")
        print("4. Chain-of-Thought")
        print("5. Negative Instructions")
        print("6. Constraint Setting")
        print("7. Error Handling")
        print("8. Task Decomposition")