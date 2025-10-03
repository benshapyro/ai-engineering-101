"""
Module 01: Basic Prompting Examples

Demonstrates fundamental prompt engineering concepts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient, count_tokens, estimate_cost


def example_1_clarity_comparison():
    """Compare vague vs specific prompts."""
    print("=" * 60)
    print("Example 1: Clarity and Specificity")
    print("=" * 60)

    client = LLMClient("openai")

    # Vague prompt
    vague_prompt = "Tell me about dogs"
    print("\nVague Prompt:", vague_prompt)

    # Estimate cost before making request
    input_tokens = count_tokens(vague_prompt)
    estimated_output = 100  # max_tokens
    print(f"\nEstimated cost for vague prompt:")
    cost_est = estimate_cost(input_tokens, estimated_output, "gpt-5")
    print(f"  Input: {input_tokens} tokens | Output: ~{estimated_output} tokens | Cost: ~${cost_est['total_cost']}")

    vague_response = client.complete(vague_prompt, temperature=0.7, max_tokens=100)
    print("Response:", vague_response[:200] + "...")

    # Specific prompt
    specific_prompt = """Write a 100-word educational summary about Golden Retrievers,
    focusing on their temperament, exercise needs, and suitability for families with young children."""

    print("\n" + "-" * 40)
    print("\nSpecific Prompt:", specific_prompt)

    # Estimate cost for specific prompt
    input_tokens = count_tokens(specific_prompt)
    estimated_output = 150
    print(f"\nEstimated cost for specific prompt:")
    cost_est = estimate_cost(input_tokens, estimated_output, "gpt-5")
    print(f"  Input: {input_tokens} tokens | Output: ~{estimated_output} tokens | Cost: ~${cost_est['total_cost']}")

    specific_response = client.complete(specific_prompt, temperature=0.7)
    print("Response:", specific_response)

    # Compare actual token usage
    vague_tokens = count_tokens(vague_response)
    specific_tokens = count_tokens(specific_response)
    print(f"\nActual token usage:")
    print(f"  Vague response: {vague_tokens} tokens")
    print(f"  Specific response: {specific_tokens} tokens")
    print(f"\n💡 Tip: Specific prompts often yield better results with similar or lower token costs!")


def example_2_temperature_effects():
    """Demonstrate temperature parameter effects."""
    print("\n" + "=" * 60)
    print("Example 2: Temperature Effects")
    print("=" * 60)

    client = LLMClient("openai")
    prompt = "Generate a creative name for a coffee shop that specializes in vintage books"

    temperatures = [0.0, 0.5, 0.9, 1.2]

    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        response = client.complete(prompt, temperature=temp, max_tokens=20)
        print(f"Response: {response}")


def example_3_using_delimiters():
    """Show how delimiters improve prompt clarity."""
    print("\n" + "=" * 60)
    print("Example 3: Using Delimiters")
    print("=" * 60)

    client = LLMClient("openai")

    # Without delimiters (confusing)
    messy_prompt = """Summarize this text in 3 bullet points:
    The Earth orbits the Sun. It takes 365.25 days. This is why we have leap years.
    The Moon orbits Earth. It takes about 27 days. This causes phases of the moon.
    Make sure each bullet point is clear and concise."""

    print("Without delimiters:")
    print(messy_prompt)
    response = client.complete(messy_prompt, temperature=0.3)
    print("\nResponse:", response)

    # With delimiters (clear)
    clear_prompt = """Summarize the text below in 3 bullet points.

Text: ###
The Earth orbits the Sun. It takes 365.25 days. This is why we have leap years.
The Moon orbits Earth. It takes about 27 days. This causes phases of the moon.
###

Requirements:
- Each bullet point should be clear and concise
- Focus on the key facts

Summary:"""

    print("\n" + "-" * 40)
    print("\nWith delimiters:")
    print(clear_prompt)
    response = client.complete(clear_prompt, temperature=0.3)
    print("\nResponse:", response)


def example_4_system_messages():
    """Demonstrate the power of system messages."""
    print("\n" + "=" * 60)
    print("Example 4: System Messages")
    print("=" * 60)

    client = LLMClient("openai")
    prompt = "Explain what recursion is"

    # Without system message
    print("Without system message:")
    response = client.complete(prompt, temperature=0.7, max_tokens=100)
    print("Response:", response)

    # With expert system message
    print("\n" + "-" * 40)
    print("\nWith Python expert system message:")
    system_message = """You are an expert Python programmer and computer science teacher.
    Explain concepts with practical Python code examples."""

    response = client.complete(
        prompt,
        system_message=system_message,
        temperature=0.7,
        max_tokens=200
    )
    print("Response:", response)

    # With different persona
    print("\n" + "-" * 40)
    print("\nWith 5-year-old explanation system message:")
    system_message = "You explain complex topics in simple terms that a 5-year-old could understand."

    response = client.complete(
        prompt,
        system_message=system_message,
        temperature=0.7,
        max_tokens=150
    )
    print("Response:", response)


def example_5_output_formatting():
    """Control output format explicitly."""
    print("\n" + "=" * 60)
    print("Example 5: Output Formatting")
    print("=" * 60)

    client = LLMClient("openai")

    base_info = "Python, JavaScript, and Java are popular programming languages"

    # Request JSON format
    json_prompt = f"""Convert this information to JSON format:
    {base_info}

    Output format:
    {{
        "languages": [
            {{"name": "...", "popularity": "..."}}
        ]
    }}

    JSON:"""

    print("JSON Format Request:")
    response = client.complete(json_prompt, temperature=0.2)
    print("Response:", response)

    # Request markdown table
    print("\n" + "-" * 40)
    table_prompt = f"""Convert this information to a markdown table:
    {base_info}

    Include columns for: Language, Type, Use Cases

    Markdown table:"""

    print("\nMarkdown Table Request:")
    response = client.complete(table_prompt, temperature=0.2)
    print("Response:", response)

    # Request numbered list
    print("\n" + "-" * 40)
    list_prompt = f"""Expand on this information as a numbered list:
    {base_info}

    For each language, include:
    - Main use cases
    - Key strength

    Numbered list:"""

    print("\nNumbered List Request:")
    response = client.complete(list_prompt, temperature=0.2)
    print("Response:", response)


def example_6_iterative_refinement():
    """Show how to iteratively improve prompts."""
    print("\n" + "=" * 60)
    print("Example 6: Iterative Refinement")
    print("=" * 60)

    client = LLMClient("openai")

    # Iteration 1: Too vague
    prompt_v1 = "Write a product description"
    print("Iteration 1 (vague):", prompt_v1)
    response = client.complete(prompt_v1, temperature=0.7, max_tokens=50)
    print("Response:", response)
    print("Issue: Too generic, no specific product")

    # Iteration 2: Add product
    print("\n" + "-" * 40)
    prompt_v2 = "Write a product description for a smartphone"
    print("\nIteration 2 (with product):", prompt_v2)
    response = client.complete(prompt_v2, temperature=0.7, max_tokens=50)
    print("Response:", response)
    print("Issue: No specific features or target audience")

    # Iteration 3: Add specifics
    print("\n" + "-" * 40)
    prompt_v3 = """Write a 100-word product description for the iPhone 15 Pro,
    highlighting camera capabilities and battery life,
    targeted at photography enthusiasts"""
    print("\nIteration 3 (specific):", prompt_v3)
    response = client.complete(prompt_v3, temperature=0.7)
    print("Response:", response)
    print("Success: Clear, specific, targeted")


def run_all_examples():
    """Run all examples in sequence."""
    examples = [
        example_1_clarity_comparison,
        example_2_temperature_effects,
        example_3_using_delimiters,
        example_4_system_messages,
        example_5_output_formatting,
        example_6_iterative_refinement
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 01: Basic Prompting Examples")
    parser.add_argument("--example", type=int, help="Run specific example (1-6)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_clarity_comparison,
            2: example_2_temperature_effects,
            3: example_3_using_delimiters,
            4: example_4_system_messages,
            5: example_5_output_formatting,
            6: example_6_iterative_refinement
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Run with --all for all examples or --example N for a specific example")
        print("\nAvailable examples:")
        print("1. Clarity and Specificity")
        print("2. Temperature Effects")
        print("3. Using Delimiters")
        print("4. System Messages")
        print("5. Output Formatting")
        print("6. Iterative Refinement")