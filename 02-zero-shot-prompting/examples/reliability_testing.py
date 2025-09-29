"""
Module 02: Reliability Testing Examples

Techniques for testing and improving zero-shot prompt consistency.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import statistics
import json
from typing import List, Dict


def example_1_temperature_impact():
    """Test how temperature affects zero-shot consistency."""
    print("=" * 60)
    print("Example 1: Temperature Impact on Consistency")
    print("=" * 60)

    client = LLMClient("openai")

    prompt = """Classify the priority level of this bug report:

    Bug: Login button sometimes doesn't respond on mobile devices.
    Users have to refresh the page to make it work.

    Priority levels: Critical, High, Medium, Low

    Priority:"""

    temperatures = [0.0, 0.2, 0.5, 0.8, 1.0]
    runs_per_temp = 5

    print("Testing consistency across temperatures...")
    print("-" * 40)

    results = {}
    for temp in temperatures:
        responses = []
        for run in range(runs_per_temp):
            response = client.complete(prompt, temperature=temp, max_tokens=20)
            # Extract just the priority level
            priority = response.strip().split()[0].rstrip(':.,')
            responses.append(priority)

        results[temp] = responses
        unique_responses = len(set(responses))
        most_common = max(set(responses), key=responses.count)
        consistency = responses.count(most_common) / len(responses) * 100

        print(f"\nTemperature {temp}:")
        print(f"  Responses: {responses}")
        print(f"  Unique answers: {unique_responses}")
        print(f"  Consistency: {consistency:.1f}%")

    print("\n" + "-" * 40)
    print("RECOMMENDATION: Use temperature 0.0-0.2 for maximum consistency")


def example_2_prompt_variation_testing():
    """Test how small prompt variations affect output."""
    print("\n" + "=" * 60)
    print("Example 2: Prompt Variation Testing")
    print("=" * 60)

    client = LLMClient("openai")

    text = "The new update is terrible. It deleted all my saved data!"

    prompt_variations = [
        "Classify sentiment: {text}\nSentiment:",
        "What is the sentiment of this text: {text}\nSentiment:",
        "Determine if this text is positive or negative: {text}\nAnswer:",
        "Sentiment analysis for: {text}\nResult:",
        "Classify the following text as Positive, Negative, or Neutral:\n{text}\nClassification:"
    ]

    print(f"Test text: '{text}'")
    print("\nTesting different prompt formulations...")
    print("-" * 40)

    results = []
    for i, prompt_template in enumerate(prompt_variations, 1):
        prompt = prompt_template.format(text=text)
        response = client.complete(prompt, temperature=0.1, max_tokens=20)
        clean_response = response.strip().lower()

        results.append({
            "variation": i,
            "prompt_preview": prompt_template.split('\n')[0][:50] + "...",
            "response": clean_response
        })

        print(f"\nVariation {i}:")
        print(f"  Prompt: {prompt_template.split('{text}')[0]}...")
        print(f"  Response: {clean_response}")

    # Check consistency
    responses = [r['response'] for r in results]
    if all('negative' in r for r in responses):
        print("\n✓ CONSISTENT: All variations identified negative sentiment")
    else:
        print("\n✗ INCONSISTENT: Responses varied across prompt formulations")


def example_3_multi_run_validation():
    """Run the same prompt multiple times to measure consistency."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Run Validation")
    print("=" * 60)

    client = LLMClient("openai")

    prompt = """Extract key information from this text:

    Text: Amazon announced Q3 earnings of $9.9 billion, beating expectations.
    The company's AWS division grew 27% year-over-year.

    Extract:
    1. Company name
    2. Earnings amount
    3. Growth percentage

    Information:"""

    num_runs = 10
    print(f"Running the same prompt {num_runs} times...")
    print("-" * 40)

    responses = []
    extracted_data = []

    for run in range(num_runs):
        response = client.complete(prompt, temperature=0.1, max_tokens=100)
        responses.append(response)

        # Try to parse the response
        try:
            lines = response.strip().split('\n')
            data = {}
            for line in lines:
                if 'company' in line.lower():
                    data['company'] = line.split(':')[-1].strip() if ':' in line else line
                elif 'earnings' in line.lower() or 'billion' in line.lower():
                    data['earnings'] = line.split(':')[-1].strip() if ':' in line else line
                elif 'growth' in line.lower() or '%' in line:
                    data['growth'] = line.split(':')[-1].strip() if ':' in line else line
            extracted_data.append(data)
        except:
            extracted_data.append({})

    # Analyze consistency
    print("\nConsistency Analysis:")

    # Check if all runs extracted the company name
    companies = [d.get('company', '') for d in extracted_data if d.get('company')]
    print(f"  Company extraction: {len(companies)}/{num_runs} successful")
    if companies:
        print(f"    Most common: {max(set(companies), key=companies.count)}")

    # Check earnings extraction
    earnings = [d.get('earnings', '') for d in extracted_data if d.get('earnings')]
    print(f"  Earnings extraction: {len(earnings)}/{num_runs} successful")

    # Check growth extraction
    growth = [d.get('growth', '') for d in extracted_data if d.get('growth')]
    print(f"  Growth extraction: {len(growth)}/{num_runs} successful")

    # Overall consistency score
    consistency_score = (len(companies) + len(earnings) + len(growth)) / (num_runs * 3) * 100
    print(f"\nOverall Consistency Score: {consistency_score:.1f}%")


def example_4_edge_case_reliability():
    """Test zero-shot reliability on edge cases."""
    print("\n" + "=" * 60)
    print("Example 4: Edge Case Reliability")
    print("=" * 60)

    client = LLMClient("openai")

    robust_prompt_template = """Classify this text into a category.

    Text: {text}

    Categories: Technology, Business, Health, Sports, Entertainment, Other

    Rules:
    - If text is empty or only whitespace, return "Error: Empty input"
    - If text is in a foreign language, return "Error: Non-English text"
    - If text is just numbers or symbols, return "Error: Invalid text"
    - If uncertain, choose "Other"
    - Return only the category name or error message

    Category:"""

    test_cases = [
        ("Apple releases new iPhone with AI features", "Technology"),  # Normal case
        ("", "Error: Empty input"),  # Empty
        ("   ", "Error: Empty input"),  # Whitespace
        ("12345 67890", "Error: Invalid text"),  # Numbers
        ("!@#$%^&*()", "Error: Invalid text"),  # Symbols
        ("Bonjour le monde", "Error: Non-English text"),  # French
        ("The weather is nice", "Other"),  # Ambiguous
    ]

    print("Testing edge case handling...")
    print("-" * 40)

    correct = 0
    for test_input, expected in test_cases:
        prompt = robust_prompt_template.format(text=test_input)
        response = client.complete(prompt, temperature=0.0, max_tokens=30)
        response_clean = response.strip()

        # Check if response matches expected
        is_correct = expected.lower() in response_clean.lower()
        correct += is_correct

        status = "✓" if is_correct else "✗"
        print(f"\n{status} Input: '{test_input[:30]}...' if '{test_input}' else '[EMPTY]'")
        print(f"  Expected: {expected}")
        print(f"  Got: {response_clean}")

    accuracy = correct / len(test_cases) * 100
    print(f"\nEdge Case Accuracy: {accuracy:.1f}%")


def example_5_consistency_optimization():
    """Demonstrate techniques to improve zero-shot consistency."""
    print("\n" + "=" * 60)
    print("Example 5: Consistency Optimization")
    print("=" * 60)

    client = LLMClient("openai")

    text = "The product works but customer service is horrible"

    # Version 1: Basic prompt (less consistent)
    basic_prompt = f"What's the sentiment of: {text}"

    # Version 2: Optimized for consistency
    optimized_prompt = f"""Analyze the sentiment of the text below.

    Text: "{text}"

    Instructions:
    - Consider both positive and negative aspects
    - Choose from: Positive, Negative, Mixed, Neutral
    - If both positive and negative elements exist, choose "Mixed"
    - Base your decision on the overall tone

    Final answer (one word only):"""

    # Version 3: Ultra-consistent with structured output
    structured_prompt = f"""Sentiment Analysis Task

    Input text: "{text}"

    Step 1: Identify positive aspects mentioned
    Step 2: Identify negative aspects mentioned
    Step 3: Determine overall sentiment

    Output format:
    {{
        "positive_aspects": [],
        "negative_aspects": [],
        "final_sentiment": "Positive|Negative|Mixed|Neutral"
    }}

    JSON output:"""

    prompts = {
        "Basic": basic_prompt,
        "Optimized": optimized_prompt,
        "Structured": structured_prompt
    }

    runs_per_prompt = 5
    print("Comparing prompt consistency...")
    print("-" * 40)

    for name, prompt in prompts.items():
        responses = []
        for _ in range(runs_per_prompt):
            response = client.complete(prompt, temperature=0.3, max_tokens=150)
            responses.append(response.strip())

        # Extract sentiment from responses
        sentiments = []
        for r in responses:
            r_lower = r.lower()
            if 'mixed' in r_lower:
                sentiments.append('mixed')
            elif 'negative' in r_lower:
                sentiments.append('negative')
            elif 'positive' in r_lower:
                sentiments.append('positive')
            else:
                sentiments.append('other')

        unique = len(set(sentiments))
        most_common = max(set(sentiments), key=sentiments.count) if sentiments else 'none'
        consistency = sentiments.count(most_common) / len(sentiments) * 100 if sentiments else 0

        print(f"\n{name} Prompt:")
        print(f"  Unique responses: {unique}")
        print(f"  Most common: {most_common}")
        print(f"  Consistency: {consistency:.1f}%")


def example_6_cross_model_reliability():
    """Test zero-shot prompts across different models."""
    print("\n" + "=" * 60)
    print("Example 6: Cross-Model Reliability")
    print("=" * 60)

    # Test with both OpenAI and Anthropic if available
    prompt = """Categorize this programming question:

    Question: "How do I reverse a string in Python?"

    Categories:
    - Syntax: Questions about language syntax
    - Algorithm: Questions about problem-solving approaches
    - Debugging: Questions about fixing errors
    - Best Practices: Questions about code quality
    - Performance: Questions about optimization

    Category:"""

    models = []

    try:
        models.append(("OpenAI", LLMClient("openai")))
    except:
        print("OpenAI client not available")

    try:
        models.append(("Anthropic", LLMClient("anthropic")))
    except:
        print("Anthropic client not available")

    if not models:
        print("No models available for testing")
        return

    print("Testing across different models...")
    print("-" * 40)

    results = {}
    for model_name, client in models:
        responses = []
        for run in range(3):
            try:
                response = client.complete(prompt, temperature=0.1, max_tokens=20)
                category = response.strip().split('\n')[0].rstrip(':.,')
                responses.append(category)
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                responses.append("Error")

        results[model_name] = responses
        print(f"\n{model_name}:")
        print(f"  Responses: {responses}")
        print(f"  Consistency: {'Yes' if len(set(responses)) == 1 else 'No'}")

    # Check cross-model agreement
    if len(results) > 1:
        all_responses = []
        for responses in results.values():
            all_responses.extend(responses)

        unique_overall = len(set(all_responses))
        print(f"\nCross-Model Agreement:")
        print(f"  Total unique responses: {unique_overall}")
        print(f"  Agreement level: {'High' if unique_overall <= 2 else 'Low'}")


def example_7_statistical_validation():
    """Use statistical methods to validate zero-shot reliability."""
    print("\n" + "=" * 60)
    print("Example 7: Statistical Validation")
    print("=" * 60)

    client = LLMClient("openai")

    prompt = """Rate the formality level of this text from 1-5:

    Text: "Hey! Wanna grab lunch tomorrow? There's this awesome new place downtown!"

    Scale:
    1 = Very informal
    2 = Informal
    3 = Neutral
    4 = Formal
    5 = Very formal

    Rating (number only):"""

    num_samples = 20
    print(f"Collecting {num_samples} samples for statistical analysis...")
    print("-" * 40)

    ratings = []
    for i in range(num_samples):
        response = client.complete(prompt, temperature=0.5, max_tokens=10)
        try:
            # Extract the number
            rating = int(''.join(filter(str.isdigit, response.strip().split()[0])))
            if 1 <= rating <= 5:
                ratings.append(rating)
        except:
            pass

    if ratings:
        mean = statistics.mean(ratings)
        median = statistics.median(ratings)
        mode = statistics.mode(ratings) if len(ratings) > 1 else ratings[0]
        stdev = statistics.stdev(ratings) if len(ratings) > 1 else 0

        print(f"\nStatistical Analysis ({len(ratings)} valid samples):")
        print(f"  Ratings: {ratings}")
        print(f"  Mean: {mean:.2f}")
        print(f"  Median: {median}")
        print(f"  Mode: {mode}")
        print(f"  Std Dev: {stdev:.2f}")

        # Interpret results
        print("\nInterpretation:")
        if stdev < 0.5:
            print("  ✓ Excellent consistency (low standard deviation)")
        elif stdev < 1.0:
            print("  ✓ Good consistency (moderate standard deviation)")
        else:
            print("  ✗ Poor consistency (high standard deviation)")

        coefficient_of_variation = (stdev / mean * 100) if mean > 0 else 0
        print(f"  Coefficient of Variation: {coefficient_of_variation:.1f}%")


def run_all_examples():
    """Run all reliability testing examples."""
    examples = [
        example_1_temperature_impact,
        example_2_prompt_variation_testing,
        example_3_multi_run_validation,
        example_4_edge_case_reliability,
        example_5_consistency_optimization,
        example_6_cross_model_reliability,
        example_7_statistical_validation
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

    parser = argparse.ArgumentParser(description="Module 02: Reliability Testing")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_temperature_impact,
            2: example_2_prompt_variation_testing,
            3: example_3_multi_run_validation,
            4: example_4_edge_case_reliability,
            5: example_5_consistency_optimization,
            6: example_6_cross_model_reliability,
            7: example_7_statistical_validation
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Run with --all for all examples or --example N for a specific example")
        print("\nAvailable examples:")
        print("1. Temperature Impact")
        print("2. Prompt Variation Testing")
        print("3. Multi-Run Validation")
        print("4. Edge Case Reliability")
        print("5. Consistency Optimization")
        print("6. Cross-Model Reliability")
        print("7. Statistical Validation")