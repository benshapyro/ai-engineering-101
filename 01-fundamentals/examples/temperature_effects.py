"""
Module 01: Fundamentals
Temperature and Parameter Effects

Learn how temperature and other parameters affect LLM outputs.
"""

import os
from dotenv import load_dotenv

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.utils import LLMClient

load_dotenv()

# Initialize LLM client
llm = LLMClient()


# Example 1: Temperature Comparison
print("=" * 50)
print("Example 1: Temperature Effects (0.0 to 2.0)")
print("=" * 50)

prompt = "Complete this sentence: The best programming language for beginners is"

temperatures = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]

print(f"\nPrompt: {prompt}\n")

for temp in temperatures:
    print(f"Temperature {temp}:")
    response = llm.complete(prompt, temperature=temp, max_tokens=50)
    print(f"  {response}")
    print()


# Example 2: Deterministic vs Creative Tasks
print("\n" + "=" * 50)
print("Example 2: Choosing Temperature for Task Type")
print("=" * 50)

# Deterministic task - use low temperature
print("\nDeterministic Task (Calculation):")
print("Temperature: 0.0 (for consistency)")
calc_prompt = "Calculate: (45 + 67) * 3 - 12. Show your work."

for i in range(3):
    response = llm.complete(calc_prompt, temperature=0.0, max_tokens=100)
    print(f"  Attempt {i+1}: {response[:100]}...")

# Creative task - use higher temperature
print("\nCreative Task (Story Idea):")
print("Temperature: 1.2 (for variety)")
story_prompt = "Create a unique story premise combining: detective, bakery, time travel"

for i in range(3):
    response = llm.complete(story_prompt, temperature=1.2, max_tokens=100)
    print(f"  Attempt {i+1}: {response[:100]}...")


# Example 3: Temperature for Different Use Cases
print("\n" + "=" * 50)
print("Example 3: Temperature Guidelines by Use Case")
print("=" * 50)

use_cases = [
    {
        "name": "Code Generation",
        "temperature": 0.2,
        "prompt": "Write a Python function to check if a string is a palindrome.",
        "reason": "Low temp for correct, consistent code"
    },
    {
        "name": "Data Extraction",
        "temperature": 0.0,
        "prompt": "Extract the email addresses from: Contact us at info@example.com or support@test.org",
        "reason": "Zero temp for factual accuracy"
    },
    {
        "name": "Content Brainstorming",
        "temperature": 1.0,
        "prompt": "Generate 5 unique blog post ideas about sustainable living.",
        "reason": "High temp for creative diversity"
    },
    {
        "name": "Classification",
        "temperature": 0.3,
        "prompt": "Classify this review as positive, negative, or neutral: 'The product works but shipping was slow.'",
        "reason": "Low temp for consistent categorization"
    },
    {
        "name": "Marketing Copy",
        "temperature": 0.8,
        "prompt": "Write an engaging tagline for an eco-friendly water bottle.",
        "reason": "Medium-high temp for creative yet focused output"
    }
]

for case in use_cases:
    print(f"\nUse Case: {case['name']}")
    print(f"Temperature: {case['temperature']}")
    print(f"Reason: {case['reason']}")
    print(f"Prompt: {case['prompt']}")

    response = llm.complete(
        case['prompt'],
        temperature=case['temperature'],
        max_tokens=150
    )
    print(f"Response: {response}")


# Example 4: Max Tokens Parameter
print("\n" + "=" * 50)
print("Example 4: Max Tokens Parameter Effects")
print("=" * 50)

prompt = "Explain what machine learning is and how it works."

token_limits = [20, 50, 100, 200]

for max_tok in token_limits:
    print(f"\nMax Tokens: {max_tok}")
    response = llm.complete(prompt, temperature=0.7, max_tokens=max_tok)
    print(f"Response: {response}")
    print(f"(Length: ~{len(response.split())} words)")


# Example 5: Top_p (Nucleus Sampling)
print("\n" + "=" * 50)
print("Example 5: Top_p Parameter (Alternative to Temperature)")
print("=" * 50)

prompt = "List 5 interesting facts about octopuses."

print("\nNote: top_p controls diversity by considering only tokens")
print("in the top probability mass. Common values: 0.1 (focused) to 1.0 (diverse)\n")

# Most LLM clients use temperature by default, but some support top_p
# Demonstrating the concept
print("Low top_p (0.1) - More focused, less random:")
response = llm.complete(prompt, temperature=0.3, max_tokens=200)
print(response)

print("\nHigh top_p (1.0) - More diverse, more random:")
response = llm.complete(prompt, temperature=0.9, max_tokens=200)
print(response)


# Example 6: Frequency and Presence Penalties
print("\n" + "=" * 50)
print("Example 6: Repetition Control Parameters")
print("=" * 50)

prompt = "Write a short paragraph about beaches."

print("\nNo penalties (may have repetition):")
response = llm.complete(
    prompt,
    temperature=0.7,
    max_tokens=150
)
print(response)

print("\nWith frequency penalty (reduces repetition):")
# Note: Some LLM APIs support frequency_penalty parameter
# Simulated here with a modified prompt
modified_prompt = prompt + " Use varied vocabulary without repeating words."
response = llm.complete(
    modified_prompt,
    temperature=0.7,
    max_tokens=150
)
print(response)


# Example 7: Combining Parameters for Optimal Results
print("\n" + "=" * 50)
print("Example 7: Parameter Combinations")
print("=" * 50)

scenarios = [
    {
        "scenario": "Technical Documentation",
        "params": {"temperature": 0.2, "max_tokens": 300},
        "prompt": "Explain how to set up a Python virtual environment.",
        "rationale": "Low temp for accuracy, sufficient tokens for detail"
    },
    {
        "scenario": "Creative Writing",
        "params": {"temperature": 1.0, "max_tokens": 500},
        "prompt": "Write the opening paragraph of a mystery novel.",
        "rationale": "High temp for creativity, more tokens for storytelling"
    },
    {
        "scenario": "Quick Summary",
        "params": {"temperature": 0.3, "max_tokens": 100},
        "prompt": "Summarize the key benefits of remote work in 2 sentences.",
        "rationale": "Low temp for consistency, limited tokens for conciseness"
    },
    {
        "scenario": "Data Analysis",
        "params": {"temperature": 0.0, "max_tokens": 200},
        "prompt": "Analyze this data and identify the trend: Sales Q1: 100K, Q2: 120K, Q3: 140K, Q4: 155K",
        "rationale": "Zero temp for factual analysis, moderate tokens for explanation"
    }
]

for scenario in scenarios:
    print(f"\n{'='*50}")
    print(f"Scenario: {scenario['scenario']}")
    print(f"Parameters: {scenario['params']}")
    print(f"Rationale: {scenario['rationale']}")
    print(f"Prompt: {scenario['prompt']}")

    response = llm.complete(
        scenario['prompt'],
        temperature=scenario['params']['temperature'],
        max_tokens=scenario['params']['max_tokens']
    )
    print(f"\nResponse:\n{response}")


# Example 8: Practical Temperature Guide
print("\n" + "=" * 50)
print("Example 8: Temperature Selection Guide")
print("=" * 50)

temperature_guide = """
Temperature Selection Guide:

0.0 - 0.3: Deterministic & Factual
├─ Code generation
├─ Data extraction
├─ Mathematical calculations
├─ Translations (formal)
└─ Classification tasks

0.3 - 0.7: Balanced & Reliable
├─ General Q&A
├─ Explanations
├─ Professional writing
├─ Documentation
└─ Summarization

0.7 - 1.0: Creative & Varied
├─ Brainstorming
├─ Marketing copy
├─ Content ideas
├─ Conversational responses
└─ Story outlines

1.0 - 2.0: Highly Creative & Experimental
├─ Poetry
├─ Creative fiction
├─ Unique ideas
├─ Artistic descriptions
└─ Innovation exercises

Pro Tips:
• Start with 0.7 as a default
• Lower temperature if outputs are inconsistent
• Raise temperature if outputs are too similar/boring
• For production systems, use 0.0-0.3 for reliability
• Test with multiple temperatures to find optimal setting
"""

print(temperature_guide)


# Example 9: A/B Testing Different Parameters
print("\n" + "=" * 50)
print("Example 9: A/B Testing Parameters")
print("=" * 50)

test_prompt = "Write a welcome email for new users of a productivity app."

print("Testing three parameter configurations:\n")

configs = [
    {"name": "Config A (Conservative)", "temp": 0.3, "tokens": 150},
    {"name": "Config B (Balanced)", "temp": 0.7, "tokens": 200},
    {"name": "Config C (Creative)", "temp": 1.0, "tokens": 250}
]

for config in configs:
    print(f"{config['name']}:")
    print(f"  Temperature: {config['temp']}, Max Tokens: {config['tokens']}")

    response = llm.complete(
        test_prompt,
        temperature=config['temp'],
        max_tokens=config['tokens']
    )
    print(f"  Response: {response}")
    print()


print("\n" + "=" * 50)
print("Key Takeaways")
print("=" * 50)
print("""
Temperature and Parameter Guidelines:

1. Temperature (0.0 - 2.0)
   - Controls randomness/creativity
   - Lower = more consistent, higher = more varied
   - Most important parameter to tune

2. Max Tokens
   - Limits response length
   - Plan for ~4 chars per token on average
   - Set based on expected response length

3. Top-p (Nucleus Sampling)
   - Alternative to temperature
   - Controls diversity of word selection
   - Usually use temperature OR top_p, not both

4. Frequency/Presence Penalties
   - Reduces repetition
   - Useful for longer text generation
   - Fine-tune if you notice repeated phrases

Best Practices:
• Test multiple temperatures for your use case
• Document your chosen parameters
• Use low temps for production reliability
• Use high temps for creative exploration
• Adjust max_tokens to avoid truncation
• Monitor outputs and adjust as needed
""")