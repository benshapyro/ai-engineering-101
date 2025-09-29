"""
Module 04: Basic Chain-of-Thought Reasoning

Fundamental patterns for implementing step-by-step reasoning in prompts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json


def example_1_zero_shot_cot():
    """The magic phrase: 'Let's think step by step'."""
    print("=" * 60)
    print("Example 1: Zero-Shot Chain-of-Thought")
    print("=" * 60)

    client = LLMClient("openai")

    problem = """
    A bookstore had 45 books. On Monday, they sold 12 books.
    On Tuesday, they received a shipment of 25 books.
    On Wednesday, they sold 8 books.
    How many books does the bookstore have now?
    """

    # Without CoT
    direct_prompt = f"Problem: {problem}\n\nAnswer:"

    print("WITHOUT CHAIN-OF-THOUGHT:")
    print("Prompt: Direct question")
    direct_response = client.complete(direct_prompt, temperature=0.2, max_tokens=50)
    print(f"Response: {direct_response.strip()}")

    # With Zero-Shot CoT
    cot_prompt = f"Problem: {problem}\n\nLet's think step by step:"

    print("\n" + "-" * 40)
    print("\nWITH CHAIN-OF-THOUGHT:")
    print("Prompt: Added 'Let's think step by step'")
    cot_response = client.complete(cot_prompt, temperature=0.2, max_tokens=200)
    print(f"Response:\n{cot_response.strip()}")

    print("\nKey Insight: Simple phrase triggers reasoning behavior")


def example_2_few_shot_cot():
    """Providing examples of reasoning patterns."""
    print("\n" + "=" * 60)
    print("Example 2: Few-Shot Chain-of-Thought")
    print("=" * 60)

    client = LLMClient("openai")

    few_shot_cot = """Solve these word problems:

Problem: Sarah has 3 bags with 4 apples each. She gives away 5 apples. How many apples does she have left?

Solution:
Step 1: Calculate total apples Sarah has
   - 3 bags × 4 apples per bag = 12 apples

Step 2: Subtract apples given away
   - 12 apples - 5 apples = 7 apples

Answer: Sarah has 7 apples left.

Problem: A train travels 60 miles per hour for 2.5 hours. How far does it travel?

Solution:
Step 1: Identify the formula
   - Distance = Speed × Time

Step 2: Apply the values
   - Distance = 60 mph × 2.5 hours = 150 miles

Answer: The train travels 150 miles.

Problem: A pizza is cut into 8 slices. Tom eats 3 slices, and Mary eats 2 slices. What fraction of the pizza is left?

Solution:"""

    print("FEW-SHOT COT:")
    print("Teaching reasoning pattern through examples")

    response = client.complete(few_shot_cot, temperature=0.2, max_tokens=200)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Model learns consistent reasoning format")


def example_3_structured_cot():
    """Using structured formats for reasoning."""
    print("\n" + "=" * 60)
    print("Example 3: Structured Chain-of-Thought")
    print("=" * 60)

    client = LLMClient("openai")

    structured_prompt = """Analyze this business scenario using structured reasoning:

Scenario: A coffee shop sells 100 cups of coffee per day at $5 each.
They want to increase the price to $6 but estimate they'll lose 15% of sales.
Should they increase the price?

Analysis Framework:
1. CURRENT STATE
   - Daily cups: [calculate]
   - Price per cup: [identify]
   - Daily revenue: [calculate]

2. PROJECTED STATE
   - New price: [identify]
   - Sales reduction: [calculate]
   - New daily cups: [calculate]
   - New daily revenue: [calculate]

3. COMPARISON
   - Revenue change: [calculate]
   - Percentage change: [calculate]

4. RECOMMENDATION
   - Decision: [based on analysis]
   - Reasoning: [explain why]

Complete the analysis:"""

    print("STRUCTURED COT:")
    print("Framework guides systematic reasoning")

    response = client.complete(structured_prompt, temperature=0.2, max_tokens=400)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Ensures comprehensive analysis")


def example_4_mathematical_cot():
    """CoT for mathematical problem solving."""
    print("\n" + "=" * 60)
    print("Example 4: Mathematical Chain-of-Thought")
    print("=" * 60)

    client = LLMClient("openai")

    math_prompt = """Solve this step-by-step, showing all work:

A rectangular garden has a length that is 3 meters more than twice its width.
If the perimeter is 36 meters, what are the dimensions of the garden?

Solution:
Let me define variables and work through this systematically."""

    print("MATHEMATICAL COT:")
    print("Systematic approach to equation solving")

    response = client.complete(math_prompt, temperature=0.2, max_tokens=400)
    print(f"\nResponse:\n{response.strip()}")

    # Verification prompt
    verify_prompt = f"""{math_prompt}

{response}

Now verify this answer by checking if the perimeter equals 36:"""

    print("\n" + "-" * 40)
    print("\nVERIFICATION STEP:")
    verify_response = client.complete(verify_prompt, temperature=0.2, max_tokens=150)
    print(f"Verification:\n{verify_response.strip()}")


def example_5_logical_deduction_cot():
    """CoT for logical reasoning and deduction."""
    print("\n" + "=" * 60)
    print("Example 5: Logical Deduction Chain-of-Thought")
    print("=" * 60)

    client = LLMClient("openai")

    logic_prompt = """Use logical reasoning to solve this puzzle:

Facts:
1. Amy, Bob, Carol, and Dave each own exactly one pet: cat, dog, fish, or bird
2. Amy is allergic to fur
3. Bob's pet can fly
4. Carol's pet lives in water
5. The person with the cat lives next to the person with the fish
6. Dave lives next to Carol

Question: Who owns which pet?

Let's reason through this step by step:"""

    print("LOGICAL DEDUCTION COT:")
    print("Systematic elimination and deduction")

    response = client.complete(logic_prompt, temperature=0.2, max_tokens=500)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Makes logical constraints explicit")


def example_6_algorithmic_cot():
    """CoT for algorithm design and analysis."""
    print("\n" + "=" * 60)
    print("Example 6: Algorithmic Chain-of-Thought")
    print("=" * 60)

    client = LLMClient("openai")

    algo_prompt = """Design an algorithm to find the second largest number in an unsorted array.

Think through this step-by-step:
1. Understand the problem requirements
2. Consider edge cases
3. Design the approach
4. Write the algorithm
5. Analyze time and space complexity

Solution:"""

    print("ALGORITHMIC COT:")
    print("Systematic algorithm development")

    response = client.complete(algo_prompt, temperature=0.3, max_tokens=500)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Comprehensive algorithm design process")


def example_7_decision_tree_cot():
    """CoT for complex decision making."""
    print("\n" + "=" * 60)
    print("Example 7: Decision Tree Chain-of-Thought")
    print("=" * 60)

    client = LLMClient("openai")

    decision_prompt = """Make a recommendation for this scenario using decision tree reasoning:

Scenario: A startup has $50,000 and must choose between:
A) Hire 2 junior developers for 6 months
B) Hire 1 senior developer for 6 months
C) Outsource development to an agency for 3 months
D) Invest in marketing and delay development

Factors to consider:
- Current team has no technical lead
- Product MVP needed in 4 months
- Competitors launching soon
- Limited runway (8 months)

Decision Analysis:
Step through each option systematically..."""

    print("DECISION TREE COT:")
    print("Systematic evaluation of options")

    response = client.complete(decision_prompt, temperature=0.3, max_tokens=600)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Transparent decision rationale")


def example_8_error_recovery_cot():
    """CoT with error detection and correction."""
    print("\n" + "=" * 60)
    print("Example 8: Error Recovery in Chain-of-Thought")
    print("=" * 60)

    client = LLMClient("openai")

    error_prompt = """Solve this problem, and if you make any errors, correct them:

A store offers a 20% discount on all items. There's an additional 15% off
for members on the already discounted price. If an item originally costs $100,
what does a member pay?

Initial attempt:
Step 1: Apply 20% discount
   $100 × 0.20 = $20 discount
   $100 - $20 = $80

Step 2: Apply 15% member discount
   $80 × 0.15 = $12 discount
   $80 + $12 = $92

Wait, that doesn't seem right. Let me recalculate Step 2:"""

    print("ERROR RECOVERY COT:")
    print("Self-correction during reasoning")

    response = client.complete(error_prompt, temperature=0.2, max_tokens=300)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Models can identify and fix their mistakes")


def run_all_examples():
    """Run all basic reasoning examples."""
    examples = [
        example_1_zero_shot_cot,
        example_2_few_shot_cot,
        example_3_structured_cot,
        example_4_mathematical_cot,
        example_5_logical_deduction_cot,
        example_6_algorithmic_cot,
        example_7_decision_tree_cot,
        example_8_error_recovery_cot
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

    parser = argparse.ArgumentParser(description="Module 04: Basic Reasoning")
    parser.add_argument("--example", type=int, help="Run specific example (1-8)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_zero_shot_cot,
            2: example_2_few_shot_cot,
            3: example_3_structured_cot,
            4: example_4_mathematical_cot,
            5: example_5_logical_deduction_cot,
            6: example_6_algorithmic_cot,
            7: example_7_decision_tree_cot,
            8: example_8_error_recovery_cot
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 04: Basic Chain-of-Thought Reasoning")
        print("\nUsage:")
        print("  python basic_reasoning.py --all        # Run all examples")
        print("  python basic_reasoning.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: Zero-Shot CoT")
        print("  2: Few-Shot CoT")
        print("  3: Structured CoT")
        print("  4: Mathematical CoT")
        print("  5: Logical Deduction CoT")
        print("  6: Algorithmic CoT")
        print("  7: Decision Tree CoT")
        print("  8: Error Recovery CoT")