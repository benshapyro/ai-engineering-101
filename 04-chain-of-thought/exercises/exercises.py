"""
Module 04: Chain-of-Thought - Exercises

Practice exercises for mastering CoT prompting techniques.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from typing import List, Dict, Tuple


# ===== Exercise 1: CoT Conversion =====

def exercise_1_cot_conversion():
    """
    Exercise 1: Convert direct prompts to CoT format.

    TODO:
    1. Take direct prompts and add CoT reasoning
    2. Test both versions and compare accuracy
    3. Identify which problems benefit most from CoT
    """
    client = LLMClient("openai")

    print("Exercise 1: Converting to Chain-of-Thought")
    print("=" * 50)

    # Direct prompts to convert
    problems = [
        "If a shirt costs $40 and is on sale for 30% off, what's the final price?",
        "A car travels 150 miles on 5 gallons of gas. How many miles per gallon?",
        "If 3 workers can paint a house in 4 days, how long for 2 workers?"
    ]

    print("TODO: Convert each direct prompt to CoT format")

    for i, problem in enumerate(problems, 1):
        print(f"\n{'-'*40}")
        print(f"\nProblem {i}: {problem}")

        # TODO: Create direct version
        direct_prompt = f"Question: {problem}\nAnswer:"

        # TODO: Create CoT version
        cot_prompt = f"""Question: {problem}

Let's solve this step by step:
Step 1:"""  # TODO: Complete the CoT structure

        print("\nTODO: Test both versions and compare")
        # TODO: Run both prompts and compare results


# ===== Exercise 2: Step Granularity =====

def exercise_2_step_granularity():
    """
    Exercise 2: Find optimal step granularity for different problems.

    TODO:
    1. Create 3-step, 5-step, and 8-step versions
    2. Evaluate clarity and completeness
    3. Determine optimal granularity
    """
    client = LLMClient("openai")

    print("Exercise 2: Step Granularity Optimization")
    print("=" * 50)

    problem = """
    A company manufactures widgets. Fixed costs are $10,000 per month.
    Variable cost per widget is $5. They sell widgets for $15 each.
    How many widgets must they sell to break even?
    """

    print(f"Problem: {problem}")

    # TODO: Create 3-step solution
    three_step = """TODO: Create a 3-step solution
Step 1:
Step 2:
Step 3:
"""

    # TODO: Create 5-step solution
    five_step = """TODO: Create a 5-step solution
Step 1:
Step 2:
Step 3:
Step 4:
Step 5:
"""

    # TODO: Create 8-step solution (might be too granular?)
    eight_step = """TODO: Create an 8-step solution
..."""

    print("\nTODO: Implement and test different granularities")


# ===== Exercise 3: Reasoning Debugger =====

def exercise_3_reasoning_debugger():
    """
    Exercise 3: Debug and fix faulty reasoning chains.

    TODO:
    1. Identify errors in given reasoning
    2. Correct the faulty steps
    3. Verify the corrected solution
    """
    client = LLMClient("openai")

    print("Exercise 3: Debugging Faulty Reasoning")
    print("=" * 50)

    # Faulty reasoning examples
    faulty_solutions = [
        {
            "problem": "A 25% increase followed by a 20% decrease. Net change?",
            "faulty_reasoning": """
            Step 1: 25% increase means multiply by 1.25
            Step 2: 20% decrease means multiply by 0.20
            Step 3: Combined effect: 1.25 × 0.20 = 0.25
            Step 4: This is a 75% decrease overall
            """,
            "error": "Step 2 calculation error"
        },
        {
            "problem": "Average speed for round trip: 60 mph one way, 40 mph return",
            "faulty_reasoning": """
            Step 1: Speed one way = 60 mph
            Step 2: Speed return = 40 mph
            Step 3: Average = (60 + 40) / 2 = 50 mph
            """,
            "error": "Incorrect average formula"
        }
    ]

    for i, case in enumerate(faulty_solutions, 1):
        print(f"\n{'-'*40}")
        print(f"\nCase {i}:")
        print(f"Problem: {case['problem']}")
        print(f"Faulty Reasoning:{case['faulty_reasoning']}")
        print(f"Known Error: {case['error']}")

        # TODO: Create corrected reasoning
        corrected_reasoning = """TODO: Fix the reasoning
        Step 1:
        ..."""

        print("\nTODO: Implement corrected reasoning")


# ===== Exercise 4: Domain-Specific CoT =====

def exercise_4_domain_specific_cot():
    """
    Exercise 4: Create domain-specific CoT templates.

    TODO:
    1. Design CoT patterns for different domains
    2. Test with domain-specific problems
    3. Refine templates based on results
    """
    client = LLMClient("openai")

    print("Exercise 4: Domain-Specific CoT Templates")
    print("=" * 50)

    # TODO: Create templates for different domains

    # Legal reasoning template
    legal_template = """Legal Analysis Framework:

    TODO: Complete this template
    Step 1: Identify relevant laws/statutes
    Step 2:
    Step 3:
    ..."""

    # Medical diagnosis template
    medical_template = """Diagnostic Reasoning Process:

    TODO: Complete this template
    Step 1: Patient history and symptoms
    Step 2:
    Step 3:
    ..."""

    # Engineering design template
    engineering_template = """Design Problem Solution:

    TODO: Complete this template
    Step 1: Define requirements and constraints
    Step 2:
    Step 3:
    ..."""

    # Financial analysis template
    financial_template = """Financial Decision Analysis:

    TODO: Complete this template
    Step 1: Identify key metrics
    Step 2:
    Step 3:
    ..."""

    print("\nTODO: Complete domain templates and test with examples")


# ===== Exercise 5: Self-Verification System =====

def exercise_5_self_verification():
    """
    Exercise 5: Build self-verifying CoT prompts.

    TODO:
    1. Add verification steps to solutions
    2. Implement multiple verification methods
    3. Create error recovery mechanisms
    """
    client = LLMClient("openai")

    print("Exercise 5: Self-Verification Systems")
    print("=" * 50)

    problem = """
    A tank can be filled by pipe A in 3 hours and by pipe B in 4 hours.
    If both pipes are open, how long to fill the tank?
    """

    # TODO: Create solution with verification
    verified_solution = """Problem: {problem}

    Solution:
    TODO: Add solution steps

    Verification Method 1: Check units
    TODO: Verify dimensional analysis

    Verification Method 2: Boundary check
    TODO: Check if answer is between fastest and slowest

    Verification Method 3: Reverse calculation
    TODO: Work backwards from answer
    """.format(problem=problem)

    print(f"\nProblem: {problem}")
    print("\nTODO: Implement self-verifying solution")


# ===== Challenge: Adaptive CoT System =====

def challenge_adaptive_cot():
    """
    Challenge: Build a system that adapts CoT complexity based on problem difficulty.

    Requirements:
    1. Analyze problem complexity
    2. Choose appropriate CoT depth
    3. Generate reasoning at correct granularity
    4. Validate output quality

    TODO: Complete the implementation
    """
    client = LLMClient("openai")

    print("Challenge: Adaptive CoT System")
    print("=" * 50)

    def assess_problem_complexity(problem: str) -> str:
        """
        TODO: Assess complexity level of a problem.

        Returns: "simple", "moderate", or "complex"
        """
        # TODO: Implement complexity assessment
        # Consider: math operations, steps required, domain knowledge

        return "moderate"  # Placeholder

    def generate_cot_template(complexity: str) -> str:
        """
        TODO: Generate appropriate CoT template based on complexity.
        """
        templates = {
            "simple": "TODO: 2-3 step template",
            "moderate": "TODO: 4-5 step template",
            "complex": "TODO: 6+ step template with verification"
        }

        # TODO: Return appropriate template
        return templates.get(complexity, templates["moderate"])

    def validate_reasoning(problem: str, solution: str) -> Dict:
        """
        TODO: Validate the quality of CoT reasoning.

        Returns: Dictionary with validation metrics
        """
        validation = {
            "has_steps": False,  # TODO: Check for step structure
            "complete": False,   # TODO: Check if reaches conclusion
            "verified": False,   # TODO: Check for verification
            "clear": False       # TODO: Check clarity
        }

        # TODO: Implement validation logic

        return validation

    # Test problems of varying complexity
    test_problems = [
        "What is 15% of 80?",  # Simple
        "A ladder leans against a wall. If the base is 3m from the wall and the ladder is 5m long, how high up the wall does it reach?",  # Moderate
        "A company must choose between three investment options with different risk profiles, time horizons, and expected returns. How should they evaluate?",  # Complex
    ]

    for problem in test_problems:
        print(f"\n{'-'*40}")
        print(f"Problem: {problem[:50]}...")

        complexity = assess_problem_complexity(problem)
        template = generate_cot_template(complexity)

        print(f"Assessed Complexity: {complexity}")
        print(f"TODO: Apply template and generate solution")

        # TODO: Generate solution using template
        # TODO: Validate the reasoning

    print("\n\nTODO: Complete adaptive CoT implementation")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 04: CoT Exercises")
    parser.add_argument("--exercise", type=int, help="Run specific exercise (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge")
    parser.add_argument("--all", action="store_true", help="Run all exercises")

    args = parser.parse_args()

    exercises = {
        1: exercise_1_cot_conversion,
        2: exercise_2_step_granularity,
        3: exercise_3_reasoning_debugger,
        4: exercise_4_domain_specific_cot,
        5: exercise_5_self_verification
    }

    if args.all:
        for ex_num, ex_func in exercises.items():
            print(f"\n{'='*60}")
            ex_func()
            print(f"\n{'='*60}\n")
        challenge_adaptive_cot()
    elif args.challenge:
        challenge_adaptive_cot()
    elif args.exercise and args.exercise in exercises:
        exercises[args.exercise]()
    else:
        print("Module 04: Chain-of-Thought - Exercises")
        print("\nUsage:")
        print("  python exercises.py --exercise N  # Run exercise N")
        print("  python exercises.py --challenge    # Run challenge")
        print("  python exercises.py --all          # Run all exercises")
        print("\nAvailable exercises:")
        print("  1: CoT Conversion")
        print("  2: Step Granularity")
        print("  3: Reasoning Debugger")
        print("  4: Domain-Specific CoT")
        print("  5: Self-Verification")
        print("  Challenge: Adaptive CoT System")