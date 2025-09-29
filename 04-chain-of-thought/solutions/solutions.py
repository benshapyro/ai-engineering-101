"""
Module 04: Chain-of-Thought - Solutions

Complete solutions for all CoT exercises.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
import re
from typing import List, Dict, Tuple


# ===== Solution 1: CoT Conversion =====

def solution_1_cot_conversion():
    """
    Solution 1: Convert direct prompts to CoT format.
    """
    client = LLMClient("openai")

    print("Solution 1: Converting to Chain-of-Thought")
    print("=" * 50)

    problems = [
        "If a shirt costs $40 and is on sale for 30% off, what's the final price?",
        "A car travels 150 miles on 5 gallons of gas. How many miles per gallon?",
        "If 3 workers can paint a house in 4 days, how long for 2 workers?"
    ]

    for i, problem in enumerate(problems, 1):
        print(f"\n{'-'*40}")
        print(f"\nProblem {i}: {problem}")

        # Direct version
        direct_prompt = f"Question: {problem}\nAnswer:"

        print("\nDIRECT APPROACH:")
        direct_response = client.complete(direct_prompt, temperature=0.2, max_tokens=50)
        print(f"Response: {direct_response.strip()}")

        # CoT version
        cot_prompt = f"""Question: {problem}

Let's solve this step by step:

Step 1: Identify what we know
Step 2: Determine what we need to find
Step 3: Apply the appropriate formula or method
Step 4: Calculate the result
Step 5: Verify our answer makes sense

Solution:"""

        print("\nCHAIN-OF-THOUGHT APPROACH:")
        cot_response = client.complete(cot_prompt, temperature=0.2, max_tokens=300)
        print(f"Response:\n{cot_response.strip()}")

        print(f"\nAnalysis: CoT provides clearer reasoning and verification")


# ===== Solution 2: Step Granularity =====

def solution_2_step_granularity():
    """
    Solution 2: Find optimal step granularity.
    """
    client = LLMClient("openai")

    print("Solution 2: Step Granularity Optimization")
    print("=" * 50)

    problem = """
    A company manufactures widgets. Fixed costs are $10,000 per month.
    Variable cost per widget is $5. They sell widgets for $15 each.
    How many widgets must they sell to break even?
    """

    print(f"Problem: {problem}")

    # 3-step solution (too coarse)
    three_step_prompt = f"""Problem: {problem}

Solve in exactly 3 steps:
Step 1: Set up the break-even equation
Step 2: Solve for quantity
Step 3: State the answer

Solution:"""

    print("\n3-STEP SOLUTION (Coarse):")
    response_3 = client.complete(three_step_prompt, temperature=0.2, max_tokens=200)
    print(response_3.strip())

    # 5-step solution (optimal)
    five_step_prompt = f"""Problem: {problem}

Solve in exactly 5 steps:
Step 1: Identify fixed and variable costs
Step 2: Determine revenue per unit
Step 3: Calculate profit per unit
Step 4: Set up break-even equation
Step 5: Solve for break-even quantity

Solution:"""

    print("\n" + "-" * 40)
    print("\n5-STEP SOLUTION (Optimal):")
    response_5 = client.complete(five_step_prompt, temperature=0.2, max_tokens=300)
    print(response_5.strip())

    # 8-step solution (too granular)
    eight_step_prompt = f"""Problem: {problem}

Solve in exactly 8 detailed steps:
Step 1: List fixed costs
Step 2: List variable costs
Step 3: Calculate total cost formula
Step 4: Determine selling price
Step 5: Calculate revenue formula
Step 6: Set up profit equation
Step 7: Apply break-even condition
Step 8: Solve and verify

Solution:"""

    print("\n" + "-" * 40)
    print("\n8-STEP SOLUTION (Too Granular):")
    response_8 = client.complete(eight_step_prompt, temperature=0.2, max_tokens=400)
    print(response_8.strip())

    print("\n\nConclusion: 5 steps provides best balance of clarity and completeness")


# ===== Solution 3: Reasoning Debugger =====

def solution_3_reasoning_debugger():
    """
    Solution 3: Debug and fix faulty reasoning chains.
    """
    client = LLMClient("openai")

    print("Solution 3: Debugging Faulty Reasoning")
    print("=" * 50)

    faulty_solutions = [
        {
            "problem": "A 25% increase followed by a 20% decrease. Net change?",
            "faulty_reasoning": """
            Step 1: 25% increase means multiply by 1.25
            Step 2: 20% decrease means multiply by 0.20
            Step 3: Combined effect: 1.25 × 0.20 = 0.25
            Step 4: This is a 75% decrease overall
            """,
            "corrected_reasoning": """
CORRECTED REASONING:
Step 1: Start with original value = 1 (or 100%)
Step 2: After 25% increase: 1 × 1.25 = 1.25
Step 3: After 20% decrease: 1.25 × 0.80 = 1.00
       (Note: 20% decrease means multiply by 0.80, not 0.20!)
Step 4: Final value is 1.00 = 100% of original
Step 5: Net change = 0% (no change)

Verification: $100 → $125 (25% up) → $100 (20% down of $125) ✓
            """
        },
        {
            "problem": "Average speed for round trip: 60 mph one way, 40 mph return",
            "faulty_reasoning": """
            Step 1: Speed one way = 60 mph
            Step 2: Speed return = 40 mph
            Step 3: Average = (60 + 40) / 2 = 50 mph
            """,
            "corrected_reasoning": """
CORRECTED REASONING:
Step 1: Let distance one way = d miles
Step 2: Time going = d/60 hours
Step 3: Time returning = d/40 hours
Step 4: Total distance = 2d miles
Step 5: Total time = d/60 + d/40 = (2d + 3d)/120 = 5d/120 = d/24 hours
Step 6: Average speed = Total distance / Total time
        = 2d / (d/24) = 2d × 24/d = 48 mph

Note: Harmonic mean is needed for average speed, not arithmetic mean!
Verification: If d = 120 miles:
  Going: 120/60 = 2 hours
  Return: 120/40 = 3 hours
  Total: 240 miles in 5 hours = 48 mph ✓
            """
        }
    ]

    for i, case in enumerate(faulty_solutions, 1):
        print(f"\n{'-'*40}")
        print(f"\nCase {i}: {case['problem']}")
        print(f"\nFAULTY REASONING:{case['faulty_reasoning']}")
        print(case['corrected_reasoning'])

    print("\nKey Lessons:")
    print("1. Always check percentage calculations carefully")
    print("2. Average speed ≠ arithmetic mean of speeds")
    print("3. Verify answers with concrete examples")


# ===== Solution 4: Domain-Specific CoT =====

def solution_4_domain_specific_cot():
    """
    Solution 4: Create domain-specific CoT templates.
    """
    client = LLMClient("openai")

    print("Solution 4: Domain-Specific CoT Templates")
    print("=" * 50)

    # Legal reasoning template
    legal_template = """Legal Analysis Framework:

Step 1: ISSUE IDENTIFICATION
  - What is the legal question?
  - Which area of law applies?

Step 2: RULE STATEMENT
  - Relevant statutes and regulations
  - Applicable case law precedents
  - Legal principles involved

Step 3: APPLICATION TO FACTS
  - How do the rules apply to this situation?
  - Compare to similar cases
  - Distinguish from opposing precedents

Step 4: CONCLUSION
  - Legal outcome based on analysis
  - Strength of position
  - Potential counterarguments"""

    # Medical diagnosis template
    medical_template = """Diagnostic Reasoning Process:

Step 1: CHIEF COMPLAINT & HISTORY
  - Primary symptoms
  - Duration and onset
  - Associated symptoms
  - Relevant medical history

Step 2: DIFFERENTIAL DIAGNOSIS
  - List possible conditions
  - Rank by probability
  - Consider serious conditions first

Step 3: DIAGNOSTIC TESTING
  - Physical examination findings
  - Laboratory tests needed
  - Imaging studies indicated

Step 4: NARROWING THE DIAGNOSIS
  - Rule out conditions
  - Confirm with test results
  - Consider complications

Step 5: TREATMENT PLAN
  - Primary diagnosis
  - Management approach
  - Follow-up needed"""

    # Engineering design template
    engineering_template = """Design Problem Solution:

Step 1: REQUIREMENTS ANALYSIS
  - Functional requirements
  - Performance specifications
  - Constraints (cost, time, resources)
  - Safety and regulatory requirements

Step 2: CONCEPTUAL DESIGN
  - Brainstorm solutions
  - Evaluate feasibility
  - Select promising approaches

Step 3: DETAILED DESIGN
  - Component specifications
  - Material selection
  - Dimensional calculations
  - Interface definitions

Step 4: ANALYSIS & VALIDATION
  - Stress/load analysis
  - Performance simulation
  - Failure mode analysis
  - Safety verification

Step 5: OPTIMIZATION
  - Cost reduction opportunities
  - Performance improvements
  - Manufacturing considerations"""

    # Financial analysis template
    financial_template = """Financial Decision Analysis:

Step 1: OBJECTIVE DEFINITION
  - Investment goals
  - Risk tolerance
  - Time horizon
  - Success metrics

Step 2: DATA GATHERING
  - Historical performance
  - Market conditions
  - Financial statements
  - Economic indicators

Step 3: QUANTITATIVE ANALYSIS
  - Calculate key ratios
  - Project cash flows
  - Assess valuations
  - Risk metrics (VaR, Beta)

Step 4: QUALITATIVE FACTORS
  - Management quality
  - Competitive position
  - Industry trends
  - Regulatory environment

Step 5: DECISION & RATIONALE
  - Recommendation
  - Expected return vs risk
  - Alternative scenarios
  - Exit strategy"""

    print("\nLEGAL TEMPLATE:")
    print(legal_template)

    print("\n" + "-" * 40)
    print("\nMEDICAL TEMPLATE:")
    print(medical_template)

    print("\n" + "-" * 40)
    print("\nENGINEERING TEMPLATE:")
    print(engineering_template)

    print("\n" + "-" * 40)
    print("\nFINANCIAL TEMPLATE:")
    print(financial_template)

    # Test with example
    print("\n" + "=" * 40)
    print("\nTEST: Applying Financial Template")

    test_prompt = financial_template + """

Problem: Should a company acquire a smaller competitor for $10 million?

Analysis:"""

    response = client.complete(test_prompt, temperature=0.3, max_tokens=500)
    print(response.strip())


# ===== Solution 5: Self-Verification System =====

def solution_5_self_verification():
    """
    Solution 5: Build self-verifying CoT prompts.
    """
    client = LLMClient("openai")

    print("Solution 5: Self-Verification Systems")
    print("=" * 50)

    problem = """
    A tank can be filled by pipe A in 3 hours and by pipe B in 4 hours.
    If both pipes are open, how long to fill the tank?
    """

    verified_solution_prompt = f"""Problem: {problem}

SOLUTION WITH VERIFICATION:

Step 1: Define rates
- Pipe A fills 1/3 of tank per hour
- Pipe B fills 1/4 of tank per hour

Step 2: Combined rate
- Together: 1/3 + 1/4 = 4/12 + 3/12 = 7/12 tank per hour

Step 3: Time to fill
- If rate = 7/12 tank/hour
- Time = 1 tank ÷ (7/12 tank/hour) = 12/7 hours ≈ 1.71 hours

VERIFICATION METHOD 1: Dimensional Analysis
- Rate units: tank/hour ✓
- Time calculation: tank ÷ (tank/hour) = hours ✓
- Units work out correctly

VERIFICATION METHOD 2: Boundary Check
- Must be faster than fastest pipe alone (3 hours) ✓
- Must be slower than if both were as fast as A (1.5 hours) ✓
- 1.71 hours is between 1.5 and 3 ✓

VERIFICATION METHOD 3: Fraction Check
- In 12/7 hours, Pipe A contributes: (12/7) × (1/3) = 4/7 tank
- In 12/7 hours, Pipe B contributes: (12/7) × (1/4) = 3/7 tank
- Total: 4/7 + 3/7 = 7/7 = 1 complete tank ✓

VERIFICATION METHOD 4: Reverse Calculation
- If answer is 12/7 hours
- Pipe A alone would fill: (12/7) ÷ 3 = 4/7 of tank
- Pipe B alone would fill: (12/7) ÷ 4 = 3/7 of tank
- Together: 4/7 + 3/7 = 1 tank ✓

All verifications pass. Answer: 12/7 hours (or 1 hour 43 minutes)"""

    print(f"Problem: {problem}")
    print("\n" + "-" * 40)
    print(verified_solution_prompt)

    # Test with another problem
    print("\n" + "=" * 40)
    print("\nAPPLYING TO NEW PROBLEM:")

    new_problem = "If 5 machines produce 100 widgets in 2 hours, how long for 3 machines to produce 150 widgets?"

    verification_template = f"""Problem: {new_problem}

Let me solve this step-by-step with multiple verifications:

SOLUTION:"""

    response = client.complete(verification_template, temperature=0.2, max_tokens=600)
    print(response.strip())


# ===== Challenge Solution: Adaptive CoT System =====

def challenge_solution_adaptive_cot():
    """
    Challenge Solution: Adaptive CoT system based on problem complexity.
    """
    client = LLMClient("openai")

    print("Challenge Solution: Adaptive CoT System")
    print("=" * 50)

    def assess_problem_complexity(problem: str) -> str:
        """
        Assess complexity level of a problem.
        """
        # Simple heuristics for complexity
        indicators = {
            "simple": ["what is", "how much is", "calculate", "find"],
            "complex": ["optimize", "analyze", "evaluate", "compare multiple"],
            "mathematical": ["equation", "solve for", "derivative", "integral"],
            "multi_step": ["then", "after that", "subsequently", "if...then"]
        }

        problem_lower = problem.lower()

        # Count complexity indicators
        complexity_score = 0

        # Check for mathematical operations
        if any(op in problem for op in ['+', '-', '*', '/', '%', '^']):
            complexity_score += 1

        # Check for multiple values
        numbers = re.findall(r'\d+', problem)
        if len(numbers) > 2:
            complexity_score += 2

        # Check for complex keywords
        if any(word in problem_lower for word in indicators["complex"]):
            complexity_score += 3

        # Check for conditional logic
        if any(word in problem_lower for word in ["if", "when", "unless"]):
            complexity_score += 2

        # Determine complexity level
        if complexity_score <= 1:
            return "simple"
        elif complexity_score <= 4:
            return "moderate"
        else:
            return "complex"

    def generate_cot_template(complexity: str) -> str:
        """
        Generate appropriate CoT template based on complexity.
        """
        templates = {
            "simple": """Direct Solution:
Step 1: Identify the calculation needed
Step 2: Apply the formula
Answer: [result]""",

            "moderate": """Step-by-step Solution:
Step 1: Understand what we're given
Step 2: Identify what we need to find
Step 3: Choose the appropriate method
Step 4: Perform the calculation
Step 5: Check the answer
Answer: [result]""",

            "complex": """Comprehensive Analysis:
Step 1: Problem decomposition
  - Break down into sub-problems
  - Identify dependencies

Step 2: Information gathering
  - List known values
  - Identify unknowns
  - Note constraints

Step 3: Solution strategy
  - Select approach
  - Consider alternatives

Step 4: Detailed execution
  - Work through calculations
  - Handle each sub-problem

Step 5: Integration
  - Combine partial solutions
  - Resolve dependencies

Step 6: Verification
  - Check against constraints
  - Validate reasonableness
  - Test edge cases

Answer: [result with confidence level]"""
        }

        return templates.get(complexity, templates["moderate"])

    def validate_reasoning(problem: str, solution: str) -> Dict:
        """
        Validate the quality of CoT reasoning.
        """
        validation = {
            "has_steps": bool(re.search(r'Step \d+:', solution)),
            "complete": bool(re.search(r'Answer:|Conclusion:|Result:', solution)),
            "verified": bool(re.search(r'Check:|Verif|Valid', solution)),
            "clear": len(solution.split('\n')) > 3
        }

        validation["quality_score"] = sum(validation.values()) / len(validation)

        return validation

    # Test problems of varying complexity
    test_problems = [
        ("What is 15% of 80?", "simple"),
        ("A ladder leans against a wall. If the base is 3m from the wall and the ladder is 5m long, how high up the wall does it reach?", "moderate"),
        ("A company must choose between three investment options: A) $100k with 15% return but high risk, B) $75k with 10% return and medium risk, C) $50k with 5% return and low risk. They have $150k budget and want to maximize return while keeping overall risk below medium. How should they allocate?", "complex"),
    ]

    for problem, expected_complexity in test_problems:
        print(f"\n{'='*40}")
        print(f"Problem: {problem[:80]}...")

        # Assess complexity
        complexity = assess_problem_complexity(problem)
        print(f"\nExpected: {expected_complexity}")
        print(f"Assessed: {complexity}")

        # Generate template
        template = generate_cot_template(complexity)
        print(f"\nTemplate Level: {complexity}")

        # Create full prompt
        full_prompt = f"Problem: {problem}\n\n{template}"

        # Generate solution
        print("\nGenerating solution...")
        solution = client.complete(full_prompt, temperature=0.3, max_tokens=500)

        # Validate
        validation = validate_reasoning(problem, solution)
        print(f"\nValidation Results:")
        for key, value in validation.items():
            if key != "quality_score":
                print(f"  {key}: {'✓' if value else '✗'}")
        print(f"  Quality Score: {validation['quality_score']:.1%}")

    print("\n" + "=" * 50)
    print("\nAdaptive CoT System Benefits:")
    print("1. Optimizes token usage based on problem complexity")
    print("2. Provides appropriate level of detail")
    print("3. Ensures quality through validation")
    print("4. Scales reasoning to match problem requirements")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 04: CoT Solutions")
    parser.add_argument("--solution", type=int, help="Run specific solution (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge solution")
    parser.add_argument("--all", action="store_true", help="Run all solutions")

    args = parser.parse_args()

    solutions = {
        1: solution_1_cot_conversion,
        2: solution_2_step_granularity,
        3: solution_3_reasoning_debugger,
        4: solution_4_domain_specific_cot,
        5: solution_5_self_verification
    }

    if args.all:
        for sol_num, sol_func in solutions.items():
            print(f"\n{'='*60}")
            sol_func()
            print(f"\n{'='*60}\n")
        challenge_solution_adaptive_cot()
    elif args.challenge:
        challenge_solution_adaptive_cot()
    elif args.solution and args.solution in solutions:
        solutions[args.solution]()
    else:
        print("Module 04: Chain-of-Thought - Solutions")
        print("\nUsage:")
        print("  python solutions.py --solution N  # Run solution N")
        print("  python solutions.py --challenge    # Run challenge solution")
        print("  python solutions.py --all          # Run all solutions")
        print("\nAvailable solutions:")
        print("  1: CoT Conversion")
        print("  2: Step Granularity")
        print("  3: Reasoning Debugger")
        print("  4: Domain-Specific CoT")
        print("  5: Self-Verification")
        print("  Challenge: Adaptive CoT System")