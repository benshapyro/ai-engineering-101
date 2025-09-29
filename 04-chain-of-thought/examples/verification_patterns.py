"""
Module 04: Verification and Validation Patterns

Techniques for self-checking, error detection, and reasoning validation in CoT.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
import re


def example_1_answer_verification():
    """Verifying final answers through multiple methods."""
    print("=" * 60)
    print("Example 1: Multi-Method Answer Verification")
    print("=" * 60)

    client = LLMClient("openai")

    verification_prompt = """Solve and verify using multiple methods:

Problem: A rectangle has an area of 48 square meters and a perimeter of 28 meters.
Find its dimensions.

Solution:
Step 1: Set up equations
Let length = l and width = w
Area: l × w = 48
Perimeter: 2(l + w) = 28, so l + w = 14

Step 2: Solve the system
From perimeter: w = 14 - l
Substitute into area: l(14 - l) = 48
14l - l² = 48
l² - 14l + 48 = 0
(l - 6)(l - 8) = 0
So l = 6 or l = 8

Step 3: Find width
If l = 6, then w = 8
If l = 8, then w = 6

Answer: The dimensions are 6m × 8m

VERIFICATION METHOD 1: Check with original conditions
- Area: 6 × 8 = 48 ✓
- Perimeter: 2(6 + 8) = 28 ✓

VERIFICATION METHOD 2: Alternative solution approach
Using quadratic formula on l² - 14l + 48 = 0:
l = (14 ± √(196 - 192))/2 = (14 ± 2)/2 = 8 or 6 ✓

VERIFICATION METHOD 3: Graphical check
Plot y = 48/x and y = 14 - x
Intersection points give the dimensions ✓

Now solve and verify this problem similarly:
A number increased by 15 equals three times the number decreased by 7.
What is the number?

Solution and Verification:"""

    print("MULTI-METHOD VERIFICATION:")
    print("Confirming answers through different approaches")

    response = client.complete(verification_prompt, temperature=0.2, max_tokens=400)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Increases confidence in correctness")


def example_2_consistency_checking():
    """Checking for internal consistency in reasoning."""
    print("\n" + "=" * 60)
    print("Example 2: Internal Consistency Checking")
    print("=" * 60)

    client = LLMClient("openai")

    consistency_prompt = """Analyze this solution for consistency:

Problem: A company's profit increased by 20% from Year 1 to Year 2,
then decreased by 25% from Year 2 to Year 3.
If the Year 3 profit is $180,000, what was the Year 1 profit?

Proposed Solution:
Step 1: Let Year 1 profit = P
Step 2: Year 2 profit = P × 1.20 = 1.2P
Step 3: Year 3 profit = 1.2P × 0.75 = 0.9P
Step 4: Since 0.9P = $180,000
Step 5: P = $180,000 / 0.9 = $200,000

CONSISTENCY CHECKS:
□ Check 1: Direction of changes
  - Year 1 to 2: Increase ✓
  - Year 2 to 3: Decrease ✓

□ Check 2: Magnitude verification
  - 20% increase followed by 25% decrease should give net decrease
  - Net factor: 1.2 × 0.75 = 0.9 (10% decrease) ✓

□ Check 3: Reverse calculation
  - Year 1: $200,000
  - Year 2: $200,000 × 1.2 = $240,000
  - Year 3: $240,000 × 0.75 = $180,000 ✓

□ Check 4: Reasonableness
  - Final profit less than initial (due to net decrease) ✓

All checks pass. Solution is internally consistent.

Now check this solution for consistency:
Problem: A train travels from City A to City B at 60 mph and returns at 40 mph.
The total trip takes 5 hours. What is the distance between the cities?

Proposed Solution:
Distance = 120 miles

CONSISTENCY CHECKS:"""

    print("CONSISTENCY CHECKING:")
    print("Validating logical coherence")

    response = client.complete(consistency_prompt, temperature=0.2, max_tokens=400)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Catches logical errors early")


def example_3_boundary_testing():
    """Testing edge cases and boundary conditions."""
    print("\n" + "=" * 60)
    print("Example 3: Boundary and Edge Case Testing")
    print("=" * 60)

    client = LLMClient("openai")

    boundary_prompt = """Test this algorithm with boundary cases:

Algorithm: Binary search for a target in a sorted array

Standard Case:
Array: [1, 3, 5, 7, 9, 11, 13]
Target: 7
Result: Found at index 3 ✓

BOUNDARY TESTS:

Test 1: Empty array
Array: []
Target: 5
Expected: Not found
Result: Correctly returns -1 ✓

Test 2: Single element (found)
Array: [5]
Target: 5
Expected: Found at index 0
Result: Correct ✓

Test 3: Single element (not found)
Array: [5]
Target: 3
Expected: Not found
Result: Correctly returns -1 ✓

Test 4: Target at first position
Array: [1, 3, 5, 7, 9]
Target: 1
Expected: Found at index 0
Result: Correct ✓

Test 5: Target at last position
Array: [1, 3, 5, 7, 9]
Target: 9
Expected: Found at index 4
Result: Correct ✓

Test 6: Target not in array (less than all)
Array: [5, 7, 9]
Target: 3
Expected: Not found
Result: Correctly returns -1 ✓

Now test this function with boundary cases:
Function: Calculate the average of a list of numbers, excluding outliers
(outliers = values more than 2 standard deviations from mean)

BOUNDARY TESTS:"""

    print("BOUNDARY TESTING:")
    print("Validating edge case handling")

    response = client.complete(boundary_prompt, temperature=0.3, max_tokens=500)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Ensures robustness")


def example_4_sanity_checks():
    """Quick sanity checks for reasonableness."""
    print("\n" + "=" * 60)
    print("Example 4: Sanity Checks for Reasonableness")
    print("=" * 60)

    client = LLMClient("openai")

    sanity_prompt = """Apply sanity checks to validate this analysis:

Analysis: Website optimization impact

Data:
- Current load time: 5.2 seconds
- Optimized load time: 1.8 seconds
- Current conversion rate: 2.3%
- Expected conversion increase: 15% per second saved

Calculation:
- Time saved: 5.2 - 1.8 = 3.4 seconds
- Conversion increase: 3.4 × 15% = 51%
- New conversion rate: 2.3% × 1.51 = 3.473%

SANITY CHECKS:
✓ Order of magnitude: 3.47% conversion is reasonable for e-commerce
✓ Direction: Faster load → higher conversion (correct direction)
✓ Benchmarks: Industry average is 2-3%, so 3.47% is achievable
? Assumption check: 15% per second seems high, typical is 7-10%
! Recalculate with 8%: 3.4 × 8% = 27.2% increase → 2.92% conversion

Revised answer after sanity check: 2.92% conversion rate

Now apply sanity checks to this:
Analysis: Data center capacity planning

Current usage: 10,000 requests/second
Growth rate: 50% monthly
Capacity needed in 6 months: ?

Calculation: 10,000 × 1.5^6 = 113,906 requests/second

SANITY CHECKS:"""

    print("SANITY CHECKS:")
    print("Reality testing for calculations")

    response = client.complete(sanity_prompt, temperature=0.3, max_tokens=400)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Catches unrealistic results")


def example_5_cross_validation():
    """Cross-validating with different approaches."""
    print("\n" + "=" * 60)
    print("Example 5: Cross-Validation Approaches")
    print("=" * 60)

    client = LLMClient("openai")

    cross_val_prompt = """Solve using multiple independent approaches:

Problem: How many ways can you arrange the letters in "MISSISSIPPI"?

APPROACH 1: Direct formula
Total letters: 11
Repeated letters: M(1), I(4), S(4), P(2)
Formula: 11! / (1! × 4! × 4! × 2!)
= 39,916,800 / (1 × 24 × 24 × 2)
= 39,916,800 / 1,152
= 34,650

APPROACH 2: Step-by-step positioning
- Choose 4 positions for I: C(11,4) = 330
- Choose 4 positions for S from remaining 7: C(7,4) = 35
- Choose 2 positions for P from remaining 3: C(3,2) = 3
- M goes in the last position: 1
Total: 330 × 35 × 3 × 1 = 34,650 ✓

APPROACH 3: Verification by pattern
Starting with all unique: 11! = 39,916,800
Divide by repetitions:
- I appears 4 times: ÷ 4! = ÷ 24
- S appears 4 times: ÷ 4! = ÷ 24
- P appears 2 times: ÷ 2! = ÷ 2
Result: 39,916,800 ÷ (24 × 24 × 2) = 34,650 ✓

All approaches agree: 34,650 arrangements

Now solve this problem using multiple approaches:
A bag contains 5 red, 3 blue, and 2 green marbles.
What's the probability of drawing exactly 2 red marbles in 3 draws (without replacement)?

APPROACH 1:"""

    print("CROSS-VALIDATION:")
    print("Multiple solution paths to same answer")

    response = client.complete(cross_val_prompt, temperature=0.2, max_tokens=500)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Confirms correctness through agreement")


def example_6_error_propagation():
    """Tracking how errors propagate through reasoning."""
    print("\n" + "=" * 60)
    print("Example 6: Error Propagation Analysis")
    print("=" * 60)

    client = LLMClient("openai")

    error_prop_prompt = """Analyze error propagation in this calculation:

Problem: Calculate final price with multiple discounts and tax

Base price: $100.00 (assumed ±0.01 precision)
Discount 1: 20% (±0.5% measurement error)
Discount 2: 10% (±0.5% measurement error)
Sales tax: 8.5% (exact)

Calculation with error propagation:

Step 1: Apply first discount
Price = $100 × (1 - 0.20) = $80.00
Error: 20% ± 0.5% means factor is 0.80 ± 0.005
Range: $100 × [0.795, 0.805] = [$79.50, $80.50]
Midpoint: $80.00 ± $0.50

Step 2: Apply second discount
Price = $80 × (1 - 0.10) = $72.00
Error: 10% ± 0.5% means factor is 0.90 ± 0.005
Range: [$79.50, $80.50] × [0.895, 0.905]
= [$71.15, $72.85]
Midpoint: $72.00 ± $0.85

Step 3: Apply sales tax
Price = $72 × 1.085 = $78.12
Range: [$71.15, $72.85] × 1.085
= [$77.20, $79.04]
Final: $78.12 ± $0.92

ERROR SUMMARY:
- Initial uncertainty: ±$0.01 (0.01%)
- After discount 1: ±$0.50 (0.63%)
- After discount 2: ±$0.85 (1.18%)
- After tax: ±$0.92 (1.18%)

Now analyze error propagation for:
Calculating the area of a rectangle where:
Length: 10.5 m ± 0.1 m
Width: 8.3 m ± 0.1 m

Step-by-step error analysis:"""

    print("ERROR PROPAGATION:")
    print("Tracking uncertainty through calculations")

    response = client.complete(error_prop_prompt, temperature=0.2, max_tokens=400)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Quantifies confidence in results")


def example_7_proof_by_contradiction():
    """Using contradiction to verify reasoning."""
    print("\n" + "=" * 60)
    print("Example 7: Proof by Contradiction")
    print("=" * 60)

    client = LLMClient("openai")

    contradiction_prompt = """Verify this conclusion using proof by contradiction:

Claim: In a group of 13 people, at least 2 must have birthdays in the same month.

Proof by contradiction:
Assume the opposite: All 13 people have birthdays in different months.

Step 1: Count available months
There are only 12 months in a year.

Step 2: Apply pigeonhole principle
If all 13 people have different birth months,
we need 13 different months.

Step 3: Reach contradiction
We need 13 months but only have 12.
This is impossible.

Step 4: Conclusion
Our assumption must be false.
Therefore, at least 2 people must share a birth month. ✓

Now verify this claim using contradiction:
Claim: If n² is even, then n must be even.

Proof by contradiction:"""

    print("PROOF BY CONTRADICTION:")
    print("Validating through logical impossibility")

    response = client.complete(contradiction_prompt, temperature=0.2, max_tokens=400)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Strong logical validation")


def run_all_examples():
    """Run all verification pattern examples."""
    examples = [
        example_1_answer_verification,
        example_2_consistency_checking,
        example_3_boundary_testing,
        example_4_sanity_checks,
        example_5_cross_validation,
        example_6_error_propagation,
        example_7_proof_by_contradiction
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

    parser = argparse.ArgumentParser(description="Module 04: Verification Patterns")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_answer_verification,
            2: example_2_consistency_checking,
            3: example_3_boundary_testing,
            4: example_4_sanity_checks,
            5: example_5_cross_validation,
            6: example_6_error_propagation,
            7: example_7_proof_by_contradiction
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 04: Verification and Validation Patterns")
        print("\nUsage:")
        print("  python verification_patterns.py --all        # Run all examples")
        print("  python verification_patterns.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: Multi-Method Verification")
        print("  2: Consistency Checking")
        print("  3: Boundary Testing")
        print("  4: Sanity Checks")
        print("  5: Cross-Validation")
        print("  6: Error Propagation")
        print("  7: Proof by Contradiction")