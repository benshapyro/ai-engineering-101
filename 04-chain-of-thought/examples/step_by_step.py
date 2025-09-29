"""
Module 04: Step-by-Step Decomposition

Advanced techniques for breaking down complex problems into manageable steps.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
import re


def example_1_granular_steps():
    """Finding the right level of step granularity."""
    print("=" * 60)
    print("Example 1: Step Granularity Control")
    print("=" * 60)

    client = LLMClient("openai")

    problem = "Calculate the compound interest on $1000 at 5% annual rate for 3 years, compounded quarterly."

    # Too coarse
    coarse_prompt = f"""Problem: {problem}

Solve in 2 steps:
Step 1:"""

    print("TOO COARSE (2 steps):")
    coarse_response = client.complete(coarse_prompt, temperature=0.2, max_tokens=200)
    print(f"{coarse_response.strip()}")

    # Just right
    balanced_prompt = f"""Problem: {problem}

Solve in 5 clear steps:
Step 1: Identify the formula
Step 2: Define variables
Step 3: Calculate quarterly rate and periods
Step 4: Apply the formula
Step 5: Calculate final interest earned

Solution:"""

    print("\n" + "-" * 40)
    print("\nBALANCED (5 steps):")
    balanced_response = client.complete(balanced_prompt, temperature=0.2, max_tokens=300)
    print(f"{balanced_response.strip()}")

    # Too granular
    granular_prompt = f"""Problem: {problem}

Solve in 10 detailed micro-steps:
Step 1:"""

    print("\n" + "-" * 40)
    print("\nTOO GRANULAR (10 steps):")
    print("[Can be excessive and confusing]")

    print("\nKey Insight: Match granularity to problem complexity")


def example_2_parallel_vs_sequential():
    """Organizing steps for parallel vs sequential execution."""
    print("\n" + "=" * 60)
    print("Example 2: Parallel vs Sequential Steps")
    print("=" * 60)

    client = LLMClient("openai")

    # Sequential steps
    sequential_prompt = """Plan a data pipeline with SEQUENTIAL steps:

Task: Process customer data for monthly reporting

Sequential Plan:
Step 1: Extract data from source systems
Step 2: Clean and validate data
Step 3: Transform to reporting schema
Step 4: Calculate metrics
Step 5: Generate report
Step 6: Distribute to stakeholders

Each step depends on the previous one completing.

Now plan the same task with PARALLEL steps where possible:

Parallel Plan:"""

    print("SEQUENTIAL VS PARALLEL DECOMPOSITION:")
    response = client.complete(sequential_prompt, temperature=0.3, max_tokens=400)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Identifies optimization opportunities")


def example_3_conditional_branching():
    """Steps with conditional logic and branches."""
    print("\n" + "=" * 60)
    print("Example 3: Conditional Branching in Steps")
    print("=" * 60)

    client = LLMClient("openai")

    branching_prompt = """Create a troubleshooting guide with conditional steps:

Problem: Website is not loading

Step 1: Check internet connection
  └─ If connected → Go to Step 2
  └─ If not connected → Restart router, then retry Step 1

Step 2: Try different browser
  └─ If works → Clear cache in original browser (END)
  └─ If doesn't work → Go to Step 3

Step 3: Check if website is down for everyone
  └─ If down for everyone → Wait for fix (END)
  └─ If only for you → Go to Step 4

Step 4: Check DNS settings
  └─ If incorrect → Fix DNS, restart browser (END)
  └─ If correct → Go to Step 5

Step 5: Check firewall/antivirus
  └─ If blocking → Add exception (END)
  └─ If not blocking → Contact support with all test results

Now create a similar conditional flow for:
Problem: Machine learning model accuracy is below threshold

Step 1:"""

    print("CONDITIONAL BRANCHING:")
    print("Decision trees in step format")

    response = client.complete(branching_prompt, temperature=0.3, max_tokens=500)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Handles complex decision flows")


def example_4_nested_substeps():
    """Hierarchical step organization with substeps."""
    print("\n" + "=" * 60)
    print("Example 4: Nested Substeps")
    print("=" * 60)

    client = LLMClient("openai")

    nested_prompt = """Break down this complex task hierarchically:

Task: Migrate a monolithic application to microservices

Phase 1: Assessment and Planning
  Step 1.1: Analyze current architecture
    └─ 1.1.1: Document existing components
    └─ 1.1.2: Identify dependencies
    └─ 1.1.3: Map data flows
  Step 1.2: Define service boundaries
    └─ 1.2.1: Apply domain-driven design
    └─ 1.2.2: Identify bounded contexts
    └─ 1.2.3: Plan API contracts
  Step 1.3: Create migration roadmap
    └─ 1.3.1: Prioritize services
    └─ 1.3.2: Set milestones
    └─ 1.3.3: Estimate timelines

Phase 2: Implementation
  Step 2.1: Set up infrastructure
    └─ 2.1.1: Container orchestration
    └─ 2.1.2: Service discovery
    └─ 2.1.3: API gateway

Continue with Phase 3: Testing and Deployment..."""

    print("NESTED SUBSTEPS:")
    print("Hierarchical task decomposition")

    response = client.complete(nested_prompt, temperature=0.3, max_tokens=500)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Manages complexity through hierarchy")


def example_5_iterative_refinement():
    """Steps that iterate and refine results."""
    print("\n" + "=" * 60)
    print("Example 5: Iterative Refinement Steps")
    print("=" * 60)

    client = LLMClient("openai")

    iterative_prompt = """Design an iterative process for optimizing a prompt:

Goal: Create the perfect prompt for sentiment analysis

Iteration 1:
  Step 1: Create basic prompt
    Prompt: "Is this positive or negative?"
  Step 2: Test with examples
    Result: 60% accuracy, too binary
  Step 3: Identify improvements
    Need: Handle neutral, mixed sentiments

Iteration 2:
  Step 1: Refine prompt
    Prompt: "Classify sentiment as positive, negative, neutral, or mixed"
  Step 2: Test with examples
    Result: 75% accuracy, better coverage
  Step 3: Identify improvements
    Need: Add examples for clarity

Iteration 3:
  Step 1: Add few-shot examples
    Prompt: [with examples]
  Step 2: Test with examples
    Result: 85% accuracy
  Step 3: Identify improvements
    Need: Handle edge cases

Continue iterations until 95% accuracy...

Now apply this iterative approach to:
Goal: Optimize a SQL query that's running slowly

Iteration 1:"""

    print("ITERATIVE REFINEMENT:")
    print("Continuous improvement through iterations")

    response = client.complete(iterative_prompt, temperature=0.3, max_tokens=500)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Systematic optimization process")


def example_6_dependency_mapping():
    """Explicitly mapping step dependencies."""
    print("\n" + "=" * 60)
    print("Example 6: Step Dependency Mapping")
    print("=" * 60)

    client = LLMClient("openai")

    dependency_prompt = """Map step dependencies for this project:

Project: Launch new product feature

Steps with Dependencies:
Step A: Market research
  Dependencies: None (can start immediately)

Step B: Design mockups
  Dependencies: A (needs market insights)

Step C: Technical specification
  Dependencies: A, B (needs requirements and design)

Step D: Backend development
  Dependencies: C (needs technical spec)

Step E: Frontend development
  Dependencies: C (needs technical spec)

Step F: API integration
  Dependencies: D, E (needs both backend and frontend)

Step G: Testing
  Dependencies: F (needs integrated system)

Step H: Documentation
  Dependencies: C (can start after spec)

Step I: Marketing materials
  Dependencies: B, H (needs design and docs)

Step J: Deployment
  Dependencies: G, I (needs tested system and materials)

Critical Path: A → B → C → D/E → F → G → J

Now create a dependency map for:
Project: Implement a recommendation system

Step A:"""

    print("DEPENDENCY MAPPING:")
    print("Explicit step relationships")

    response = client.complete(dependency_prompt, temperature=0.3, max_tokens=500)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Identifies bottlenecks and parallelization")


def example_7_step_validation():
    """Validating each step before proceeding."""
    print("\n" + "=" * 60)
    print("Example 7: Step Validation Checkpoints")
    print("=" * 60)

    client = LLMClient("openai")

    validation_prompt = """Create a data processing pipeline with validation at each step:

Pipeline: ETL for Financial Data

Step 1: EXTRACT
  Action: Pull data from multiple sources
  Validation:
    ✓ Check row counts match expected
    ✓ Verify all required fields present
    ✓ Confirm date ranges correct
  If validation fails: Log error, retry extraction

Step 2: TRANSFORM
  Action: Clean and standardize data
  Validation:
    ✓ No null values in required fields
    ✓ Data types are correct
    ✓ Values within expected ranges
    ✓ No duplicate records
  If validation fails: Quarantine bad records, continue with clean data

Step 3: ENRICH
  Action: Add calculated fields and lookups
  Validation:
    ✓ All lookups resolved
    ✓ Calculations produce valid results
    ✓ No division by zero errors
  If validation fails: Flag records for manual review

Step 4: LOAD
  Action: Insert into data warehouse
  Validation:
    ✓ Foreign key constraints satisfied
    ✓ Row counts match
    ✓ Checksums verify integrity
  If validation fails: Rollback transaction

Now create a similar validated pipeline for:
Process: Training a machine learning model

Step 1:"""

    print("STEP VALIDATION:")
    print("Quality gates at each step")

    response = client.complete(validation_prompt, temperature=0.3, max_tokens=500)
    print(f"\nResponse:\n{response.strip()}")

    print("\nBenefit: Ensures quality throughout process")


def run_all_examples():
    """Run all step-by-step examples."""
    examples = [
        example_1_granular_steps,
        example_2_parallel_vs_sequential,
        example_3_conditional_branching,
        example_4_nested_substeps,
        example_5_iterative_refinement,
        example_6_dependency_mapping,
        example_7_step_validation
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

    parser = argparse.ArgumentParser(description="Module 04: Step-by-Step")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_granular_steps,
            2: example_2_parallel_vs_sequential,
            3: example_3_conditional_branching,
            4: example_4_nested_substeps,
            5: example_5_iterative_refinement,
            6: example_6_dependency_mapping,
            7: example_7_step_validation
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 04: Step-by-Step Decomposition")
        print("\nUsage:")
        print("  python step_by_step.py --all        # Run all examples")
        print("  python step_by_step.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: Granularity Control")
        print("  2: Parallel vs Sequential")
        print("  3: Conditional Branching")
        print("  4: Nested Substeps")
        print("  5: Iterative Refinement")
        print("  6: Dependency Mapping")
        print("  7: Step Validation")