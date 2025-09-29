"""
Module 05: Sequential Prompt Chains

Linear workflow patterns for multi-step prompt processing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
import time
from typing import List, Dict, Any


def example_1_basic_sequential():
    """Basic sequential chain: output of one becomes input to next."""
    print("=" * 60)
    print("Example 1: Basic Sequential Chain")
    print("=" * 60)

    client = LLMClient("openai")

    # Input document
    document = """
    The quarterly sales report shows strong growth in the Asia-Pacific region,
    with revenue up 45% year-over-year to $12.3 million. Europe remained flat
    at $8.7 million, while North America declined 10% to $15.2 million.
    Customer satisfaction scores improved across all regions, with APAC leading
    at 94%, followed by Europe at 89% and North America at 85%.
    """

    print(f"Original Document:\n{document}\n")

    # Step 1: Extract key metrics
    extract_prompt = f"""Extract the key metrics from this document in a structured format:

    {document}

    Extracted Metrics:"""

    print("STEP 1: Extracting metrics...")
    extracted = client.complete(extract_prompt, temperature=0.2, max_tokens=200)
    print(f"Extracted:\n{extracted.strip()}\n")

    # Step 2: Analyze trends
    analyze_prompt = f"""Based on these metrics, analyze the trends and patterns:

    {extracted}

    Trend Analysis:"""

    print("STEP 2: Analyzing trends...")
    analysis = client.complete(analyze_prompt, temperature=0.3, max_tokens=200)
    print(f"Analysis:\n{analysis.strip()}\n")

    # Step 3: Generate recommendations
    recommend_prompt = f"""Based on this analysis, provide strategic recommendations:

    {analysis}

    Strategic Recommendations:"""

    print("STEP 3: Generating recommendations...")
    recommendations = client.complete(recommend_prompt, temperature=0.4, max_tokens=200)
    print(f"Recommendations:\n{recommendations.strip()}\n")

    print("Chain Complete: Document → Metrics → Analysis → Recommendations")


def example_2_data_transformation_pipeline():
    """Multi-step data transformation pipeline."""
    print("\n" + "=" * 60)
    print("Example 2: Data Transformation Pipeline")
    print("=" * 60)

    client = LLMClient("openai")

    # Raw data
    raw_data = """
    john_doe@email.com,35,new york,software engineer,85000
    jane.smith@company.org,28,san francisco,data scientist,95000
    bob-wilson@mail.net,42,chicago,product manager,110000
    """

    print(f"Raw Data:\n{raw_data}")

    # Step 1: Parse and structure
    parse_prompt = f"""Parse this CSV data and structure it as JSON:

    {raw_data}

    JSON Output:"""

    print("\nSTEP 1: Parsing to JSON...")
    parsed = client.complete(parse_prompt, temperature=0.1, max_tokens=300)
    print(f"Parsed:\n{parsed.strip()[:200]}...")

    # Step 2: Validate and clean
    validate_prompt = f"""Validate and clean this data (check emails, standardize locations):

    {parsed}

    Cleaned Data:"""

    print("\nSTEP 2: Validating and cleaning...")
    cleaned = client.complete(validate_prompt, temperature=0.1, max_tokens=300)
    print(f"Cleaned:\n{cleaned.strip()[:200]}...")

    # Step 3: Enrich with categories
    enrich_prompt = f"""Enrich this data by adding salary categories (low/medium/high) and location regions:

    {cleaned}

    Enriched Data:"""

    print("\nSTEP 3: Enriching data...")
    enriched = client.complete(enrich_prompt, temperature=0.2, max_tokens=400)
    print(f"Enriched:\n{enriched.strip()[:200]}...")

    # Step 4: Generate summary statistics
    summary_prompt = f"""Generate summary statistics from this data:

    {enriched}

    Summary Statistics:"""

    print("\nSTEP 4: Generating summary...")
    summary = client.complete(summary_prompt, temperature=0.2, max_tokens=200)
    print(f"Summary:\n{summary.strip()}")

    print("\nPipeline Complete: Raw CSV → JSON → Cleaned → Enriched → Summary")


def example_3_document_processing_chain():
    """Complex document processing with multiple stages."""
    print("\n" + "=" * 60)
    print("Example 3: Document Processing Chain")
    print("=" * 60)

    client = LLMClient("openai")

    # Technical document
    document = """
    Title: Implementing Microservices Architecture

    Our monolithic application is becoming difficult to maintain and scale.
    The codebase has grown to over 500,000 lines, deployment takes 2 hours,
    and a single bug can bring down the entire system. We need to modernize
    our architecture to improve reliability and developer productivity.

    The proposed solution is to gradually migrate to microservices, starting
    with the authentication service, followed by the payment processing and
    user management modules. This will require containerization, API gateway
    implementation, and a robust service mesh for communication.
    """

    print("Document Processing Chain:\n")

    # Step 1: Classification
    classify_prompt = f"""Classify this document:

    {document}

    Classification (type, domain, technical_level):"""

    print("STEP 1: Classification...")
    classification = client.complete(classify_prompt, temperature=0.2, max_tokens=50)
    print(f"→ {classification.strip()}\n")

    # Step 2: Key concept extraction
    concepts_prompt = f"""Extract key technical concepts and technologies mentioned:

    {document}

    Key Concepts:"""

    print("STEP 2: Concept extraction...")
    concepts = client.complete(concepts_prompt, temperature=0.2, max_tokens=150)
    print(f"→ {concepts.strip()}\n")

    # Step 3: Problem-solution mapping
    mapping_prompt = f"""Map problems to proposed solutions:

    {document}

    Problem-Solution Mapping:"""

    print("STEP 3: Problem-solution mapping...")
    mapping = client.complete(mapping_prompt, temperature=0.3, max_tokens=200)
    print(f"→ {mapping.strip()}\n")

    # Step 4: Risk assessment
    risk_prompt = f"""Based on this document and the proposed solutions:
    {mapping}

    Identify potential risks and challenges:"""

    print("STEP 4: Risk assessment...")
    risks = client.complete(risk_prompt, temperature=0.4, max_tokens=200)
    print(f"→ {risks.strip()}\n")

    # Step 5: Action items generation
    actions_prompt = f"""Based on the document, solutions, and risks:
    Document: {document[:200]}...
    Risks: {risks}

    Generate prioritized action items:"""

    print("STEP 5: Action items...")
    actions = client.complete(actions_prompt, temperature=0.3, max_tokens=200)
    print(f"→ {actions.strip()}")


def example_4_iterative_refinement_chain():
    """Iteratively refine output through multiple passes."""
    print("\n" + "=" * 60)
    print("Example 4: Iterative Refinement Chain")
    print("=" * 60)

    client = LLMClient("openai")

    # Initial draft
    initial_draft = "We should use AI to make things better and more efficient."

    print(f"Initial Draft: {initial_draft}\n")

    refinements = [
        ("Add specificity", "Make this statement more specific with concrete examples:"),
        ("Add metrics", "Add measurable outcomes and success metrics:"),
        ("Add timeline", "Add realistic timelines and milestones:"),
        ("Professional tone", "Rewrite in a professional, executive-ready tone:"),
    ]

    current = initial_draft

    for i, (improvement, instruction) in enumerate(refinements, 1):
        print(f"REFINEMENT {i}: {improvement}")

        prompt = f"""{instruction}

        Current version:
        {current}

        Improved version:"""

        current = client.complete(prompt, temperature=0.3, max_tokens=200).strip()
        print(f"Result: {current}\n")
        time.sleep(0.5)  # Rate limiting

    print(f"Final Version After {len(refinements)} Refinements:")
    print(current)


def example_5_validation_chain():
    """Chain with validation at each step."""
    print("\n" + "=" * 60)
    print("Example 5: Validation Chain")
    print("=" * 60)

    client = LLMClient("openai")

    # Input data to validate
    user_input = """
    Please process order #12345 for customer John Smith.
    Amount: $1,250.00
    Items: 3x Widget Pro, 2x Gadget Plus
    Shipping: Express to 123 Main St, Anytown, ST 12345
    Payment: Credit card ending in 4567
    """

    print(f"Input Order:\n{user_input}\n")

    # Step 1: Extract order details
    extract_prompt = f"""Extract order details as structured data:

    {user_input}

    Order Details:"""

    print("STEP 1: Extract → Validate...")
    extracted = client.complete(extract_prompt, temperature=0.1, max_tokens=200)

    # Validate extraction
    validate_extract_prompt = f"""Validate this extraction is complete (has order#, customer, amount, items):

    {extracted}

    Validation Result (PASS/FAIL with reason):"""

    validation1 = client.complete(validate_extract_prompt, temperature=0.1, max_tokens=50)
    print(f"Extraction: {validation1.strip()}\n")

    # Step 2: Check business rules
    rules_prompt = f"""Check business rules for this order:
    - Amount must be under $10,000
    - Express shipping only for amounts over $100
    - Credit card must be validated

    Order: {extracted}

    Business Rules Check:"""

    print("STEP 2: Business Rules → Validate...")
    rules_check = client.complete(rules_prompt, temperature=0.2, max_tokens=150)

    # Validate rules
    validate_rules_prompt = f"""Confirm all business rules passed:

    {rules_check}

    All Rules Passed? (YES/NO with details):"""

    validation2 = client.complete(validate_rules_prompt, temperature=0.1, max_tokens=50)
    print(f"Rules Check: {validation2.strip()}\n")

    # Step 3: Generate confirmation
    confirm_prompt = f"""Generate order confirmation if all validations passed:

    Validations:
    - Extraction: {validation1}
    - Business Rules: {validation2}

    Order: {extracted}

    Confirmation Message:"""

    print("STEP 3: Generate Confirmation...")
    confirmation = client.complete(confirm_prompt, temperature=0.3, max_tokens=200)
    print(f"Final Output:\n{confirmation.strip()}")


def example_6_accumulative_context_chain():
    """Build context progressively through the chain."""
    print("\n" + "=" * 60)
    print("Example 6: Accumulative Context Chain")
    print("=" * 60)

    client = LLMClient("openai")

    # Research topic
    topic = "quantum computing impact on cryptography"

    print(f"Research Topic: {topic}\n")

    # Chain that builds comprehensive understanding
    chain_steps = [
        ("Define", "Define the key terms"),
        ("Current State", "Describe the current state of the field"),
        ("Challenges", "Identify main challenges"),
        ("Opportunities", "Explore opportunities"),
        ("Timeline", "Project future timeline"),
        ("Implications", "Analyze broader implications")
    ]

    accumulated_context = f"Topic: {topic}\n\n"

    for i, (step_name, instruction) in enumerate(chain_steps, 1):
        print(f"STEP {i}: {step_name}")

        prompt = f"""Research Context:
        {accumulated_context}

        {instruction} related to {topic}:"""

        response = client.complete(prompt, temperature=0.3, max_tokens=150).strip()
        print(f"→ {response[:100]}...\n")

        # Add to accumulated context
        accumulated_context += f"{step_name}:\n{response}\n\n"

        time.sleep(0.5)  # Rate limiting

    # Final synthesis using all accumulated context
    print("FINAL STEP: Comprehensive Synthesis")
    synthesis_prompt = f"""Based on this comprehensive research:

    {accumulated_context}

    Provide an executive summary synthesizing all findings:"""

    synthesis = client.complete(synthesis_prompt, temperature=0.3, max_tokens=300)
    print(f"\nExecutive Summary:\n{synthesis.strip()}")


def example_7_branching_pipeline():
    """Pipeline that branches based on intermediate results."""
    print("\n" + "=" * 60)
    print("Example 7: Branching Pipeline")
    print("=" * 60)

    client = LLMClient("openai")

    # Customer query
    query = "My laptop won't turn on and I have an important presentation tomorrow!"

    print(f"Customer Query: {query}\n")

    # Step 1: Assess urgency
    urgency_prompt = f"""Assess the urgency level of this query:

    Query: {query}

    Urgency Level (CRITICAL/HIGH/MEDIUM/LOW):"""

    print("STEP 1: Assess Urgency...")
    urgency = client.complete(urgency_prompt, temperature=0.1, max_tokens=20).strip()
    print(f"Urgency: {urgency}\n")

    # Branch based on urgency
    if "CRITICAL" in urgency or "HIGH" in urgency:
        # High priority path
        print("→ Routing to HIGH PRIORITY path\n")

        # Step 2A: Quick solutions
        quick_prompt = f"""Provide immediate troubleshooting steps for:

        {query}

        Quick Solutions (numbered list):"""

        print("STEP 2A: Quick Solutions...")
        quick_solutions = client.complete(quick_prompt, temperature=0.2, max_tokens=200)
        print(f"{quick_solutions.strip()}\n")

        # Step 3A: Escalation options
        escalate_prompt = f"""Provide escalation and alternative options:

        Situation: {query}

        Escalation Options:"""

        print("STEP 3A: Escalation Options...")
        escalation = client.complete(escalate_prompt, temperature=0.3, max_tokens=150)
        print(f"{escalation.strip()}\n")

    else:
        # Standard priority path
        print("→ Routing to STANDARD path\n")

        # Step 2B: Diagnostic questions
        diagnostic_prompt = f"""Generate diagnostic questions for:

        {query}

        Diagnostic Questions:"""

        print("STEP 2B: Diagnostic Questions...")
        diagnostics = client.complete(diagnostic_prompt, temperature=0.3, max_tokens=150)
        print(f"{diagnostics.strip()}\n")

        # Step 3B: Schedule follow-up
        schedule_prompt = f"""Suggest follow-up scheduling:

        Issue: {query}

        Scheduling Suggestion:"""

        print("STEP 3B: Schedule Follow-up...")
        schedule = client.complete(schedule_prompt, temperature=0.3, max_tokens=100)
        print(f"{schedule.strip()}\n")

    print("Pipeline Complete: Different paths based on urgency assessment")


def run_all_examples():
    """Run all sequential chain examples."""
    examples = [
        example_1_basic_sequential,
        example_2_data_transformation_pipeline,
        example_3_document_processing_chain,
        example_4_iterative_refinement_chain,
        example_5_validation_chain,
        example_6_accumulative_context_chain,
        example_7_branching_pipeline
    ]

    for example in examples:
        try:
            example()
            print("\n" + "=" * 60 + "\n")
            time.sleep(1)  # Rate limiting between examples
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 05: Sequential Chains")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_basic_sequential,
            2: example_2_data_transformation_pipeline,
            3: example_3_document_processing_chain,
            4: example_4_iterative_refinement_chain,
            5: example_5_validation_chain,
            6: example_6_accumulative_context_chain,
            7: example_7_branching_pipeline
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 05: Sequential Prompt Chains")
        print("\nUsage:")
        print("  python sequential_chains.py --all        # Run all examples")
        print("  python sequential_chains.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: Basic Sequential")
        print("  2: Data Transformation Pipeline")
        print("  3: Document Processing Chain")
        print("  4: Iterative Refinement")
        print("  5: Validation Chain")
        print("  6: Accumulative Context")
        print("  7: Branching Pipeline")