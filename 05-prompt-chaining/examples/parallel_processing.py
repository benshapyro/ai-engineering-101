"""
Module 05: Parallel Processing Patterns

Concurrent prompt execution, aggregation, and optimization strategies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
import time
import asyncio
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass
class ParallelTask:
    """Represents a task to be executed in parallel."""
    name: str
    prompt: str
    temperature: float = 0.3
    max_tokens: int = 200
    dependencies: List[str] = None


def example_1_simple_parallel():
    """Execute multiple independent prompts in parallel."""
    print("=" * 60)
    print("Example 1: Simple Parallel Execution")
    print("=" * 60)

    client = LLMClient("openai")

    # Document to analyze from multiple perspectives
    document = """
    The new product launch exceeded expectations with 10,000 units sold
    in the first week. Customer reviews are mostly positive (4.2/5 stars)
    but there are concerns about the price point ($299) being too high
    for the target market. Competitors are already planning similar products
    at lower prices. Manufacturing costs could be reduced by 15% with
    minor design changes.
    """

    print(f"Document to analyze:\n{document}\n")

    # Define parallel analysis tasks
    tasks = [
        ("Sentiment Analysis", f"Analyze the sentiment of this text:\n\n{document}\n\nSentiment:"),
        ("Key Metrics", f"Extract key business metrics from:\n\n{document}\n\nMetrics:"),
        ("Risk Assessment", f"Identify business risks in:\n\n{document}\n\nRisks:"),
        ("Opportunities", f"Identify opportunities mentioned in:\n\n{document}\n\nOpportunities:"),
        ("Action Items", f"Generate action items based on:\n\n{document}\n\nAction Items:")
    ]

    print("Executing 5 analyses in parallel...\n")
    start_time = time.time()

    # Execute in parallel using ThreadPoolExecutor
    results = {}

    def execute_task(task_info):
        name, prompt = task_info
        result = client.complete(prompt, temperature=0.3, max_tokens=150)
        return name, result.strip()

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        futures = {executor.submit(execute_task, task): task[0] for task in tasks}

        # Collect results as they complete
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
            print(f"✓ Completed: {name}")

    elapsed = time.time() - start_time
    print(f"\nAll tasks completed in {elapsed:.2f} seconds")

    # Display results
    print("\n" + "-" * 40)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"{result[:100]}...")


def example_2_map_reduce_pattern():
    """Map-reduce pattern for processing multiple items."""
    print("\n" + "=" * 60)
    print("Example 2: Map-Reduce Pattern")
    print("=" * 60)

    client = LLMClient("openai")

    # Multiple customer reviews to process
    reviews = [
        "This product is amazing! Best purchase ever.",
        "Terrible quality, broke after one day. Want refund.",
        "Good value for money, but shipping was slow.",
        "Exceeded expectations. Highly recommend!",
        "Not bad, but I've seen better. It's okay."
    ]

    print("Processing customer reviews with map-reduce pattern\n")

    # MAP PHASE: Process each review in parallel
    print("MAP PHASE: Analyzing each review...")

    def analyze_review(review):
        prompt = f"""Analyze this review:

        Review: {review}

        Analysis (sentiment, key_point, rating_guess):"""

        result = client.complete(prompt, temperature=0.2, max_tokens=100)
        return {"review": review, "analysis": result.strip()}

    # Execute map phase in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_review, review) for review in reviews]
        mapped_results = [future.result() for future in as_completed(futures)]

    print(f"✓ Processed {len(mapped_results)} reviews\n")

    # Display mapped results
    for i, result in enumerate(mapped_results[:3], 1):  # Show first 3
        print(f"Review {i}: {result['review']}")
        print(f"Analysis: {result['analysis'][:80]}...\n")

    # REDUCE PHASE: Aggregate all analyses
    print("REDUCE PHASE: Aggregating insights...")

    analyses_text = "\n".join([r["analysis"] for r in mapped_results])
    reduce_prompt = f"""Synthesize these review analyses into overall insights:

    Individual Analyses:
    {analyses_text}

    Overall Summary:
    - Average Sentiment:
    - Common Themes:
    - Recommended Actions:"""

    summary = client.complete(reduce_prompt, temperature=0.3, max_tokens=200)
    print(f"\nAggregated Insights:\n{summary.strip()}")


def example_3_parallel_with_dependencies():
    """Handle dependencies between parallel tasks."""
    print("\n" + "=" * 60)
    print("Example 3: Parallel Processing with Dependencies")
    print("=" * 60)

    client = LLMClient("openai")

    # Project description
    project = """
    Build a mobile app for food delivery that connects restaurants with customers.
    Must support real-time tracking, payment processing, and ratings.
    """

    print(f"Project: {project}\n")

    # Define tasks with dependencies
    # Phase 1: Can run in parallel (no dependencies)
    phase1_tasks = {
        "requirements": f"List technical requirements for: {project}",
        "competitors": f"Identify main competitors for: {project}",
        "target_users": f"Define target user segments for: {project}"
    }

    # Execute Phase 1 in parallel
    print("PHASE 1: Initial Analysis (parallel execution)")
    phase1_results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(client.complete, prompt, 0.3, 150): name
            for name, prompt in phase1_tasks.items()
        }

        for future in as_completed(futures):
            name = futures[future]
            phase1_results[name] = future.result().strip()
            print(f"✓ Completed: {name}")

    # Phase 2: Depends on Phase 1 results
    print("\nPHASE 2: Detailed Planning (depends on Phase 1)")

    phase2_tasks = {
        "architecture": f"""Design technical architecture based on:
        Requirements: {phase1_results['requirements'][:100]}...

        Architecture:""",

        "differentiation": f"""Create differentiation strategy based on:
        Competitors: {phase1_results['competitors'][:100]}...

        Strategy:""",

        "user_stories": f"""Write user stories for:
        Target Users: {phase1_results['target_users'][:100]}...

        User Stories:"""
    }

    phase2_results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(client.complete, prompt, 0.3, 150): name
            for name, prompt in phase2_tasks.items()
        }

        for future in as_completed(futures):
            name = futures[future]
            phase2_results[name] = future.result().strip()
            print(f"✓ Completed: {name}")

    # Phase 3: Final synthesis (depends on Phase 2)
    print("\nPHASE 3: Final Synthesis (depends on Phase 2)")

    synthesis_prompt = f"""Create project roadmap based on:

    Architecture: {phase2_results['architecture'][:100]}...
    Differentiation: {phase2_results['differentiation'][:100]}...
    User Stories: {phase2_results['user_stories'][:100]}...

    Project Roadmap:"""

    roadmap = client.complete(synthesis_prompt, temperature=0.3, max_tokens=200)
    print(f"\nFinal Roadmap:\n{roadmap.strip()}")


def example_4_competitive_parallel():
    """Run competing approaches in parallel and select best."""
    print("\n" + "=" * 60)
    print("Example 4: Competitive Parallel Approaches")
    print("=" * 60)

    client = LLMClient("openai")

    # Problem to solve
    problem = """
    Optimize database queries that are taking 30+ seconds to complete.
    The database has 10 million records across 15 tables with complex joins.
    """

    print(f"Problem: {problem}\n")

    # Define competing solution approaches
    approaches = {
        "indexing": {
            "prompt": f"""Solve through indexing strategy:
            Problem: {problem}

            Indexing Solution:""",
            "temperature": 0.2
        },
        "denormalization": {
            "prompt": f"""Solve through denormalization:
            Problem: {problem}

            Denormalization Solution:""",
            "temperature": 0.3
        },
        "caching": {
            "prompt": f"""Solve through caching strategy:
            Problem: {problem}

            Caching Solution:""",
            "temperature": 0.3
        },
        "query_rewrite": {
            "prompt": f"""Solve through query optimization:
            Problem: {problem}

            Query Optimization Solution:""",
            "temperature": 0.2
        }
    }

    print("Running 4 solution approaches in parallel...\n")

    # Execute all approaches in parallel
    solutions = {}

    def execute_approach(name, config):
        result = client.complete(config["prompt"], config["temperature"], 150)
        return name, result.strip()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(execute_approach, name, config): name
            for name, config in approaches.items()
        }

        for future in as_completed(futures):
            name, solution = future.result()
            solutions[name] = solution
            print(f"✓ Generated: {name} approach")

    # Evaluate all solutions
    print("\nEvaluating solutions...")

    evaluation_prompt = f"""Evaluate and rank these database optimization solutions:

    1. INDEXING: {solutions['indexing'][:100]}...
    2. DENORMALIZATION: {solutions['denormalization'][:100]}...
    3. CACHING: {solutions['caching'][:100]}...
    4. QUERY REWRITE: {solutions['query_rewrite'][:100]}...

    Evaluation (rank by effectiveness, feasibility, and impact):"""

    evaluation = client.complete(evaluation_prompt, temperature=0.2, max_tokens=200)
    print(f"\nEvaluation:\n{evaluation.strip()}")

    # Select best approach
    selection_prompt = f"""Based on the evaluation, which approach should be implemented first?

    {evaluation}

    Recommended Approach and Why:"""

    recommendation = client.complete(selection_prompt, temperature=0.2, max_tokens=100)
    print(f"\nRecommendation:\n{recommendation.strip()}")


def example_5_batch_processing():
    """Process batches of items in parallel."""
    print("\n" + "=" * 60)
    print("Example 5: Batch Processing")
    print("=" * 60)

    client = LLMClient("openai")

    # Large dataset to process
    items = [
        {"id": 1, "text": "Request for pricing information"},
        {"id": 2, "text": "Bug report: app crashes on startup"},
        {"id": 3, "text": "Feature request: dark mode"},
        {"id": 4, "text": "Complaint about customer service"},
        {"id": 5, "text": "Question about API integration"},
        {"id": 6, "text": "Positive feedback on new feature"},
        {"id": 7, "text": "Security vulnerability report"},
        {"id": 8, "text": "Request for partnership"},
    ]

    print(f"Processing {len(items)} items in batches\n")

    # Process in batches
    batch_size = 3
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    print(f"Created {len(batches)} batches of size {batch_size}\n")

    def process_batch(batch_num, batch):
        # Create a single prompt for the entire batch
        batch_text = "\n".join([f"{item['id']}. {item['text']}" for item in batch])

        prompt = f"""Classify these items by category and priority:

        Items:
        {batch_text}

        For each item, provide:
        Category: [SUPPORT/BUG/FEATURE/FEEDBACK/SECURITY/BUSINESS]
        Priority: [HIGH/MEDIUM/LOW]

        Classifications:"""

        result = client.complete(prompt, temperature=0.2, max_tokens=200)
        return batch_num, result.strip()

    # Process batches in parallel
    batch_results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(process_batch, i, batch): i
            for i, batch in enumerate(batches, 1)
        }

        for future in as_completed(futures):
            batch_num, result = future.result()
            batch_results[batch_num] = result
            print(f"✓ Processed batch {batch_num}")

    # Display results
    print("\nBatch Processing Results:")
    for batch_num in sorted(batch_results.keys()):
        print(f"\nBatch {batch_num}:")
        print(batch_results[batch_num][:150] + "...")


def example_6_parallel_aggregation():
    """Aggregate results from multiple parallel analyses."""
    print("\n" + "=" * 60)
    print("Example 6: Parallel Aggregation")
    print("=" * 60)

    client = LLMClient("openai")

    # Complex scenario to analyze
    scenario = """
    A SaaS company is experiencing rapid growth but facing challenges:
    - Customer churn increased from 5% to 12% in 6 months
    - Support tickets doubled but team size remained same
    - Feature requests backlog grew to 200+ items
    - Competitors launched similar features faster
    - Revenue grew 40% but costs grew 60%
    """

    print(f"Business Scenario:\n{scenario}\n")

    # Multiple parallel analyses
    analyses = {
        "financial": f"Analyze financial implications of: {scenario}",
        "operational": f"Analyze operational challenges in: {scenario}",
        "strategic": f"Analyze strategic position given: {scenario}",
        "customer": f"Analyze from customer perspective: {scenario}",
        "competitive": f"Analyze competitive threats in: {scenario}"
    }

    print("Running 5 parallel analyses...")
    analysis_results = {}

    # Execute analyses in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(client.complete, prompt, 0.3, 150): name
            for name, prompt in analyses.items()
        }

        for future in as_completed(futures):
            name = futures[future]
            analysis_results[name] = future.result().strip()
            print(f"✓ Completed: {name} analysis")

    # Parallel aggregation strategies
    print("\nRunning 3 aggregation strategies in parallel...")

    aggregations = {
        "executive_summary": f"""Create executive summary from these analyses:
        Financial: {analysis_results['financial'][:100]}...
        Operational: {analysis_results['operational'][:100]}...
        Strategic: {analysis_results['strategic'][:100]}...

        Executive Summary:""",

        "action_plan": f"""Create prioritized action plan from:
        Customer: {analysis_results['customer'][:100]}...
        Competitive: {analysis_results['competitive'][:100]}...
        Operational: {analysis_results['operational'][:100]}...

        Action Plan:""",

        "risk_mitigation": f"""Identify risks and mitigation from:
        Financial: {analysis_results['financial'][:100]}...
        Strategic: {analysis_results['strategic'][:100]}...

        Risk Mitigation Strategy:"""
    }

    final_results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(client.complete, prompt, 0.3, 200): name
            for name, prompt in aggregations.items()
        }

        for future in as_completed(futures):
            name = futures[future]
            final_results[name] = future.result().strip()
            print(f"✓ Generated: {name}")

    # Display final aggregated insights
    print("\n" + "=" * 40)
    print("AGGREGATED INSIGHTS:")
    print("=" * 40)

    for name, result in final_results.items():
        print(f"\n{name.upper().replace('_', ' ')}:")
        print(result[:200] + "...")


def example_7_pipeline_parallelization():
    """Optimize pipeline by parallelizing independent stages."""
    print("\n" + "=" * 60)
    print("Example 7: Pipeline Parallelization")
    print("=" * 60)

    client = LLMClient("openai")

    # Content to process through pipeline
    content = """
    Artificial intelligence is transforming healthcare through diagnostic imaging,
    drug discovery, and personalized treatment plans. However, concerns about
    data privacy, algorithmic bias, and regulatory compliance remain significant
    challenges that must be addressed for widespread adoption.
    """

    print(f"Content: {content}\n")

    # Pipeline with parallel and sequential stages
    print("STAGE 1: Parallel Initial Processing")

    stage1_tasks = {
        "entities": f"Extract named entities from: {content}",
        "sentiment": f"Analyze sentiment of: {content}",
        "summary": f"Summarize in one sentence: {content}",
        "keywords": f"Extract keywords from: {content}"
    }

    stage1_results = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(client.complete, prompt, 0.2, 100): name
            for name, prompt in stage1_tasks.items()
        }

        for future in as_completed(futures):
            name = futures[future]
            stage1_results[name] = future.result().strip()
            print(f"  ✓ {name}")

    print("\nSTAGE 2: Sequential Processing (depends on Stage 1)")

    # This stage must run after stage 1
    enrichment_prompt = f"""Enrich the analysis with context:

    Entities: {stage1_results['entities']}
    Keywords: {stage1_results['keywords']}

    Provide context and relationships:"""

    enrichment = client.complete(enrichment_prompt, temperature=0.3, max_tokens=150)
    print(f"  ✓ enrichment")

    print("\nSTAGE 3: Parallel Final Processing")

    stage3_tasks = {
        "implications": f"""Analyze implications based on:
        Summary: {stage1_results['summary']}
        Context: {enrichment.strip()[:100]}...

        Implications:""",

        "recommendations": f"""Generate recommendations from:
        Sentiment: {stage1_results['sentiment']}
        Entities: {stage1_results['entities']}

        Recommendations:""",

        "questions": f"""Generate follow-up questions about:
        Content: {content[:100]}...
        Keywords: {stage1_results['keywords']}

        Questions:"""
    }

    stage3_results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(client.complete, prompt, 0.3, 100): name
            for name, prompt in stage3_tasks.items()
        }

        for future in as_completed(futures):
            name = futures[future]
            stage3_results[name] = future.result().strip()
            print(f"  ✓ {name}")

    print("\nPIPELINE COMPLETE")
    print("-" * 40)
    print("Final outputs available:")
    for key in stage3_results:
        print(f"  - {key}: {stage3_results[key][:50]}...")


def run_all_examples():
    """Run all parallel processing examples."""
    examples = [
        example_1_simple_parallel,
        example_2_map_reduce_pattern,
        example_3_parallel_with_dependencies,
        example_4_competitive_parallel,
        example_5_batch_processing,
        example_6_parallel_aggregation,
        example_7_pipeline_parallelization
    ]

    for example in examples:
        try:
            example()
            print("\n" + "=" * 60 + "\n")
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 05: Parallel Processing")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_simple_parallel,
            2: example_2_map_reduce_pattern,
            3: example_3_parallel_with_dependencies,
            4: example_4_competitive_parallel,
            5: example_5_batch_processing,
            6: example_6_parallel_aggregation,
            7: example_7_pipeline_parallelization
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 05: Parallel Processing Patterns")
        print("\nUsage:")
        print("  python parallel_processing.py --all        # Run all examples")
        print("  python parallel_processing.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: Simple Parallel")
        print("  2: Map-Reduce Pattern")
        print("  3: Parallel with Dependencies")
        print("  4: Competitive Approaches")
        print("  5: Batch Processing")
        print("  6: Parallel Aggregation")
        print("  7: Pipeline Parallelization")