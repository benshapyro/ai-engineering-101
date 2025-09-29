"""
Module 05: Prompt Chaining - Exercises

Practice exercises for mastering prompt chaining techniques.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor


# ===== Exercise 1: Build a Sequential Pipeline =====

def exercise_1_sequential_pipeline():
    """
    Exercise 1: Build a multi-step data processing pipeline.

    TODO:
    1. Create a 4-step pipeline for processing customer feedback
    2. Each step should use output from the previous step
    3. Implement error handling between steps
    4. Track state throughout the pipeline
    """
    client = LLMClient("openai")

    print("Exercise 1: Sequential Pipeline")
    print("=" * 50)

    # Sample customer feedback
    feedback = """
    The product quality is excellent but the shipping took forever!
    Customer service was unhelpful when I called. The packaging was
    damaged but the item inside was fine. Price is too high compared
    to competitors. Would maybe recommend to others.
    """

    print(f"Customer Feedback:\n{feedback}\n")

    # TODO: Implement pipeline steps
    # Step 1: Extract key points
    # Step 2: Categorize issues (product, service, shipping, pricing)
    # Step 3: Determine sentiment for each category
    # Step 4: Generate action items based on analysis

    pipeline_state = {
        "original": feedback,
        "step_results": [],
        "errors": []
    }

    print("TODO: Implement the following pipeline:")
    print("1. Extract → 2. Categorize → 3. Sentiment → 4. Actions")

    # TODO: Step 1 - Extract key points
    step1_prompt = """TODO: Create extraction prompt"""

    # TODO: Step 2 - Categorize issues
    # Use output from step 1

    # TODO: Step 3 - Sentiment analysis per category
    # Use output from step 2

    # TODO: Step 4 - Generate action items
    # Use outputs from all previous steps

    print("\nTODO: Complete pipeline implementation")


# ===== Exercise 2: Implement Conditional Routing =====

def exercise_2_conditional_routing():
    """
    Exercise 2: Create a routing system with conditional logic.

    TODO:
    1. Classify input into multiple categories
    2. Route to different processing paths based on classification
    3. Implement fallback handling for unknown categories
    4. Merge results from different paths
    """
    client = LLMClient("openai")

    print("Exercise 2: Conditional Routing")
    print("=" * 50)

    # Various types of requests
    requests = [
        "How do I reset my password?",
        "I want to cancel my subscription immediately",
        "Can you explain your pricing plans?",
        "Bug: The app crashes when I upload photos"
    ]

    print("Requests to route:")
    for req in requests:
        print(f"  - {req}")

    print("\nTODO: Implement routing logic:")

    for request in requests[:2]:  # Process first 2 for exercise
        print(f"\n" + "-" * 40)
        print(f"Processing: {request}")

        # TODO: Step 1 - Classify request type
        # Categories: TECHNICAL, BILLING, SALES, BUG_REPORT

        # TODO: Step 2 - Route based on classification
        # if TECHNICAL: provide solution
        # if BILLING: check account and provide options
        # if SALES: provide information
        # if BUG_REPORT: create ticket

        # TODO: Step 3 - Implement fallback for unclassified

        print("TODO: Implement classification and routing")


# ===== Exercise 3: Parallel Processing =====

def exercise_3_parallel_processing():
    """
    Exercise 3: Process multiple items in parallel.

    TODO:
    1. Split work into parallel tasks
    2. Execute tasks concurrently
    3. Aggregate results
    4. Handle failures in parallel execution
    """
    client = LLMClient("openai")

    print("Exercise 3: Parallel Processing")
    print("=" * 50)

    # Article to analyze from multiple angles
    article = """
    The rise of remote work has fundamentally changed how companies operate.
    Productivity has increased for many workers, but collaboration challenges
    persist. Mental health concerns have grown as work-life boundaries blur.
    Companies are saving on office costs but investing more in digital tools.
    """

    print(f"Article:\n{article}\n")

    # TODO: Define parallel analysis tasks
    parallel_tasks = {
        "summary": "TODO: Summarize the article",
        "pros_cons": "TODO: List pros and cons mentioned",
        "predictions": "TODO: Predict future trends based on this",
        "questions": "TODO: Generate discussion questions"
    }

    print("TODO: Execute these analyses in parallel:")
    for task_name in parallel_tasks:
        print(f"  - {task_name}")

    # TODO: Implement parallel execution using ThreadPoolExecutor
    def analyze_aspect(task_name, prompt):
        """TODO: Execute single analysis task"""
        pass

    # TODO: Aggregate results into final report

    print("\nTODO: Complete parallel processing implementation")


# ===== Exercise 4: Error Recovery Chain =====

def exercise_4_error_recovery():
    """
    Exercise 4: Build a chain with robust error recovery.

    TODO:
    1. Implement retry logic with exponential backoff
    2. Create fallback strategies for each step
    3. Log errors and recovery attempts
    4. Ensure chain completes even with failures
    """
    client = LLMClient("openai")

    print("Exercise 4: Error Recovery Chain")
    print("=" * 50)

    # Complex task that might fail at various steps
    task = """
    Translate this technical document to Spanish, then summarize it,
    then extract action items, then format as an email.
    """

    document = """
    The system architecture needs refactoring. Database queries are slow.
    We should implement caching and add monitoring. Deploy by Friday.
    """

    print(f"Task: {task}")
    print(f"Document: {document}\n")

    # TODO: Implement chain with error recovery

    class ResilientChain:
        def __init__(self, client):
            self.client = client
            self.max_retries = 3
            self.retry_delay = 1

        def execute_with_retry(self, prompt, step_name):
            """TODO: Implement retry logic"""
            pass

        def execute_with_fallback(self, primary_prompt, fallback_prompt, step_name):
            """TODO: Implement fallback logic"""
            pass

        def run_chain(self, steps):
            """TODO: Run chain with error recovery"""
            pass

    # TODO: Define chain steps with fallbacks
    chain_steps = [
        {
            "name": "translate",
            "primary": "TODO: Translation prompt",
            "fallback": "TODO: Simpler translation prompt"
        },
        # TODO: Add more steps
    ]

    print("TODO: Implement resilient chain execution")


# ===== Exercise 5: State Management =====

def exercise_5_state_management():
    """
    Exercise 5: Implement state management across chain execution.

    TODO:
    1. Track state throughout chain execution
    2. Allow steps to access previous results
    3. Implement state persistence
    4. Handle state rollback on failures
    """
    client = LLMClient("openai")

    print("Exercise 5: State Management")
    print("=" * 50)

    # Multi-step conversation requiring state
    conversation_flow = [
        "My name is Alice and I'm interested in your Pro plan",
        "What's included in that plan?",
        "Can I get a discount?",
        "Ok, I'll take it. How do I sign up?"
    ]

    print("Conversation flow requiring state management:\n")
    for msg in conversation_flow:
        print(f"  User: {msg}")

    # TODO: Implement state manager
    class ChainState:
        def __init__(self):
            self.context = {}
            self.history = []
            self.extracted_info = {}

        def update(self, key, value):
            """TODO: Update state with new information"""
            pass

        def get_context_prompt(self):
            """TODO: Generate context prompt from current state"""
            pass

        def rollback(self):
            """TODO: Rollback to previous state"""
            pass

    # TODO: Process conversation with state management
    state = ChainState()

    for message in conversation_flow:
        print(f"\nProcessing: {message}")

        # TODO: Update state with user info
        # TODO: Generate context-aware response
        # TODO: Extract and store key information

    print("\nTODO: Complete state management implementation")


# ===== Challenge: Adaptive Workflow Orchestrator =====

def challenge_adaptive_orchestrator():
    """
    Challenge: Build an adaptive workflow orchestrator.

    Requirements:
    1. Dynamically select chain strategy based on input
    2. Monitor performance and adapt execution
    3. Balance between speed and quality
    4. Implement caching for repeated operations
    5. Provide detailed execution analytics

    TODO: Complete the implementation
    """
    client = LLMClient("openai")

    print("Challenge: Adaptive Workflow Orchestrator")
    print("=" * 50)

    class AdaptiveOrchestrator:
        def __init__(self, client):
            self.client = client
            self.cache = {}
            self.metrics = {
                "total_calls": 0,
                "cache_hits": 0,
                "avg_latency": 0,
                "errors": 0
            }

        def analyze_task_complexity(self, task):
            """
            TODO: Analyze task to determine complexity level
            Returns: 'simple', 'moderate', 'complex'
            """
            pass

        def select_strategy(self, complexity, requirements):
            """
            TODO: Select execution strategy based on complexity
            Options: 'sequential', 'parallel', 'hybrid'
            """
            pass

        def execute_sequential(self, steps):
            """TODO: Execute steps sequentially"""
            pass

        def execute_parallel(self, tasks):
            """TODO: Execute tasks in parallel"""
            pass

        def execute_hybrid(self, workflow):
            """TODO: Execute hybrid workflow with parallel and sequential parts"""
            pass

        def cache_result(self, key, result):
            """TODO: Cache result with TTL"""
            pass

        def get_cached_result(self, key):
            """TODO: Retrieve cached result if valid"""
            pass

        def optimize_execution(self, workflow, constraints):
            """
            TODO: Optimize workflow execution based on constraints
            Constraints: time_limit, cost_limit, quality_target
            """
            pass

        def generate_analytics(self):
            """TODO: Generate execution analytics report"""
            pass

    # Test scenarios
    test_scenarios = [
        {
            "task": "Analyze customer feedback and generate report",
            "constraints": {"time_limit": 5, "quality_target": "high"}
        },
        {
            "task": "Process 100 documents for classification",
            "constraints": {"cost_limit": 1.0, "time_limit": 60}
        },
        {
            "task": "Real-time chat response with context",
            "constraints": {"time_limit": 2, "quality_target": "medium"}
        }
    ]

    print("Test Scenarios:")
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. Task: {scenario['task']}")
        print(f"   Constraints: {scenario['constraints']}")

    print("\nTODO: Implement adaptive orchestrator")

    orchestrator = AdaptiveOrchestrator(client)

    # TODO: Process each scenario with adaptive strategy
    # TODO: Monitor and report performance
    # TODO: Demonstrate caching benefits
    # TODO: Show execution analytics


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 05: Prompt Chaining Exercises")
    parser.add_argument("--exercise", type=int, help="Run specific exercise (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge")
    parser.add_argument("--all", action="store_true", help="Run all exercises")

    args = parser.parse_args()

    exercises = {
        1: exercise_1_sequential_pipeline,
        2: exercise_2_conditional_routing,
        3: exercise_3_parallel_processing,
        4: exercise_4_error_recovery,
        5: exercise_5_state_management
    }

    if args.all:
        for ex_num, ex_func in exercises.items():
            print(f"\n{'='*60}")
            ex_func()
            print(f"\n{'='*60}\n")
        challenge_adaptive_orchestrator()
    elif args.challenge:
        challenge_adaptive_orchestrator()
    elif args.exercise and args.exercise in exercises:
        exercises[args.exercise]()
    else:
        print("Module 05: Prompt Chaining - Exercises")
        print("\nUsage:")
        print("  python exercises.py --exercise N  # Run exercise N")
        print("  python exercises.py --challenge    # Run challenge")
        print("  python exercises.py --all          # Run all exercises")
        print("\nAvailable exercises:")
        print("  1: Sequential Pipeline")
        print("  2: Conditional Routing")
        print("  3: Parallel Processing")
        print("  4: Error Recovery")
        print("  5: State Management")
        print("  Challenge: Adaptive Workflow Orchestrator")