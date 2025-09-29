"""
Module 12: Prompt Optimization
Exercises

Practice implementing various prompt optimization techniques.
"""

import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import tiktoken
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Exercise 1: Token Counting and Analysis
print("=" * 50)
print("Exercise 1: Token Counting and Analysis")
print("=" * 50)
print("""
Task: Create a function that analyzes token usage across different models
and recommends the most cost-effective model for a given prompt.

Requirements:
1. Count tokens for GPT-3.5 and GPT-4 models
2. Calculate estimated costs
3. Consider quality requirements
4. Return recommendation with justification
""")


def analyze_token_usage(prompt: str, quality_requirement: float = 0.8) -> Dict:
    """
    Analyze token usage and recommend optimal model.

    Args:
        prompt: The prompt to analyze
        quality_requirement: Minimum quality score (0-1)

    Returns:
        Dictionary with token counts, costs, and recommendation
    """
    # TODO: Implement token analysis
    # Hint: Use tiktoken for different models
    # Consider both input and estimated output tokens
    pass


# Test case
test_prompt = """
Analyze the following customer feedback and provide:
1. Overall sentiment
2. Key issues mentioned
3. Suggested improvements

Feedback: The product quality is good but shipping was slow.
Customer service was helpful when I called.
"""

# result = analyze_token_usage(test_prompt, quality_requirement=0.7)
# print(f"Recommendation: {result}")


# Exercise 2: Prompt Compression
print("\n" + "=" * 50)
print("Exercise 2: Prompt Compression")
print("=" * 50)
print("""
Task: Implement a prompt compression function that reduces token count
while maintaining meaning and effectiveness.

Requirements:
1. Remove redundant words and phrases
2. Simplify complex instructions
3. Maintain critical information
4. Achieve at least 30% reduction
""")


def compress_prompt(original_prompt: str, target_reduction: float = 0.3) -> Tuple[str, Dict]:
    """
    Compress a prompt while maintaining effectiveness.

    Args:
        original_prompt: The prompt to compress
        target_reduction: Target reduction percentage (0-1)

    Returns:
        Tuple of (compressed_prompt, metrics)
    """
    # TODO: Implement compression logic
    # Consider:
    # - Removing filler words
    # - Consolidating instructions
    # - Using abbreviations where clear
    # - Restructuring for efficiency
    pass


# Test case
verbose_prompt = """
I would like you to please carefully analyze and evaluate the following
piece of text and provide me with a comprehensive summary that includes
all of the main points, key ideas, and important details that are mentioned
in the text. Please make sure to be thorough and complete in your analysis.

Text: Machine learning models require large amounts of data for training.
"""

# compressed, metrics = compress_prompt(verbose_prompt)
# print(f"Compressed: {compressed}")
# print(f"Metrics: {metrics}")


# Exercise 3: A/B Testing Implementation
print("\n" + "=" * 50)
print("Exercise 3: A/B Testing Implementation")
print("=" * 50)
print("""
Task: Create a simple A/B testing framework for comparing two prompt variants.

Requirements:
1. Run tests on multiple inputs
2. Measure response quality (length, keywords, structure)
3. Calculate statistical significance
4. Provide clear winner recommendation
""")


class SimpleABTester:
    """Simple A/B testing framework for prompts."""

    def __init__(self):
        self.results_a = []
        self.results_b = []

    def run_test(self,
                 prompt_a: str,
                 prompt_b: str,
                 test_inputs: List[str]) -> Dict:
        """
        Run A/B test on two prompt variants.

        Args:
            prompt_a: Control prompt
            prompt_b: Test prompt
            test_inputs: List of test inputs

        Returns:
            Test results and winner
        """
        # TODO: Implement A/B testing
        # For each input:
        # 1. Run both prompts (can be simulated)
        # 2. Measure quality metrics
        # 3. Store results
        # 4. Calculate statistics
        # 5. Determine winner
        pass


# Test case
control = "Summarize: {input}"
variant = "Provide a brief summary of: {input}"
test_data = [
    "Artificial intelligence is transforming industries.",
    "Climate change requires immediate action.",
    "Remote work has become more common."
]

# tester = SimpleABTester()
# results = tester.run_test(control, variant, test_data)
# print(f"A/B Test Results: {results}")


# Exercise 4: Cost-Aware Routing
print("\n" + "=" * 50)
print("Exercise 4: Cost-Aware Routing")
print("=" * 50)
print("""
Task: Implement a routing system that directs requests to different models
based on complexity and cost constraints.

Requirements:
1. Classify prompt complexity (simple/medium/complex)
2. Route to appropriate model
3. Stay within budget constraints
4. Track routing decisions and costs
""")


class CostRouter:
    """Route requests based on cost and complexity."""

    def __init__(self, daily_budget: float = 10.0):
        self.daily_budget = daily_budget
        self.spent_today = 0.0
        self.routing_log = []

    def route_request(self, prompt: str, max_cost: Optional[float] = None) -> Dict:
        """
        Route request to optimal model.

        Args:
            prompt: The prompt to route
            max_cost: Maximum cost for this request

        Returns:
            Routing decision with model and estimated cost
        """
        # TODO: Implement routing logic
        # 1. Analyze prompt complexity
        # 2. Check budget constraints
        # 3. Select appropriate model
        # 4. Update spending tracker
        # 5. Log decision
        pass

    def get_complexity(self, prompt: str) -> str:
        """Determine prompt complexity."""
        # TODO: Implement complexity analysis
        # Consider:
        # - Length
        # - Keywords (analyze, create, list, etc.)
        # - Number of tasks
        pass


# Test case
router = CostRouter(daily_budget=5.0)
test_prompts = [
    "What is 2+2?",
    "Analyze the economic impact of AI on employment",
    "Create a detailed marketing strategy for a startup"
]

# for prompt in test_prompts:
#     decision = router.route_request(prompt, max_cost=0.05)
#     print(f"Prompt: {prompt[:30]}...")
#     print(f"Decision: {decision}")


# Exercise 5: Response Caching System
print("\n" + "=" * 50)
print("Exercise 5: Response Caching System")
print("=" * 50)
print("""
Task: Build a caching system that identifies and reuses similar prompts.

Requirements:
1. Generate cache keys from prompts
2. Implement similarity matching
3. Handle cache expiration
4. Track cache performance metrics
""")


class ResponseCache:
    """Cache system for prompt responses."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.cache = {}
        self.similarity_threshold = similarity_threshold
        self.metrics = {"hits": 0, "misses": 0}

    def get(self, prompt: str) -> Optional[str]:
        """
        Retrieve cached response if available.

        Args:
            prompt: The prompt to look up

        Returns:
            Cached response or None
        """
        # TODO: Implement cache retrieval
        # 1. Check exact match
        # 2. Check similar prompts
        # 3. Update metrics
        pass

    def set(self, prompt: str, response: str):
        """
        Cache a response.

        Args:
            prompt: The prompt
            response: The response to cache
        """
        # TODO: Implement cache storage
        # Include timestamp for expiration
        pass

    def calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity between two prompts."""
        # TODO: Implement similarity calculation
        # Can use simple word overlap or more sophisticated methods
        pass


# Test case
cache = ResponseCache()
cache.set("What is machine learning?", "Machine learning is...")
cache.set("Explain ML", "ML is a subset of AI...")

# test_prompts = [
#     "What is machine learning?",  # Exact match
#     "What's machine learning?",    # Similar
#     "Explain deep learning"        # Different
# ]

# for prompt in test_prompts:
#     response = cache.get(prompt)
#     print(f"Prompt: {prompt}")
#     print(f"Cached: {response is not None}")


# Challenge Exercise: Complete Optimization Pipeline
print("\n" + "=" * 50)
print("CHALLENGE: Complete Optimization Pipeline")
print("=" * 50)
print("""
Task: Create an optimization pipeline that combines multiple techniques
to minimize costs while maintaining quality.

Requirements:
1. Implement token analysis and compression
2. Use caching for repeated queries
3. Route to appropriate models
4. Batch similar requests
5. Track all metrics and generate report

The pipeline should:
- Accept a list of prompts
- Apply all optimizations
- Process efficiently
- Return results and optimization report
""")


class OptimizationPipeline:
    """Complete optimization pipeline for prompt processing."""

    def __init__(self, daily_budget: float = 20.0):
        self.daily_budget = daily_budget
        self.cache = ResponseCache()
        self.router = CostRouter(daily_budget)
        self.batch_queue = []
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "compressed": 0,
            "batched": 0,
            "total_cost": 0,
            "tokens_saved": 0
        }

    def process_requests(self, requests: List[Dict]) -> List[Dict]:
        """
        Process multiple requests with full optimization.

        Args:
            requests: List of request dictionaries with 'prompt' and 'priority'

        Returns:
            List of results with responses and optimization details
        """
        # TODO: Implement complete pipeline
        # For each request:
        # 1. Check cache
        # 2. Compress if beneficial
        # 3. Consider batching (low priority)
        # 4. Route to optimal model
        # 5. Cache results
        # 6. Track all metrics
        pass

    def generate_report(self) -> str:
        """Generate optimization report."""
        # TODO: Create comprehensive report
        # Include:
        # - Total requests processed
        # - Cache hit rate
        # - Average compression
        # - Cost savings
        # - Model distribution
        pass


# Test implementation
pipeline = OptimizationPipeline()

test_requests = [
    {"prompt": "What is AI?", "priority": "high"},
    {"prompt": "Explain machine learning in detail", "priority": "normal"},
    {"prompt": "What is AI?", "priority": "low"},  # Duplicate
    {"prompt": "List benefits of cloud computing", "priority": "low"},
    {"prompt": "Analyze market trends for technology sector", "priority": "high"}
]

# results = pipeline.process_requests(test_requests)
# for i, result in enumerate(results):
#     print(f"\nRequest {i+1}:")
#     print(f"  Optimizations: {result.get('optimizations', [])}")
#     print(f"  Cost: ${result.get('cost', 0):.4f}")

# print("\n" + pipeline.generate_report())


print("\n" + "=" * 50)
print("Exercises Complete!")
print("=" * 50)
print("""
These exercises cover key optimization techniques:
1. Token analysis for cost estimation
2. Prompt compression while maintaining quality
3. A/B testing for prompt improvement
4. Cost-aware routing strategies
5. Caching for efficiency
6. Complete optimization pipeline

Try implementing each exercise and experiment with different approaches!
""")