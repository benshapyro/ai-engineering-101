"""
Module 12: Prompt Optimization
Solutions

Complete implementations for all optimization exercises.
"""

import os
import time
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import tiktoken
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Solution 1: Token Counting and Analysis
print("=" * 50)
print("Solution 1: Token Counting and Analysis")
print("=" * 50)


def analyze_token_usage(prompt: str, quality_requirement: float = 0.8) -> Dict:
    """
    Analyze token usage and recommend optimal model.

    Args:
        prompt: The prompt to analyze
        quality_requirement: Minimum quality score (0-1)

    Returns:
        Dictionary with token counts, costs, and recommendation
    """
    models = {
        "gpt-3.5-turbo": {
            "encoder": "cl100k_base",
            "max_tokens": 4096,
            "input_cost": 0.0005,
            "output_cost": 0.0015,
            "quality": 0.7
        },
        "gpt-4": {
            "encoder": "cl100k_base",
            "max_tokens": 8192,
            "input_cost": 0.03,
            "output_cost": 0.06,
            "quality": 0.95
        },
        "gpt-4-turbo-preview": {
            "encoder": "cl100k_base",
            "max_tokens": 128000,
            "input_cost": 0.01,
            "output_cost": 0.03,
            "quality": 0.93
        }
    }

    results = {}
    recommendations = []

    for model_name, config in models.items():
        # Count tokens
        encoder = tiktoken.get_encoding(config["encoder"])
        input_tokens = len(encoder.encode(prompt))

        # Estimate output tokens (typically 1.5-2x input)
        estimated_output = int(input_tokens * 1.5)

        # Check if prompt fits
        if input_tokens > config["max_tokens"]:
            results[model_name] = {
                "fits": False,
                "reason": f"Exceeds max tokens ({input_tokens} > {config['max_tokens']})"
            }
            continue

        # Calculate costs
        input_cost = (input_tokens / 1000) * config["input_cost"]
        output_cost = (estimated_output / 1000) * config["output_cost"]
        total_cost = input_cost + output_cost

        results[model_name] = {
            "fits": True,
            "input_tokens": input_tokens,
            "estimated_output": estimated_output,
            "total_tokens": input_tokens + estimated_output,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "quality": config["quality"]
        }

        # Check if meets quality requirement
        if config["quality"] >= quality_requirement:
            recommendations.append({
                "model": model_name,
                "cost": total_cost,
                "quality": config["quality"],
                "value_score": config["quality"] / total_cost  # Quality per dollar
            })

    # Sort recommendations by value score
    recommendations.sort(key=lambda x: x["value_score"], reverse=True)

    return {
        "analysis": results,
        "recommendation": recommendations[0] if recommendations else None,
        "reason": f"Best value for quality requirement {quality_requirement}"
    }


# Test
test_prompt = """
Analyze the following customer feedback and provide:
1. Overall sentiment
2. Key issues mentioned
3. Suggested improvements

Feedback: The product quality is good but shipping was slow.
Customer service was helpful when I called.
"""

result = analyze_token_usage(test_prompt, quality_requirement=0.7)
print(f"Recommended model: {result['recommendation']['model']}")
print(f"Cost: ${result['recommendation']['cost']:.4f}")
print(f"Quality: {result['recommendation']['quality']}")


# Solution 2: Prompt Compression
print("\n" + "=" * 50)
print("Solution 2: Prompt Compression")
print("=" * 50)


def compress_prompt(original_prompt: str, target_reduction: float = 0.3) -> Tuple[str, Dict]:
    """
    Compress a prompt while maintaining effectiveness.

    Args:
        original_prompt: The prompt to compress
        target_reduction: Target reduction percentage (0-1)

    Returns:
        Tuple of (compressed_prompt, metrics)
    """
    # Redundant phrase mappings
    replacements = {
        "I would like you to please": "",
        "Please make sure to": "",
        "carefully analyze and evaluate": "analyze",
        "comprehensive summary": "summary",
        "all of the": "all",
        "main points, key ideas, and important details": "key points",
        "that are mentioned": "",
        "Please make sure to be": "Be",
        "thorough and complete": "thorough",
        "in your analysis": "",
        "piece of text": "text",
        "provide me with": "provide",
        "includes": "with",
    }

    compressed = original_prompt

    # Apply replacements
    for verbose, concise in replacements.items():
        compressed = compressed.replace(verbose, concise)

    # Remove double spaces
    while "  " in compressed:
        compressed = compressed.replace("  ", " ")

    # Remove unnecessary line breaks
    compressed = "\n".join(line.strip() for line in compressed.split("\n") if line.strip())

    # Calculate metrics
    encoder = tiktoken.get_encoding("cl100k_base")
    original_tokens = len(encoder.encode(original_prompt))
    compressed_tokens = len(encoder.encode(compressed))
    reduction = 1 - (compressed_tokens / original_tokens)

    metrics = {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "reduction_percentage": reduction * 100,
        "target_met": reduction >= target_reduction,
        "characters_saved": len(original_prompt) - len(compressed)
    }

    # If still not meeting target, be more aggressive
    if not metrics["target_met"]:
        # Additional compression strategies
        aggressive_replacements = {
            "following": "",
            "that includes": "with",
            "mentioned in the text": "",
        }

        for verbose, concise in aggressive_replacements.items():
            compressed = compressed.replace(verbose, concise)

        compressed_tokens = len(encoder.encode(compressed))
        reduction = 1 - (compressed_tokens / original_tokens)
        metrics["compressed_tokens"] = compressed_tokens
        metrics["reduction_percentage"] = reduction * 100
        metrics["target_met"] = reduction >= target_reduction

    return compressed.strip(), metrics


# Test
verbose_prompt = """
I would like you to please carefully analyze and evaluate the following
piece of text and provide me with a comprehensive summary that includes
all of the main points, key ideas, and important details that are mentioned
in the text. Please make sure to be thorough and complete in your analysis.

Text: Machine learning models require large amounts of data for training.
"""

compressed, metrics = compress_prompt(verbose_prompt)
print(f"Original: {len(verbose_prompt)} chars, {metrics['original_tokens']} tokens")
print(f"Compressed: {len(compressed)} chars, {metrics['compressed_tokens']} tokens")
print(f"Reduction: {metrics['reduction_percentage']:.1f}%")
print(f"\nCompressed prompt:\n{compressed}")


# Solution 3: A/B Testing Implementation
print("\n" + "=" * 50)
print("Solution 3: A/B Testing Implementation")
print("=" * 50)


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
        for test_input in test_inputs:
            # Simulate responses (in practice, would call API)
            response_a = self._simulate_response(prompt_a.format(input=test_input))
            response_b = self._simulate_response(prompt_b.format(input=test_input))

            # Calculate quality metrics
            metrics_a = self._calculate_metrics(response_a, test_input)
            metrics_b = self._calculate_metrics(response_b, test_input)

            self.results_a.append(metrics_a)
            self.results_b.append(metrics_b)

        # Analyze results
        analysis = self._analyze_results()

        return analysis

    def _simulate_response(self, prompt: str) -> str:
        """Simulate API response based on prompt."""
        # Simple simulation based on prompt characteristics
        if "brief" in prompt.lower() or "concise" in prompt.lower():
            return "Short focused response with key points."
        elif "detailed" in prompt.lower():
            return "Comprehensive response with extensive analysis and multiple perspectives considered."
        else:
            return "Standard response with moderate detail level."

    def _calculate_metrics(self, response: str, input_text: str) -> Dict:
        """Calculate quality metrics for response."""
        # Length score (brevity is good for summaries)
        length_score = min(100, 100 - len(response))

        # Keyword coverage (simplified)
        input_words = set(input_text.lower().split())
        response_words = set(response.lower().split())
        coverage = len(input_words & response_words) / max(len(input_words), 1)

        # Structure score (has punctuation, sentences)
        structure_score = (
            response.count('.') * 10 +
            response.count(',') * 5
        )

        return {
            "length_score": length_score,
            "coverage": coverage * 100,
            "structure_score": min(100, structure_score),
            "overall": (length_score + coverage * 100 + structure_score) / 3
        }

    def _analyze_results(self) -> Dict:
        """Analyze test results and determine winner."""
        # Calculate averages
        avg_a = np.mean([r["overall"] for r in self.results_a])
        avg_b = np.mean([r["overall"] for r in self.results_b])

        # Calculate standard deviations
        std_a = np.std([r["overall"] for r in self.results_a])
        std_b = np.std([r["overall"] for r in self.results_b])

        # Perform t-test if enough samples
        if len(self.results_a) >= 2 and len(self.results_b) >= 2:
            t_stat, p_value = stats.ttest_ind(
                [r["overall"] for r in self.results_a],
                [r["overall"] for r in self.results_b]
            )
        else:
            p_value = 1.0  # Not enough samples

        # Determine winner
        winner = None
        if p_value < 0.05:  # Statistically significant
            winner = "A" if avg_a > avg_b else "B"

        return {
            "variant_a": {
                "average": avg_a,
                "std_dev": std_a,
                "samples": len(self.results_a)
            },
            "variant_b": {
                "average": avg_b,
                "std_dev": std_b,
                "samples": len(self.results_b)
            },
            "p_value": p_value,
            "statistically_significant": p_value < 0.05,
            "winner": winner,
            "improvement": ((avg_b - avg_a) / avg_a * 100) if avg_a > 0 else 0
        }


# Test
control = "Summarize: {input}"
variant = "Provide a brief summary of: {input}"
test_data = [
    "Artificial intelligence is transforming industries.",
    "Climate change requires immediate action.",
    "Remote work has become more common."
]

tester = SimpleABTester()
results = tester.run_test(control, variant, test_data)
print(f"Variant A average: {results['variant_a']['average']:.2f}")
print(f"Variant B average: {results['variant_b']['average']:.2f}")
print(f"Winner: {results['winner'] or 'No significant difference'}")
print(f"Improvement: {results['improvement']:.1f}%")


# Solution 4: Cost-Aware Routing
print("\n" + "=" * 50)
print("Solution 4: Cost-Aware Routing")
print("=" * 50)


class CostRouter:
    """Route requests based on cost and complexity."""

    def __init__(self, daily_budget: float = 10.0):
        self.daily_budget = daily_budget
        self.spent_today = 0.0
        self.routing_log = []
        self.models = {
            "gpt-3.5-turbo": {"cost": 0.002, "quality": 0.7},
            "gpt-4-turbo-preview": {"cost": 0.01, "quality": 0.93},
            "gpt-4": {"cost": 0.06, "quality": 0.95}
        }

    def route_request(self, prompt: str, max_cost: Optional[float] = None) -> Dict:
        """
        Route request to optimal model.

        Args:
            prompt: The prompt to route
            max_cost: Maximum cost for this request

        Returns:
            Routing decision with model and estimated cost
        """
        complexity = self.get_complexity(prompt)

        # Map complexity to model
        complexity_model_map = {
            "simple": "gpt-3.5-turbo",
            "medium": "gpt-4-turbo-preview",
            "complex": "gpt-4"
        }

        selected_model = complexity_model_map[complexity]
        estimated_cost = self.models[selected_model]["cost"]

        # Check cost constraints
        if max_cost and estimated_cost > max_cost:
            # Downgrade model
            for model in ["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4"]:
                if self.models[model]["cost"] <= max_cost:
                    selected_model = model
                    estimated_cost = self.models[model]["cost"]
                    break

        # Check daily budget
        if self.spent_today + estimated_cost > self.daily_budget:
            return {
                "error": "Daily budget exceeded",
                "budget_remaining": self.daily_budget - self.spent_today
            }

        # Update tracking
        self.spent_today += estimated_cost
        decision = {
            "model": selected_model,
            "complexity": complexity,
            "estimated_cost": estimated_cost,
            "quality": self.models[selected_model]["quality"],
            "budget_remaining": self.daily_budget - self.spent_today
        }

        self.routing_log.append({
            "timestamp": datetime.now(),
            "prompt_preview": prompt[:50],
            **decision
        })

        return decision

    def get_complexity(self, prompt: str) -> str:
        """Determine prompt complexity."""
        # Simple heuristics
        prompt_lower = prompt.lower()
        word_count = len(prompt.split())

        # Check for complex indicators
        complex_indicators = ["analyze", "evaluate", "create", "design", "compare", "synthesize"]
        medium_indicators = ["explain", "describe", "summarize", "list", "outline"]
        simple_indicators = ["what", "when", "who", "define", "is"]

        # Count indicators
        complex_score = sum(1 for ind in complex_indicators if ind in prompt_lower)
        medium_score = sum(1 for ind in medium_indicators if ind in prompt_lower)
        simple_score = sum(1 for ind in simple_indicators if ind in prompt_lower)

        # Also consider length
        if word_count > 100 or complex_score > 0:
            return "complex"
        elif word_count > 30 or medium_score > 0:
            return "medium"
        else:
            return "simple"


# Test
router = CostRouter(daily_budget=5.0)
test_prompts = [
    "What is 2+2?",
    "Analyze the economic impact of AI on employment",
    "Create a detailed marketing strategy for a startup"
]

for prompt in test_prompts:
    decision = router.route_request(prompt, max_cost=0.05)
    print(f"\nPrompt: {prompt[:30]}...")
    if "error" not in decision:
        print(f"  Model: {decision['model']}")
        print(f"  Complexity: {decision['complexity']}")
        print(f"  Cost: ${decision['estimated_cost']:.3f}")
        print(f"  Budget remaining: ${decision['budget_remaining']:.2f}")
    else:
        print(f"  Error: {decision['error']}")


# Solution 5: Response Caching System
print("\n" + "=" * 50)
print("Solution 5: Response Caching System")
print("=" * 50)


class ResponseCache:
    """Cache system for prompt responses."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.cache = {}
        self.similarity_threshold = similarity_threshold
        self.metrics = {"hits": 0, "misses": 0}
        self.ttl = 3600  # 1 hour TTL

    def get(self, prompt: str) -> Optional[str]:
        """
        Retrieve cached response if available.

        Args:
            prompt: The prompt to look up

        Returns:
            Cached response or None
        """
        # Check exact match first
        cache_key = self._get_cache_key(prompt)
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if self._is_valid(entry):
                self.metrics["hits"] += 1
                return entry["response"]

        # Check similar prompts
        for key, entry in self.cache.items():
            if self._is_valid(entry):
                similarity = self.calculate_similarity(prompt, entry["prompt"])
                if similarity >= self.similarity_threshold:
                    self.metrics["hits"] += 1
                    return entry["response"]

        self.metrics["misses"] += 1
        return None

    def set(self, prompt: str, response: str):
        """
        Cache a response.

        Args:
            prompt: The prompt
            response: The response to cache
        """
        cache_key = self._get_cache_key(prompt)
        self.cache[cache_key] = {
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now()
        }

    def calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity between two prompts."""
        # Simple word overlap similarity
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        # Jaccard similarity
        return len(intersection) / len(union)

    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def _is_valid(self, entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        age = (datetime.now() - entry["timestamp"]).total_seconds()
        return age < self.ttl

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.metrics["hits"] + self.metrics["misses"]
        hit_rate = self.metrics["hits"] / total if total > 0 else 0

        return {
            "hit_rate": hit_rate,
            "hits": self.metrics["hits"],
            "misses": self.metrics["misses"],
            "cache_size": len(self.cache)
        }


# Test
cache = ResponseCache()
cache.set("What is machine learning?", "Machine learning is a subset of AI...")
cache.set("Explain ML", "ML is a type of artificial intelligence...")

test_prompts = [
    "What is machine learning?",  # Exact match
    "What's machine learning?",    # Similar
    "Explain deep learning"        # Different
]

for prompt in test_prompts:
    response = cache.get(prompt)
    print(f"Prompt: {prompt}")
    print(f"  Cached: {response is not None}")
    if response:
        print(f"  Response preview: {response[:30]}...")

print(f"\nCache stats: {cache.get_stats()}")


# Challenge Solution: Complete Optimization Pipeline
print("\n" + "=" * 50)
print("CHALLENGE SOLUTION: Complete Optimization Pipeline")
print("=" * 50)


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
        results = []
        batch_candidates = []

        for request in requests:
            self.metrics["total_requests"] += 1
            optimizations = []

            # 1. Check cache
            cached_response = self.cache.get(request["prompt"])
            if cached_response:
                self.metrics["cache_hits"] += 1
                optimizations.append("cache_hit")
                results.append({
                    "request": request,
                    "response": cached_response,
                    "optimizations": optimizations,
                    "cost": 0
                })
                continue

            # 2. Compress if beneficial
            if len(request["prompt"]) > 200:
                compressed, compress_metrics = compress_prompt(request["prompt"])
                if compress_metrics["reduction_percentage"] > 20:
                    self.metrics["compressed"] += 1
                    self.metrics["tokens_saved"] += (
                        compress_metrics["original_tokens"] -
                        compress_metrics["compressed_tokens"]
                    )
                    optimizations.append(f"compressed_{compress_metrics['reduction_percentage']:.0f}%")
                    prompt = compressed
                else:
                    prompt = request["prompt"]
            else:
                prompt = request["prompt"]

            # 3. Consider batching for low priority
            if request.get("priority") == "low" and len(prompt) < 200:
                batch_candidates.append({
                    "original_request": request,
                    "processed_prompt": prompt,
                    "optimizations": optimizations.copy()
                })
                continue

            # 4. Route to optimal model
            routing = self.router.route_request(prompt)
            if "error" in routing:
                results.append({
                    "request": request,
                    "error": routing["error"],
                    "optimizations": optimizations
                })
                continue

            optimizations.append(f"routed_to_{routing['model']}")

            # 5. Simulate processing and cache result
            response = f"Processed via {routing['model']}: {prompt[:50]}..."
            self.cache.set(request["prompt"], response)

            self.metrics["total_cost"] += routing["estimated_cost"]

            results.append({
                "request": request,
                "response": response,
                "optimizations": optimizations,
                "cost": routing["estimated_cost"],
                "model": routing["model"]
            })

        # Process batched requests
        if batch_candidates:
            batch_results = self._process_batch(batch_candidates)
            results.extend(batch_results)

        return results

    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process batched requests together."""
        if not batch:
            return []

        self.metrics["batched"] += len(batch)

        # Combine prompts
        combined = "Answer these questions:\n"
        for i, item in enumerate(batch):
            combined += f"{i+1}. {item['processed_prompt']}\n"

        # Route the combined prompt
        routing = self.router.route_request(combined)

        # Simulate batch processing
        batch_response = f"Batch processed via {routing['model']}"

        results = []
        for item in batch:
            # Cache individual responses
            response = f"[Batched] {batch_response}"
            self.cache.set(item["original_request"]["prompt"], response)

            item["optimizations"].append("batched")
            item["optimizations"].append(f"routed_to_{routing['model']}")

            results.append({
                "request": item["original_request"],
                "response": response,
                "optimizations": item["optimizations"],
                "cost": routing["estimated_cost"] / len(batch)  # Split cost
            })

        self.metrics["total_cost"] += routing["estimated_cost"]

        return results

    def generate_report(self) -> str:
        """Generate optimization report."""
        report = "\n" + "=" * 40
        report += "\nOptimization Pipeline Report"
        report += "\n" + "=" * 40

        if self.metrics["total_requests"] == 0:
            return report + "\nNo requests processed yet."

        cache_hit_rate = (self.metrics["cache_hits"] / self.metrics["total_requests"]) * 100
        compression_rate = (self.metrics["compressed"] / self.metrics["total_requests"]) * 100
        batch_rate = (self.metrics["batched"] / self.metrics["total_requests"]) * 100

        report += f"\n\nTotal requests: {self.metrics['total_requests']}"
        report += f"\n\nOptimizations Applied:"
        report += f"\n  Cache hits: {self.metrics['cache_hits']} ({cache_hit_rate:.1f}%)"
        report += f"\n  Compressed: {self.metrics['compressed']} ({compression_rate:.1f}%)"
        report += f"\n  Batched: {self.metrics['batched']} ({batch_rate:.1f}%)"

        report += f"\n\nCost Metrics:"
        report += f"\n  Total cost: ${self.metrics['total_cost']:.2f}"
        report += f"\n  Average per request: ${self.metrics['total_cost'] / self.metrics['total_requests']:.4f}"
        report += f"\n  Budget remaining: ${self.daily_budget - self.metrics['total_cost']:.2f}"

        report += f"\n\nEfficiency:"
        report += f"\n  Tokens saved: {self.metrics['tokens_saved']}"
        report += f"\n  Cache efficiency: {cache_hit_rate:.1f}%"

        # Estimated savings
        baseline_cost = self.metrics["total_requests"] * 0.06  # Assume GPT-4 for all
        actual_cost = self.metrics["total_cost"]
        savings = baseline_cost - actual_cost
        savings_percent = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

        report += f"\n\nEstimated Savings:"
        report += f"\n  Baseline cost (all GPT-4): ${baseline_cost:.2f}"
        report += f"\n  Actual cost: ${actual_cost:.2f}"
        report += f"\n  Total saved: ${savings:.2f} ({savings_percent:.1f}%)"

        return report


# Test implementation
pipeline = OptimizationPipeline()

test_requests = [
    {"prompt": "What is AI?", "priority": "high"},
    {"prompt": "Explain machine learning in detail with examples and use cases", "priority": "normal"},
    {"prompt": "What is AI?", "priority": "low"},  # Duplicate - should hit cache
    {"prompt": "List benefits of cloud computing", "priority": "low"},
    {"prompt": "Analyze market trends for technology sector and provide investment recommendations", "priority": "high"}
]

results = pipeline.process_requests(test_requests)
for i, result in enumerate(results):
    print(f"\nRequest {i+1}: {result['request']['prompt'][:30]}...")
    if "error" in result:
        print(f"  Error: {result['error']}")
    else:
        print(f"  Optimizations: {', '.join(result['optimizations'])}")
        print(f"  Cost: ${result.get('cost', 0):.4f}")

print(pipeline.generate_report())


print("\n" + "=" * 50)
print("All Solutions Implemented!")
print("=" * 50)