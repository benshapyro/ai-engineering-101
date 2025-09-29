"""
Module 12: Prompt Optimization
Performance Testing Examples

This file demonstrates techniques for testing and measuring prompt performance:
1. A/B testing framework for prompts
2. Automated prompt evaluation
3. Response quality metrics
4. Performance benchmarking
5. Statistical significance testing
6. Multi-model comparison
7. Production performance testing system

Each example includes comprehensive performance measurement and analysis.
"""

import os
import time
import json
import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
from scipy import stats
import pandas as pd
from dotenv import load_dotenv
import openai
import tiktoken

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class MetricType(Enum):
    """Types of metrics to evaluate."""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    LATENCY = "latency"
    TOKEN_COUNT = "token_count"
    COST = "cost"


@dataclass
class TestResult:
    """Result from a prompt test."""
    prompt_id: str
    response: str
    metrics: Dict[str, float]
    timestamp: datetime
    duration: float
    tokens_used: int
    cost: float
    error: Optional[str] = None


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    name: str
    variant_a: str  # Control prompt
    variant_b: str  # Test prompt
    sample_size: int
    confidence_level: float = 0.95
    metrics: List[MetricType] = field(default_factory=lambda: [MetricType.ACCURACY])


# Example 1: A/B Testing Framework for Prompts
print("=" * 50)
print("Example 1: A/B Testing Framework")
print("=" * 50)


class ABTestFramework:
    """Framework for A/B testing prompts."""

    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.model = model
        self.encoder = tiktoken.encoding_for_model(model)
        self.results_a: List[TestResult] = []
        self.results_b: List[TestResult] = []

    def run_variant(self, prompt: str, test_input: str) -> TestResult:
        """Run a single test with a prompt variant."""
        start_time = time.time()

        try:
            full_prompt = prompt.format(input=test_input)
            tokens = len(self.encoder.encode(full_prompt))

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.3
            )

            result_text = response.choices[0].message.content
            duration = time.time() - start_time

            # Calculate metrics
            metrics = self.calculate_metrics(result_text, test_input)

            # Calculate cost (approximate)
            total_tokens = response.usage.total_tokens
            cost = (total_tokens / 1000) * 0.01  # Approximate pricing

            return TestResult(
                prompt_id=prompt[:50],
                response=result_text,
                metrics=metrics,
                timestamp=datetime.now(),
                duration=duration,
                tokens_used=total_tokens,
                cost=cost
            )
        except Exception as e:
            return TestResult(
                prompt_id=prompt[:50],
                response="",
                metrics={},
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                tokens_used=0,
                cost=0,
                error=str(e)
            )

    def calculate_metrics(self, response: str, input_text: str) -> Dict[str, float]:
        """Calculate quality metrics for a response."""
        metrics = {}

        # Response length ratio
        metrics["length_ratio"] = len(response) / max(len(input_text), 1)

        # Contains key terms (simple relevance check)
        key_terms = input_text.lower().split()[:5]
        matches = sum(1 for term in key_terms if term in response.lower())
        metrics["relevance"] = matches / max(len(key_terms), 1)

        # Coherence (sentence count as proxy)
        sentences = response.count('.') + response.count('!') + response.count('?')
        metrics["coherence"] = min(sentences / 3, 1.0)  # Normalize

        return metrics

    def run_ab_test(self, config: ABTestConfig, test_inputs: List[str]) -> Dict:
        """Run a complete A/B test."""
        print(f"Running A/B test: {config.name}")
        print(f"Sample size: {config.sample_size}")

        # Run tests for each variant
        for i, test_input in enumerate(test_inputs[:config.sample_size]):
            # Randomly assign to variant
            if np.random.random() > 0.5:
                # Test A first
                result_a = self.run_variant(config.variant_a, test_input)
                result_b = self.run_variant(config.variant_b, test_input)
            else:
                # Test B first (to avoid order bias)
                result_b = self.run_variant(config.variant_b, test_input)
                result_a = self.run_variant(config.variant_a, test_input)

            self.results_a.append(result_a)
            self.results_b.append(result_b)

            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{config.sample_size} tests")

        # Analyze results
        return self.analyze_results(config)

    def analyze_results(self, config: ABTestConfig) -> Dict:
        """Analyze A/B test results with statistical significance."""
        analysis = {
            "test_name": config.name,
            "sample_size": len(self.results_a),
            "metrics": {}
        }

        for metric in config.metrics:
            metric_name = metric.value

            # Extract metric values
            values_a = [r.metrics.get(metric_name, 0) for r in self.results_a if not r.error]
            values_b = [r.metrics.get(metric_name, 0) for r in self.results_b if not r.error]

            if values_a and values_b:
                # Calculate statistics
                mean_a = np.mean(values_a)
                mean_b = np.mean(values_b)
                std_a = np.std(values_a)
                std_b = np.std(values_b)

                # Perform t-test
                t_stat, p_value = stats.ttest_ind(values_a, values_b)

                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt((std_a ** 2 + std_b ** 2) / 2)
                effect_size = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0

                analysis["metrics"][metric_name] = {
                    "variant_a_mean": mean_a,
                    "variant_b_mean": mean_b,
                    "difference": mean_b - mean_a,
                    "relative_change": ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0,
                    "p_value": p_value,
                    "significant": p_value < (1 - config.confidence_level),
                    "effect_size": effect_size,
                    "winner": "B" if mean_b > mean_a else "A"
                }

        # Performance metrics
        latency_a = [r.duration for r in self.results_a if not r.error]
        latency_b = [r.duration for r in self.results_b if not r.error]

        analysis["performance"] = {
            "variant_a_avg_latency": np.mean(latency_a) if latency_a else 0,
            "variant_b_avg_latency": np.mean(latency_b) if latency_b else 0,
            "variant_a_avg_tokens": np.mean([r.tokens_used for r in self.results_a]),
            "variant_b_avg_tokens": np.mean([r.tokens_used for r in self.results_b]),
            "variant_a_total_cost": sum(r.cost for r in self.results_a),
            "variant_b_total_cost": sum(r.cost for r in self.results_b)
        }

        return analysis


# Example usage
ab_tester = ABTestFramework()

# Define test configuration
test_config = ABTestConfig(
    name="Instruction Clarity Test",
    variant_a="Summarize the following text: {input}",
    variant_b="Provide a concise summary of the main points in the following text: {input}",
    sample_size=20,
    confidence_level=0.95,
    metrics=[MetricType.RELEVANCE, MetricType.COHERENCE]
)

# Sample test inputs
test_texts = [
    "Machine learning is a subset of artificial intelligence...",
    "Climate change refers to long-term shifts in temperatures...",
    "The stock market experienced significant volatility today..."
] * 7  # Repeat for sample size

# Run A/B test (commented to avoid API calls)
# results = ab_tester.run_ab_test(test_config, test_texts)
# print(json.dumps(results, indent=2, default=str))


# Example 2: Automated Prompt Evaluation
print("\n" + "=" * 50)
print("Example 2: Automated Prompt Evaluation")
print("=" * 50)


class AutomatedEvaluator:
    """Automated evaluation system for prompts."""

    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.model = model
        self.evaluation_criteria = {
            "accuracy": "How accurate and factually correct is the response?",
            "relevance": "How relevant is the response to the input?",
            "completeness": "How complete and comprehensive is the response?",
            "clarity": "How clear and easy to understand is the response?",
            "conciseness": "How concise is the response without losing important information?"
        }

    async def evaluate_response(self,
                               prompt: str,
                               response: str,
                               ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Evaluate a response using LLM-as-judge approach."""
        scores = {}

        for criterion, description in self.evaluation_criteria.items():
            eval_prompt = f"""
            Evaluate the following response on a scale of 0-100 for {criterion}.
            {description}

            Original prompt: {prompt}
            Response: {response}
            {f'Expected answer: {ground_truth}' if ground_truth else ''}

            Provide only a numerical score between 0 and 100.
            Score:
            """

            try:
                result = await self._get_evaluation_score(eval_prompt)
                scores[criterion] = result
            except Exception as e:
                scores[criterion] = -1  # Error indicator

        # Calculate composite score
        valid_scores = [s for s in scores.values() if s >= 0]
        scores["composite"] = np.mean(valid_scores) if valid_scores else 0

        return scores

    async def _get_evaluation_score(self, eval_prompt: str) -> float:
        """Get numerical evaluation score from LLM."""
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0,
            max_tokens=10
        )

        try:
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return min(max(score, 0), 100)  # Clamp between 0-100
        except:
            return 50  # Default middle score on parse error

    async def batch_evaluate(self, test_cases: List[Dict]) -> pd.DataFrame:
        """Evaluate multiple test cases and return results as DataFrame."""
        results = []

        for i, test_case in enumerate(test_cases):
            print(f"Evaluating test case {i+1}/{len(test_cases)}")

            scores = await self.evaluate_response(
                prompt=test_case["prompt"],
                response=test_case["response"],
                ground_truth=test_case.get("ground_truth")
            )

            results.append({
                "test_id": test_case.get("id", i),
                "prompt": test_case["prompt"][:50] + "...",
                **scores
            })

        return pd.DataFrame(results)


# Example 3: Response Quality Metrics
print("\n" + "=" * 50)
print("Example 3: Response Quality Metrics")
print("=" * 50)


class QualityMetrics:
    """Calculate various quality metrics for prompt responses."""

    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def calculate_readability(self, text: str) -> float:
        """Calculate Flesch Reading Ease score."""
        sentences = text.count('.') + text.count('!') + text.count('?')
        words = len(text.split())
        syllables = sum(self._count_syllables(word) for word in text.split())

        if sentences == 0 or words == 0:
            return 0

        # Flesch Reading Ease formula
        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        return max(0, min(100, score))

    def _count_syllables(self, word: str) -> int:
        """Simple syllable counter."""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count += 1
        return count

    def calculate_diversity(self, text: str) -> float:
        """Calculate lexical diversity (type-token ratio)."""
        words = text.lower().split()
        if not words:
            return 0
        unique_words = len(set(words))
        total_words = len(words)
        return unique_words / total_words

    def calculate_sentiment_consistency(self, sentences: List[str]) -> float:
        """Calculate consistency of sentiment across sentences."""
        if len(sentences) < 2:
            return 1.0

        # Simple sentiment scoring based on positive/negative words
        positive_words = {'good', 'great', 'excellent', 'best', 'amazing', 'wonderful'}
        negative_words = {'bad', 'poor', 'terrible', 'worst', 'awful', 'horrible'}

        sentiments = []
        for sentence in sentences:
            words = set(sentence.lower().split())
            pos_count = len(words & positive_words)
            neg_count = len(words & negative_words)

            if pos_count > neg_count:
                sentiments.append(1)
            elif neg_count > pos_count:
                sentiments.append(-1)
            else:
                sentiments.append(0)

        # Calculate consistency
        if all(s >= 0 for s in sentiments) or all(s <= 0 for s in sentiments):
            return 1.0
        else:
            changes = sum(1 for i in range(1, len(sentiments))
                         if sentiments[i] != sentiments[i-1])
            return 1 - (changes / (len(sentiments) - 1))

    def calculate_information_density(self, text: str) -> float:
        """Calculate information density (unique concepts per token)."""
        tokens = self.encoder.encode(text)

        # Extract potential concept words (nouns, verbs, adjectives)
        words = text.split()
        concept_words = [w for w in words if len(w) > 3 and w.isalpha()]
        unique_concepts = len(set(concept_words))

        if len(tokens) == 0:
            return 0

        return unique_concepts / len(tokens)

    def get_comprehensive_metrics(self, text: str) -> Dict[str, float]:
        """Get all quality metrics for a text."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        return {
            "readability": self.calculate_readability(text),
            "diversity": self.calculate_diversity(text),
            "sentiment_consistency": self.calculate_sentiment_consistency(sentences),
            "information_density": self.calculate_information_density(text),
            "avg_sentence_length": np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            "token_count": len(self.encoder.encode(text))
        }


# Example usage
metrics_calculator = QualityMetrics()

sample_response = """
Artificial intelligence has revolutionized many industries.
It enables machines to learn from data and make intelligent decisions.
The applications range from healthcare to finance to transportation.
"""

quality_scores = metrics_calculator.get_comprehensive_metrics(sample_response)
print("Quality Metrics:")
for metric, score in quality_scores.items():
    print(f"  {metric}: {score:.2f}")


# Example 4: Performance Benchmarking
print("\n" + "=" * 50)
print("Example 4: Performance Benchmarking")
print("=" * 50)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking system."""

    def __init__(self):
        self.models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
        self.benchmark_results = []

    async def benchmark_prompt(self,
                              prompt: str,
                              test_inputs: List[str],
                              iterations: int = 3) -> pd.DataFrame:
        """Benchmark a prompt across models and inputs."""

        for model in self.models:
            print(f"\nBenchmarking model: {model}")

            for test_input in test_inputs:
                for iteration in range(iterations):
                    result = await self._run_single_benchmark(
                        model, prompt, test_input, iteration
                    )
                    self.benchmark_results.append(result)

        # Create DataFrame for analysis
        df = pd.DataFrame(self.benchmark_results)

        # Calculate aggregated statistics
        summary = df.groupby(['model', 'input_id']).agg({
            'latency': ['mean', 'std', 'min', 'max'],
            'tokens': ['mean', 'sum'],
            'cost': ['mean', 'sum'],
            'quality_score': 'mean'
        }).round(3)

        return summary

    async def _run_single_benchmark(self,
                                   model: str,
                                   prompt: str,
                                   test_input: str,
                                   iteration: int) -> Dict:
        """Run a single benchmark test."""
        start_time = time.time()

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt.format(input=test_input)}],
                temperature=0.3
            )

            latency = time.time() - start_time
            tokens = response.usage.total_tokens

            # Calculate approximate cost
            cost_per_1k = {"gpt-3.5-turbo": 0.002, "gpt-4": 0.03, "gpt-4-turbo-preview": 0.01}
            cost = (tokens / 1000) * cost_per_1k.get(model, 0.01)

            # Simple quality score (length and structure)
            response_text = response.choices[0].message.content
            quality_score = min(100, len(response_text) / 10)

            return {
                "model": model,
                "input_id": hash(test_input) % 1000,
                "iteration": iteration,
                "latency": latency,
                "tokens": tokens,
                "cost": cost,
                "quality_score": quality_score,
                "timestamp": datetime.now()
            }
        except Exception as e:
            return {
                "model": model,
                "input_id": hash(test_input) % 1000,
                "iteration": iteration,
                "latency": time.time() - start_time,
                "tokens": 0,
                "cost": 0,
                "quality_score": 0,
                "error": str(e),
                "timestamp": datetime.now()
            }

    def plot_results(self, df: pd.DataFrame):
        """Generate performance comparison plots."""
        print("\nPerformance Comparison:")
        print("-" * 50)

        for model in self.models:
            model_data = [r for r in self.benchmark_results if r["model"] == model]
            if model_data:
                avg_latency = np.mean([r["latency"] for r in model_data])
                avg_tokens = np.mean([r["tokens"] for r in model_data])
                avg_cost = np.mean([r["cost"] for r in model_data])

                print(f"\n{model}:")
                print(f"  Avg Latency: {avg_latency:.2f}s")
                print(f"  Avg Tokens: {avg_tokens:.0f}")
                print(f"  Avg Cost: ${avg_cost:.4f}")


# Example 5: Statistical Significance Testing
print("\n" + "=" * 50)
print("Example 5: Statistical Significance Testing")
print("=" * 50)


class StatisticalTester:
    """Advanced statistical testing for prompt optimization."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def calculate_sample_size(self,
                            effect_size: float = 0.5,
                            power: float = 0.8) -> int:
        """Calculate required sample size for detecting effect."""
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - self.alpha / 2)
        z_beta = norm.ppf(power)

        n = ((z_alpha + z_beta) ** 2 * 2) / (effect_size ** 2)
        return int(np.ceil(n))

    def run_sequential_testing(self,
                             results_a: List[float],
                             results_b: List[float]) -> Dict:
        """Run sequential hypothesis testing for early stopping."""
        n = min(len(results_a), len(results_b))

        sequential_results = []
        for i in range(2, n):  # Need at least 2 samples
            subset_a = results_a[:i]
            subset_b = results_b[:i]

            # Run t-test
            t_stat, p_value = stats.ttest_ind(subset_a, subset_b)

            # Calculate running averages
            mean_a = np.mean(subset_a)
            mean_b = np.mean(subset_b)

            # Check for early stopping
            can_stop = p_value < self.alpha or p_value > (1 - self.alpha)

            sequential_results.append({
                "sample_size": i,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "p_value": p_value,
                "significant": p_value < self.alpha,
                "can_stop": can_stop
            })

            if can_stop and i >= 10:  # Minimum 10 samples
                break

        return {
            "final_result": sequential_results[-1] if sequential_results else None,
            "stopped_early": len(sequential_results) < n - 2,
            "samples_needed": len(sequential_results) + 2,
            "history": sequential_results
        }

    def bayesian_ab_test(self,
                        successes_a: int,
                        trials_a: int,
                        successes_b: int,
                        trials_b: int) -> Dict:
        """Perform Bayesian A/B testing."""
        from scipy.stats import beta

        # Calculate posterior distributions
        posterior_a = beta(successes_a + 1, trials_a - successes_a + 1)
        posterior_b = beta(successes_b + 1, trials_b - successes_b + 1)

        # Monte Carlo simulation for P(B > A)
        samples = 10000
        samples_a = posterior_a.rvs(samples)
        samples_b = posterior_b.rvs(samples)

        prob_b_better = np.mean(samples_b > samples_a)

        # Calculate expected improvement
        improvement = np.mean(samples_b - samples_a)

        # Credible intervals
        ci_a = posterior_a.interval(self.confidence_level)
        ci_b = posterior_b.interval(self.confidence_level)

        return {
            "prob_b_better": prob_b_better,
            "expected_improvement": improvement,
            "credible_interval_a": ci_a,
            "credible_interval_b": ci_b,
            "recommendation": "B" if prob_b_better > 0.95 else "A" if prob_b_better < 0.05 else "Continue testing"
        }


# Example usage
stat_tester = StatisticalTester()

# Calculate required sample size
sample_size = stat_tester.calculate_sample_size(effect_size=0.3, power=0.8)
print(f"Required sample size for effect size 0.3: {sample_size}")

# Simulate test results
np.random.seed(42)
results_control = np.random.normal(0.75, 0.1, 50)
results_variant = np.random.normal(0.78, 0.1, 50)  # Small improvement

# Run sequential testing
sequential_result = stat_tester.run_sequential_testing(results_control, results_variant)
print(f"\nSequential testing stopped at: {sequential_result['samples_needed']} samples")
print(f"Early stopping: {sequential_result['stopped_early']}")

# Bayesian A/B test
bayesian_result = stat_tester.bayesian_ab_test(
    successes_a=35, trials_a=50,
    successes_b=40, trials_b=50
)
print(f"\nBayesian A/B Test:")
print(f"  P(B > A): {bayesian_result['prob_b_better']:.2%}")
print(f"  Recommendation: {bayesian_result['recommendation']}")


# Example 6: Multi-Model Comparison
print("\n" + "=" * 50)
print("Example 6: Multi-Model Comparison")
print("=" * 50)


class MultiModelComparator:
    """Compare prompt performance across multiple models."""

    def __init__(self):
        self.models = {
            "gpt-3.5-turbo": {"cost_per_1k": 0.002, "speed": "fast", "quality": "good"},
            "gpt-4": {"cost_per_1k": 0.03, "speed": "slow", "quality": "excellent"},
            "gpt-4-turbo-preview": {"cost_per_1k": 0.01, "speed": "medium", "quality": "excellent"}
        }
        self.comparison_results = []

    async def compare_models(self,
                           prompt_template: str,
                           test_cases: List[Dict],
                           evaluation_criteria: List[str]) -> pd.DataFrame:
        """Compare models on various criteria."""

        for model_name, model_info in self.models.items():
            print(f"\nTesting model: {model_name}")

            model_scores = {
                "model": model_name,
                "cost_per_1k": model_info["cost_per_1k"],
                "speed_rating": model_info["speed"],
                "quality_rating": model_info["quality"]
            }

            # Run tests for each criterion
            for criterion in evaluation_criteria:
                scores = []

                for test_case in test_cases:
                    score = await self._evaluate_model_on_test(
                        model_name,
                        prompt_template,
                        test_case,
                        criterion
                    )
                    scores.append(score)

                model_scores[f"{criterion}_avg"] = np.mean(scores)
                model_scores[f"{criterion}_std"] = np.std(scores)

            # Calculate composite score
            criterion_scores = [model_scores[f"{c}_avg"] for c in evaluation_criteria]
            model_scores["composite_score"] = np.mean(criterion_scores)

            self.comparison_results.append(model_scores)

        return pd.DataFrame(self.comparison_results)

    async def _evaluate_model_on_test(self,
                                     model: str,
                                     prompt: str,
                                     test_case: Dict,
                                     criterion: str) -> float:
        """Evaluate a model on a specific test and criterion."""
        # Simulate evaluation (in practice, would call API)

        # Mock scoring based on model characteristics
        base_scores = {
            "gpt-3.5-turbo": {"accuracy": 0.8, "creativity": 0.7, "consistency": 0.85},
            "gpt-4": {"accuracy": 0.95, "creativity": 0.9, "consistency": 0.95},
            "gpt-4-turbo-preview": {"accuracy": 0.93, "creativity": 0.88, "consistency": 0.92}
        }

        base_score = base_scores.get(model, {}).get(criterion, 0.5)
        # Add some random variation
        return base_score + np.random.normal(0, 0.05)

    def recommend_model(self,
                       df: pd.DataFrame,
                       constraints: Dict) -> str:
        """Recommend best model based on constraints."""
        recommendations = []

        for _, row in df.iterrows():
            score = 0
            reasons = []

            # Check cost constraint
            if "max_cost_per_1k" in constraints:
                if row["cost_per_1k"] <= constraints["max_cost_per_1k"]:
                    score += 1
                    reasons.append(f"Within cost limit (${row['cost_per_1k']}/1k)")

            # Check quality requirement
            if "min_quality" in constraints:
                if row["composite_score"] >= constraints["min_quality"]:
                    score += 2  # Quality is more important
                    reasons.append(f"Meets quality threshold ({row['composite_score']:.2f})")

            # Check speed requirement
            if "speed_requirement" in constraints:
                speed_scores = {"fast": 3, "medium": 2, "slow": 1}
                if speed_scores.get(row["speed_rating"], 0) >= speed_scores.get(constraints["speed_requirement"], 0):
                    score += 1
                    reasons.append(f"Meets speed requirement ({row['speed_rating']})")

            recommendations.append({
                "model": row["model"],
                "score": score,
                "reasons": reasons,
                "composite_score": row["composite_score"]
            })

        # Sort by score, then by composite score
        recommendations.sort(key=lambda x: (x["score"], x["composite_score"]), reverse=True)

        best = recommendations[0]
        print(f"\nRecommendation: {best['model']}")
        print(f"Reasons: {', '.join(best['reasons'])}")

        return best["model"]


# Example 7: Production Performance Testing System
print("\n" + "=" * 50)
print("Example 7: Production Performance Testing System")
print("=" * 50)


class ProductionPerformanceTester:
    """Complete production-ready performance testing system."""

    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.test_queue = []
        self.results_cache = {}
        self.alert_thresholds = {
            "latency_p95": 2.0,  # seconds
            "error_rate": 0.05,   # 5%
            "cost_per_request": 0.10  # $0.10
        }

    async def continuous_monitoring(self,
                                  prompts: List[str],
                                  interval: int = 300):  # 5 minutes
        """Continuously monitor prompt performance in production."""

        while True:
            print(f"\n[{datetime.now()}] Running performance tests...")

            for prompt in prompts:
                metrics = await self.test_prompt_performance(prompt)

                # Check for alerts
                alerts = self.check_alerts(metrics)
                if alerts:
                    await self.send_alerts(alerts, prompt)

                # Store metrics
                await self.store_metrics(prompt, metrics)

            # Generate report
            report = self.generate_performance_report()
            print(report)

            # Wait for next interval
            await asyncio.sleep(interval)

    async def test_prompt_performance(self, prompt: str) -> Dict:
        """Test a single prompt's performance."""
        test_inputs = self.generate_test_inputs()

        latencies = []
        errors = []
        tokens_used = []
        costs = []

        for test_input in test_inputs:
            start_time = time.time()

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": prompt.format(input=test_input)}],
                    timeout=10
                )

                latency = time.time() - start_time
                latencies.append(latency)

                tokens = response.usage.total_tokens
                tokens_used.append(tokens)

                cost = (tokens / 1000) * 0.01
                costs.append(cost)

            except Exception as e:
                errors.append(str(e))
                latencies.append(10.0)  # Timeout

        # Calculate metrics
        metrics = {
            "timestamp": datetime.now(),
            "latency_p50": np.percentile(latencies, 50) if latencies else 0,
            "latency_p95": np.percentile(latencies, 95) if latencies else 0,
            "latency_p99": np.percentile(latencies, 99) if latencies else 0,
            "error_rate": len(errors) / len(test_inputs),
            "avg_tokens": np.mean(tokens_used) if tokens_used else 0,
            "avg_cost": np.mean(costs) if costs else 0,
            "total_tests": len(test_inputs)
        }

        return metrics

    def generate_test_inputs(self) -> List[str]:
        """Generate realistic test inputs."""
        return [
            "Explain quantum computing in simple terms",
            "What are the benefits of microservices architecture?",
            "How does machine learning differ from traditional programming?",
            "Describe the process of photosynthesis",
            "What are best practices for API design?"
        ]

    def check_alerts(self, metrics: Dict) -> List[str]:
        """Check if metrics exceed alert thresholds."""
        alerts = []

        if metrics["latency_p95"] > self.alert_thresholds["latency_p95"]:
            alerts.append(f"High latency: P95 = {metrics['latency_p95']:.2f}s")

        if metrics["error_rate"] > self.alert_thresholds["error_rate"]:
            alerts.append(f"High error rate: {metrics['error_rate']:.1%}")

        if metrics["avg_cost"] > self.alert_thresholds["cost_per_request"]:
            alerts.append(f"High cost: ${metrics['avg_cost']:.3f}/request")

        return alerts

    async def send_alerts(self, alerts: List[str], prompt: str):
        """Send alerts for performance issues."""
        print(f"\n⚠️  ALERTS for prompt: {prompt[:50]}...")
        for alert in alerts:
            print(f"   - {alert}")

        # In production, would send to monitoring system
        if self.redis_client:
            await self.redis_client.lpush("performance_alerts", json.dumps({
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt[:100],
                "alerts": alerts
            }))

    async def store_metrics(self, prompt: str, metrics: Dict):
        """Store metrics for historical analysis."""
        key = f"metrics:{hash(prompt)}"

        if key not in self.results_cache:
            self.results_cache[key] = []

        self.results_cache[key].append(metrics)

        # Keep only last 100 measurements
        if len(self.results_cache[key]) > 100:
            self.results_cache[key] = self.results_cache[key][-100:]

        # Store in Redis if available
        if self.redis_client:
            await self.redis_client.zadd(
                f"performance:{key}",
                {json.dumps(metrics, default=str): time.time()}
            )

    def generate_performance_report(self) -> str:
        """Generate performance summary report."""
        report = "\n" + "=" * 50
        report += "\nPerformance Summary Report"
        report += "\n" + "=" * 50

        for key, metrics_list in self.results_cache.items():
            if metrics_list:
                latest = metrics_list[-1]
                report += f"\n\nPrompt ID: {key}"
                report += f"\n  Latest P95 Latency: {latest['latency_p95']:.2f}s"
                report += f"\n  Error Rate: {latest['error_rate']:.1%}"
                report += f"\n  Avg Cost: ${latest['avg_cost']:.4f}"

                # Trend analysis
                if len(metrics_list) > 1:
                    latency_trend = latest['latency_p95'] - metrics_list[-2]['latency_p95']
                    trend_symbol = "↑" if latency_trend > 0 else "↓" if latency_trend < 0 else "→"
                    report += f"\n  Latency Trend: {trend_symbol} {abs(latency_trend):.2f}s"

        return report

    async def run_regression_test(self,
                                 new_prompt: str,
                                 baseline_prompt: str,
                                 test_suite: List[Dict]) -> Dict:
        """Run regression test comparing new vs baseline prompt."""

        print("\nRunning regression test...")

        baseline_results = []
        new_results = []

        for test_case in test_suite:
            # Test baseline
            baseline_metrics = await self.test_single_case(
                baseline_prompt,
                test_case["input"],
                test_case.get("expected_output")
            )
            baseline_results.append(baseline_metrics)

            # Test new prompt
            new_metrics = await self.test_single_case(
                new_prompt,
                test_case["input"],
                test_case.get("expected_output")
            )
            new_results.append(new_metrics)

        # Compare results
        regression_report = {
            "passed": True,
            "baseline_metrics": self.aggregate_metrics(baseline_results),
            "new_metrics": self.aggregate_metrics(new_results),
            "improvements": {},
            "regressions": {}
        }

        # Check for regressions
        for metric in ["accuracy", "latency", "cost"]:
            baseline_val = regression_report["baseline_metrics"].get(metric, 0)
            new_val = regression_report["new_metrics"].get(metric, 0)

            if metric == "latency" or metric == "cost":
                # Lower is better
                if new_val > baseline_val * 1.1:  # 10% regression threshold
                    regression_report["regressions"][metric] = {
                        "baseline": baseline_val,
                        "new": new_val,
                        "change": (new_val - baseline_val) / baseline_val
                    }
                    regression_report["passed"] = False
                elif new_val < baseline_val * 0.9:
                    regression_report["improvements"][metric] = {
                        "baseline": baseline_val,
                        "new": new_val,
                        "change": (new_val - baseline_val) / baseline_val
                    }
            else:
                # Higher is better
                if new_val < baseline_val * 0.9:  # 10% regression threshold
                    regression_report["regressions"][metric] = {
                        "baseline": baseline_val,
                        "new": new_val,
                        "change": (new_val - baseline_val) / baseline_val
                    }
                    regression_report["passed"] = False
                elif new_val > baseline_val * 1.1:
                    regression_report["improvements"][metric] = {
                        "baseline": baseline_val,
                        "new": new_val,
                        "change": (new_val - baseline_val) / baseline_val
                    }

        return regression_report

    async def test_single_case(self,
                              prompt: str,
                              test_input: str,
                              expected_output: Optional[str]) -> Dict:
        """Test a single test case."""
        start_time = time.time()

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt.format(input=test_input)}],
                temperature=0
            )

            result = response.choices[0].message.content
            latency = time.time() - start_time

            # Calculate accuracy if expected output provided
            accuracy = 1.0
            if expected_output:
                # Simple similarity check
                common_words = set(result.lower().split()) & set(expected_output.lower().split())
                accuracy = len(common_words) / max(len(expected_output.split()), 1)

            return {
                "accuracy": accuracy,
                "latency": latency,
                "cost": (response.usage.total_tokens / 1000) * 0.01,
                "success": True
            }
        except Exception as e:
            return {
                "accuracy": 0,
                "latency": time.time() - start_time,
                "cost": 0,
                "success": False,
                "error": str(e)
            }

    def aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Aggregate metrics from multiple test results."""
        successful = [r for r in results if r.get("success", False)]

        if not successful:
            return {"accuracy": 0, "latency": 0, "cost": 0, "success_rate": 0}

        return {
            "accuracy": np.mean([r["accuracy"] for r in successful]),
            "latency": np.mean([r["latency"] for r in successful]),
            "cost": np.mean([r["cost"] for r in successful]),
            "success_rate": len(successful) / len(results)
        }


# Create production tester instance
production_tester = ProductionPerformanceTester()

# Example regression test suite
test_suite = [
    {"input": "What is AI?", "expected_output": "Artificial Intelligence"},
    {"input": "Explain cloud computing", "expected_output": "distributed computing resources"},
    {"input": "Define machine learning", "expected_output": "algorithms that learn from data"}
]

# Example usage (commented to avoid API calls)
"""
# Run regression test
baseline = "Answer the question: {input}"
new_prompt = "Provide a concise answer to: {input}"

regression_results = asyncio.run(
    production_tester.run_regression_test(new_prompt, baseline, test_suite)
)

print(f"Regression test passed: {regression_results['passed']}")
if regression_results['improvements']:
    print("Improvements:", regression_results['improvements'])
if regression_results['regressions']:
    print("Regressions:", regression_results['regressions'])
"""

print("\n✅ All performance testing examples completed!")
print("\nKey Takeaways:")
print("1. A/B testing with statistical significance is crucial for prompt optimization")
print("2. Automated evaluation enables scalable testing")
print("3. Quality metrics should cover multiple dimensions")
print("4. Benchmarking across models helps optimize cost/quality tradeoffs")
print("5. Statistical testing prevents false positives in optimization")
print("6. Multi-model comparison reveals best fit for use cases")
print("7. Production monitoring ensures consistent performance")