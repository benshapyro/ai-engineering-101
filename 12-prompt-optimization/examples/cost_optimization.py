"""
Module 12: Prompt Optimization
Cost Optimization Examples

This file demonstrates techniques for optimizing prompt costs:
1. Model selection strategies
2. Caching and memoization
3. Batch processing optimization
4. Token-aware routing
5. Response compression
6. Tiered processing strategies
7. Complete cost optimization system

Each example includes cost calculations and optimization strategies.
"""

import os
import time
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from functools import lru_cache
import pickle
import redis
import numpy as np
from dotenv import load_dotenv
import openai
import tiktoken

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class ModelConfig:
    """Configuration for different models."""
    name: str
    input_cost_per_1k: float  # USD per 1000 tokens
    output_cost_per_1k: float
    max_tokens: int
    quality_score: float  # 0-1 scale
    speed: str  # fast, medium, slow


# Model configurations with pricing
MODELS = {
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        max_tokens=4096,
        quality_score=0.7,
        speed="fast"
    ),
    "gpt-3.5-turbo-16k": ModelConfig(
        name="gpt-3.5-turbo-16k",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.004,
        max_tokens=16384,
        quality_score=0.7,
        speed="fast"
    ),
    "gpt-4": ModelConfig(
        name="gpt-4",
        input_cost_per_1k=0.03,
        output_cost_per_1k=0.06,
        max_tokens=8192,
        quality_score=0.95,
        speed="slow"
    ),
    "gpt-4-turbo-preview": ModelConfig(
        name="gpt-4-turbo-preview",
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.03,
        max_tokens=128000,
        quality_score=0.93,
        speed="medium"
    )
}


# Example 1: Model Selection Strategies
print("=" * 50)
print("Example 1: Model Selection Strategies")
print("=" * 50)


class ModelSelector:
    """Intelligent model selection based on task requirements."""

    def __init__(self, budget_per_request: float = 0.10):
        self.budget_per_request = budget_per_request
        self.models = MODELS
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def select_model(self,
                    prompt: str,
                    quality_requirement: float = 0.8,
                    max_latency: Optional[float] = None) -> ModelConfig:
        """Select optimal model based on requirements."""

        # Estimate token count
        token_count = len(self.encoder.encode(prompt))
        estimated_output_tokens = token_count * 2  # Rough estimate

        candidates = []

        for model_name, config in self.models.items():
            # Check if model can handle token count
            if token_count > config.max_tokens:
                continue

            # Check quality requirement
            if config.quality_score < quality_requirement:
                continue

            # Check latency requirement
            if max_latency:
                if config.speed == "slow" and max_latency < 5:
                    continue
                elif config.speed == "medium" and max_latency < 2:
                    continue

            # Calculate estimated cost
            input_cost = (token_count / 1000) * config.input_cost_per_1k
            output_cost = (estimated_output_tokens / 1000) * config.output_cost_per_1k
            total_cost = input_cost + output_cost

            # Check budget
            if total_cost > self.budget_per_request:
                continue

            candidates.append({
                "model": config,
                "estimated_cost": total_cost,
                "quality": config.quality_score,
                "tokens_fit": token_count <= config.max_tokens
            })

        if not candidates:
            # Fallback to cheapest model that fits
            return self.models["gpt-3.5-turbo"]

        # Sort by cost-effectiveness (quality per dollar)
        candidates.sort(key=lambda x: x["quality"] / x["estimated_cost"], reverse=True)

        selected = candidates[0]
        print(f"Selected model: {selected['model'].name}")
        print(f"  Estimated cost: ${selected['estimated_cost']:.4f}")
        print(f"  Quality score: {selected['quality']}")

        return selected["model"]

    def route_by_complexity(self, prompt: str) -> ModelConfig:
        """Route to different models based on task complexity."""

        # Analyze prompt complexity
        complexity_indicators = {
            "simple": ["what is", "define", "list", "name"],
            "medium": ["explain", "describe", "compare", "summarize"],
            "complex": ["analyze", "evaluate", "design", "create", "solve"]
        }

        prompt_lower = prompt.lower()
        complexity = "simple"

        for level, indicators in complexity_indicators.items():
            if any(ind in prompt_lower for ind in indicators):
                complexity = level
                break

        # Map complexity to model
        model_mapping = {
            "simple": "gpt-3.5-turbo",
            "medium": "gpt-4-turbo-preview",
            "complex": "gpt-4"
        }

        selected_model = self.models[model_mapping[complexity]]
        print(f"Complexity: {complexity} → Model: {selected_model.name}")

        return selected_model


# Example usage
selector = ModelSelector(budget_per_request=0.05)

test_prompt = "Analyze the economic implications of artificial intelligence on labor markets"
selected = selector.select_model(test_prompt, quality_requirement=0.9)
routed = selector.route_by_complexity(test_prompt)


# Example 2: Caching and Memoization
print("\n" + "=" * 50)
print("Example 2: Caching and Memoization")
print("=" * 50)


class CostOptimizedCache:
    """Advanced caching system for cost optimization."""

    def __init__(self, redis_client=None, ttl: int = 3600):
        self.redis_client = redis_client
        self.ttl = ttl  # Time to live in seconds
        self.local_cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "cost_saved": 0.0
        }

    def _generate_cache_key(self, prompt: str, model: str, temperature: float) -> str:
        """Generate deterministic cache key."""
        content = f"{prompt}:{model}:{temperature}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def get_cached_response(self,
                                 prompt: str,
                                 model: str,
                                 temperature: float = 0) -> Optional[Dict]:
        """Retrieve cached response if available."""

        cache_key = self._generate_cache_key(prompt, model, temperature)

        # Check local cache first
        if cache_key in self.local_cache:
            entry = self.local_cache[cache_key]
            if datetime.now() - entry["timestamp"] < timedelta(seconds=self.ttl):
                self.cache_stats["hits"] += 1
                self._calculate_cost_saved(model, prompt)
                print(f"Cache hit! Saved ${self.cache_stats['cost_saved']:.4f} total")
                return entry["response"]

        # Check Redis if available
        if self.redis_client:
            cached = await self.redis_client.get(cache_key)
            if cached:
                entry = json.loads(cached)
                self.local_cache[cache_key] = entry
                self.cache_stats["hits"] += 1
                self._calculate_cost_saved(model, prompt)
                return entry["response"]

        self.cache_stats["misses"] += 1
        return None

    async def cache_response(self,
                           prompt: str,
                           model: str,
                           temperature: float,
                           response: Dict):
        """Cache a response."""

        cache_key = self._generate_cache_key(prompt, model, temperature)

        entry = {
            "response": response,
            "timestamp": datetime.now(),
            "model": model,
            "prompt_length": len(prompt)
        }

        # Store in local cache
        self.local_cache[cache_key] = entry

        # Store in Redis if available
        if self.redis_client:
            await self.redis_client.setex(
                cache_key,
                self.ttl,
                json.dumps(entry, default=str)
            )

    def _calculate_cost_saved(self, model: str, prompt: str):
        """Calculate cost saved by cache hit."""
        if model in MODELS:
            config = MODELS[model]
            encoder = tiktoken.encoding_for_model(model)
            tokens = len(encoder.encode(prompt))
            cost = (tokens / 1000) * config.input_cost_per_1k
            self.cache_stats["cost_saved"] += cost

    def get_cache_efficiency(self) -> Dict:
        """Calculate cache efficiency metrics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]

        if total_requests == 0:
            return {"hit_rate": 0, "cost_saved": 0}

        return {
            "hit_rate": self.cache_stats["hits"] / total_requests,
            "total_requests": total_requests,
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "cost_saved": self.cache_stats["cost_saved"],
            "avg_savings_per_hit": (
                self.cache_stats["cost_saved"] / self.cache_stats["hits"]
                if self.cache_stats["hits"] > 0 else 0
            )
        }


# Semantic caching for similar prompts
class SemanticCache(CostOptimizedCache):
    """Cache that recognizes semantically similar prompts."""

    def __init__(self, similarity_threshold: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.similarity_threshold = similarity_threshold
        self.embeddings_cache = {}

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]

        # In production, would call embedding API
        # For demo, using simple hash-based pseudo-embedding
        hash_val = hashlib.md5(text.encode()).hexdigest()
        embedding = [float(int(hash_val[i:i+2], 16))/255 for i in range(0, 32, 2)]

        self.embeddings_cache[text] = embedding
        return embedding

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a ** 2 for a in embedding1) ** 0.5
        norm2 = sum(b ** 2 for b in embedding2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    async def find_similar_cached(self, prompt: str, model: str) -> Optional[Dict]:
        """Find similar cached prompts."""
        prompt_embedding = await self.get_embedding(prompt)

        for cache_key, entry in self.local_cache.items():
            if entry["model"] != model:
                continue

            cached_prompt = entry.get("prompt", "")
            if not cached_prompt:
                continue

            cached_embedding = await self.get_embedding(cached_prompt)
            similarity = self.calculate_similarity(prompt_embedding, cached_embedding)

            if similarity >= self.similarity_threshold:
                print(f"Found similar cached prompt (similarity: {similarity:.2f})")
                self.cache_stats["hits"] += 1
                self._calculate_cost_saved(model, prompt)
                return entry["response"]

        return None


# Example 3: Batch Processing Optimization
print("\n" + "=" * 50)
print("Example 3: Batch Processing Optimization")
print("=" * 50)


class BatchProcessor:
    """Optimize costs through intelligent batching."""

    def __init__(self, batch_size: int = 10, wait_time: float = 1.0):
        self.batch_size = batch_size
        self.wait_time = wait_time  # Max wait time in seconds
        self.pending_requests = []
        self.processing = False

    async def add_request(self, prompt: str, callback) -> None:
        """Add request to batch queue."""
        request = {
            "prompt": prompt,
            "callback": callback,
            "timestamp": datetime.now()
        }
        self.pending_requests.append(request)

        # Check if we should process
        if len(self.pending_requests) >= self.batch_size:
            await self.process_batch()
        elif not self.processing:
            # Start timer for batch processing
            asyncio.create_task(self._wait_and_process())

    async def _wait_and_process(self):
        """Wait for more requests or timeout."""
        await asyncio.sleep(self.wait_time)
        if self.pending_requests and not self.processing:
            await self.process_batch()

    async def process_batch(self):
        """Process batch of requests together."""
        if self.processing or not self.pending_requests:
            return

        self.processing = True
        batch = self.pending_requests[:self.batch_size]
        self.pending_requests = self.pending_requests[self.batch_size:]

        print(f"Processing batch of {len(batch)} requests")

        # Combine prompts for batch processing
        combined_prompt = self._create_batch_prompt(batch)

        # Calculate cost savings
        individual_cost = self._calculate_individual_cost(batch)
        batch_cost = self._calculate_batch_cost(combined_prompt)
        savings = individual_cost - batch_cost

        print(f"Batch processing savings: ${savings:.4f}")
        print(f"  Individual cost: ${individual_cost:.4f}")
        print(f"  Batch cost: ${batch_cost:.4f}")

        # Process batch (simulated)
        results = await self._process_combined_prompt(combined_prompt)

        # Distribute results
        for i, request in enumerate(batch):
            if i < len(results):
                await request["callback"](results[i])

        self.processing = False

    def _create_batch_prompt(self, batch: List[Dict]) -> str:
        """Create combined prompt for batch processing."""
        prompts = [f"{i+1}. {req['prompt']}" for i, req in enumerate(batch)]
        combined = "Please answer the following questions concisely:\n\n"
        combined += "\n\n".join(prompts)
        combined += "\n\nProvide answers in the same numbered format."
        return combined

    def _calculate_individual_cost(self, batch: List[Dict]) -> float:
        """Calculate cost if processed individually."""
        encoder = tiktoken.get_encoding("cl100k_base")
        total_cost = 0

        for request in batch:
            tokens = len(encoder.encode(request["prompt"]))
            # Assume GPT-3.5 for calculation
            cost = (tokens / 1000) * MODELS["gpt-3.5-turbo"].input_cost_per_1k
            # Add estimated output cost
            cost += (tokens / 1000) * MODELS["gpt-3.5-turbo"].output_cost_per_1k
            total_cost += cost

        return total_cost

    def _calculate_batch_cost(self, combined_prompt: str) -> float:
        """Calculate cost for batch processing."""
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = len(encoder.encode(combined_prompt))
        # Single API call cost
        cost = (tokens / 1000) * MODELS["gpt-3.5-turbo"].input_cost_per_1k
        # Estimated output (slightly more than individual due to format)
        cost += (tokens * 1.2 / 1000) * MODELS["gpt-3.5-turbo"].output_cost_per_1k
        return cost

    async def _process_combined_prompt(self, prompt: str) -> List[str]:
        """Process the combined prompt (simulated)."""
        # In production, would call API
        # For demo, return mock results
        num_questions = prompt.count("\n\n") - 1
        return [f"Answer {i+1}" for i in range(num_questions)]


# Example 4: Token-Aware Routing
print("\n" + "=" * 50)
print("Example 4: Token-Aware Routing")
print("=" * 50)


class TokenAwareRouter:
    """Route requests based on token count and cost optimization."""

    def __init__(self):
        self.encoders = {
            model: tiktoken.encoding_for_model(model)
            for model in MODELS.keys()
        }
        self.routing_stats = {
            "total_requests": 0,
            "routing_decisions": {},
            "total_cost": 0
        }

    def route_request(self, prompt: str, context: Optional[str] = None) -> Tuple[str, Dict]:
        """Route request to optimal model based on tokens."""

        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\n{prompt}"

        # Count tokens for each model
        token_counts = {}
        for model, encoder in self.encoders.items():
            token_counts[model] = len(encoder.encode(full_prompt))

        # Find models that can handle the request
        viable_models = []
        for model, count in token_counts.items():
            if count <= MODELS[model].max_tokens:
                # Calculate cost
                cost = (count / 1000) * MODELS[model].input_cost_per_1k
                # Estimate output cost
                cost += (count * 0.5 / 1000) * MODELS[model].output_cost_per_1k

                viable_models.append({
                    "model": model,
                    "tokens": count,
                    "cost": cost,
                    "quality": MODELS[model].quality_score
                })

        if not viable_models:
            # Need to truncate or split
            return self._handle_oversized_request(full_prompt)

        # Select best model based on cost/quality ratio
        viable_models.sort(key=lambda x: x["quality"] / x["cost"], reverse=True)
        selected = viable_models[0]

        # Update stats
        self.routing_stats["total_requests"] += 1
        self.routing_stats["total_cost"] += selected["cost"]

        if selected["model"] not in self.routing_stats["routing_decisions"]:
            self.routing_stats["routing_decisions"][selected["model"]] = 0
        self.routing_stats["routing_decisions"][selected["model"]] += 1

        print(f"Routed to {selected['model']}:")
        print(f"  Tokens: {selected['tokens']}")
        print(f"  Est. Cost: ${selected['cost']:.4f}")
        print(f"  Quality: {selected['quality']}")

        return selected["model"], selected

    def _handle_oversized_request(self, prompt: str) -> Tuple[str, Dict]:
        """Handle requests that exceed token limits."""
        print("Request exceeds token limits, implementing fallback strategy")

        # Strategy 1: Use model with highest token limit
        largest_model = max(MODELS.items(), key=lambda x: x[1].max_tokens)
        model_name = largest_model[0]
        config = largest_model[1]

        # Strategy 2: Truncate prompt to fit
        encoder = self.encoders[model_name]
        tokens = encoder.encode(prompt)
        max_allowed = int(config.max_tokens * 0.8)  # Leave room for output

        if len(tokens) > max_allowed:
            truncated_tokens = tokens[:max_allowed]
            truncated_prompt = encoder.decode(truncated_tokens)
            print(f"Truncated prompt from {len(tokens)} to {max_allowed} tokens")
        else:
            truncated_prompt = prompt

        cost = (len(truncated_tokens) / 1000) * config.input_cost_per_1k

        return model_name, {
            "model": model_name,
            "tokens": len(truncated_tokens),
            "cost": cost,
            "truncated": len(tokens) > max_allowed
        }

    def get_routing_report(self) -> str:
        """Generate routing statistics report."""
        report = "\nToken-Aware Routing Report\n" + "-" * 40

        if self.routing_stats["total_requests"] == 0:
            return report + "\nNo requests processed yet."

        report += f"\nTotal requests: {self.routing_stats['total_requests']}"
        report += f"\nTotal estimated cost: ${self.routing_stats['total_cost']:.4f}"
        report += f"\nAverage cost per request: ${self.routing_stats['total_cost'] / self.routing_stats['total_requests']:.4f}"

        report += "\n\nRouting distribution:"
        for model, count in self.routing_stats["routing_decisions"].items():
            percentage = (count / self.routing_stats["total_requests"]) * 100
            report += f"\n  {model}: {count} ({percentage:.1f}%)"

        return report


# Example usage
router = TokenAwareRouter()

test_prompts = [
    "What is 2+2?",  # Very short
    "Explain quantum computing" * 100,  # Medium
    "Write a detailed analysis of..." * 1000  # Very long
]

for prompt in test_prompts[:2]:  # Limit for demo
    model, details = router.route_request(prompt)

print(router.get_routing_report())


# Example 5: Response Compression
print("\n" + "=" * 50)
print("Example 5: Response Compression")
print("=" * 50)


class ResponseCompressor:
    """Compress responses to reduce token usage in conversations."""

    def __init__(self):
        self.compression_strategies = {
            "summarize": self.summarize_response,
            "extract_key_points": self.extract_key_points,
            "remove_redundancy": self.remove_redundancy,
            "structured_format": self.convert_to_structured
        }

    def compress_response(self,
                         response: str,
                         target_reduction: float = 0.5,
                         preserve_key_info: bool = True) -> Tuple[str, Dict]:
        """Compress response while preserving information."""

        original_tokens = len(tiktoken.get_encoding("cl100k_base").encode(response))
        target_tokens = int(original_tokens * (1 - target_reduction))

        print(f"Compressing response: {original_tokens} → {target_tokens} tokens")

        # Try compression strategies in order of preference
        best_result = response
        best_metrics = {"tokens": original_tokens, "reduction": 0}

        for strategy_name, strategy_func in self.compression_strategies.items():
            compressed = strategy_func(response)
            compressed_tokens = len(tiktoken.get_encoding("cl100k_base").encode(compressed))

            if compressed_tokens <= target_tokens:
                reduction = 1 - (compressed_tokens / original_tokens)
                if reduction > best_metrics["reduction"]:
                    best_result = compressed
                    best_metrics = {
                        "tokens": compressed_tokens,
                        "reduction": reduction,
                        "strategy": strategy_name
                    }

                    if compressed_tokens <= target_tokens:
                        break

        # Calculate cost savings
        cost_saved = self._calculate_cost_savings(
            original_tokens - best_metrics["tokens"]
        )
        best_metrics["cost_saved"] = cost_saved

        print(f"Best compression: {best_metrics['strategy']}")
        print(f"  Reduction: {best_metrics['reduction']:.1%}")
        print(f"  Cost saved: ${cost_saved:.4f}")

        return best_result, best_metrics

    def summarize_response(self, text: str) -> str:
        """Summarize response to key points."""
        lines = text.split('\n')
        if len(lines) <= 3:
            return text

        # Simple summarization: keep first and last parts
        summary = lines[0]
        if len(lines) > 4:
            # Add middle key sentence
            middle = lines[len(lines)//2]
            if len(middle) > 20:
                summary += f" {middle}"
        summary += f" {lines[-1]}"

        return summary

    def extract_key_points(self, text: str) -> str:
        """Extract key points from response."""
        # Look for structured content
        key_indicators = ["important:", "note:", "key:", "main:", "summary:"]

        key_lines = []
        lines = text.split('\n')

        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in key_indicators):
                key_lines.append(line)
            elif len(line) > 50 and '.' in line:
                # Include substantial sentences
                key_lines.append(line.split('.')[0] + '.')

        if not key_lines:
            # Fallback to first and last sentences
            sentences = text.split('.')
            if len(sentences) > 2:
                return f"{sentences[0]}. {sentences[-2]}."
            return text

        return ' '.join(key_lines[:3])  # Limit to 3 key points

    def remove_redundancy(self, text: str) -> str:
        """Remove redundant information."""
        sentences = text.split('.')
        unique_sentences = []
        seen_concepts = set()

        for sentence in sentences:
            # Simple deduplication based on key words
            words = set(sentence.lower().split())
            if len(words) < 3:
                continue

            # Check overlap with seen concepts
            overlap = len(words & seen_concepts)
            if overlap < len(words) * 0.5:  # Less than 50% overlap
                unique_sentences.append(sentence)
                seen_concepts.update(words)

        return '. '.join(unique_sentences) + '.'

    def convert_to_structured(self, text: str) -> str:
        """Convert to structured format for compression."""
        lines = text.split('\n')

        # Create structured summary
        structured = []
        for i, line in enumerate(lines):
            if len(line) > 20:
                # Compress to bullet point
                key_words = [w for w in line.split() if len(w) > 4][:5]
                if key_words:
                    structured.append(f"• {' '.join(key_words)}")

        return '\n'.join(structured[:5])  # Limit points

    def _calculate_cost_savings(self, tokens_saved: int) -> float:
        """Calculate cost savings from compression."""
        # Assume GPT-4 pricing for context
        return (tokens_saved / 1000) * MODELS["gpt-4-turbo-preview"].input_cost_per_1k


# Example 6: Tiered Processing Strategies
print("\n" + "=" * 50)
print("Example 6: Tiered Processing Strategies")
print("=" * 50)


class TieredProcessor:
    """Implement tiered processing for cost optimization."""

    def __init__(self):
        self.tiers = {
            "screening": {
                "model": "gpt-3.5-turbo",
                "purpose": "Initial filtering and classification",
                "max_cost": 0.01
            },
            "processing": {
                "model": "gpt-4-turbo-preview",
                "purpose": "Main processing and analysis",
                "max_cost": 0.05
            },
            "refinement": {
                "model": "gpt-4",
                "purpose": "Final quality check and refinement",
                "max_cost": 0.10
            }
        }
        self.processing_stats = []

    async def process_with_tiers(self,
                                input_data: str,
                                quality_threshold: float = 0.8) -> Tuple[str, Dict]:
        """Process through multiple tiers as needed."""

        total_cost = 0
        processing_path = []

        # Tier 1: Screening
        screen_result = await self._process_tier("screening", input_data)
        total_cost += screen_result["cost"]
        processing_path.append("screening")

        # Check if screening is sufficient
        if screen_result["confidence"] > 0.9:
            print("High confidence from screening tier, stopping early")
            return screen_result["response"], {
                "path": processing_path,
                "total_cost": total_cost,
                "final_tier": "screening"
            }

        # Tier 2: Main processing
        if screen_result["requires_processing"]:
            process_result = await self._process_tier(
                "processing",
                input_data,
                context=screen_result["response"]
            )
            total_cost += process_result["cost"]
            processing_path.append("processing")

            # Check if processing is sufficient
            if process_result["quality"] >= quality_threshold:
                return process_result["response"], {
                    "path": processing_path,
                    "total_cost": total_cost,
                    "final_tier": "processing"
                }

            # Tier 3: Refinement if needed
            if process_result["quality"] < quality_threshold:
                refine_result = await self._process_tier(
                    "refinement",
                    input_data,
                    context=process_result["response"]
                )
                total_cost += refine_result["cost"]
                processing_path.append("refinement")

                return refine_result["response"], {
                    "path": processing_path,
                    "total_cost": total_cost,
                    "final_tier": "refinement"
                }

        # Default return
        return screen_result["response"], {
            "path": processing_path,
            "total_cost": total_cost,
            "final_tier": processing_path[-1]
        }

    async def _process_tier(self,
                          tier_name: str,
                          input_data: str,
                          context: Optional[str] = None) -> Dict:
        """Process at a specific tier."""

        tier_config = self.tiers[tier_name]
        print(f"\nProcessing at tier: {tier_name}")
        print(f"  Model: {tier_config['model']}")
        print(f"  Purpose: {tier_config['purpose']}")

        # Prepare prompt based on tier
        if tier_name == "screening":
            prompt = f"Quickly assess: {input_data}\nIs this complex? (yes/no)"
        elif tier_name == "processing":
            prompt = f"Process this request: {input_data}"
            if context:
                prompt = f"Initial assessment: {context}\n\n{prompt}"
        else:  # refinement
            prompt = f"Refine and improve this response:\n{context}\n\nFor request: {input_data}"

        # Calculate cost
        encoder = tiktoken.encoding_for_model(tier_config["model"])
        tokens = len(encoder.encode(prompt))
        model_config = MODELS[tier_config["model"]]
        cost = (tokens / 1000) * model_config.input_cost_per_1k

        # Simulate processing
        if tier_name == "screening":
            return {
                "response": "Initial assessment complete",
                "confidence": 0.7,
                "requires_processing": True,
                "cost": cost
            }
        elif tier_name == "processing":
            return {
                "response": "Processed response with good quality",
                "quality": 0.85,
                "cost": cost
            }
        else:
            return {
                "response": "Refined high-quality response",
                "quality": 0.95,
                "cost": cost
            }

    def analyze_tier_efficiency(self) -> Dict:
        """Analyze efficiency of tiered processing."""
        if not self.processing_stats:
            return {"message": "No processing data available"}

        tier_usage = {}
        total_cost = 0
        early_exits = 0

        for stat in self.processing_stats:
            final_tier = stat["final_tier"]
            tier_usage[final_tier] = tier_usage.get(final_tier, 0) + 1
            total_cost += stat["total_cost"]

            if final_tier != "refinement":
                early_exits += 1

        avg_cost = total_cost / len(self.processing_stats)
        early_exit_rate = early_exits / len(self.processing_stats)

        # Calculate savings from early exits
        max_cost = sum(t["max_cost"] for t in self.tiers.values())
        avg_max_cost = max_cost
        savings_rate = 1 - (avg_cost / avg_max_cost)

        return {
            "total_processed": len(self.processing_stats),
            "tier_usage": tier_usage,
            "early_exit_rate": early_exit_rate,
            "average_cost": avg_cost,
            "savings_rate": savings_rate,
            "total_cost": total_cost
        }


# Example 7: Complete Cost Optimization System
print("\n" + "=" * 50)
print("Example 7: Complete Cost Optimization System")
print("=" * 50)


class CostOptimizationSystem:
    """Comprehensive cost optimization system."""

    def __init__(self, daily_budget: float = 100.0):
        self.daily_budget = daily_budget
        self.model_selector = ModelSelector()
        self.cache = SemanticCache()
        self.batch_processor = BatchProcessor()
        self.router = TokenAwareRouter()
        self.compressor = ResponseCompressor()
        self.tiered_processor = TieredProcessor()

        self.daily_stats = {
            "date": datetime.now().date(),
            "requests": 0,
            "total_cost": 0,
            "cache_hits": 0,
            "optimizations_applied": []
        }

    async def process_request(self,
                            prompt: str,
                            priority: str = "normal",
                            quality_requirement: float = 0.8) -> Tuple[str, Dict]:
        """Process request with full optimization stack."""

        optimization_report = {
            "optimizations": [],
            "original_cost": 0,
            "optimized_cost": 0,
            "savings": 0
        }

        # Check budget
        if not self._check_budget():
            return "Daily budget exceeded", {"error": "budget_exceeded"}

        # 1. Check cache first
        cached = await self.cache.find_similar_cached(prompt, "gpt-4-turbo-preview")
        if cached:
            optimization_report["optimizations"].append("cache_hit")
            self.daily_stats["cache_hits"] += 1
            return cached["content"], optimization_report

        # 2. Route based on tokens
        model, routing_info = self.router.route_request(prompt)
        optimization_report["optimizations"].append(f"routed_to_{model}")

        # 3. Check if batching is appropriate
        if priority == "low" and len(prompt) < 500:
            optimization_report["optimizations"].append("batched")
            # Would add to batch queue
            # await self.batch_processor.add_request(prompt, callback)

        # 4. Apply tiered processing if needed
        if quality_requirement < 0.7:
            result, tier_info = await self.tiered_processor.process_with_tiers(
                prompt,
                quality_requirement
            )
            optimization_report["optimizations"].append(f"tiered_{tier_info['final_tier']}")
            optimization_report["optimized_cost"] = tier_info["total_cost"]
        else:
            # Simulate regular processing
            optimization_report["optimized_cost"] = routing_info["cost"]
            result = f"Processed: {prompt[:50]}..."

        # 5. Compress response if needed
        original_tokens = len(tiktoken.get_encoding("cl100k_base").encode(result))
        if original_tokens > 1000:
            compressed, compression_info = self.compressor.compress_response(
                result,
                target_reduction=0.3
            )
            optimization_report["optimizations"].append(
                f"compressed_{compression_info['reduction']:.0%}"
            )
            result = compressed

        # 6. Cache the result
        await self.cache.cache_response(prompt, model, 0, {"content": result})

        # Update statistics
        self.daily_stats["requests"] += 1
        self.daily_stats["total_cost"] += optimization_report["optimized_cost"]

        # Calculate savings
        base_cost = self._calculate_base_cost(prompt)
        optimization_report["original_cost"] = base_cost
        optimization_report["savings"] = base_cost - optimization_report["optimized_cost"]

        return result, optimization_report

    def _check_budget(self) -> bool:
        """Check if within daily budget."""
        if self.daily_stats["date"] != datetime.now().date():
            # Reset for new day
            self.daily_stats = {
                "date": datetime.now().date(),
                "requests": 0,
                "total_cost": 0,
                "cache_hits": 0,
                "optimizations_applied": []
            }

        return self.daily_stats["total_cost"] < self.daily_budget

    def _calculate_base_cost(self, prompt: str) -> float:
        """Calculate baseline cost without optimizations."""
        encoder = tiktoken.encoding_for_model("gpt-4")
        tokens = len(encoder.encode(prompt))
        input_cost = (tokens / 1000) * MODELS["gpt-4"].input_cost_per_1k
        output_cost = (tokens / 1000) * MODELS["gpt-4"].output_cost_per_1k
        return input_cost + output_cost

    def get_optimization_report(self) -> str:
        """Generate comprehensive optimization report."""
        report = "\n" + "=" * 50
        report += "\nCost Optimization Report"
        report += "\n" + "=" * 50

        report += f"\n\nDate: {self.daily_stats['date']}"
        report += f"\nTotal Requests: {self.daily_stats['requests']}"
        report += f"\nTotal Cost: ${self.daily_stats['total_cost']:.2f}"
        report += f"\nBudget Remaining: ${self.daily_budget - self.daily_stats['total_cost']:.2f}"
        report += f"\nCache Hit Rate: {self.daily_stats['cache_hits'] / max(self.daily_stats['requests'], 1):.1%}"

        # Model usage distribution
        report += "\n\nModel Usage:"
        for model, count in self.router.routing_stats["routing_decisions"].items():
            report += f"\n  {model}: {count}"

        # Cache efficiency
        cache_stats = self.cache.get_cache_efficiency()
        report += f"\n\nCache Performance:"
        report += f"\n  Hit Rate: {cache_stats['hit_rate']:.1%}"
        report += f"\n  Cost Saved: ${cache_stats['cost_saved']:.2f}"

        # Tier efficiency
        tier_stats = self.tiered_processor.analyze_tier_efficiency()
        if "early_exit_rate" in tier_stats:
            report += f"\n\nTiered Processing:"
            report += f"\n  Early Exit Rate: {tier_stats['early_exit_rate']:.1%}"
            report += f"\n  Savings Rate: {tier_stats['savings_rate']:.1%}"

        return report

    async def optimize_conversation_history(self,
                                          messages: List[Dict],
                                          max_tokens: int = 4000) -> List[Dict]:
        """Optimize conversation history to fit within token budget."""

        encoder = tiktoken.get_encoding("cl100k_base")
        current_tokens = sum(len(encoder.encode(m["content"])) for m in messages)

        if current_tokens <= max_tokens:
            return messages

        print(f"Optimizing conversation: {current_tokens} → {max_tokens} tokens")

        optimized = []
        strategies_applied = []

        # Strategy 1: Remove system messages except first
        system_msgs = [m for m in messages if m["role"] == "system"]
        if len(system_msgs) > 1:
            messages = [system_msgs[0]] + [m for m in messages if m["role"] != "system"]
            strategies_applied.append("removed_extra_system")

        # Strategy 2: Compress older messages
        for i, msg in enumerate(messages):
            if i < len(messages) - 4:  # Keep recent messages intact
                compressed, _ = self.compressor.compress_response(
                    msg["content"],
                    target_reduction=0.5
                )
                optimized.append({"role": msg["role"], "content": compressed})
                strategies_applied.append(f"compressed_msg_{i}")
            else:
                optimized.append(msg)

        # Strategy 3: Remove oldest messages if still over budget
        while sum(len(encoder.encode(m["content"])) for m in optimized) > max_tokens:
            if len(optimized) > 3:  # Keep minimum context
                optimized.pop(1)  # Remove second message (keep system)
                strategies_applied.append("removed_old_message")
            else:
                break

        final_tokens = sum(len(encoder.encode(m["content"])) for m in optimized)
        print(f"Optimization complete: {final_tokens} tokens")
        print(f"Strategies: {', '.join(strategies_applied)}")

        return optimized


# Create system instance
optimizer = CostOptimizationSystem(daily_budget=10.0)

# Example usage (commented to avoid async issues in sync context)
"""
# Process sample request
result, report = asyncio.run(
    optimizer.process_request(
        "Explain the benefits of renewable energy",
        priority="normal",
        quality_requirement=0.85
    )
)

print(f"Result: {result[:100]}...")
print(f"Optimizations applied: {report['optimizations']}")
print(f"Cost savings: ${report['savings']:.4f}")

# Generate report
print(optimizer.get_optimization_report())
"""

print("\n✅ All cost optimization examples completed!")
print("\nKey Takeaways:")
print("1. Model selection based on task complexity reduces costs significantly")
print("2. Caching and semantic similarity matching eliminate redundant API calls")
print("3. Batch processing amortizes costs across multiple requests")
print("4. Token-aware routing ensures optimal model selection")
print("5. Response compression reduces context window usage")
print("6. Tiered processing allows early exit for simple tasks")
print("7. Comprehensive optimization systems combine multiple strategies")