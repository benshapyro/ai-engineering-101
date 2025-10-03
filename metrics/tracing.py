"""
Tracing and metrics collection for LLM applications.

Provides decorators and utilities for:
- Latency tracking
- Token counting
- Cost calculation
- Cache hit rate monitoring
- Failure tracking
"""

import time
import functools
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime
import json
import threading


@dataclass
class CallMetrics:
    """Metrics for a single LLM call."""
    timestamp: str
    function_name: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    cache_hit: bool
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Collect and aggregate metrics across calls.

    Thread-safe metrics collection for production use.
    """

    def __init__(self):
        """Initialize collector."""
        self._calls: List[CallMetrics] = []
        self._lock = threading.Lock()

        # Aggregate counters
        self._total_calls = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_latency = 0.0
        self._cache_hits = 0
        self._failures = 0

        # Per-model stats
        self._model_stats = defaultdict(lambda: {
            "calls": 0,
            "tokens": 0,
            "cost": 0.0,
            "latency": 0.0
        })

        # Latency histogram (buckets in ms)
        self._latency_buckets = {
            "0-100": 0,
            "100-500": 0,
            "500-1000": 0,
            "1000-5000": 0,
            "5000+": 0
        }

    def record(self, metrics: CallMetrics):
        """
        Record a single call's metrics.

        Args:
            metrics: CallMetrics instance
        """
        with self._lock:
            self._calls.append(metrics)

            # Update aggregates
            self._total_calls += 1
            self._total_tokens += metrics.total_tokens
            self._total_cost += metrics.cost
            self._total_latency += metrics.latency_ms

            if metrics.cache_hit:
                self._cache_hits += 1

            if not metrics.success:
                self._failures += 1

            # Update per-model stats
            model_stats = self._model_stats[metrics.model]
            model_stats["calls"] += 1
            model_stats["tokens"] += metrics.total_tokens
            model_stats["cost"] += metrics.cost
            model_stats["latency"] += metrics.latency_ms

            # Update latency histogram
            latency = metrics.latency_ms
            if latency < 100:
                self._latency_buckets["0-100"] += 1
            elif latency < 500:
                self._latency_buckets["100-500"] += 1
            elif latency < 1000:
                self._latency_buckets["500-1000"] += 1
            elif latency < 5000:
                self._latency_buckets["1000-5000"] += 1
            else:
                self._latency_buckets["5000+"] += 1

    def get_summary(self) -> Dict[str, Any]:
        """
        Get aggregate metrics summary.

        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            if self._total_calls == 0:
                return {"message": "No calls recorded"}

            summary = {
                "total_calls": self._total_calls,
                "total_tokens": self._total_tokens,
                "total_cost": round(self._total_cost, 4),
                "avg_latency_ms": round(self._total_latency / self._total_calls, 2),
                "cache_hit_rate": round(self._cache_hits / self._total_calls, 3),
                "failure_rate": round(self._failures / self._total_calls, 3),
                "latency_distribution": dict(self._latency_buckets),
                "per_model": {}
            }

            # Add per-model stats
            for model, stats in self._model_stats.items():
                summary["per_model"][model] = {
                    "calls": stats["calls"],
                    "tokens": stats["tokens"],
                    "cost": round(stats["cost"], 4),
                    "avg_latency_ms": round(stats["latency"] / stats["calls"], 2)
                }

            return summary

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent call metrics.

        Args:
            n: Number of recent calls to return

        Returns:
            List of recent call metrics
        """
        with self._lock:
            return [asdict(call) for call in self._calls[-n:]]

    def get_percentiles(self, percentiles: List[int] = [50, 95, 99]) -> Dict[int, float]:
        """
        Get latency percentiles.

        Args:
            percentiles: List of percentiles to compute

        Returns:
            Dict mapping percentile to latency in ms
        """
        with self._lock:
            if not self._calls:
                return {p: 0.0 for p in percentiles}

            latencies = sorted([call.latency_ms for call in self._calls])
            n = len(latencies)

            result = {}
            for p in percentiles:
                idx = int((p / 100.0) * n)
                idx = min(idx, n - 1)
                result[p] = latencies[idx]

            return result

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        summary = self.get_summary()

        if "message" in summary:
            return "# No metrics"

        lines = []

        # Total calls
        lines.append(f"# HELP llm_calls_total Total number of LLM calls")
        lines.append(f"# TYPE llm_calls_total counter")
        lines.append(f"llm_calls_total {summary['total_calls']}")

        # Total tokens
        lines.append(f"# HELP llm_tokens_total Total tokens processed")
        lines.append(f"# TYPE llm_tokens_total counter")
        lines.append(f"llm_tokens_total {summary['total_tokens']}")

        # Total cost
        lines.append(f"# HELP llm_cost_total Total cost in USD")
        lines.append(f"# TYPE llm_cost_total counter")
        lines.append(f"llm_cost_total {summary['total_cost']}")

        # Cache hit rate
        lines.append(f"# HELP llm_cache_hit_rate Cache hit rate")
        lines.append(f"# TYPE llm_cache_hit_rate gauge")
        lines.append(f"llm_cache_hit_rate {summary['cache_hit_rate']}")

        # Latency histogram
        lines.append(f"# HELP llm_latency_histogram Latency distribution")
        lines.append(f"# TYPE llm_latency_histogram histogram")
        for bucket, count in summary['latency_distribution'].items():
            lines.append(f'llm_latency_histogram{{le="{bucket}"}} {count}')

        return "\n".join(lines)

    def export_jsonl(self, filepath: str):
        """
        Export detailed call logs to JSONL file.

        Args:
            filepath: Output file path
        """
        with self._lock:
            with open(filepath, 'w') as f:
                for call in self._calls:
                    f.write(json.dumps(asdict(call)) + "\n")

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._calls.clear()
            self._total_calls = 0
            self._total_tokens = 0
            self._total_cost = 0.0
            self._total_latency = 0.0
            self._cache_hits = 0
            self._failures = 0
            self._model_stats.clear()
            for bucket in self._latency_buckets:
                self._latency_buckets[bucket] = 0


# Global collector instance
_global_collector = MetricsCollector()


def get_global_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    return _global_collector


def reset_metrics():
    """Reset global metrics collector."""
    _global_collector.reset()


def trace_call(
    model: str = "unknown",
    track_cost: bool = True
):
    """
    Decorator to trace LLM function calls.

    Captures:
    - Latency
    - Token counts (if available in result)
    - Cost (if enabled)
    - Success/failure
    - Cache hits (if available)

    Args:
        model: Model name (can be overridden at runtime)
        track_cost: Whether to calculate cost

    Example:
        @trace_call(model="gpt-5")
        def my_llm_call(prompt):
            response = client.generate(prompt)
            return response

        # Metrics automatically collected
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Determine model (check kwargs, then use default)
            actual_model = kwargs.get('model', model)

            # Initialize metrics
            input_tokens = 0
            output_tokens = 0
            cost = 0.0
            cache_hit = False
            success = True
            error = None

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Extract token counts if available
                if hasattr(result, 'usage'):
                    input_tokens = getattr(result.usage, 'prompt_tokens', 0)
                    output_tokens = getattr(result.usage, 'completion_tokens', 0)
                elif isinstance(result, dict):
                    usage = result.get('usage', {})
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)

                # Check for cache hit
                if hasattr(result, 'from_cache'):
                    cache_hit = result.from_cache
                elif isinstance(result, dict):
                    cache_hit = result.get('from_cache', False)

                return result

            except Exception as e:
                success = False
                error = str(e)
                raise

            finally:
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000

                # Calculate cost (simple estimate)
                if track_cost:
                    # Rough estimate: $5/$15 per 1M tokens
                    cost = (input_tokens * 5 / 1_000_000) + (output_tokens * 15 / 1_000_000)

                # Record metrics
                metrics = CallMetrics(
                    timestamp=datetime.now().isoformat(),
                    function_name=func.__name__,
                    model=actual_model,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    cost=cost,
                    cache_hit=cache_hit,
                    success=success,
                    error=error
                )

                _global_collector.record(metrics)

        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    print("Metrics Collection Example")
    print("=" * 60)

    # Simulate some calls
    @trace_call(model="gpt-5")
    def example_call(prompt: str, sleep_ms: int = 100):
        """Example traced function."""
        import time
        time.sleep(sleep_ms / 1000.0)

        # Simulate response with usage
        return {
            "output": f"Response to: {prompt}",
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 50
            }
        }

    # Make some calls
    for i in range(10):
        example_call(f"Test prompt {i}", sleep_ms=100 + i * 50)

    # Get summary
    collector = get_global_collector()
    summary = collector.get_summary()

    print("\nMetrics Summary:")
    print(json.dumps(summary, indent=2))

    # Get percentiles
    percentiles = collector.get_percentiles([50, 95, 99])
    print("\nLatency Percentiles:")
    for p, latency in percentiles.items():
        print(f"  p{p}: {latency:.2f}ms")

    # Prometheus export
    print("\nPrometheus Metrics:")
    print(collector.export_prometheus())

    # Recent calls
    print("\nRecent Calls:")
    for call in collector.get_recent(n=3):
        print(f"  {call['function_name']}: {call['latency_ms']:.2f}ms, "
              f"{call['total_tokens']} tokens, ${call['cost']:.4f}")
