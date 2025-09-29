"""
Module 14: Production Patterns
Error Resilience Examples

Learn to build robust error handling and recovery for production LLM applications.
"""

import os
import time
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import json
from functools import wraps
from dotenv import load_dotenv

load_dotenv()


# Example 1: Retry Strategies
print("=" * 50)
print("Example 1: Retry Strategies")
print("=" * 50)


class RetryStrategy(Enum):
    """Different retry strategies."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    CUSTOM = "custom"


class RetryHandler:
    """Advanced retry handler for API calls."""

    def __init__(self,
                 max_retries: int = 3,
                 strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.strategy = strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_history = []

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on strategy."""
        if self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == RetryStrategy.FIBONACCI:
            delay = self._fibonacci_delay(attempt)
        else:
            delay = self.base_delay

        # Apply max delay cap
        delay = min(delay, self.max_delay)

        # Add jitter to prevent thundering herd
        if self.jitter:
            delay *= (0.5 + random.random())

        return delay

    def _fibonacci_delay(self, n: int) -> float:
        """Calculate Fibonacci delay."""
        if n <= 1:
            return self.base_delay
        a, b = self.base_delay, self.base_delay
        for _ in range(n - 1):
            a, b = b, a + b
        return b

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry."""
        # Don't retry if max attempts reached
        if attempt >= self.max_retries:
            return False

        # Define retryable exceptions
        retryable_exceptions = (
            ConnectionError,
            TimeoutError,
            OSError,
        )

        # Check if exception is retryable
        if isinstance(exception, retryable_exceptions):
            return True

        # Check for specific error messages
        error_message = str(exception).lower()
        retryable_keywords = [
            "rate limit",
            "timeout",
            "connection",
            "temporary",
            "unavailable"
        ]

        return any(keyword in error_message for keyword in retryable_keywords)

    async def execute_with_retry(self, func: Callable,
                                *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)

                # Record successful attempt
                self.retry_history.append({
                    "timestamp": datetime.now(),
                    "attempt": attempt,
                    "success": True
                })

                return result

            except Exception as e:
                last_exception = e

                # Record failed attempt
                self.retry_history.append({
                    "timestamp": datetime.now(),
                    "attempt": attempt,
                    "success": False,
                    "error": str(e)
                })

                if not self.should_retry(e, attempt):
                    raise

                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    print(f"Retry {attempt + 1}/{self.max_retries} after {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)

        # All retries exhausted
        raise last_exception

    def get_retry_stats(self) -> Dict:
        """Get retry statistics."""
        if not self.retry_history:
            return {}

        successes = sum(1 for h in self.retry_history if h["success"])
        failures = len(self.retry_history) - successes

        return {
            "total_attempts": len(self.retry_history),
            "successes": successes,
            "failures": failures,
            "success_rate": successes / len(self.retry_history),
            "avg_attempts": sum(h["attempt"] + 1 for h in self.retry_history) / len(self.retry_history)
        }


# Test retry handler
async def flaky_api_call():
    """Simulate a flaky API call."""
    if random.random() < 0.7:  # 70% failure rate
        raise ConnectionError("API temporarily unavailable")
    return {"status": "success", "data": "Response data"}


async def test_retry():
    retry_handler = RetryHandler(
        max_retries=5,
        strategy=RetryStrategy.EXPONENTIAL,
        base_delay=0.5
    )

    try:
        result = await retry_handler.execute_with_retry(flaky_api_call)
        print(f"Success: {result}")
    except Exception as e:
        print(f"Failed after retries: {e}")

    stats = retry_handler.get_retry_stats()
    print(f"Retry Stats: {stats}")

asyncio.run(test_retry())


# Example 2: Circuit Breaker Pattern
print("\n" + "=" * 50)
print("Example 2: Circuit Breaker Pattern")
print("=" * 50)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.success_count = 0
        self.call_count = 0

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker."""
        self.call_count += 1

        # Update state based on conditions
        self._update_state()

        if self.state == CircuitState.OPEN:
            raise Exception(f"Circuit breaker is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

    def _update_state(self):
        """Update circuit state based on conditions."""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
                if time_since_failure >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    print(f"Circuit breaker: OPEN → HALF_OPEN")

    def _on_success(self):
        """Handle successful call."""
        self.success_count += 1

        if self.state == CircuitState.HALF_OPEN:
            # Service recovered, close circuit
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            print(f"Circuit breaker: HALF_OPEN → CLOSED (service recovered)")

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            # Still failing, reopen circuit
            self.state = CircuitState.OPEN
            print(f"Circuit breaker: HALF_OPEN → OPEN (still failing)")

        elif self.failure_count >= self.failure_threshold:
            # Too many failures, open circuit
            self.state = CircuitState.OPEN
            print(f"Circuit breaker: CLOSED → OPEN (threshold reached)")

    def get_status(self) -> Dict:
        """Get circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "call_count": self.call_count,
            "failure_rate": self.failure_count / max(self.call_count, 1)
        }

    def reset(self):
        """Reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None


# Test circuit breaker
def unreliable_service():
    """Simulate an unreliable service."""
    if random.random() < 0.8:  # 80% failure rate initially
        raise ConnectionError("Service unavailable")
    return "Success"


breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=2)

print("Testing Circuit Breaker:")
for i in range(15):
    try:
        result = breaker.call(unreliable_service)
        print(f"  Call {i+1}: {result}")
    except Exception as e:
        print(f"  Call {i+1}: {e}")

    if i == 7:
        print("  [Waiting for recovery timeout...]")
        time.sleep(2.5)

print(f"\nFinal Status: {breaker.get_status()}")


# Example 3: Fallback Mechanisms
print("\n" + "=" * 50)
print("Example 3: Fallback Mechanisms")
print("=" * 50)


class FallbackHandler:
    """Handle fallbacks for failed operations."""

    def __init__(self):
        self.fallback_chain = []
        self.fallback_history = []
        self.cache = {}

    def add_fallback(self, name: str, handler: Callable,
                     priority: int = 0):
        """Add a fallback handler."""
        self.fallback_chain.append({
            "name": name,
            "handler": handler,
            "priority": priority
        })
        # Sort by priority (higher priority first)
        self.fallback_chain.sort(key=lambda x: x["priority"], reverse=True)

    async def execute_with_fallback(self, primary_func: Callable,
                                   *args, **kwargs) -> Any:
        """Execute with fallback chain."""
        # Try primary function
        try:
            result = await primary_func(*args, **kwargs)
            self.fallback_history.append({
                "timestamp": datetime.now(),
                "used": "primary",
                "success": True
            })
            return result

        except Exception as primary_error:
            print(f"Primary failed: {primary_error}")

            # Try fallbacks in order
            for fallback in self.fallback_chain:
                try:
                    result = await fallback["handler"](*args, **kwargs)
                    self.fallback_history.append({
                        "timestamp": datetime.now(),
                        "used": fallback["name"],
                        "success": True
                    })
                    print(f"Fallback '{fallback['name']}' succeeded")
                    return result

                except Exception as fallback_error:
                    print(f"Fallback '{fallback['name']}' failed: {fallback_error}")
                    continue

            # All fallbacks failed
            self.fallback_history.append({
                "timestamp": datetime.now(),
                "used": "none",
                "success": False
            })
            raise Exception("All fallbacks failed")

    def cached_fallback(self, key: str) -> Optional[Any]:
        """Return cached result as fallback."""
        if key in self.cache:
            age = (datetime.now() - self.cache[key]["timestamp"]).total_seconds()
            # Return cache if less than 5 minutes old
            if age < 300:
                return self.cache[key]["value"]
        return None

    def update_cache(self, key: str, value: Any):
        """Update fallback cache."""
        self.cache[key] = {
            "value": value,
            "timestamp": datetime.now()
        }

    def get_fallback_stats(self) -> Dict:
        """Get fallback usage statistics."""
        if not self.fallback_history:
            return {}

        usage_counts = defaultdict(int)
        for entry in self.fallback_history:
            usage_counts[entry["used"]] += 1

        return {
            "total_calls": len(self.fallback_history),
            "usage_counts": dict(usage_counts),
            "primary_success_rate": usage_counts.get("primary", 0) / len(self.fallback_history),
            "fallback_usage_rate": 1 - (usage_counts.get("primary", 0) / len(self.fallback_history))
        }


# Test fallback mechanisms
async def primary_llm_call(prompt: str):
    """Primary LLM API call."""
    if random.random() < 0.7:
        raise Exception("Primary LLM unavailable")
    return f"Primary response to: {prompt}"


async def secondary_llm_call(prompt: str):
    """Secondary LLM API call."""
    if random.random() < 0.5:
        raise Exception("Secondary LLM unavailable")
    return f"Secondary response to: {prompt}"


async def cached_response(prompt: str):
    """Return cached response."""
    # Simulate cache lookup
    cached_responses = {
        "test": "Cached response for test",
        "hello": "Cached response for hello"
    }
    if prompt in cached_responses:
        return cached_responses[prompt]
    raise Exception("No cached response")


async def default_response(prompt: str):
    """Return default response."""
    return "I'm having trouble processing your request. Please try again later."


async def test_fallback():
    handler = FallbackHandler()

    # Add fallbacks in priority order
    handler.add_fallback("secondary_llm", secondary_llm_call, priority=3)
    handler.add_fallback("cache", cached_response, priority=2)
    handler.add_fallback("default", default_response, priority=1)

    # Test multiple calls
    prompts = ["test", "hello", "analyze this", "translate that"]

    for prompt in prompts:
        try:
            result = await handler.execute_with_fallback(
                primary_llm_call,
                prompt
            )
            print(f"Result for '{prompt}': {result}")
        except Exception as e:
            print(f"Failed for '{prompt}': {e}")

    print(f"\nFallback Stats: {handler.get_fallback_stats()}")

asyncio.run(test_fallback())


# Example 4: Graceful Degradation
print("\n" + "=" * 50)
print("Example 4: Graceful Degradation")
print("=" * 50)


class ServiceLevel(Enum):
    """Service levels for degradation."""
    FULL = "full"
    DEGRADED = "degraded"
    ESSENTIAL = "essential"
    MAINTENANCE = "maintenance"


class GracefulDegradation:
    """Manage graceful degradation of services."""

    def __init__(self):
        self.current_level = ServiceLevel.FULL
        self.feature_availability = {
            ServiceLevel.FULL: [
                "advanced_analysis",
                "real_time_processing",
                "personalization",
                "caching",
                "logging"
            ],
            ServiceLevel.DEGRADED: [
                "basic_analysis",
                "batch_processing",
                "caching",
                "logging"
            ],
            ServiceLevel.ESSENTIAL: [
                "basic_processing",
                "critical_logging"
            ],
            ServiceLevel.MAINTENANCE: []
        }
        self.degradation_history = []
        self.health_metrics = {
            "cpu_usage": 0.5,
            "memory_usage": 0.5,
            "error_rate": 0.01,
            "latency_p95": 1.0
        }

    def update_health_metrics(self, metrics: Dict):
        """Update system health metrics."""
        self.health_metrics.update(metrics)
        self._evaluate_service_level()

    def _evaluate_service_level(self):
        """Evaluate and adjust service level based on health."""
        old_level = self.current_level

        # Determine appropriate level based on metrics
        if (self.health_metrics["error_rate"] > 0.1 or
            self.health_metrics["cpu_usage"] > 0.9):
            new_level = ServiceLevel.ESSENTIAL

        elif (self.health_metrics["error_rate"] > 0.05 or
              self.health_metrics["cpu_usage"] > 0.8 or
              self.health_metrics["latency_p95"] > 3.0):
            new_level = ServiceLevel.DEGRADED

        else:
            new_level = ServiceLevel.FULL

        if new_level != old_level:
            self._change_service_level(new_level)

    def _change_service_level(self, new_level: ServiceLevel):
        """Change service level."""
        old_level = self.current_level
        self.current_level = new_level

        self.degradation_history.append({
            "timestamp": datetime.now(),
            "from_level": old_level.value,
            "to_level": new_level.value,
            "metrics": dict(self.health_metrics)
        })

        print(f"Service level changed: {old_level.value} → {new_level.value}")

    def is_feature_available(self, feature: str) -> bool:
        """Check if a feature is available at current service level."""
        return feature in self.feature_availability[self.current_level]

    async def execute_with_degradation(self, feature: str,
                                      func: Callable,
                                      fallback_func: Callable = None,
                                      *args, **kwargs) -> Any:
        """Execute function with degradation support."""
        if self.is_feature_available(feature):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                print(f"Feature '{feature}' failed: {e}")
                if fallback_func:
                    return await fallback_func(*args, **kwargs)
                raise
        else:
            print(f"Feature '{feature}' not available at {self.current_level.value} level")
            if fallback_func:
                return await fallback_func(*args, **kwargs)
            return None

    def get_degradation_report(self) -> Dict:
        """Get degradation status report."""
        return {
            "current_level": self.current_level.value,
            "available_features": self.feature_availability[self.current_level],
            "health_metrics": self.health_metrics,
            "degradation_events": len(self.degradation_history)
        }


# Test graceful degradation
async def advanced_analysis(data: str):
    """Advanced analysis feature."""
    return f"Advanced analysis of: {data}"


async def basic_analysis(data: str):
    """Basic analysis fallback."""
    return f"Basic analysis of: {data}"


async def test_degradation():
    degradation = GracefulDegradation()

    # Test at full service
    print("Testing at FULL service level:")
    result = await degradation.execute_with_degradation(
        "advanced_analysis",
        advanced_analysis,
        basic_analysis,
        "test data"
    )
    print(f"  Result: {result}")

    # Simulate high load
    print("\nSimulating high load...")
    degradation.update_health_metrics({
        "cpu_usage": 0.85,
        "error_rate": 0.06
    })

    # Test at degraded level
    print("Testing at DEGRADED service level:")
    result = await degradation.execute_with_degradation(
        "advanced_analysis",
        advanced_analysis,
        basic_analysis,
        "test data"
    )
    print(f"  Result: {result}")

    print(f"\nDegradation Report: {degradation.get_degradation_report()}")

asyncio.run(test_degradation())


# Example 5: Error Aggregation and Analysis
print("\n" + "=" * 50)
print("Example 5: Error Aggregation and Analysis")
print("=" * 50)


@dataclass
class ErrorEvent:
    """Represents an error event."""
    timestamp: datetime
    error_type: str
    error_message: str
    operation: str
    user_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class ErrorAggregator:
    """Aggregate and analyze errors."""

    def __init__(self, window_size: int = 1000):
        self.errors = deque(maxlen=window_size)
        self.error_patterns = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.user_errors = defaultdict(list)

    def record_error(self, error: Exception, operation: str,
                    user_id: str = None, **metadata):
        """Record an error event."""
        event = ErrorEvent(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            operation=operation,
            user_id=user_id,
            metadata=metadata
        )

        self.errors.append(event)
        self.error_counts[event.error_type] += 1

        if user_id:
            self.user_errors[user_id].append(event)

        # Detect patterns
        self._detect_pattern(event)

    def _detect_pattern(self, event: ErrorEvent):
        """Detect error patterns."""
        # Group similar errors
        pattern_key = f"{event.error_type}:{event.operation}"
        self.error_patterns[pattern_key].append(event)

    def get_error_summary(self, time_window: timedelta = None) -> Dict:
        """Get error summary statistics."""
        if time_window:
            cutoff = datetime.now() - time_window
            recent_errors = [e for e in self.errors if e.timestamp > cutoff]
        else:
            recent_errors = list(self.errors)

        if not recent_errors:
            return {"total_errors": 0}

        # Calculate statistics
        error_types = defaultdict(int)
        operation_errors = defaultdict(int)

        for error in recent_errors:
            error_types[error.error_type] += 1
            operation_errors[error.operation] += 1

        return {
            "total_errors": len(recent_errors),
            "error_types": dict(error_types),
            "operation_errors": dict(operation_errors),
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0],
            "most_affected_operation": max(operation_errors.items(), key=lambda x: x[1])[0]
        }

    def get_error_trends(self, bucket_minutes: int = 5) -> List[Dict]:
        """Get error trends over time."""
        if not self.errors:
            return []

        trends = defaultdict(lambda: {"count": 0, "types": set()})

        for error in self.errors:
            # Round timestamp to bucket
            bucket_time = error.timestamp.replace(
                minute=error.timestamp.minute // bucket_minutes * bucket_minutes,
                second=0,
                microsecond=0
            )

            trends[bucket_time]["count"] += 1
            trends[bucket_time]["types"].add(error.error_type)

        return [
            {
                "timestamp": ts.isoformat(),
                "count": data["count"],
                "unique_types": len(data["types"])
            }
            for ts, data in sorted(trends.items())
        ]

    def get_user_error_report(self, user_id: str) -> Dict:
        """Get error report for specific user."""
        if user_id not in self.user_errors:
            return {"user_id": user_id, "errors": []}

        user_error_list = self.user_errors[user_id]

        return {
            "user_id": user_id,
            "total_errors": len(user_error_list),
            "recent_errors": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "type": e.error_type,
                    "operation": e.operation
                }
                for e in user_error_list[-5:]  # Last 5 errors
            ]
        }

    def suggest_fixes(self) -> List[str]:
        """Suggest fixes based on error patterns."""
        suggestions = []

        # Check for rate limiting errors
        if self.error_counts.get("RateLimitError", 0) > 10:
            suggestions.append(
                "Implement exponential backoff for rate limit errors"
            )

        # Check for timeout patterns
        timeout_errors = sum(
            1 for e in self.errors
            if "timeout" in e.error_message.lower()
        )
        if timeout_errors > 5:
            suggestions.append(
                "Increase timeout values or optimize slow operations"
            )

        # Check for connection errors
        if self.error_counts.get("ConnectionError", 0) > 5:
            suggestions.append(
                "Implement connection pooling and retry mechanisms"
            )

        # Check for user-specific issues
        for user_id, errors in self.user_errors.items():
            if len(errors) > 10:
                suggestions.append(
                    f"User {user_id} experiencing high error rate - investigate"
                )

        return suggestions


# Test error aggregation
aggregator = ErrorAggregator()

# Simulate various errors
error_scenarios = [
    (ConnectionError("Connection timeout"), "api_call", "user_1"),
    (ValueError("Invalid input format"), "validation", "user_2"),
    (ConnectionError("Connection refused"), "api_call", "user_1"),
    (RuntimeError("Rate limit exceeded"), "api_call", "user_3"),
    (ConnectionError("Connection timeout"), "api_call", "user_2"),
    (RuntimeError("Rate limit exceeded"), "api_call", "user_1"),
]

for error, operation, user_id in error_scenarios * 3:  # Repeat for pattern detection
    aggregator.record_error(error, operation, user_id)

# Get error analysis
print("Error Summary:")
summary = aggregator.get_error_summary(timedelta(hours=1))
for key, value in summary.items():
    print(f"  {key}: {value}")

print("\nError Trends:")
trends = aggregator.get_error_trends(bucket_minutes=5)
for trend in trends[-3:]:  # Show last 3 buckets
    print(f"  {trend}")

print("\nSuggested Fixes:")
for suggestion in aggregator.suggest_fixes():
    print(f"  - {suggestion}")


# Example 6: Health Check System
print("\n" + "=" * 50)
print("Example 6: Health Check System")
print("=" * 50)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Represents a health check."""
    name: str
    check_fn: Callable
    critical: bool = False
    timeout: float = 5.0


class HealthCheckSystem:
    """Comprehensive health check system."""

    def __init__(self):
        self.health_checks = []
        self.check_results = {}
        self.check_history = deque(maxlen=100)

    def register_check(self, check: HealthCheck):
        """Register a health check."""
        self.health_checks.append(check)

    async def run_checks(self) -> Dict:
        """Run all health checks."""
        results = {}
        overall_status = HealthStatus.HEALTHY

        for check in self.health_checks:
            try:
                # Run check with timeout
                result = await asyncio.wait_for(
                    check.check_fn(),
                    timeout=check.timeout
                )

                results[check.name] = {
                    "status": "healthy" if result else "unhealthy",
                    "timestamp": datetime.now().isoformat(),
                    "critical": check.critical
                }

                if not result:
                    if check.critical:
                        overall_status = HealthStatus.UNHEALTHY
                    elif overall_status != HealthStatus.UNHEALTHY:
                        overall_status = HealthStatus.DEGRADED

            except asyncio.TimeoutError:
                results[check.name] = {
                    "status": "timeout",
                    "timestamp": datetime.now().isoformat(),
                    "critical": check.critical
                }
                if check.critical:
                    overall_status = HealthStatus.UNHEALTHY

            except Exception as e:
                results[check.name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "critical": check.critical
                }
                if check.critical:
                    overall_status = HealthStatus.UNHEALTHY

        self.check_results = results
        self.check_history.append({
            "timestamp": datetime.now(),
            "status": overall_status,
            "checks": len(results),
            "failures": sum(1 for r in results.values() if r["status"] != "healthy")
        })

        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": results
        }

    def get_health_trends(self) -> Dict:
        """Get health check trends."""
        if not self.check_history:
            return {}

        total_checks = len(self.check_history)
        healthy_checks = sum(
            1 for h in self.check_history
            if h["status"] == HealthStatus.HEALTHY
        )

        return {
            "total_checks": total_checks,
            "healthy_percentage": healthy_checks / total_checks,
            "recent_failures": [
                h for h in list(self.check_history)[-10:]
                if h["failures"] > 0
            ]
        }


# Test health checks
async def check_api_connectivity():
    """Check API connectivity."""
    # Simulate API check
    return random.random() < 0.8  # 80% healthy


async def check_database():
    """Check database connection."""
    # Simulate DB check
    return random.random() < 0.9  # 90% healthy


async def check_cache():
    """Check cache service."""
    # Simulate cache check
    return random.random() < 0.95  # 95% healthy


async def check_disk_space():
    """Check available disk space."""
    # Simulate disk check
    return random.random() < 0.99  # 99% healthy


async def test_health_checks():
    health_system = HealthCheckSystem()

    # Register health checks
    health_system.register_check(
        HealthCheck("api", check_api_connectivity, critical=True)
    )
    health_system.register_check(
        HealthCheck("database", check_database, critical=True)
    )
    health_system.register_check(
        HealthCheck("cache", check_cache, critical=False)
    )
    health_system.register_check(
        HealthCheck("disk", check_disk_space, critical=False)
    )

    # Run health checks
    print("Running health checks...")
    results = await health_system.run_checks()

    print(f"Overall Status: {results['status']}")
    print("Individual Checks:")
    for name, result in results["checks"].items():
        print(f"  {name}: {result['status']}")

    # Run multiple times for trends
    for _ in range(5):
        await asyncio.sleep(0.5)
        await health_system.run_checks()

    trends = health_system.get_health_trends()
    print(f"\nHealth Trends: {trends}")

asyncio.run(test_health_checks())


# Example 7: Complete Resilience Framework
print("\n" + "=" * 50)
print("Example 7: Complete Resilience Framework")
print("=" * 50)


class ResilienceFramework:
    """Complete resilience framework for production systems."""

    def __init__(self, service_name: str):
        self.service_name = service_name

        # Initialize components
        self.retry_handler = RetryHandler(max_retries=3)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5)
        self.fallback_handler = FallbackHandler()
        self.degradation = GracefulDegradation()
        self.error_aggregator = ErrorAggregator()
        self.health_system = HealthCheckSystem()

        # Metrics
        self.request_count = 0
        self.success_count = 0
        self.fallback_count = 0

    async def execute_resilient_request(self,
                                       operation: str,
                                       primary_func: Callable,
                                       fallback_func: Callable = None,
                                       user_id: str = None,
                                       **kwargs) -> Dict:
        """Execute request with full resilience."""
        self.request_count += 1
        start_time = time.time()
        result = None
        used_fallback = False

        try:
            # Check circuit breaker
            if self.circuit_breaker.state == CircuitState.OPEN:
                print(f"Circuit open - using fallback for {operation}")
                if fallback_func:
                    result = await fallback_func(**kwargs)
                    used_fallback = True
                else:
                    raise Exception("Circuit open and no fallback available")

            else:
                # Try primary with retry
                try:
                    result = await self.retry_handler.execute_with_retry(
                        primary_func,
                        **kwargs
                    )
                    self.circuit_breaker._on_success()
                    self.success_count += 1

                except Exception as e:
                    self.circuit_breaker._on_failure()
                    self.error_aggregator.record_error(
                        e, operation, user_id
                    )

                    # Try fallback
                    if fallback_func:
                        result = await fallback_func(**kwargs)
                        used_fallback = True
                        self.fallback_count += 1
                    else:
                        raise

            # Record metrics
            duration = time.time() - start_time

            return {
                "success": True,
                "result": result,
                "duration": duration,
                "used_fallback": used_fallback,
                "circuit_state": self.circuit_breaker.state.value
            }

        except Exception as e:
            # Record error
            self.error_aggregator.record_error(
                e, operation, user_id
            )

            duration = time.time() - start_time

            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "circuit_state": self.circuit_breaker.state.value
            }

    def get_resilience_metrics(self) -> Dict:
        """Get comprehensive resilience metrics."""
        success_rate = self.success_count / max(self.request_count, 1)
        fallback_rate = self.fallback_count / max(self.request_count, 1)

        return {
            "service": self.service_name,
            "total_requests": self.request_count,
            "success_rate": f"{success_rate:.1%}",
            "fallback_rate": f"{fallback_rate:.1%}",
            "circuit_state": self.circuit_breaker.state.value,
            "service_level": self.degradation.current_level.value,
            "retry_stats": self.retry_handler.get_retry_stats(),
            "error_summary": self.error_aggregator.get_error_summary(
                timedelta(hours=1)
            ),
            "suggestions": self.error_aggregator.suggest_fixes()
        }


# Test complete resilience framework
async def llm_api_call(prompt: str):
    """Simulate LLM API call."""
    if random.random() < 0.6:  # 60% failure rate
        raise ConnectionError("LLM API unavailable")
    return f"LLM response to: {prompt}"


async def cached_llm_response(prompt: str):
    """Fallback cached response."""
    return f"Cached response for: {prompt}"


async def test_resilience_framework():
    framework = ResilienceFramework("llm_service")

    # Register health checks
    framework.health_system.register_check(
        HealthCheck("llm_api", check_api_connectivity, critical=True)
    )

    # Run multiple requests
    prompts = [
        "Analyze this text",
        "Translate to Spanish",
        "Summarize the document",
        "Generate code",
        "Answer question"
    ]

    print("Testing Resilience Framework:")
    for i, prompt in enumerate(prompts * 2):  # Test 10 requests
        result = await framework.execute_resilient_request(
            operation="llm_call",
            primary_func=llm_api_call,
            fallback_func=cached_llm_response,
            user_id=f"user_{i % 3}",
            prompt=prompt
        )

        status = "✓" if result["success"] else "✗"
        fallback = " (fallback)" if result.get("used_fallback") else ""
        print(f"  Request {i+1}: {status}{fallback} - {result['circuit_state']}")

        # Simulate some recovery
        if i == 5:
            print("  [Waiting for partial recovery...]")
            await asyncio.sleep(0.5)

    # Get final metrics
    print("\nResilience Metrics:")
    metrics = framework.get_resilience_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        elif isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")

asyncio.run(test_resilience_framework())


print("\n" + "=" * 50)
print("Error Resilience Examples Complete!")
print("=" * 50)
print("""
These examples demonstrated:
1. Advanced retry strategies
2. Circuit breaker pattern
3. Fallback mechanisms
4. Graceful degradation
5. Error aggregation and analysis
6. Health check systems
7. Complete resilience framework

Key concepts for production resilience:
- Implement intelligent retry logic
- Use circuit breakers to prevent cascading failures
- Provide fallback mechanisms
- Support graceful degradation
- Aggregate and analyze errors
- Monitor system health
- Build comprehensive resilience
""")