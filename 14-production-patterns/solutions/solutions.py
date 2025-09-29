"""
Module 14: Production Patterns
Solutions

Complete implementations for all production pattern exercises.
"""

import os
import time
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import random
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()


# Solution 1: Load Balancer with Health Checks
print("=" * 50)
print("Solution 1: Load Balancer with Health Checks")
print("=" * 50)


@dataclass
class Endpoint:
    """Represents an LLM endpoint."""
    name: str
    url: str
    weight: int = 1
    healthy: bool = True
    current_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_latency: float = 0.0
    last_health_check: Optional[datetime] = None


class LoadBalancer:
    """Production-ready load balancer with health monitoring."""

    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.endpoints = []
        self.current_index = 0
        self.request_queue = deque(maxlen=1000)
        self.health_check_interval = 30  # seconds
        self.max_connections_per_endpoint = 10

    def add_endpoint(self, endpoint: Endpoint):
        """Add an endpoint to the load balancer."""
        self.endpoints.append(endpoint)
        print(f"Added endpoint: {endpoint.name}")

    def select_endpoint(self) -> Optional[Endpoint]:
        """Select an endpoint based on the strategy."""
        # Filter healthy endpoints
        healthy_endpoints = [e for e in self.endpoints if e.healthy]

        if not healthy_endpoints:
            return None

        if self.strategy == "round_robin":
            # Simple round-robin
            endpoint = healthy_endpoints[self.current_index % len(healthy_endpoints)]
            self.current_index += 1
            return endpoint

        elif self.strategy == "least_connections":
            # Select endpoint with fewest active connections
            return min(healthy_endpoints, key=lambda e: e.current_connections)

        elif self.strategy == "weighted":
            # Weighted random selection
            weights = [e.weight for e in healthy_endpoints]
            total_weight = sum(weights)
            r = random.uniform(0, total_weight)

            cumulative = 0
            for endpoint, weight in zip(healthy_endpoints, weights):
                cumulative += weight
                if r <= cumulative:
                    return endpoint

            return healthy_endpoints[-1]

        else:
            return healthy_endpoints[0]

    async def health_check(self, endpoint: Endpoint) -> bool:
        """Check health of an endpoint."""
        try:
            # Simulate health check ping
            start_time = time.time()

            # Simulate API call
            await asyncio.sleep(random.uniform(0.01, 0.05))

            # Random failure for demo
            if random.random() < 0.1:  # 10% failure rate
                raise Exception("Health check failed")

            # Update latency
            latency = time.time() - start_time
            endpoint.avg_latency = (endpoint.avg_latency * 0.9 + latency * 0.1)

            endpoint.healthy = True
            endpoint.last_health_check = datetime.now()
            return True

        except Exception as e:
            print(f"Health check failed for {endpoint.name}: {e}")
            endpoint.healthy = False
            endpoint.last_health_check = datetime.now()
            return False

    async def execute_request(self, request: Dict) -> Dict:
        """Execute request through load balancer."""
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            endpoint = self.select_endpoint()

            if not endpoint:
                await self._queue_request(request)
                return {"error": "All endpoints unhealthy", "queued": True}

            # Check connection limit
            if endpoint.current_connections >= self.max_connections_per_endpoint:
                continue

            try:
                # Track connection
                endpoint.current_connections += 1
                endpoint.total_requests += 1

                # Simulate API call
                start_time = time.time()
                await asyncio.sleep(random.uniform(0.1, 0.5))

                # Random failure for demo
                if random.random() < 0.2:  # 20% failure rate
                    raise Exception("Request failed")

                # Update metrics
                latency = time.time() - start_time
                endpoint.avg_latency = (endpoint.avg_latency * 0.95 + latency * 0.05)

                return {
                    "success": True,
                    "endpoint": endpoint.name,
                    "response": f"Response from {endpoint.name}",
                    "latency": latency
                }

            except Exception as e:
                last_error = e
                endpoint.total_errors += 1

                # Mark unhealthy if too many errors
                if endpoint.total_errors > 5:
                    endpoint.healthy = False

            finally:
                endpoint.current_connections -= 1

        return {"error": str(last_error), "attempts": max_retries}

    async def _queue_request(self, request: Dict):
        """Queue request when all endpoints are busy."""
        self.request_queue.append({
            "request": request,
            "timestamp": datetime.now()
        })

    async def process_queue(self):
        """Process queued requests."""
        while self.request_queue:
            item = self.request_queue.popleft()
            result = await self.execute_request(item["request"])
            print(f"Processed queued request: {result}")

    async def monitor_health(self):
        """Continuously monitor endpoint health."""
        while True:
            for endpoint in self.endpoints:
                # Check if health check is due
                if endpoint.last_health_check:
                    time_since_check = (datetime.now() - endpoint.last_health_check).seconds
                    if time_since_check < self.health_check_interval:
                        continue

                await self.health_check(endpoint)

            await asyncio.sleep(5)

    def get_metrics(self) -> Dict:
        """Get load balancer metrics."""
        total_requests = sum(e.total_requests for e in self.endpoints)
        total_errors = sum(e.total_errors for e in self.endpoints)

        return {
            "total_endpoints": len(self.endpoints),
            "healthy_endpoints": sum(1 for e in self.endpoints if e.healthy),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_requests, 1),
            "queued_requests": len(self.request_queue),
            "endpoints": [
                {
                    "name": e.name,
                    "healthy": e.healthy,
                    "connections": e.current_connections,
                    "requests": e.total_requests,
                    "errors": e.total_errors,
                    "avg_latency": f"{e.avg_latency:.3f}s"
                }
                for e in self.endpoints
            ]
        }


# Test load balancer
async def test_load_balancer():
    balancer = LoadBalancer("least_connections")
    balancer.add_endpoint(Endpoint("gpt4-1", "https://api1.example.com", weight=2))
    balancer.add_endpoint(Endpoint("gpt4-2", "https://api2.example.com", weight=1))
    balancer.add_endpoint(Endpoint("gpt4-3", "https://api3.example.com", weight=1))

    # Run some requests
    results = []
    for i in range(10):
        result = await balancer.execute_request({"prompt": f"test_{i}"})
        results.append(result)

    print("\nRequest Results:")
    for i, result in enumerate(results):
        status = "✓" if result.get("success") else "✗"
        print(f"  Request {i+1}: {status} - {result.get('endpoint', 'failed')}")

    print(f"\nMetrics: {balancer.get_metrics()}")

asyncio.run(test_load_balancer())


# Solution 2: Monitoring Dashboard
print("\n" + "=" * 50)
print("Solution 2: Monitoring Dashboard")
print("=" * 50)


class MonitoringDashboard:
    """Comprehensive monitoring dashboard for LLM applications."""

    def __init__(self, refresh_interval: int = 60):
        self.refresh_interval = refresh_interval
        self.metrics = defaultdict(list)
        self.alerts = []
        self.anomaly_baseline = {}
        self.cost_tracker = {
            "gpt-4": {"rate": 0.03, "usage": 0},
            "gpt-3.5-turbo": {"rate": 0.001, "usage": 0}
        }

    def record_metric(self, name: str, value: float,
                     tags: Dict[str, str] = None):
        """Record a metric value with timestamp."""
        self.metrics[name].append({
            "value": value,
            "timestamp": datetime.now(),
            "tags": tags or {}
        })

        # Maintain window of last 1000 points
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]

        # Check for anomalies
        if self.detect_anomaly(name):
            self.generate_alert(f"Anomaly detected in {name}", "warning")

    def calculate_rate(self, metric_name: str,
                       window_seconds: int = 60) -> float:
        """Calculate rate for a metric over time window."""
        if metric_name not in self.metrics:
            return 0.0

        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        recent = [m for m in self.metrics[metric_name]
                 if m["timestamp"] > cutoff]

        if not recent:
            return 0.0

        # Count events in window
        return len(recent) / window_seconds

    def detect_anomaly(self, metric_name: str,
                       threshold_factor: float = 2.0) -> bool:
        """Detect anomalies using statistical methods."""
        if metric_name not in self.metrics:
            return False

        values = [m["value"] for m in self.metrics[metric_name]]

        if len(values) < 20:  # Need minimum data
            return False

        # Calculate baseline from older values
        if len(values) > 50:
            baseline_values = values[-50:-10]  # Use middle values
            recent_values = values[-10:]  # Recent values
        else:
            baseline_values = values[:-10]
            recent_values = values[-10:]

        if not baseline_values or not recent_values:
            return False

        baseline_mean = np.mean(baseline_values)
        baseline_std = np.std(baseline_values)
        recent_mean = np.mean(recent_values)

        # Check if recent mean is outside threshold
        if baseline_std > 0:
            z_score = abs(recent_mean - baseline_mean) / baseline_std
            return z_score > threshold_factor

        return abs(recent_mean - baseline_mean) > baseline_mean * 0.5

    def generate_alert(self, message: str, severity: str = "warning"):
        """Generate and store an alert."""
        alert = {
            "id": len(self.alerts) + 1,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now(),
            "acknowledged": False
        }
        self.alerts.append(alert)
        print(f"🚨 Alert: {message} ({severity})")

    def get_dashboard_data(self) -> Dict:
        """Compile comprehensive dashboard data."""
        # Calculate current metrics
        request_rate = self.calculate_rate("requests", 60)
        error_rate = self.calculate_rate("errors", 60)
        avg_latency = self._calculate_average("latency", 60)

        # Get token usage
        token_metrics = self._get_token_metrics()

        # Get active alerts
        active_alerts = [a for a in self.alerts if not a["acknowledged"]]

        # Identify top issues
        top_issues = self._identify_top_issues()

        return {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": {
                "request_rate": f"{request_rate:.2f} req/s",
                "error_rate": f"{error_rate:.2f} err/s",
                "avg_latency": f"{avg_latency:.3f}s",
                "active_alerts": len(active_alerts)
            },
            "token_usage": token_metrics,
            "alerts": active_alerts[-5:],  # Latest 5 alerts
            "top_issues": top_issues,
            "trends": self._calculate_trends()
        }

    def _calculate_average(self, metric_name: str,
                          window_seconds: int) -> float:
        """Calculate average value over time window."""
        if metric_name not in self.metrics:
            return 0.0

        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        recent = [m["value"] for m in self.metrics[metric_name]
                 if m["timestamp"] > cutoff]

        return np.mean(recent) if recent else 0.0

    def _get_token_metrics(self) -> Dict:
        """Get token usage metrics."""
        total_tokens = sum(
            len(m["tags"].get("tokens", []))
            for metrics in self.metrics.values()
            for m in metrics
        )

        return {
            "total_tokens_processed": total_tokens,
            "tokens_per_minute": total_tokens / max(len(self.metrics.get("requests", [])), 1)
        }

    def _identify_top_issues(self) -> List[str]:
        """Identify top issues from metrics."""
        issues = []

        # High error rate
        error_rate = self.calculate_rate("errors", 60)
        if error_rate > 0.1:
            issues.append(f"High error rate: {error_rate:.2f} errors/s")

        # High latency
        avg_latency = self._calculate_average("latency", 60)
        if avg_latency > 2.0:
            issues.append(f"High latency: {avg_latency:.2f}s average")

        # Anomalies
        for metric_name in self.metrics.keys():
            if self.detect_anomaly(metric_name, 1.5):
                issues.append(f"Anomaly in {metric_name}")

        return issues[:3]  # Top 3 issues

    def _calculate_trends(self) -> Dict:
        """Calculate metric trends."""
        trends = {}

        for metric_name in ["requests", "errors", "latency"]:
            if metric_name not in self.metrics:
                continue

            # Compare last hour to previous hour
            now = datetime.now()
            last_hour = [m["value"] for m in self.metrics[metric_name]
                        if now - timedelta(hours=1) < m["timestamp"] <= now]
            prev_hour = [m["value"] for m in self.metrics[metric_name]
                        if now - timedelta(hours=2) < m["timestamp"] <= now - timedelta(hours=1)]

            if last_hour and prev_hour:
                last_avg = np.mean(last_hour)
                prev_avg = np.mean(prev_hour)
                change = ((last_avg - prev_avg) / prev_avg) * 100 if prev_avg > 0 else 0
                trends[metric_name] = f"{'+' if change > 0 else ''}{change:.1f}%"

        return trends

    def get_cost_analysis(self) -> Dict:
        """Analyze costs based on usage."""
        total_cost = 0
        model_costs = {}

        for model, data in self.cost_tracker.items():
            cost = data["usage"] * data["rate"] / 1000  # Cost per 1K tokens
            model_costs[model] = cost
            total_cost += cost

        # Project monthly cost
        days_tracked = 1  # Simplified for demo
        monthly_projection = total_cost * 30 / days_tracked

        return {
            "current_cost": f"${total_cost:.2f}",
            "model_breakdown": {k: f"${v:.2f}" for k, v in model_costs.items()},
            "monthly_projection": f"${monthly_projection:.2f}",
            "cost_per_request": f"${total_cost / max(len(self.metrics.get('requests', [])), 1):.4f}"
        }


# Test monitoring dashboard
dashboard = MonitoringDashboard()

# Simulate metrics
for i in range(100):
    dashboard.record_metric("requests", 1)
    dashboard.record_metric("tokens", random.randint(100, 500), {"model": "gpt-4"})
    dashboard.record_metric("latency", random.uniform(0.5, 2.0))

    # Occasionally record errors
    if random.random() < 0.1:
        dashboard.record_metric("errors", 1)

    # Simulate anomaly
    if i == 80:
        dashboard.record_metric("latency", 5.0)  # Spike

# Update cost tracker
dashboard.cost_tracker["gpt-4"]["usage"] = 10000

print("\nDashboard Data:")
data = dashboard.get_dashboard_data()
for key, value in data.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    else:
        print(f"{key}: {value}")

print("\nCost Analysis:")
costs = dashboard.get_cost_analysis()
for key, value in costs.items():
    print(f"  {key}: {value}")


# Solution 3: Resilient API Wrapper
print("\n" + "=" * 50)
print("Solution 3: Resilient API Wrapper")
print("=" * 50)


class ResilientAPIWrapper:
    """Resilient wrapper for LLM API calls."""

    def __init__(self, primary_endpoint: str, max_retries: int = 3):
        self.primary_endpoint = primary_endpoint
        self.max_retries = max_retries

        # Circuit breaker settings
        self.circuit_open = False
        self.failure_count = 0
        self.failure_threshold = 5
        self.circuit_open_until = None
        self.recovery_timeout = 60  # seconds

        # Retry settings
        self.base_delay = 1.0
        self.max_delay = 60.0

    async def call_with_retry(self, request: Dict) -> Dict:
        """Call API with exponential backoff retry."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Validate request first
                if not self.validate_request(request):
                    raise ValueError("Invalid request format")

                # Simulate API call
                await asyncio.sleep(random.uniform(0.1, 0.3))

                # Random failure for demo
                if random.random() < 0.4:  # 40% failure rate
                    raise ConnectionError("API unavailable")

                response = {
                    "success": True,
                    "data": f"Response to: {request.get('prompt', '')}",
                    "tokens": random.randint(100, 300)
                }

                # Validate response
                if self.validate_response(response):
                    # Reset circuit breaker on success
                    self.failure_count = 0
                    return response
                else:
                    raise ValueError("Invalid response format")

            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (2 ** attempt) + random.uniform(0, 1),
                        self.max_delay
                    )
                    print(f"Retry {attempt + 1}/{self.max_retries} after {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)

        # All retries failed
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self._open_circuit()

        raise last_exception

    def check_circuit(self) -> bool:
        """Check if circuit breaker should trip."""
        if self.circuit_open:
            # Check if recovery timeout has passed
            if self.circuit_open_until and datetime.now() > self.circuit_open_until:
                print("Circuit breaker: Attempting recovery")
                self.circuit_open = False
                self.failure_count = 0
                return False
            return True
        return False

    def _open_circuit(self):
        """Open the circuit breaker."""
        self.circuit_open = True
        self.circuit_open_until = datetime.now() + timedelta(seconds=self.recovery_timeout)
        print(f"Circuit breaker OPEN - will retry at {self.circuit_open_until}")

    async def execute_with_fallback(self, request: Dict,
                                   fallbacks: List[Callable] = None) -> Dict:
        """Execute request with fallback chain."""
        # Check circuit breaker
        if self.check_circuit():
            print("Circuit breaker is OPEN - trying fallbacks")
            if fallbacks:
                return await self._try_fallbacks(request, fallbacks)
            raise Exception("Circuit breaker open and no fallbacks available")

        # Try primary endpoint
        try:
            return await self.call_with_retry(request)
        except Exception as primary_error:
            print(f"Primary failed: {primary_error}")

            # Try fallbacks
            if fallbacks:
                return await self._try_fallbacks(request, fallbacks)

            raise primary_error

    async def _try_fallbacks(self, request: Dict,
                           fallbacks: List[Callable]) -> Dict:
        """Try fallback functions in order."""
        last_error = None

        for i, fallback in enumerate(fallbacks):
            try:
                print(f"Trying fallback {i + 1}/{len(fallbacks)}")
                result = await fallback(request)
                print(f"Fallback {i + 1} succeeded")
                return result
            except Exception as e:
                print(f"Fallback {i + 1} failed: {e}")
                last_error = e
                continue

        raise Exception(f"All fallbacks failed. Last error: {last_error}")

    def validate_request(self, request: Dict) -> bool:
        """Validate request format and parameters."""
        # Check required fields
        if "prompt" not in request:
            return False

        # Check prompt length
        if len(request["prompt"]) > 10000:
            return False

        # Check optional parameters
        if "max_tokens" in request:
            if not (1 <= request["max_tokens"] <= 4000):
                return False

        if "temperature" in request:
            if not (0.0 <= request["temperature"] <= 2.0):
                return False

        return True

    def validate_response(self, response: Dict) -> bool:
        """Validate response from API."""
        # Check required fields
        if "success" not in response:
            return False

        if response["success"] and "data" not in response:
            return False

        # Check for error responses
        if not response["success"] and "error" not in response:
            return False

        return True


# Test resilient API wrapper
async def fallback_cache(request: Dict):
    """Cached response fallback."""
    return {
        "success": True,
        "data": f"Cached response for: {request['prompt']}",
        "cached": True
    }


async def fallback_simple_model(request: Dict):
    """Simpler model fallback."""
    return {
        "success": True,
        "data": f"Simple model response for: {request['prompt']}",
        "model": "simple"
    }


async def test_resilient_wrapper():
    wrapper = ResilientAPIWrapper("https://api.example.com")

    # Test requests
    requests = [
        {"prompt": "Test prompt 1", "max_tokens": 100},
        {"prompt": "Test prompt 2", "temperature": 0.7},
        {"prompt": "Test prompt 3"},
    ]

    for request in requests:
        try:
            response = await wrapper.execute_with_fallback(
                request,
                [fallback_cache, fallback_simple_model]
            )
            print(f"Success: {response}")
        except Exception as e:
            print(f"Failed: {e}")

asyncio.run(test_resilient_wrapper())


# Solution 4: Deployment Pipeline
print("\n" + "=" * 50)
print("Solution 4: Deployment Pipeline")
print("=" * 50)


class DeploymentStage(Enum):
    """Deployment stages."""
    DEV = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentPipeline:
    """Automated deployment pipeline with testing and rollback."""

    def __init__(self):
        self.stages = [DeploymentStage.DEV, DeploymentStage.STAGING,
                      DeploymentStage.PRODUCTION]
        self.current_versions = {
            DeploymentStage.DEV: None,
            DeploymentStage.STAGING: None,
            DeploymentStage.PRODUCTION: None
        }
        self.deployment_history = []
        self.test_suites = {
            DeploymentStage.DEV: ["unit", "integration"],
            DeploymentStage.STAGING: ["integration", "performance", "security"],
            DeploymentStage.PRODUCTION: ["smoke", "canary"]
        }

    async def validate_deployment(self, version: str,
                                 stage: DeploymentStage) -> bool:
        """Validate deployment readiness."""
        print(f"Validating {version} for {stage.value}")

        # Check version format
        if not version.startswith("v"):
            print("Invalid version format")
            return False

        # Check dependencies
        dependencies_ok = await self._check_dependencies(version)
        if not dependencies_ok:
            print("Dependency check failed")
            return False

        # Verify configuration
        config_ok = await self._verify_configuration(stage)
        if not config_ok:
            print("Configuration verification failed")
            return False

        return True

    async def _check_dependencies(self, version: str) -> bool:
        """Check deployment dependencies."""
        # Simulate dependency check
        await asyncio.sleep(0.1)
        return random.random() < 0.9  # 90% success

    async def _verify_configuration(self, stage: DeploymentStage) -> bool:
        """Verify stage configuration."""
        # Simulate config verification
        await asyncio.sleep(0.1)
        return random.random() < 0.95  # 95% success

    async def run_tests(self, stage: DeploymentStage) -> Dict:
        """Run tests for a deployment stage."""
        print(f"Running tests for {stage.value}")
        test_results = {
            "stage": stage.value,
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "passed": True
        }

        for test_type in self.test_suites.get(stage, []):
            # Simulate test execution
            await asyncio.sleep(0.2)
            passed = random.random() < 0.85  # 85% pass rate

            test_results["tests"][test_type] = {
                "passed": passed,
                "duration": random.uniform(1, 5),
                "message": "Test passed" if passed else "Test failed"
            }

            if not passed:
                test_results["passed"] = False

        return test_results

    async def deploy_to_stage(self, version: str,
                             stage: DeploymentStage) -> bool:
        """Deploy version to a stage."""
        print(f"Deploying {version} to {stage.value}")

        # Validate first
        if not await self.validate_deployment(version, stage):
            return False

        # Store previous version for rollback
        previous_version = self.current_versions[stage]

        try:
            # Simulate deployment
            print(f"  Deploying application...")
            await asyncio.sleep(0.5)

            # Update version
            self.current_versions[stage] = version

            # Run smoke tests
            print(f"  Running smoke tests...")
            smoke_test_passed = random.random() < 0.9  # 90% success

            if not smoke_test_passed:
                raise Exception("Smoke tests failed")

            # Update routing (for production)
            if stage == DeploymentStage.PRODUCTION:
                print(f"  Updating routing...")
                await asyncio.sleep(0.2)

            # Record deployment
            self.deployment_history.append({
                "version": version,
                "stage": stage.value,
                "timestamp": datetime.now(),
                "status": "success"
            })

            print(f"✓ Successfully deployed {version} to {stage.value}")
            return True

        except Exception as e:
            print(f"✗ Deployment failed: {e}")

            # Rollback
            if previous_version:
                await self.rollback(stage)

            self.deployment_history.append({
                "version": version,
                "stage": stage.value,
                "timestamp": datetime.now(),
                "status": "failed",
                "error": str(e)
            })

            return False

    async def rollback(self, stage: DeploymentStage):
        """Rollback deployment in a stage."""
        # Get previous version from history
        previous_deployments = [
            d for d in self.deployment_history
            if d["stage"] == stage.value and d["status"] == "success"
        ]

        if len(previous_deployments) < 2:
            print("No previous version to rollback to")
            return

        previous_version = previous_deployments[-2]["version"]
        print(f"Rolling back {stage.value} to {previous_version}")

        # Simulate rollback
        await asyncio.sleep(0.3)
        self.current_versions[stage] = previous_version

        print(f"✓ Rolled back to {previous_version}")

    async def promote_version(self, version: str,
                             from_stage: DeploymentStage,
                             to_stage: DeploymentStage) -> bool:
        """Promote version between stages."""
        print(f"Promoting {version} from {from_stage.value} to {to_stage.value}")

        # Verify version is in source stage
        if self.current_versions[from_stage] != version:
            print(f"Version {version} not found in {from_stage.value}")
            return False

        # Run tests in source stage
        test_results = await self.run_tests(from_stage)
        if not test_results["passed"]:
            print(f"Tests failed in {from_stage.value}")
            return False

        # Deploy to target stage
        success = await self.deploy_to_stage(version, to_stage)

        if success:
            print(f"✓ Promoted {version} to {to_stage.value}")

            # Monitor for issues (simplified)
            await asyncio.sleep(0.5)
            issues_detected = random.random() < 0.1  # 10% issue rate

            if issues_detected:
                print(f"Issues detected in {to_stage.value}, rolling back")
                await self.rollback(to_stage)
                return False

        return success

    def get_deployment_status(self) -> Dict:
        """Get current deployment status across all stages."""
        recent_deployments = self.deployment_history[-5:] if self.deployment_history else []

        return {
            "current_versions": {
                stage.value: version
                for stage, version in self.current_versions.items()
            },
            "recent_deployments": [
                {
                    "version": d["version"],
                    "stage": d["stage"],
                    "status": d["status"],
                    "timestamp": d["timestamp"].isoformat()
                }
                for d in recent_deployments
            ],
            "health_status": {
                stage.value: "healthy"  # Simplified
                for stage in self.stages
            }
        }


# Test deployment pipeline
async def test_deployment_pipeline():
    pipeline = DeploymentPipeline()

    # Deploy to dev
    success = await pipeline.deploy_to_stage("v1.2.0", DeploymentStage.DEV)

    if success:
        # Run tests
        test_results = await pipeline.run_tests(DeploymentStage.DEV)
        print(f"Test Results: {test_results['passed']}")

        if test_results["passed"]:
            # Promote to staging
            await pipeline.promote_version(
                "v1.2.0",
                DeploymentStage.DEV,
                DeploymentStage.STAGING
            )

            # Promote to production
            await pipeline.promote_version(
                "v1.2.0",
                DeploymentStage.STAGING,
                DeploymentStage.PRODUCTION
            )

    # Get status
    status = pipeline.get_deployment_status()
    print(f"\nDeployment Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")

asyncio.run(test_deployment_pipeline())


# Solution 5: Cost Optimization System
print("\n" + "=" * 50)
print("Solution 5: Cost Optimization System")
print("=" * 50)


class CostOptimizer:
    """Advanced cost optimization system for LLM usage."""

    def __init__(self):
        self.model_costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015}
        }

        self.usage_data = defaultdict(lambda: {
            "tokens": 0,
            "cost": 0,
            "requests": 0
        })

        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.budgets = {}
        self.model_performance = {
            "gpt-4": {"accuracy": 0.95, "speed": 2.0},
            "gpt-3.5-turbo": {"accuracy": 0.85, "speed": 0.5},
            "claude-3-opus": {"accuracy": 0.93, "speed": 1.5}
        }

    def track_usage(self, user_id: str, model: str,
                   input_tokens: int, output_tokens: int):
        """Track token usage and calculate costs."""
        # Calculate cost
        if model in self.model_costs:
            input_cost = (input_tokens / 1000) * self.model_costs[model]["input"]
            output_cost = (output_tokens / 1000) * self.model_costs[model]["output"]
            total_cost = input_cost + output_cost

            # Update usage
            self.usage_data[user_id]["tokens"] += input_tokens + output_tokens
            self.usage_data[user_id]["cost"] += total_cost
            self.usage_data[user_id]["requests"] += 1

            # Check budget
            if user_id in self.budgets:
                if self.usage_data[user_id]["cost"] > self.budgets[user_id]:
                    print(f"⚠️ User {user_id} exceeded budget!")

    def check_budget(self, user_id: str) -> bool:
        """Check if user is within budget."""
        if user_id not in self.budgets:
            return True  # No budget set

        current_usage = self.usage_data[user_id]["cost"]
        budget = self.budgets[user_id]

        return current_usage < budget

    def suggest_optimization(self, request: Dict) -> Dict:
        """Provide intelligent optimization suggestions."""
        suggestions = {
            "optimizations": [],
            "estimated_savings": 0,
            "recommended_model": None
        }

        prompt = request.get("prompt", "")
        model = request.get("model", "gpt-4")

        # Check cache first
        request_hash = hashlib.md5(prompt.encode()).hexdigest()
        if request_hash in self.cache:
            suggestions["optimizations"].append("Use cached response")
            suggestions["estimated_savings"] = self.model_costs[model]["input"] * len(prompt.split()) / 1000
            return suggestions

        # Analyze prompt complexity
        prompt_length = len(prompt.split())
        is_simple = prompt_length < 50 and not any(
            keyword in prompt.lower()
            for keyword in ["analyze", "complex", "detailed", "comprehensive"]
        )

        # Suggest model based on complexity
        if is_simple and model == "gpt-4":
            suggestions["optimizations"].append("Use gpt-3.5-turbo for simple queries")
            suggestions["recommended_model"] = "gpt-3.5-turbo"

            # Calculate potential savings
            gpt4_cost = self.model_costs["gpt-4"]["input"] * prompt_length / 1000
            gpt35_cost = self.model_costs["gpt-3.5-turbo"]["input"] * prompt_length / 1000
            suggestions["estimated_savings"] = gpt4_cost - gpt35_cost

        # Suggest prompt optimization
        if prompt_length > 200:
            suggestions["optimizations"].append("Reduce prompt length by removing redundant information")
            suggestions["estimated_savings"] += self.model_costs[model]["input"] * 50 / 1000

        # Suggest batching for similar requests
        if request.get("batch_compatible", False):
            suggestions["optimizations"].append("Batch similar requests to reduce overhead")

        return suggestions

    def cache_response(self, request_hash: str, response: Dict,
                      ttl: int = 3600):
        """Cache response with TTL."""
        self.cache[request_hash] = {
            "response": response,
            "timestamp": datetime.now(),
            "ttl": ttl,
            "hits": 0
        }

    def get_cached_response(self, request_hash: str) -> Optional[Dict]:
        """Retrieve cached response if valid."""
        if request_hash not in self.cache:
            self.cache_misses += 1
            return None

        cache_entry = self.cache[request_hash]
        age = (datetime.now() - cache_entry["timestamp"]).seconds

        if age > cache_entry["ttl"]:
            # Expired
            del self.cache[request_hash]
            self.cache_misses += 1
            return None

        # Valid cache hit
        cache_entry["hits"] += 1
        self.cache_hits += 1
        return cache_entry["response"]

    def generate_cost_report(self, user_id: str = None) -> Dict:
        """Generate comprehensive cost report with recommendations."""
        if user_id:
            users_data = {user_id: self.usage_data[user_id]}
        else:
            users_data = dict(self.usage_data)

        total_cost = sum(u["cost"] for u in users_data.values())
        total_tokens = sum(u["tokens"] for u in users_data.values())
        total_requests = sum(u["requests"] for u in users_data.values())

        # Calculate cache savings
        cache_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        estimated_cache_savings = total_cost * cache_rate * 0.8  # 80% of cached requests save cost

        # Generate recommendations
        recommendations = []

        if total_cost > 100:
            recommendations.append("Consider implementing request batching to reduce API calls")

        if cache_rate < 0.2:
            recommendations.append("Increase cache usage - current hit rate is only {:.1%}".format(cache_rate))

        # Find heavy users
        heavy_users = [
            uid for uid, data in users_data.items()
            if data["cost"] > total_cost * 0.3
        ]
        if heavy_users:
            recommendations.append(f"Users {heavy_users} account for >30% of costs - consider optimization")

        return {
            "total_cost": f"${total_cost:.2f}",
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "avg_cost_per_request": f"${total_cost / max(total_requests, 1):.4f}",
            "cache_hit_rate": f"{cache_rate:.1%}",
            "estimated_cache_savings": f"${estimated_cache_savings:.2f}",
            "user_breakdown": {
                uid: {
                    "cost": f"${data['cost']:.2f}",
                    "tokens": data["tokens"],
                    "requests": data["requests"]
                }
                for uid, data in users_data.items()
            },
            "recommendations": recommendations
        }


# Test cost optimizer
optimizer = CostOptimizer()

# Set budgets
optimizer.budgets["user_1"] = 10.0
optimizer.budgets["user_2"] = 5.0

# Simulate usage
users = ["user_1", "user_2", "user_3"]
models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]

for _ in range(20):
    user = random.choice(users)
    model = random.choice(models)

    # Track usage
    optimizer.track_usage(
        user,
        model,
        random.randint(100, 500),
        random.randint(200, 800)
    )

# Test optimization suggestions
request = {
    "model": "gpt-4",
    "prompt": "What is the capital of France?"  # Simple question
}
suggestions = optimizer.suggest_optimization(request)
print("\nOptimization Suggestions:")
for key, value in suggestions.items():
    print(f"  {key}: {value}")

# Generate report
report = optimizer.generate_cost_report()
print("\nCost Report:")
for key, value in report.items():
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


print("\n" + "=" * 50)
print("Solutions Complete!")
print("=" * 50)
print("""
These solutions demonstrate production-ready implementations of:
1. Load balancer with health monitoring
2. Comprehensive monitoring dashboard
3. Resilient API wrapper with circuit breaker
4. Automated deployment pipeline
5. Cost optimization system

Each solution includes best practices for production LLM applications!
""")