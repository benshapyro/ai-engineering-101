"""
Module 14: Production Patterns
Deployment Strategies Examples

This file demonstrates various deployment strategies for LLM applications:
1. Microservices architecture
2. API gateway pattern
3. Blue-green deployment
4. Canary deployment
5. Load balancing
6. Service mesh integration
7. Production deployment pipeline

Each example shows production-ready deployment patterns.
"""

import os
import asyncio
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import random
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example 1: Microservices Architecture
print("=" * 50)
print("Example 1: Microservices Architecture")
print("=" * 50)


class ServiceHealth(Enum):
    """Service health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ServiceConfig:
    """Configuration for a microservice."""
    name: str
    host: str
    port: int
    version: str
    capabilities: List[str]
    dependencies: List[str] = field(default_factory=list)
    health_endpoint: str = "/health"
    max_retries: int = 3
    timeout: int = 30


class LLMMicroservice:
    """Microservice wrapper for LLM functionality."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.app = FastAPI(title=config.name)
        self.health_status = ServiceHealth.HEALTHY
        self.metrics = defaultdict(int)
        self.circuit_breaker = CircuitBreaker()

        self._setup_routes()
        self._setup_middleware()

    def _setup_routes(self):
        """Setup service routes."""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "service": self.config.name,
                "status": self.health_status.value,
                "version": self.config.version,
                "uptime": self.get_uptime(),
                "metrics": self.get_metrics_summary()
            }

        @self.app.post("/process")
        async def process_request(request: Dict):
            """Process LLM request."""
            return await self.handle_request(request)

        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.metrics

    def _setup_middleware(self):
        """Setup middleware for the service."""

        @self.app.middleware("http")
        async def add_metrics(request: Request, call_next):
            """Track request metrics."""
            start_time = time.time()
            self.metrics["total_requests"] += 1

            try:
                response = await call_next(request)
                self.metrics["successful_requests"] += 1
                return response
            except Exception as e:
                self.metrics["failed_requests"] += 1
                raise
            finally:
                duration = time.time() - start_time
                self.metrics["total_duration"] += duration

    async def handle_request(self, request: Dict) -> Dict:
        """Handle incoming request with production safeguards."""
        # Circuit breaker check
        if self.circuit_breaker.is_open():
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")

        try:
            # Process request based on capabilities
            if "completion" in self.config.capabilities:
                result = await self.process_completion(request)
            elif "embedding" in self.config.capabilities:
                result = await self.process_embedding(request)
            elif "classification" in self.config.capabilities:
                result = await self.process_classification(request)
            else:
                result = {"error": "No matching capability"}

            self.circuit_breaker.record_success()
            return result

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Service {self.config.name} error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def process_completion(self, request: Dict) -> Dict:
        """Process completion request."""
        # Simulate LLM completion
        await asyncio.sleep(0.1)  # Simulate processing
        return {
            "type": "completion",
            "response": f"Processed by {self.config.name}",
            "tokens": random.randint(10, 100)
        }

    async def process_embedding(self, request: Dict) -> Dict:
        """Process embedding request."""
        # Simulate embedding generation
        await asyncio.sleep(0.05)
        return {
            "type": "embedding",
            "embedding": [random.random() for _ in range(128)],
            "model": self.config.name
        }

    async def process_classification(self, request: Dict) -> Dict:
        """Process classification request."""
        # Simulate classification
        await asyncio.sleep(0.03)
        return {
            "type": "classification",
            "labels": ["positive", "negative"],
            "scores": [0.7, 0.3]
        }

    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        # In production, track actual start time
        return time.time()

    def get_metrics_summary(self) -> Dict:
        """Get metrics summary."""
        total_requests = self.metrics.get("total_requests", 1)
        return {
            "requests": total_requests,
            "success_rate": self.metrics.get("successful_requests", 0) / max(total_requests, 1),
            "avg_latency": self.metrics.get("total_duration", 0) / max(total_requests, 1)
        }


class CircuitBreaker:
    """Circuit breaker for failure protection."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def is_open(self) -> bool:
        """Check if circuit is open."""
        if self.state == "OPEN":
            if self.should_attempt_reset():
                self.state = "HALF_OPEN"
                return False
            return True
        return False

    def should_attempt_reset(self) -> bool:
        """Check if should attempt to reset."""
        if self.last_failure_time:
            return (time.time() - self.last_failure_time) > self.recovery_timeout
        return False

    def record_success(self):
        """Record successful call."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
        self.failure_count = 0

    def record_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning("Circuit breaker opened")


# Create microservices
services = [
    LLMMicroservice(ServiceConfig(
        name="completion-service",
        host="localhost",
        port=8001,
        version="1.0.0",
        capabilities=["completion"]
    )),
    LLMMicroservice(ServiceConfig(
        name="embedding-service",
        host="localhost",
        port=8002,
        version="1.0.0",
        capabilities=["embedding"]
    )),
    LLMMicroservice(ServiceConfig(
        name="classification-service",
        host="localhost",
        port=8003,
        version="1.0.0",
        capabilities=["classification"]
    ))
]

print(f"Created {len(services)} microservices")


# Example 2: API Gateway Pattern
print("\n" + "=" * 50)
print("Example 2: API Gateway Pattern")
print("=" * 50)


class RateLimiter:
    """Rate limiting for API gateway."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_counts = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        self.request_counts[client_id] = [
            t for t in self.request_counts[client_id]
            if t > minute_ago
        ]

        # Check limit
        if len(self.request_counts[client_id]) < self.requests_per_minute:
            self.request_counts[client_id].append(now)
            return True

        return False


class ResponseCache:
    """Response caching for API gateway."""

    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.cache = {}

    def get(self, key: str) -> Optional[Dict]:
        """Get cached response."""
        if key in self.cache:
            entry, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return entry
            del self.cache[key]
        return None

    def set(self, key: str, value: Dict):
        """Cache response."""
        self.cache[key] = (value, time.time())

    def generate_key(self, request: Dict) -> str:
        """Generate cache key from request."""
        content = json.dumps(request, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()


class APIGateway:
    """API Gateway for LLM services."""

    def __init__(self):
        self.services = {}
        self.rate_limiter = RateLimiter()
        self.cache = ResponseCache()
        self.router = ServiceRouter()
        self.auth_manager = AuthenticationManager()
        self.metrics = defaultdict(int)

    def register_service(self, service: ServiceConfig):
        """Register a service with the gateway."""
        self.services[service.name] = service
        logger.info(f"Registered service: {service.name}")

    async def handle_request(self, request: Dict, client_id: str) -> Dict:
        """Handle incoming request through gateway."""
        try:
            # Rate limiting
            if not self.rate_limiter.is_allowed(client_id):
                self.metrics["rate_limited"] += 1
                return {"error": "Rate limit exceeded", "status": 429}

            # Authentication
            if not self.auth_manager.authenticate(request.get("token")):
                self.metrics["auth_failed"] += 1
                return {"error": "Unauthorized", "status": 401}

            # Check cache
            cache_key = self.cache.generate_key(request)
            cached = self.cache.get(cache_key)
            if cached:
                self.metrics["cache_hits"] += 1
                return cached

            # Route to service
            service = self.router.route_request(request, self.services)
            if not service:
                self.metrics["routing_failed"] += 1
                return {"error": "No service available", "status": 503}

            # Forward request
            response = await self.forward_request(service, request)

            # Cache successful responses
            if response.get("status", 200) == 200:
                self.cache.set(cache_key, response)

            self.metrics["successful_requests"] += 1
            return response

        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Gateway error: {e}")
            return {"error": "Internal server error", "status": 500}

    async def forward_request(self, service: ServiceConfig, request: Dict) -> Dict:
        """Forward request to service."""
        # In production, use actual HTTP client
        async with httpx.AsyncClient() as client:
            try:
                url = f"http://{service.host}:{service.port}/process"
                response = await client.post(
                    url,
                    json=request,
                    timeout=service.timeout
                )
                return response.json()
            except Exception as e:
                logger.error(f"Error forwarding to {service.name}: {e}")
                raise

    def get_metrics(self) -> Dict:
        """Get gateway metrics."""
        total = self.metrics.get("successful_requests", 0) + self.metrics.get("errors", 0)
        return {
            "total_requests": total,
            "successful_requests": self.metrics.get("successful_requests", 0),
            "cache_hit_rate": self.metrics.get("cache_hits", 0) / max(total, 1),
            "rate_limited": self.metrics.get("rate_limited", 0),
            "auth_failed": self.metrics.get("auth_failed", 0),
            "errors": self.metrics.get("errors", 0)
        }


class ServiceRouter:
    """Route requests to appropriate services."""

    def route_request(self, request: Dict, services: Dict[str, ServiceConfig]) -> Optional[ServiceConfig]:
        """Route request based on type."""
        request_type = request.get("type", "completion")

        # Find services with matching capability
        matching_services = [
            service for service in services.values()
            if request_type in service.capabilities
        ]

        if not matching_services:
            return None

        # Simple round-robin selection
        # In production, use more sophisticated load balancing
        return random.choice(matching_services)


class AuthenticationManager:
    """Manage API authentication."""

    def __init__(self):
        self.valid_tokens = set()
        # In production, use proper token management
        self.valid_tokens.add("test_token_123")

    def authenticate(self, token: str) -> bool:
        """Authenticate request token."""
        return token in self.valid_tokens


# Example usage
gateway = APIGateway()

# Register services
for service in services:
    gateway.register_service(service.config)

print(f"API Gateway initialized with {len(services)} services")
print(f"Gateway features: Rate limiting, Caching, Authentication, Routing")


# Example 3: Blue-Green Deployment
print("\n" + "=" * 50)
print("Example 3: Blue-Green Deployment")
print("=" * 50)


@dataclass
class Environment:
    """Deployment environment."""
    name: str
    version: str
    status: str = "inactive"
    health_check_url: str = "/health"
    services: List[ServiceConfig] = field(default_factory=list)


class LoadBalancer:
    """Load balancer for traffic management."""

    def __init__(self):
        self.active_environment = "blue"
        self.environments = {
            "blue": None,
            "green": None
        }

    def switch_traffic(self, target: str):
        """Switch traffic to target environment."""
        if target not in self.environments:
            raise ValueError(f"Invalid environment: {target}")

        logger.info(f"Switching traffic from {self.active_environment} to {target}")
        self.active_environment = target

    def get_active_backend(self) -> Optional[Environment]:
        """Get active backend environment."""
        return self.environments.get(self.active_environment)


class BlueGreenDeployment:
    """Blue-green deployment strategy."""

    def __init__(self):
        self.blue_env = Environment(name="blue", version="")
        self.green_env = Environment(name="green", version="")
        self.load_balancer = LoadBalancer()
        self.active = "blue"

    async def deploy_new_version(self, new_version: str, services: List[ServiceConfig]) -> Dict:
        """Deploy new version using blue-green strategy."""
        logger.info(f"Starting blue-green deployment for version {new_version}")

        # Determine target environment
        target_env = self.green_env if self.active == "blue" else self.blue_env
        target_name = target_env.name

        try:
            # Phase 1: Deploy to inactive environment
            logger.info(f"Phase 1: Deploying to {target_name}")
            await self.deploy_to_environment(target_env, new_version, services)

            # Phase 2: Health checks
            logger.info(f"Phase 2: Running health checks on {target_name}")
            health_status = await self.health_check(target_env)
            if not health_status["healthy"]:
                raise Exception(f"Health check failed: {health_status}")

            # Phase 3: Smoke tests
            logger.info(f"Phase 3: Running smoke tests on {target_name}")
            smoke_results = await self.run_smoke_tests(target_env)
            if not smoke_results["passed"]:
                raise Exception(f"Smoke tests failed: {smoke_results}")

            # Phase 4: Switch traffic
            logger.info(f"Phase 4: Switching traffic to {target_name}")
            self.load_balancer.switch_traffic(target_name)
            self.active = target_name

            # Phase 5: Monitor
            logger.info("Phase 5: Monitoring new deployment")
            monitoring_results = await self.monitor_deployment(duration=60)

            return {
                "status": "success",
                "version": new_version,
                "environment": target_name,
                "health": health_status,
                "smoke_tests": smoke_results,
                "monitoring": monitoring_results
            }

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            # Rollback if needed
            if self.active != target_name:
                logger.info("No rollback needed - traffic not switched")
            return {
                "status": "failed",
                "error": str(e),
                "version": new_version
            }

    async def deploy_to_environment(self, env: Environment, version: str, services: List[ServiceConfig]):
        """Deploy services to environment."""
        env.version = version
        env.services = services
        env.status = "deploying"

        # Simulate deployment
        await asyncio.sleep(2)

        env.status = "deployed"
        logger.info(f"Deployed version {version} to {env.name}")

    async def health_check(self, env: Environment) -> Dict:
        """Perform health checks on environment."""
        checks = []

        for service in env.services:
            # Simulate health check
            health = random.choice([True, True, True, False])  # 75% success
            checks.append({
                "service": service.name,
                "healthy": health,
                "response_time": random.uniform(10, 100)
            })

        all_healthy = all(check["healthy"] for check in checks)

        return {
            "healthy": all_healthy,
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }

    async def run_smoke_tests(self, env: Environment) -> Dict:
        """Run smoke tests on environment."""
        tests = [
            "test_basic_completion",
            "test_embedding_generation",
            "test_classification",
            "test_error_handling"
        ]

        results = []
        for test in tests:
            # Simulate test execution
            passed = random.choice([True, True, True, True, False])  # 80% success
            results.append({
                "test": test,
                "passed": passed,
                "duration": random.uniform(0.1, 1.0)
            })

        all_passed = all(r["passed"] for r in results)

        return {
            "passed": all_passed,
            "total": len(tests),
            "passed_count": sum(1 for r in results if r["passed"]),
            "results": results
        }

    async def monitor_deployment(self, duration: int) -> Dict:
        """Monitor deployment for issues."""
        # Simulate monitoring
        await asyncio.sleep(min(duration, 2))  # Don't actually wait full duration

        metrics = {
            "error_rate": random.uniform(0, 0.02),  # 0-2% error rate
            "avg_latency": random.uniform(50, 200),  # ms
            "throughput": random.uniform(100, 1000),  # requests/sec
            "cpu_usage": random.uniform(20, 80),  # percentage
            "memory_usage": random.uniform(30, 70)  # percentage
        }

        alerts = []
        if metrics["error_rate"] > 0.01:
            alerts.append("High error rate detected")
        if metrics["avg_latency"] > 150:
            alerts.append("High latency detected")

        return {
            "duration": duration,
            "metrics": metrics,
            "alerts": alerts,
            "status": "healthy" if not alerts else "degraded"
        }

    async def rollback(self):
        """Rollback to previous environment."""
        previous = "blue" if self.active == "green" else "green"
        logger.warning(f"Rolling back from {self.active} to {previous}")
        self.load_balancer.switch_traffic(previous)
        self.active = previous


# Example usage
deployer = BlueGreenDeployment()

# Simulate deployment
async def demo_blue_green():
    result = await deployer.deploy_new_version(
        "2.0.0",
        services=[s.config for s in services]
    )
    print(f"Deployment result: {result['status']}")
    if result["status"] == "success":
        print(f"  Active environment: {result['environment']}")
        print(f"  Version: {result['version']}")
        print(f"  Monitoring: {result['monitoring']['status']}")


# Run demo
asyncio.run(demo_blue_green())


# Example 4: Canary Deployment
print("\n" + "=" * 50)
print("Example 4: Canary Deployment")
print("=" * 50)


@dataclass
class CanaryStage:
    """Canary deployment stage."""
    percentage: float
    duration: int  # seconds
    success_criteria: Dict[str, float]


class TrafficManager:
    """Manage traffic distribution for canary."""

    def __init__(self):
        self.canary_percentage = 0.0
        self.stable_version = "1.0.0"
        self.canary_version = None

    def set_canary_traffic(self, percentage: float):
        """Set canary traffic percentage."""
        self.canary_percentage = max(0, min(100, percentage))
        logger.info(f"Canary traffic set to {self.canary_percentage}%")

    def should_route_to_canary(self) -> bool:
        """Determine if request should go to canary."""
        return random.random() * 100 < self.canary_percentage


class MetricsCollector:
    """Collect metrics for canary analysis."""

    def __init__(self):
        self.metrics = defaultdict(lambda: defaultdict(list))

    def record(self, version: str, metric_name: str, value: float):
        """Record a metric value."""
        self.metrics[version][metric_name].append(value)

    def get_comparative_metrics(self) -> Dict:
        """Get comparative metrics between versions."""
        comparison = {}

        for metric_name in set().union(*[m.keys() for m in self.metrics.values()]):
            comparison[metric_name] = {}

            for version, metrics in self.metrics.items():
                if metric_name in metrics:
                    values = metrics[metric_name]
                    comparison[metric_name][version] = {
                        "mean": np.mean(values),
                        "p50": np.percentile(values, 50),
                        "p95": np.percentile(values, 95),
                        "p99": np.percentile(values, 99)
                    }

        return comparison


class CanaryDeployment:
    """Canary deployment strategy."""

    def __init__(self):
        self.traffic_manager = TrafficManager()
        self.metrics_collector = MetricsCollector()
        self.stages = self.default_stages()

    def default_stages(self) -> List[CanaryStage]:
        """Default canary stages."""
        return [
            CanaryStage(
                percentage=5,
                duration=300,  # 5 minutes
                success_criteria={
                    "error_rate": 0.01,  # < 1%
                    "p95_latency": 200,  # < 200ms
                }
            ),
            CanaryStage(
                percentage=25,
                duration=600,  # 10 minutes
                success_criteria={
                    "error_rate": 0.01,
                    "p95_latency": 200,
                }
            ),
            CanaryStage(
                percentage=50,
                duration=900,  # 15 minutes
                success_criteria={
                    "error_rate": 0.01,
                    "p95_latency": 200,
                }
            ),
            CanaryStage(
                percentage=100,
                duration=0,  # Final stage
                success_criteria={}
            )
        ]

    async def deploy_canary(self, new_version: str) -> Dict:
        """Deploy using canary strategy."""
        logger.info(f"Starting canary deployment for version {new_version}")
        self.traffic_manager.canary_version = new_version

        deployment_results = {
            "version": new_version,
            "stages": [],
            "status": "in_progress"
        }

        try:
            for i, stage in enumerate(self.stages):
                logger.info(f"Stage {i+1}: {stage.percentage}% traffic")

                # Update traffic split
                self.traffic_manager.set_canary_traffic(stage.percentage)

                # Wait and collect metrics
                if stage.duration > 0:
                    await self.collect_metrics(stage.duration)

                    # Analyze metrics
                    metrics = self.metrics_collector.get_comparative_metrics()
                    stage_result = self.evaluate_stage(stage, metrics)

                    deployment_results["stages"].append({
                        "percentage": stage.percentage,
                        "metrics": metrics,
                        "passed": stage_result["passed"],
                        "issues": stage_result.get("issues", [])
                    })

                    if not stage_result["passed"]:
                        raise Exception(f"Stage failed: {stage_result['issues']}")

            deployment_results["status"] = "success"
            logger.info("Canary deployment completed successfully")

        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            await self.rollback()
            deployment_results["status"] = "failed"
            deployment_results["error"] = str(e)

        return deployment_results

    async def collect_metrics(self, duration: int):
        """Collect metrics for specified duration."""
        # Simulate metric collection
        end_time = time.time() + min(duration, 2)  # Don't actually wait full duration

        while time.time() < end_time:
            # Simulate requests to both versions
            for _ in range(10):
                if self.traffic_manager.should_route_to_canary():
                    version = self.traffic_manager.canary_version
                    # Slightly worse metrics for canary (for demo)
                    error = random.random() < 0.008
                    latency = random.uniform(50, 180)
                else:
                    version = self.traffic_manager.stable_version
                    error = random.random() < 0.005
                    latency = random.uniform(40, 150)

                self.metrics_collector.record(version, "error_rate", 1 if error else 0)
                self.metrics_collector.record(version, "latency", latency)

            await asyncio.sleep(0.1)

    def evaluate_stage(self, stage: CanaryStage, metrics: Dict) -> Dict:
        """Evaluate if stage meets success criteria."""
        canary_version = self.traffic_manager.canary_version
        issues = []
        passed = True

        for criterion, threshold in stage.success_criteria.items():
            if criterion == "error_rate":
                if "error_rate" in metrics:
                    canary_error_rate = metrics["error_rate"].get(canary_version, {}).get("mean", 0)
                    if canary_error_rate > threshold:
                        issues.append(f"Error rate {canary_error_rate:.3f} exceeds threshold {threshold}")
                        passed = False

            elif criterion == "p95_latency":
                if "latency" in metrics:
                    canary_latency = metrics["latency"].get(canary_version, {}).get("p95", 0)
                    if canary_latency > threshold:
                        issues.append(f"P95 latency {canary_latency:.1f}ms exceeds threshold {threshold}ms")
                        passed = False

        return {"passed": passed, "issues": issues}

    async def rollback(self):
        """Rollback canary deployment."""
        logger.warning("Rolling back canary deployment")
        self.traffic_manager.set_canary_traffic(0)
        self.traffic_manager.canary_version = None


# Example usage
canary = CanaryDeployment()

# Simulate canary deployment
async def demo_canary():
    result = await canary.deploy_canary("2.1.0")
    print(f"Canary deployment result: {result['status']}")
    if result["status"] == "success":
        print(f"  Successfully deployed version {result['version']}")
        print(f"  Stages completed: {len(result['stages'])}")
    else:
        print(f"  Error: {result.get('error')}")


# Run demo
asyncio.run(demo_canary())


# Example 5: Load Balancing
print("\n" + "=" * 50)
print("Example 5: Load Balancing")
print("=" * 50)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    RANDOM = "random"
    IP_HASH = "ip_hash"


class ServiceInstance:
    """Service instance for load balancing."""

    def __init__(self, instance_id: str, host: str, port: int, weight: int = 1):
        self.instance_id = instance_id
        self.host = host
        self.port = port
        self.weight = weight
        self.active_connections = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.health_score = 1.0

    @property
    def is_healthy(self) -> bool:
        """Check if instance is healthy."""
        return self.health_score > 0.5

    def update_health_score(self):
        """Update health score based on failures."""
        if self.total_requests > 0:
            success_rate = 1 - (self.failed_requests / self.total_requests)
            self.health_score = 0.9 * self.health_score + 0.1 * success_rate


class LoadBalancerAdvanced:
    """Advanced load balancer with multiple strategies."""

    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.instances = []
        self.current_index = 0

    def add_instance(self, instance: ServiceInstance):
        """Add service instance."""
        self.instances.append(instance)
        logger.info(f"Added instance {instance.instance_id} to load balancer")

    def remove_instance(self, instance_id: str):
        """Remove service instance."""
        self.instances = [i for i in self.instances if i.instance_id != instance_id]
        logger.info(f"Removed instance {instance_id} from load balancer")

    def get_next_instance(self, client_ip: Optional[str] = None) -> Optional[ServiceInstance]:
        """Get next instance based on strategy."""
        healthy_instances = [i for i in self.instances if i.is_healthy]

        if not healthy_instances:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            return self._weighted(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            return self._ip_hash(healthy_instances, client_ip)

    def _round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin selection."""
        instance = instances[self.current_index % len(instances)]
        self.current_index += 1
        return instance

    def _least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least connections."""
        return min(instances, key=lambda i: i.active_connections)

    def _weighted(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted random selection."""
        weights = [i.weight * i.health_score for i in instances]
        return random.choices(instances, weights=weights)[0]

    def _random(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Random selection."""
        return random.choice(instances)

    def _ip_hash(self, instances: List[ServiceInstance], client_ip: Optional[str]) -> ServiceInstance:
        """IP hash-based selection."""
        if not client_ip:
            return self._random(instances)

        hash_value = hashlib.md5(client_ip.encode()).hexdigest()
        index = int(hash_value, 16) % len(instances)
        return instances[index]

    def update_instance_metrics(self, instance_id: str, success: bool):
        """Update instance metrics after request."""
        instance = next((i for i in self.instances if i.instance_id == instance_id), None)
        if instance:
            instance.total_requests += 1
            if not success:
                instance.failed_requests += 1
            instance.update_health_score()

    def get_load_distribution(self) -> Dict:
        """Get current load distribution."""
        return {
            instance.instance_id: {
                "active_connections": instance.active_connections,
                "total_requests": instance.total_requests,
                "health_score": instance.health_score,
                "weight": instance.weight
            }
            for instance in self.instances
        }


# Example usage
lb = LoadBalancerAdvanced(LoadBalancingStrategy.LEAST_CONNECTIONS)

# Add instances
for i in range(3):
    lb.add_instance(ServiceInstance(
        instance_id=f"instance-{i}",
        host="localhost",
        port=8000 + i,
        weight=random.randint(1, 3)
    ))

# Simulate load balancing
print("\nLoad balancing simulation:")
for req in range(10):
    instance = lb.get_next_instance(client_ip=f"192.168.1.{req}")
    if instance:
        # Simulate request
        instance.active_connections += 1
        success = random.random() > 0.1  # 90% success rate
        instance.active_connections -= 1
        lb.update_instance_metrics(instance.instance_id, success)
        print(f"  Request {req} -> {instance.instance_id}")

print("\nLoad distribution:")
distribution = lb.get_load_distribution()
for instance_id, metrics in distribution.items():
    print(f"  {instance_id}: {metrics['total_requests']} requests, "
          f"health={metrics['health_score']:.2f}")


# Example 6: Service Mesh Integration
print("\n" + "=" * 50)
print("Example 6: Service Mesh Integration")
print("=" * 50)


class ServiceMeshProxy:
    """Service mesh sidecar proxy."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.circuit_breaker = CircuitBreaker()
        self.retry_policy = RetryPolicy()
        self.timeout_policy = TimeoutPolicy()
        self.metrics = defaultdict(int)

    async def intercept_request(self, request: Dict) -> Dict:
        """Intercept and handle outbound request."""
        # Add tracing headers
        request["headers"] = request.get("headers", {})
        request["headers"]["x-trace-id"] = str(uuid.uuid4())
        request["headers"]["x-source-service"] = self.service_name

        # Apply policies
        try:
            # Timeout
            response = await self.timeout_policy.execute(
                self._send_request,
                request
            )

            # Retry on failure
            if response.get("status", 500) >= 500:
                response = await self.retry_policy.execute(
                    self._send_request,
                    request
                )

            self.metrics["successful_requests"] += 1
            return response

        except Exception as e:
            self.metrics["failed_requests"] += 1
            self.circuit_breaker.record_failure()
            raise

    async def _send_request(self, request: Dict) -> Dict:
        """Send actual request (simulated)."""
        await asyncio.sleep(0.1)
        # Simulate response
        return {
            "status": 200 if random.random() > 0.1 else 500,
            "data": "Response data"
        }


class RetryPolicy:
    """Retry policy for service mesh."""

    def __init__(self, max_retries: int = 3, backoff_multiplier: float = 2.0):
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier

    async def execute(self, func, *args, **kwargs):
        """Execute with retry policy."""
        last_exception = None
        backoff = 1.0

        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(backoff)
                    backoff *= self.backoff_multiplier

        raise last_exception


class TimeoutPolicy:
    """Timeout policy for service mesh."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    async def execute(self, func, *args, **kwargs):
        """Execute with timeout."""
        return await asyncio.wait_for(
            func(*args, **kwargs),
            timeout=self.timeout
        )


class ServiceMesh:
    """Service mesh control plane."""

    def __init__(self):
        self.services = {}
        self.proxies = {}
        self.traffic_policies = {}
        self.observability = ObservabilityStack()

    def register_service(self, service_name: str, config: ServiceConfig):
        """Register service with mesh."""
        self.services[service_name] = config
        self.proxies[service_name] = ServiceMeshProxy(service_name)
        logger.info(f"Registered {service_name} with service mesh")

    def set_traffic_policy(self, service_name: str, policy: Dict):
        """Set traffic management policy."""
        self.traffic_policies[service_name] = policy
        logger.info(f"Updated traffic policy for {service_name}")

    async def route_request(self, source: str, target: str, request: Dict) -> Dict:
        """Route request through mesh."""
        if source not in self.proxies:
            raise ValueError(f"Unknown source service: {source}")

        # Apply traffic policies
        if target in self.traffic_policies:
            policy = self.traffic_policies[target]
            if "canary" in policy:
                # Route to canary if applicable
                if random.random() < policy["canary"]["percentage"]:
                    target = policy["canary"]["version"]

        # Route through proxy
        proxy = self.proxies[source]
        response = await proxy.intercept_request(request)

        # Collect telemetry
        self.observability.record_request(source, target, response)

        return response

    def get_mesh_topology(self) -> Dict:
        """Get service mesh topology."""
        return {
            "services": list(self.services.keys()),
            "total_services": len(self.services),
            "policies": {
                service: policy
                for service, policy in self.traffic_policies.items()
            }
        }


class ObservabilityStack:
    """Observability for service mesh."""

    def __init__(self):
        self.traces = []
        self.metrics = defaultdict(lambda: defaultdict(int))

    def record_request(self, source: str, target: str, response: Dict):
        """Record request telemetry."""
        trace = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "target": target,
            "status": response.get("status"),
            "duration": random.uniform(10, 200)  # Simulated duration
        }
        self.traces.append(trace)

        # Update metrics
        self.metrics[source]["outbound_requests"] += 1
        self.metrics[target]["inbound_requests"] += 1

        if response.get("status", 500) >= 400:
            self.metrics[source]["outbound_errors"] += 1
            self.metrics[target]["inbound_errors"] += 1


# Example usage
mesh = ServiceMesh()

# Register services
for service_name in ["auth-service", "user-service", "payment-service"]:
    mesh.register_service(
        service_name,
        ServiceConfig(
            name=service_name,
            host="localhost",
            port=8000,
            version="1.0.0",
            capabilities=[]
        )
    )

# Set traffic policies
mesh.set_traffic_policy("payment-service", {
    "canary": {
        "version": "payment-service-v2",
        "percentage": 0.1
    }
})

print(f"Service mesh initialized")
print(f"Topology: {mesh.get_mesh_topology()}")


# Example 7: Production Deployment Pipeline
print("\n" + "=" * 50)
print("Example 7: Production Deployment Pipeline")
print("=" * 50)


class DeploymentPipeline:
    """Complete production deployment pipeline."""

    def __init__(self):
        self.stages = [
            "build",
            "test",
            "security_scan",
            "staging_deploy",
            "integration_test",
            "performance_test",
            "approval",
            "production_deploy",
            "smoke_test",
            "monitoring"
        ]
        self.deployment_history = []

    async def execute_pipeline(self, version: str, config: Dict) -> Dict:
        """Execute full deployment pipeline."""
        logger.info(f"Starting deployment pipeline for version {version}")

        pipeline_result = {
            "version": version,
            "start_time": datetime.now(),
            "stages": {},
            "status": "in_progress"
        }

        try:
            for stage in self.stages:
                logger.info(f"Executing stage: {stage}")

                # Execute stage
                stage_result = await self.execute_stage(stage, version, config)
                pipeline_result["stages"][stage] = stage_result

                if not stage_result["success"]:
                    raise Exception(f"Stage {stage} failed: {stage_result.get('error')}")

                # Check for manual approval
                if stage == "approval":
                    if not await self.get_approval():
                        raise Exception("Deployment not approved")

            pipeline_result["status"] = "success"
            pipeline_result["end_time"] = datetime.now()

            # Record deployment
            self.deployment_history.append(pipeline_result)

            logger.info(f"Pipeline completed successfully for version {version}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_result["status"] = "failed"
            pipeline_result["error"] = str(e)
            pipeline_result["end_time"] = datetime.now()

            # Rollback if in production stage
            if "production_deploy" in pipeline_result["stages"]:
                await self.rollback(version)

        return pipeline_result

    async def execute_stage(self, stage: str, version: str, config: Dict) -> Dict:
        """Execute a pipeline stage."""
        stage_handlers = {
            "build": self.stage_build,
            "test": self.stage_test,
            "security_scan": self.stage_security_scan,
            "staging_deploy": self.stage_staging_deploy,
            "integration_test": self.stage_integration_test,
            "performance_test": self.stage_performance_test,
            "approval": self.stage_approval,
            "production_deploy": self.stage_production_deploy,
            "smoke_test": self.stage_smoke_test,
            "monitoring": self.stage_monitoring
        }

        handler = stage_handlers.get(stage, self.stage_default)
        return await handler(version, config)

    async def stage_build(self, version: str, config: Dict) -> Dict:
        """Build stage."""
        await asyncio.sleep(0.5)  # Simulate build
        return {
            "success": True,
            "artifacts": [f"app-{version}.tar.gz"],
            "duration": 45.2
        }

    async def stage_test(self, version: str, config: Dict) -> Dict:
        """Test stage."""
        await asyncio.sleep(0.3)
        test_results = {
            "unit_tests": {"passed": 150, "failed": 0},
            "integration_tests": {"passed": 45, "failed": 0}
        }
        return {
            "success": True,
            "test_results": test_results,
            "coverage": 85.5
        }

    async def stage_security_scan(self, version: str, config: Dict) -> Dict:
        """Security scan stage."""
        await asyncio.sleep(0.4)
        return {
            "success": True,
            "vulnerabilities": {
                "critical": 0,
                "high": 0,
                "medium": 2,
                "low": 5
            }
        }

    async def stage_staging_deploy(self, version: str, config: Dict) -> Dict:
        """Deploy to staging."""
        await asyncio.sleep(0.6)
        return {
            "success": True,
            "environment": "staging",
            "url": f"https://staging.example.com/v{version}"
        }

    async def stage_integration_test(self, version: str, config: Dict) -> Dict:
        """Integration testing."""
        await asyncio.sleep(0.5)
        return {
            "success": True,
            "tests_passed": 30,
            "tests_failed": 0
        }

    async def stage_performance_test(self, version: str, config: Dict) -> Dict:
        """Performance testing."""
        await asyncio.sleep(0.7)
        return {
            "success": True,
            "metrics": {
                "throughput": "1000 req/s",
                "p95_latency": "150ms",
                "error_rate": "0.01%"
            }
        }

    async def stage_approval(self, version: str, config: Dict) -> Dict:
        """Manual approval stage."""
        # In production, would wait for actual approval
        return {
            "success": True,
            "approved_by": "admin",
            "approved_at": datetime.now().isoformat()
        }

    async def stage_production_deploy(self, version: str, config: Dict) -> Dict:
        """Deploy to production."""
        await asyncio.sleep(0.8)
        return {
            "success": True,
            "environment": "production",
            "strategy": config.get("deployment_strategy", "blue-green")
        }

    async def stage_smoke_test(self, version: str, config: Dict) -> Dict:
        """Smoke testing in production."""
        await asyncio.sleep(0.3)
        return {
            "success": True,
            "critical_paths_tested": 5,
            "all_passed": True
        }

    async def stage_monitoring(self, version: str, config: Dict) -> Dict:
        """Setup monitoring."""
        await asyncio.sleep(0.2)
        return {
            "success": True,
            "dashboards_configured": True,
            "alerts_configured": True
        }

    async def stage_default(self, version: str, config: Dict) -> Dict:
        """Default stage handler."""
        return {"success": True}

    async def get_approval(self) -> bool:
        """Get deployment approval (simulated)."""
        # In production, would integrate with approval system
        return True

    async def rollback(self, version: str):
        """Rollback deployment."""
        logger.warning(f"Rolling back deployment of version {version}")
        # Implement rollback logic

    def get_deployment_metrics(self) -> Dict:
        """Get deployment metrics."""
        if not self.deployment_history:
            return {}

        successful = [d for d in self.deployment_history if d["status"] == "success"]
        failed = [d for d in self.deployment_history if d["status"] == "failed"]

        return {
            "total_deployments": len(self.deployment_history),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.deployment_history) if self.deployment_history else 0,
            "last_deployment": self.deployment_history[-1] if self.deployment_history else None
        }


# Example usage
pipeline = DeploymentPipeline()

# Simulate deployment
async def demo_pipeline():
    result = await pipeline.execute_pipeline(
        "3.0.0",
        {
            "deployment_strategy": "canary",
            "target_environment": "production"
        }
    )

    print(f"Pipeline result: {result['status']}")
    if result["status"] == "success":
        print(f"  Version {result['version']} deployed successfully")
        print(f"  Stages completed: {len(result['stages'])}")
    else:
        print(f"  Error: {result.get('error')}")

    print(f"\nDeployment metrics: {pipeline.get_deployment_metrics()}")


# Run demo
asyncio.run(demo_pipeline())

print("\n" + "=" * 50)
print("All Deployment Strategy Examples Complete!")
print("=" * 50)