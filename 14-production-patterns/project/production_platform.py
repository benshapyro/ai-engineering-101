"""
Module 14: Production Patterns
Final Project - Complete Production LLM Platform

A production-ready LLM platform with all enterprise features including
deployment strategies, monitoring, resilience, and cost optimization.
"""

import os
import asyncio
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import random
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Data Models ====================

class RequestModel(BaseModel):
    """LLM request model."""
    prompt: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(default="gpt-3.5-turbo")
    max_tokens: int = Field(default=500, ge=1, le=4000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=5)


class DeploymentStage(Enum):
    """Deployment stages."""
    CANARY = "canary"
    BLUE = "blue"
    GREEN = "green"


class ServiceStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# ==================== Core Components ====================

class LoadBalancer:
    """Advanced load balancer with multiple strategies."""

    def __init__(self):
        self.endpoints = []
        self.strategy = "least_connections"
        self.health_check_interval = 30

    def add_endpoint(self, name: str, url: str, weight: int = 1):
        """Add an endpoint."""
        self.endpoints.append({
            "name": name,
            "url": url,
            "weight": weight,
            "healthy": True,
            "connections": 0,
            "total_requests": 0,
            "avg_latency": 0.0
        })

    def select_endpoint(self) -> Optional[Dict]:
        """Select best endpoint based on strategy."""
        healthy = [e for e in self.endpoints if e["healthy"]]
        if not healthy:
            return None

        if self.strategy == "least_connections":
            return min(healthy, key=lambda e: e["connections"])
        elif self.strategy == "weighted":
            weights = [e["weight"] for e in healthy]
            return random.choices(healthy, weights=weights)[0]
        else:  # round-robin
            return healthy[0]

    async def health_check(self, endpoint: Dict) -> bool:
        """Check endpoint health."""
        try:
            # Simulate health check
            await asyncio.sleep(0.1)
            endpoint["healthy"] = random.random() < 0.95
            return endpoint["healthy"]
        except:
            endpoint["healthy"] = False
            return False


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset circuit."""
        if not self.last_failure_time:
            return True
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class MetricsCollector:
    """Comprehensive metrics collection."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.window_size = 1000

    def record(self, name: str, value: float, tags: Dict = None):
        """Record a metric."""
        self.metrics[name].append({
            "value": value,
            "timestamp": datetime.now(),
            "tags": tags or {}
        })

        # Maintain window
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name] = self.metrics[name][-self.window_size:]

    def get_stats(self, name: str, window_seconds: int = 60) -> Dict:
        """Get statistics for a metric."""
        if name not in self.metrics:
            return {}

        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        recent = [m["value"] for m in self.metrics[name]
                 if m["timestamp"] > cutoff]

        if not recent:
            return {}

        return {
            "count": len(recent),
            "mean": np.mean(recent),
            "median": np.median(recent),
            "p95": np.percentile(recent, 95),
            "p99": np.percentile(recent, 99)
        }


class CostTracker:
    """Track and optimize costs."""

    MODEL_COSTS = {
        "gpt-4": 0.03,
        "gpt-3.5-turbo": 0.001,
        "claude-3": 0.015
    }

    def __init__(self):
        self.usage = defaultdict(lambda: {"tokens": 0, "cost": 0.0})
        self.budgets = {}

    def track(self, user_id: str, model: str, tokens: int):
        """Track usage and cost."""
        cost = (tokens / 1000) * self.MODEL_COSTS.get(model, 0.001)
        self.usage[user_id]["tokens"] += tokens
        self.usage[user_id]["cost"] += cost

        # Check budget
        if user_id in self.budgets:
            if self.usage[user_id]["cost"] > self.budgets[user_id]:
                logger.warning(f"User {user_id} exceeded budget")
                return False
        return True

    def get_report(self, user_id: str = None) -> Dict:
        """Get cost report."""
        if user_id:
            return self.usage.get(user_id, {})

        total_cost = sum(u["cost"] for u in self.usage.values())
        total_tokens = sum(u["tokens"] for u in self.usage.values())

        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "users": len(self.usage),
            "avg_cost_per_user": total_cost / max(len(self.usage), 1)
        }


class ResponseCache:
    """Intelligent response caching."""

    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl
        self.hits = 0
        self.misses = 0

    def get_key(self, prompt: str, model: str) -> str:
        """Generate cache key."""
        content = f"{prompt}:{model}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        """Get cached response."""
        if key in self.cache:
            entry = self.cache[key]
            age = (datetime.now() - entry["timestamp"]).seconds

            if age < self.ttl:
                self.hits += 1
                entry["hits"] += 1
                return entry["response"]

            del self.cache[key]

        self.misses += 1
        return None

    def set(self, key: str, response: str):
        """Cache a response."""
        self.cache[key] = {
            "response": response,
            "timestamp": datetime.now(),
            "hits": 0
        }

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(total, 1),
            "cached_items": len(self.cache)
        }


# ==================== Production Platform ====================

class ProductionLLMPlatform:
    """Complete production LLM platform."""

    def __init__(self):
        # Core components
        self.load_balancer = LoadBalancer()
        self.circuit_breaker = CircuitBreaker()
        self.metrics = MetricsCollector()
        self.cost_tracker = CostTracker()
        self.cache = ResponseCache()

        # Deployment configuration
        self.deployment_stage = DeploymentStage.BLUE
        self.canary_percentage = 10  # 10% traffic to canary
        self.service_status = ServiceStatus.HEALTHY

        # Request queue
        self.request_queue = asyncio.Queue(maxsize=1000)
        self.processing = False

        # Initialize endpoints
        self._initialize_endpoints()

    def _initialize_endpoints(self):
        """Initialize LLM endpoints."""
        self.load_balancer.add_endpoint("primary", "https://api1.example.com", weight=2)
        self.load_balancer.add_endpoint("secondary", "https://api2.example.com", weight=1)
        self.load_balancer.add_endpoint("backup", "https://api3.example.com", weight=1)

    async def process_request(self, request: RequestModel) -> Dict:
        """Process an LLM request with full production features."""
        start_time = time.time()

        try:
            # Record request metric
            self.metrics.record("requests", 1, {"model": request.model})

            # Check cache first
            cache_key = self.cache.get_key(request.prompt, request.model)
            cached = self.cache.get(cache_key)

            if cached:
                self.metrics.record("cache_hits", 1)
                return {
                    "success": True,
                    "response": cached,
                    "cached": True,
                    "latency": time.time() - start_time
                }

            # Check user budget
            if request.user_id:
                budget_ok = self.cost_tracker.track(
                    request.user_id,
                    request.model,
                    request.max_tokens
                )
                if not budget_ok:
                    raise HTTPException(status_code=429, detail="Budget exceeded")

            # Route based on deployment stage
            if self._should_use_canary():
                endpoint = self._get_canary_endpoint()
            else:
                endpoint = self.load_balancer.select_endpoint()

            if not endpoint:
                raise HTTPException(status_code=503, detail="No healthy endpoints")

            # Execute with circuit breaker
            response = await self._execute_llm_call(endpoint, request)

            # Cache successful response
            self.cache.set(cache_key, response)

            # Record metrics
            latency = time.time() - start_time
            self.metrics.record("latency", latency, {"model": request.model})
            self.metrics.record("tokens", request.max_tokens, {"model": request.model})

            return {
                "success": True,
                "response": response,
                "cached": False,
                "latency": latency,
                "endpoint": endpoint["name"]
            }

        except Exception as e:
            # Record error
            self.metrics.record("errors", 1, {"type": type(e).__name__})
            logger.error(f"Request failed: {e}")

            # Try fallback
            fallback = await self._fallback_response(request)
            if fallback:
                return {
                    "success": True,
                    "response": fallback,
                    "fallback": True,
                    "latency": time.time() - start_time
                }

            raise

    def _should_use_canary(self) -> bool:
        """Determine if request should go to canary."""
        if self.deployment_stage != DeploymentStage.CANARY:
            return False
        return random.randint(1, 100) <= self.canary_percentage

    def _get_canary_endpoint(self) -> Dict:
        """Get canary deployment endpoint."""
        # Simplified - would select actual canary endpoint
        return self.load_balancer.endpoints[0]

    async def _execute_llm_call(self, endpoint: Dict, request: RequestModel) -> str:
        """Execute actual LLM API call."""
        # Simulate API call
        endpoint["connections"] += 1

        try:
            await asyncio.sleep(random.uniform(0.5, 2.0))

            # Simulate occasional failures
            if random.random() < 0.1:
                raise Exception("API call failed")

            response = f"Response to '{request.prompt[:50]}...' from {endpoint['name']}"
            endpoint["total_requests"] += 1

            return response

        finally:
            endpoint["connections"] -= 1

    async def _fallback_response(self, request: RequestModel) -> Optional[str]:
        """Generate fallback response."""
        # Try simpler model
        if request.model == "gpt-4":
            request.model = "gpt-3.5-turbo"
            return await self._execute_llm_call(
                self.load_balancer.endpoints[0],
                request
            )

        # Return default response
        return "I apologize, but I'm currently unable to process your request. Please try again later."

    async def health_check(self) -> Dict:
        """Comprehensive health check."""
        checks = {
            "endpoints": [],
            "cache": self.cache.get_stats(),
            "metrics": {},
            "status": ServiceStatus.HEALTHY.value
        }

        # Check endpoints
        for endpoint in self.load_balancer.endpoints:
            healthy = await self.load_balancer.health_check(endpoint)
            checks["endpoints"].append({
                "name": endpoint["name"],
                "healthy": healthy,
                "connections": endpoint["connections"]
            })

        # Check metrics
        checks["metrics"] = {
            "request_rate": self.metrics.get_stats("requests", 60),
            "error_rate": self.metrics.get_stats("errors", 60),
            "avg_latency": self.metrics.get_stats("latency", 60)
        }

        # Determine overall health
        healthy_endpoints = sum(1 for e in checks["endpoints"] if e["healthy"])
        error_rate = checks["metrics"].get("error_rate", {}).get("mean", 0)

        if healthy_endpoints == 0 or error_rate > 0.1:
            checks["status"] = ServiceStatus.UNHEALTHY.value
        elif healthy_endpoints < len(self.load_balancer.endpoints) or error_rate > 0.05:
            checks["status"] = ServiceStatus.DEGRADED.value

        self.service_status = ServiceStatus(checks["status"])

        return checks

    async def deploy(self, version: str, stage: DeploymentStage) -> Dict:
        """Deploy new version with strategy."""
        logger.info(f"Deploying {version} to {stage.value}")

        if stage == DeploymentStage.CANARY:
            # Canary deployment
            self.deployment_stage = DeploymentStage.CANARY
            self.canary_percentage = 10

            # Monitor for issues
            await asyncio.sleep(5)  # Simplified monitoring

            # Check metrics
            error_rate = self.metrics.get_stats("errors", 60).get("mean", 0)

            if error_rate > 0.1:
                # Rollback
                logger.warning("High error rate, rolling back")
                self.deployment_stage = DeploymentStage.BLUE
                return {"success": False, "reason": "High error rate"}

            # Gradually increase traffic
            for percentage in [25, 50, 75, 100]:
                self.canary_percentage = percentage
                await asyncio.sleep(2)

                # Check health
                health = await self.health_check()
                if health["status"] == ServiceStatus.UNHEALTHY.value:
                    # Rollback
                    self.deployment_stage = DeploymentStage.BLUE
                    return {"success": False, "reason": "Health check failed"}

            # Success - promote to full
            self.deployment_stage = DeploymentStage.GREEN

        elif stage in [DeploymentStage.BLUE, DeploymentStage.GREEN]:
            # Blue-green deployment
            old_stage = self.deployment_stage
            self.deployment_stage = stage

            # Health check
            await asyncio.sleep(2)
            health = await self.health_check()

            if health["status"] == ServiceStatus.UNHEALTHY.value:
                # Rollback
                self.deployment_stage = old_stage
                return {"success": False, "reason": "Deployment unhealthy"}

        return {
            "success": True,
            "version": version,
            "stage": stage.value,
            "timestamp": datetime.now().isoformat()
        }

    def get_dashboard(self) -> Dict:
        """Get comprehensive dashboard data."""
        return {
            "service_status": self.service_status.value,
            "deployment_stage": self.deployment_stage.value,
            "metrics": {
                "requests": self.metrics.get_stats("requests", 300),
                "errors": self.metrics.get_stats("errors", 300),
                "latency": self.metrics.get_stats("latency", 300),
                "tokens": self.metrics.get_stats("tokens", 300)
            },
            "cache": self.cache.get_stats(),
            "costs": self.cost_tracker.get_report(),
            "endpoints": [
                {
                    "name": e["name"],
                    "healthy": e["healthy"],
                    "requests": e["total_requests"],
                    "connections": e["connections"]
                }
                for e in self.load_balancer.endpoints
            ],
            "circuit_breaker": {
                "state": self.circuit_breaker.state,
                "failures": self.circuit_breaker.failure_count
            }
        }


# ==================== FastAPI Application ====================

app = FastAPI(title="Production LLM Platform", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize platform
platform = ProductionLLMPlatform()


@app.post("/api/generate")
async def generate(request: RequestModel):
    """Generate LLM response."""
    try:
        result = await platform.process_request(request)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return await platform.health_check()


@app.get("/metrics")
async def get_metrics():
    """Get platform metrics."""
    return platform.get_dashboard()


@app.post("/deploy")
async def deploy(version: str, stage: str):
    """Deploy new version."""
    try:
        deployment_stage = DeploymentStage(stage)
        result = await platform.deploy(version, deployment_stage)
        return result
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid deployment stage")


@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring."""
    await websocket.accept()

    try:
        while True:
            # Send dashboard data every 5 seconds
            dashboard = platform.get_dashboard()
            await websocket.send_json(dashboard)
            await asyncio.sleep(5)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


@app.post("/api/batch")
async def batch_generate(requests: List[RequestModel]):
    """Batch process multiple requests."""
    results = []

    for request in requests:
        try:
            result = await platform.process_request(request)
            results.append(result)
        except Exception as e:
            results.append({
                "success": False,
                "error": str(e)
            })

    return {"results": results}


@app.put("/config/budget/{user_id}")
async def set_budget(user_id: str, budget: float):
    """Set user budget."""
    platform.cost_tracker.budgets[user_id] = budget
    return {"user_id": user_id, "budget": budget}


@app.get("/costs/{user_id}")
async def get_user_costs(user_id: str):
    """Get user cost report."""
    return platform.cost_tracker.get_report(user_id)


# ==================== Background Tasks ====================

async def monitor_health():
    """Background task to monitor health."""
    while True:
        await platform.health_check()
        await asyncio.sleep(30)


async def process_queue():
    """Background task to process queued requests."""
    while True:
        if not platform.request_queue.empty():
            request = await platform.request_queue.get()
            await platform.process_request(request)
        await asyncio.sleep(1)


async def cleanup_cache():
    """Background task to clean expired cache entries."""
    while True:
        # Clean expired entries
        now = datetime.now()
        expired = []

        for key, entry in platform.cache.cache.items():
            age = (now - entry["timestamp"]).seconds
            if age > platform.cache.ttl:
                expired.append(key)

        for key in expired:
            del platform.cache.cache[key]

        await asyncio.sleep(300)  # Every 5 minutes


@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup."""
    asyncio.create_task(monitor_health())
    asyncio.create_task(process_queue())
    asyncio.create_task(cleanup_cache())
    logger.info("Production LLM Platform started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Production LLM Platform shutting down")


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Production LLM Platform")
    print("=" * 50)
    print("""
Features:
- Load balancing across multiple endpoints
- Circuit breaker for fault tolerance
- Response caching with TTL
- Cost tracking and budget enforcement
- Comprehensive metrics and monitoring
- Blue-green and canary deployments
- WebSocket real-time monitoring
- Batch processing support
- Health checks and auto-recovery

API Endpoints:
- POST /api/generate - Generate LLM response
- POST /api/batch - Batch process requests
- GET /health - Health check
- GET /metrics - Platform metrics
- POST /deploy - Deploy new version
- PUT /config/budget/{user_id} - Set user budget
- GET /costs/{user_id} - Get user costs
- WS /ws/monitor - Real-time monitoring

Starting server on http://localhost:8000
    """)

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)