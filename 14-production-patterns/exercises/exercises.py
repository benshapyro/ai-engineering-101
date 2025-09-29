"""
Module 14: Production Patterns
Exercises

Practice implementing production-ready LLM application patterns.
"""

import os
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import random
from dotenv import load_dotenv

load_dotenv()


# Exercise 1: Implement Load Balancer with Health Checks
print("=" * 50)
print("Exercise 1: Implement Load Balancer with Health Checks")
print("=" * 50)
print("""
Task: Create a load balancer that distributes requests across multiple
LLM endpoints with health monitoring.

Requirements:
1. Support multiple load balancing strategies (round-robin, least connections, weighted)
2. Monitor endpoint health and remove unhealthy ones
3. Implement request queuing when all endpoints are busy
4. Track metrics for each endpoint
""")


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


class LoadBalancer:
    """Load balancer to implement."""

    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.endpoints = []
        self.current_index = 0
        # TODO: Initialize other necessary attributes

    def add_endpoint(self, endpoint: Endpoint):
        """
        Add an endpoint to the load balancer.

        Args:
            endpoint: Endpoint to add
        """
        # TODO: Implement endpoint addition
        pass

    def select_endpoint(self) -> Optional[Endpoint]:
        """
        Select an endpoint based on the strategy.

        Returns:
            Selected endpoint or None if all are unhealthy
        """
        # TODO: Implement endpoint selection based on strategy
        # - round_robin: Rotate through endpoints
        # - least_connections: Choose endpoint with fewest connections
        # - weighted: Select based on endpoint weights
        pass

    async def health_check(self, endpoint: Endpoint) -> bool:
        """
        Check health of an endpoint.

        Args:
            endpoint: Endpoint to check

        Returns:
            True if healthy, False otherwise
        """
        # TODO: Implement health check
        # - Ping the endpoint
        # - Check response time
        # - Update endpoint health status
        pass

    async def execute_request(self, request: Dict) -> Dict:
        """
        Execute request through load balancer.

        Args:
            request: Request to execute

        Returns:
            Response from endpoint
        """
        # TODO: Implement request execution
        # - Select endpoint
        # - Execute request
        # - Update metrics
        # - Handle failures with retry on different endpoint
        pass

    def get_metrics(self) -> Dict:
        """
        Get load balancer metrics.

        Returns:
            Metrics for all endpoints
        """
        # TODO: Return comprehensive metrics
        pass


# Test your implementation
# balancer = LoadBalancer("least_connections")
# balancer.add_endpoint(Endpoint("gpt4-1", "https://api1.example.com"))
# balancer.add_endpoint(Endpoint("gpt4-2", "https://api2.example.com"))
# result = asyncio.run(balancer.execute_request({"prompt": "test"}))
# print(f"Result: {result}")
# print(f"Metrics: {balancer.get_metrics()}")


# Exercise 2: Build Monitoring Dashboard
print("\n" + "=" * 50)
print("Exercise 2: Build Monitoring Dashboard")
print("=" * 50)
print("""
Task: Create a monitoring dashboard that tracks key metrics for your
LLM application.

Requirements:
1. Track request rate, error rate, and latency
2. Monitor token usage and costs
3. Show alerts and anomalies
4. Provide real-time and historical views
""")


class MonitoringDashboard:
    """Monitoring dashboard to implement."""

    def __init__(self, refresh_interval: int = 60):
        self.refresh_interval = refresh_interval
        self.metrics = defaultdict(list)
        self.alerts = []
        # TODO: Initialize other monitoring components

    def record_metric(self, name: str, value: float,
                     tags: Dict[str, str] = None):
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        # TODO: Implement metric recording
        # Store with timestamp
        # Apply tags for filtering
        pass

    def calculate_rate(self, metric_name: str,
                       window_seconds: int = 60) -> float:
        """
        Calculate rate for a metric.

        Args:
            metric_name: Name of metric
            window_seconds: Time window

        Returns:
            Rate per second
        """
        # TODO: Calculate rate over time window
        pass

    def detect_anomaly(self, metric_name: str,
                      threshold_factor: float = 2.0) -> bool:
        """
        Detect anomalies in metrics.

        Args:
            metric_name: Metric to check
            threshold_factor: Factor for anomaly detection

        Returns:
            True if anomaly detected
        """
        # TODO: Implement anomaly detection
        # Compare recent values to historical baseline
        pass

    def generate_alert(self, message: str, severity: str = "warning"):
        """
        Generate an alert.

        Args:
            message: Alert message
            severity: Alert severity
        """
        # TODO: Create and store alert
        pass

    def get_dashboard_data(self) -> Dict:
        """
        Get data for dashboard display.

        Returns:
            Dashboard data including metrics, alerts, and trends
        """
        # TODO: Compile dashboard data
        # - Current metrics
        # - Trends
        # - Active alerts
        # - Top issues
        pass

    def get_cost_analysis(self) -> Dict:
        """
        Analyze costs based on usage.

        Returns:
            Cost breakdown and projections
        """
        # TODO: Calculate costs
        # - By model
        # - By operation
        # - Projected monthly cost
        pass


# Test your implementation
# dashboard = MonitoringDashboard()
# dashboard.record_metric("requests", 1)
# dashboard.record_metric("tokens", 150, {"model": "gpt-4"})
# dashboard.record_metric("latency", 1.2)
# if dashboard.detect_anomaly("latency"):
#     dashboard.generate_alert("High latency detected", "warning")
# print(f"Dashboard: {dashboard.get_dashboard_data()}")


# Exercise 3: Create Resilient API Wrapper
print("\n" + "=" * 50)
print("Exercise 3: Create Resilient API Wrapper")
print("=" * 50)
print("""
Task: Build a resilient wrapper for LLM API calls with comprehensive
error handling.

Requirements:
1. Implement exponential backoff retry
2. Add circuit breaker functionality
3. Support multiple fallback options
4. Include request/response validation
""")


class ResilientAPIWrapper:
    """Resilient API wrapper to implement."""

    def __init__(self, primary_endpoint: str,
                 max_retries: int = 3):
        self.primary_endpoint = primary_endpoint
        self.max_retries = max_retries
        self.circuit_open = False
        self.failure_count = 0
        # TODO: Initialize other resilience components

    async def call_with_retry(self, request: Dict) -> Dict:
        """
        Call API with retry logic.

        Args:
            request: API request

        Returns:
            API response
        """
        # TODO: Implement retry with exponential backoff
        # - Start with base delay
        # - Double delay on each retry
        # - Add jitter to prevent thundering herd
        pass

    def check_circuit(self) -> bool:
        """
        Check if circuit breaker should trip.

        Returns:
            True if circuit should open
        """
        # TODO: Implement circuit breaker logic
        # - Track failure count
        # - Open circuit after threshold
        # - Implement recovery timeout
        pass

    async def execute_with_fallback(self, request: Dict,
                                   fallbacks: List[Callable] = None) -> Dict:
        """
        Execute request with fallback options.

        Args:
            request: Request to execute
            fallbacks: List of fallback functions

        Returns:
            Response from primary or fallback
        """
        # TODO: Implement fallback chain
        # - Try primary endpoint
        # - On failure, try each fallback in order
        # - Track which option succeeded
        pass

    def validate_request(self, request: Dict) -> bool:
        """
        Validate request before sending.

        Args:
            request: Request to validate

        Returns:
            True if valid
        """
        # TODO: Implement request validation
        # - Check required fields
        # - Validate prompt length
        # - Check parameter ranges
        pass

    def validate_response(self, response: Dict) -> bool:
        """
        Validate response from API.

        Args:
            response: Response to validate

        Returns:
            True if valid
        """
        # TODO: Implement response validation
        # - Check response structure
        # - Validate content
        # - Detect error responses
        pass


# Test your implementation
# wrapper = ResilientAPIWrapper("https://api.example.com")
# request = {"prompt": "Test prompt", "max_tokens": 100}
# if wrapper.validate_request(request):
#     response = asyncio.run(wrapper.call_with_retry(request))
#     if wrapper.validate_response(response):
#         print(f"Success: {response}")


# Exercise 4: Design Deployment Pipeline
print("\n" + "=" * 50)
print("Exercise 4: Design Deployment Pipeline")
print("=" * 50)
print("""
Task: Create a deployment pipeline for rolling out LLM application updates.

Requirements:
1. Implement staged deployment (dev → staging → production)
2. Add automated testing at each stage
3. Support rollback on failure
4. Include deployment metrics and monitoring
""")


class DeploymentStage(Enum):
    """Deployment stages."""
    DEV = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentPipeline:
    """Deployment pipeline to implement."""

    def __init__(self):
        self.stages = [DeploymentStage.DEV, DeploymentStage.STAGING,
                      DeploymentStage.PRODUCTION]
        self.current_version = None
        self.deployment_history = []
        # TODO: Initialize other pipeline components

    async def validate_deployment(self, version: str,
                                 stage: DeploymentStage) -> bool:
        """
        Validate deployment readiness.

        Args:
            version: Version to deploy
            stage: Target stage

        Returns:
            True if validation passes
        """
        # TODO: Implement validation
        # - Run tests
        # - Check dependencies
        # - Verify configuration
        pass

    async def run_tests(self, stage: DeploymentStage) -> Dict:
        """
        Run tests for a deployment stage.

        Args:
            stage: Deployment stage

        Returns:
            Test results
        """
        # TODO: Implement test execution
        # - Unit tests
        # - Integration tests
        # - Performance tests
        # - Return pass/fail status and details
        pass

    async def deploy_to_stage(self, version: str,
                             stage: DeploymentStage) -> bool:
        """
        Deploy version to a stage.

        Args:
            version: Version to deploy
            stage: Target stage

        Returns:
            True if deployment succeeds
        """
        # TODO: Implement deployment
        # - Validate first
        # - Deploy to stage
        # - Run smoke tests
        # - Update routing
        pass

    async def rollback(self, stage: DeploymentStage):
        """
        Rollback deployment in a stage.

        Args:
            stage: Stage to rollback
        """
        # TODO: Implement rollback
        # - Get previous version
        # - Deploy previous version
        # - Verify rollback success
        pass

    async def promote_version(self, version: str,
                             from_stage: DeploymentStage,
                             to_stage: DeploymentStage) -> bool:
        """
        Promote version between stages.

        Args:
            version: Version to promote
            from_stage: Source stage
            to_stage: Target stage

        Returns:
            True if promotion succeeds
        """
        # TODO: Implement promotion
        # - Verify version in source stage
        # - Run target stage validation
        # - Deploy to target
        # - Monitor for issues
        pass

    def get_deployment_status(self) -> Dict:
        """
        Get current deployment status.

        Returns:
            Status of all stages
        """
        # TODO: Return deployment status
        # - Current version per stage
        # - Health status
        # - Recent deployments
        pass


# Test your implementation
# pipeline = DeploymentPipeline()
# success = asyncio.run(pipeline.deploy_to_stage("v1.2.0", DeploymentStage.DEV))
# if success:
#     test_results = asyncio.run(pipeline.run_tests(DeploymentStage.DEV))
#     print(f"Test results: {test_results}")
#     if test_results.get("passed"):
#         asyncio.run(pipeline.promote_version("v1.2.0",
#                                             DeploymentStage.DEV,
#                                             DeploymentStage.STAGING))


# Exercise 5: Build Cost Optimization System
print("\n" + "=" * 50)
print("Exercise 5: Build Cost Optimization System")
print("=" * 50)
print("""
Task: Create a system to optimize and reduce LLM usage costs.

Requirements:
1. Track costs by model, user, and operation
2. Implement intelligent caching
3. Suggest cheaper alternatives when appropriate
4. Set and enforce budget limits
""")


class CostOptimizer:
    """Cost optimization system to implement."""

    def __init__(self):
        self.model_costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "claude-3-opus": {"input": 0.015, "output": 0.075}
        }
        self.usage_data = defaultdict(lambda: {"tokens": 0, "cost": 0})
        self.cache = {}
        self.budgets = {}
        # TODO: Initialize other components

    def track_usage(self, user_id: str, model: str,
                   input_tokens: int, output_tokens: int):
        """
        Track token usage and costs.

        Args:
            user_id: User identifier
            model: Model used
            input_tokens: Input token count
            output_tokens: Output token count
        """
        # TODO: Record usage and calculate cost
        pass

    def check_budget(self, user_id: str) -> bool:
        """
        Check if user is within budget.

        Args:
            user_id: User to check

        Returns:
            True if within budget
        """
        # TODO: Check against budget limits
        pass

    def suggest_optimization(self, request: Dict) -> Dict:
        """
        Suggest cost optimizations for a request.

        Args:
            request: Request to optimize

        Returns:
            Optimization suggestions
        """
        # TODO: Analyze request and suggest optimizations
        # - Use cheaper model for simple tasks
        # - Reduce token count
        # - Use cached response if available
        pass

    def cache_response(self, request_hash: str, response: Dict,
                      ttl: int = 3600):
        """
        Cache a response for reuse.

        Args:
            request_hash: Hash of request
            response: Response to cache
            ttl: Time to live in seconds
        """
        # TODO: Implement caching with TTL
        pass

    def get_cached_response(self, request_hash: str) -> Optional[Dict]:
        """
        Get cached response if available.

        Args:
            request_hash: Hash of request

        Returns:
            Cached response or None
        """
        # TODO: Retrieve from cache if not expired
        pass

    def generate_cost_report(self, user_id: str = None) -> Dict:
        """
        Generate cost report.

        Args:
            user_id: Optional user filter

        Returns:
            Cost report with breakdown and recommendations
        """
        # TODO: Generate comprehensive cost report
        # - Total costs
        # - Breakdown by model/operation
        # - Savings from caching
        # - Optimization recommendations
        pass


# Test your implementation
# optimizer = CostOptimizer()
# optimizer.budgets["user_1"] = 10.0  # $10 budget
# optimizer.track_usage("user_1", "gpt-4", 1000, 500)
# if not optimizer.check_budget("user_1"):
#     print("Budget exceeded!")
# suggestions = optimizer.suggest_optimization({"model": "gpt-4", "prompt": "simple question"})
# print(f"Suggestions: {suggestions}")
# report = optimizer.generate_cost_report("user_1")
# print(f"Cost report: {report}")


# Challenge Exercise: Complete Production-Ready LLM Service
print("\n" + "=" * 50)
print("CHALLENGE: Complete Production-Ready LLM Service")
print("=" * 50)
print("""
Task: Build a complete production-ready LLM service combining all patterns.

Requirements:
1. Load balancing across multiple endpoints
2. Comprehensive monitoring and alerting
3. Error resilience with retries and fallbacks
4. Deployment pipeline with staging
5. Cost optimization and budgeting
6. Health checks and graceful degradation
7. Request/response validation
8. Caching and performance optimization

The service should:
- Handle high traffic with auto-scaling
- Recover gracefully from failures
- Minimize costs while maintaining quality
- Provide detailed observability
- Support safe deployments with rollback
""")


class ProductionLLMService:
    """Complete production LLM service to implement."""

    def __init__(self, service_name: str):
        self.service_name = service_name

        # TODO: Initialize all components
        self.load_balancer = None
        self.monitoring = None
        self.resilience = None
        self.deployment = None
        self.cost_optimizer = None
        self.health_checks = []

    async def initialize(self):
        """
        Initialize all service components.
        """
        # TODO: Set up all components
        # - Configure load balancer
        # - Set up monitoring
        # - Initialize resilience mechanisms
        # - Configure deployment pipeline
        # - Set up cost optimization
        pass

    async def process_request(self, request: Dict,
                             user_id: str = None) -> Dict:
        """
        Process an LLM request through the complete pipeline.

        Args:
            request: Request to process
            user_id: Optional user identifier

        Returns:
            Processed response
        """
        # TODO: Implement complete request processing
        # 1. Validate request
        # 2. Check user budget
        # 3. Check cache
        # 4. Select endpoint via load balancer
        # 5. Execute with resilience (retry, fallback)
        # 6. Validate response
        # 7. Update cache
        # 8. Record metrics
        # 9. Track costs
        # 10. Return response
        pass

    async def health_check(self) -> Dict:
        """
        Run comprehensive health checks.

        Returns:
            Health status of all components
        """
        # TODO: Check health of all components
        # - Endpoint availability
        # - System resources
        # - Error rates
        # - Performance metrics
        pass

    async def auto_scale(self):
        """
        Auto-scale based on load and performance.
        """
        # TODO: Implement auto-scaling
        # - Monitor current load
        # - Check performance metrics
        # - Scale up/down as needed
        # - Update load balancer
        pass

    async def deploy_update(self, version: str) -> bool:
        """
        Deploy an update to the service.

        Args:
            version: Version to deploy

        Returns:
            True if deployment succeeds
        """
        # TODO: Implement safe deployment
        # - Validate new version
        # - Deploy to staging
        # - Run tests
        # - Gradual rollout to production
        # - Monitor for issues
        # - Automatic rollback if needed
        pass

    def get_service_dashboard(self) -> Dict:
        """
        Get comprehensive service dashboard data.

        Returns:
            Dashboard with all key metrics and status
        """
        # TODO: Compile dashboard data
        # - Current traffic
        # - Error rates
        # - Latency percentiles
        # - Cost metrics
        # - Health status
        # - Active alerts
        # - Endpoint status
        pass

    async def graceful_shutdown(self):
        """
        Perform graceful shutdown of the service.
        """
        # TODO: Implement graceful shutdown
        # - Stop accepting new requests
        # - Complete in-flight requests
        # - Save state
        # - Clean up resources
        pass


# Test your complete implementation
# service = ProductionLLMService("llm_api_service")
# asyncio.run(service.initialize())
#
# # Process a request
# request = {
#     "prompt": "Explain quantum computing",
#     "max_tokens": 200,
#     "temperature": 0.7
# }
# response = asyncio.run(service.process_request(request, "user_123"))
# print(f"Response: {response}")
#
# # Check health
# health = asyncio.run(service.health_check())
# print(f"Health: {health}")
#
# # Get dashboard
# dashboard = service.get_service_dashboard()
# print(f"Dashboard: {dashboard}")


print("\n" + "=" * 50)
print("Exercises Complete!")
print("=" * 50)
print("""
These exercises cover essential production patterns:
1. Load balancing with health monitoring
2. Comprehensive monitoring dashboards
3. Resilient API wrappers
4. Deployment pipelines
5. Cost optimization systems
6. Complete production-ready service

Try implementing each exercise to build robust production LLM applications!
""")