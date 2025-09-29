# Module 14: Production Patterns

## Learning Objectives
By the end of this module, you will:
- Master enterprise deployment strategies for LLM applications
- Implement comprehensive monitoring and observability
- Build robust error handling and recovery systems
- Understand compliance, security, and governance requirements
- Deploy and maintain LLM systems at scale

## Key Concepts

### 1. Production Architecture

#### Microservices Pattern
```python
class LLMService:
    """Microservice wrapper for LLM functionality."""

    def __init__(self, service_name, model_config):
        self.name = service_name
        self.model = self.load_model(model_config)
        self.health_check = HealthCheck()
        self.metrics = MetricsCollector()
        self.circuit_breaker = CircuitBreaker()

    async def handle_request(self, request):
        """Handle incoming request with production safeguards."""
        # Rate limiting
        if not self.rate_limiter.allow(request.client_id):
            return {'error': 'Rate limit exceeded', 'status': 429}

        # Circuit breaker check
        if self.circuit_breaker.is_open():
            return {'error': 'Service temporarily unavailable', 'status': 503}

        try:
            # Process request
            with self.metrics.timer('request_duration'):
                result = await self.process(request)

            self.metrics.increment('successful_requests')
            return result

        except Exception as e:
            self.metrics.increment('failed_requests')
            self.circuit_breaker.record_failure()
            return self.handle_error(e)
```

#### API Gateway Pattern
```python
class LLMGateway:
    """API Gateway for LLM services."""

    def __init__(self):
        self.services = {}
        self.router = Router()
        self.auth = AuthenticationMiddleware()
        self.cache = CacheLayer()
        self.logger = Logger()

    async def route_request(self, request):
        """Route request to appropriate service."""
        # Authentication
        if not await self.auth.validate(request):
            return {'error': 'Unauthorized', 'status': 401}

        # Check cache
        cache_key = self.generate_cache_key(request)
        cached = await self.cache.get(cache_key)
        if cached:
            self.logger.info(f"Cache hit for {cache_key}")
            return cached

        # Route to service
        service = self.router.get_service(request.path)
        response = await service.handle(request)

        # Cache successful responses
        if response.get('status') == 200:
            await self.cache.set(cache_key, response, ttl=300)

        return response
```

### 2. Monitoring & Observability

#### Comprehensive Metrics
```python
class LLMMetrics:
    """Production metrics for LLM systems."""

    def __init__(self):
        self.prometheus = PrometheusClient()

        # Define metrics
        self.request_count = Counter('llm_requests_total')
        self.request_duration = Histogram('llm_request_duration_seconds')
        self.token_usage = Counter('llm_tokens_used_total')
        self.error_count = Counter('llm_errors_total')
        self.model_latency = Histogram('llm_model_latency_seconds')
        self.cache_hit_rate = Gauge('llm_cache_hit_rate')

    def track_request(self, request, response, duration):
        """Track comprehensive request metrics."""
        labels = {
            'model': request.model,
            'endpoint': request.endpoint,
            'status': response.status
        }

        self.request_count.inc(labels)
        self.request_duration.observe(duration, labels)

        if response.token_count:
            self.token_usage.inc(response.token_count, labels)

        if response.status >= 400:
            self.error_count.inc(labels)
```

#### Distributed Tracing
```python
class LLMTracer:
    """Distributed tracing for LLM pipelines."""

    def __init__(self):
        self.tracer = OpenTelemetryTracer()

    def trace_llm_call(self, func):
        """Decorator for tracing LLM calls."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with self.tracer.start_span(func.__name__) as span:
                # Add metadata
                span.set_attribute('llm.model', kwargs.get('model'))
                span.set_attribute('llm.prompt_tokens', count_tokens(kwargs.get('prompt')))

                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute('llm.response_tokens', count_tokens(result))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status.ERROR)
                    raise

        return wrapper
```

#### Logging Strategy
```python
class ProductionLogger:
    """Structured logging for production."""

    def __init__(self):
        self.logger = structlog.get_logger()

    def log_llm_request(self, request, response, metadata):
        """Log LLM request with context."""
        self.logger.info(
            "llm_request",
            request_id=request.id,
            model=request.model,
            prompt_hash=hashlib.md5(request.prompt.encode()).hexdigest(),
            response_status=response.status,
            duration_ms=metadata.get('duration_ms'),
            tokens_used=metadata.get('tokens'),
            cost_estimate=metadata.get('cost'),
            user_id=request.user_id,
            trace_id=metadata.get('trace_id')
        )

    def log_error(self, error, context):
        """Log errors with full context."""
        self.logger.error(
            "llm_error",
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            **context
        )
```

### 3. Error Handling & Recovery

#### Retry Strategies
```python
class RetryStrategy:
    """Advanced retry mechanisms."""

    def __init__(self):
        self.strategies = {
            'exponential': self.exponential_backoff,
            'linear': self.linear_backoff,
            'fibonacci': self.fibonacci_backoff
        }

    async def retry_with_backoff(self, func, strategy='exponential', max_retries=3):
        """Execute with intelligent retry."""
        backoff_func = self.strategies[strategy]

        for attempt in range(max_retries):
            try:
                return await func()
            except RetriableError as e:
                if attempt == max_retries - 1:
                    raise

                wait_time = backoff_func(attempt)
                await asyncio.sleep(wait_time)

                # Jitter to prevent thundering herd
                jitter = random.uniform(0, wait_time * 0.1)
                await asyncio.sleep(jitter)

        raise MaxRetriesExceeded()
```

#### Circuit Breaker
```python
class CircuitBreaker:
    """Prevent cascading failures."""

    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func):
        """Execute function with circuit breaker."""
        if self.state == 'OPEN':
            if self.should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise CircuitOpenError()

        try:
            result = func()
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_failure(self):
        """Record failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

    def on_success(self):
        """Record success."""
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
        self.failure_count = 0
```

### 4. Security & Compliance

#### Input Sanitization
```python
class InputSanitizer:
    """Sanitize inputs for security."""

    def __init__(self):
        self.validators = []
        self.filters = []

    def sanitize(self, input_text):
        """Clean and validate input."""
        # Remove potential injection attempts
        cleaned = self.remove_injection_patterns(input_text)

        # Filter sensitive information
        cleaned = self.filter_sensitive_data(cleaned)

        # Validate content
        if not self.validate_content(cleaned):
            raise InvalidInputError()

        return cleaned

    def remove_injection_patterns(self, text):
        """Remove prompt injection attempts."""
        patterns = [
            r'ignore previous instructions',
            r'disregard all prior',
            r'system:\s*',
            r'</?(script|style|iframe)',
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text

    def filter_sensitive_data(self, text):
        """Remove PII and sensitive information."""
        # SSN pattern
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]', text)

        # Credit card pattern
        text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CC REDACTED]', text)

        return text
```

#### Audit Logging
```python
class AuditLogger:
    """Compliance audit logging."""

    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.encryption = EncryptionService()

    def log_llm_interaction(self, request, response, metadata):
        """Log interaction for audit."""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request.id,
            'user_id': request.user_id,
            'model': request.model,
            'prompt_hash': self.hash_content(request.prompt),
            'response_hash': self.hash_content(response.content),
            'metadata': metadata,
            'compliance_flags': self.check_compliance(request, response)
        }

        # Encrypt sensitive fields
        encrypted = self.encryption.encrypt(audit_entry)

        # Store with immutability
        self.storage.append_only_write(encrypted)
```

### 5. Deployment Strategies

#### Blue-Green Deployment
```python
class BlueGreenDeployment:
    """Zero-downtime deployment strategy."""

    def __init__(self):
        self.blue_env = Environment('blue')
        self.green_env = Environment('green')
        self.load_balancer = LoadBalancer()
        self.active = 'blue'

    async def deploy_new_version(self, new_version):
        """Deploy with blue-green strategy."""
        inactive = 'green' if self.active == 'blue' else 'blue'
        target_env = self.green_env if inactive == 'green' else self.blue_env

        # Deploy to inactive environment
        await target_env.deploy(new_version)

        # Run health checks
        if not await target_env.health_check():
            raise DeploymentError("Health check failed")

        # Run smoke tests
        if not await self.run_smoke_tests(target_env):
            raise DeploymentError("Smoke tests failed")

        # Switch traffic
        await self.load_balancer.switch_traffic(inactive)
        self.active = inactive

        # Monitor for issues
        await self.monitor_deployment(duration=300)
```

#### Canary Deployment
```python
class CanaryDeployment:
    """Gradual rollout strategy."""

    def __init__(self):
        self.traffic_manager = TrafficManager()
        self.metrics_monitor = MetricsMonitor()

    async def deploy_canary(self, new_version, stages):
        """Progressive canary deployment."""
        for stage in stages:
            # Update traffic split
            await self.traffic_manager.set_canary_traffic(
                stage['percentage']
            )

            # Monitor metrics
            await asyncio.sleep(stage['duration'])
            metrics = await self.metrics_monitor.get_comparative_metrics()

            # Check success criteria
            if not self.meets_criteria(metrics, stage['criteria']):
                await self.rollback()
                raise CanaryFailure(f"Failed at {stage['percentage']}%")

        # Full rollout
        await self.traffic_manager.set_canary_traffic(100)
```

### 6. Cost Management

#### Usage Tracking
```python
class CostTracker:
    """Track and optimize LLM costs."""

    def __init__(self):
        self.usage_db = UsageDatabase()
        self.pricing = PricingCalculator()

    def track_usage(self, request, response):
        """Track token usage and costs."""
        usage = {
            'timestamp': datetime.utcnow(),
            'user_id': request.user_id,
            'model': request.model,
            'prompt_tokens': count_tokens(request.prompt),
            'response_tokens': count_tokens(response.content),
            'total_tokens': None,  # Will be calculated
            'estimated_cost': None  # Will be calculated
        }

        usage['total_tokens'] = usage['prompt_tokens'] + usage['response_tokens']
        usage['estimated_cost'] = self.pricing.calculate(
            model=usage['model'],
            tokens=usage['total_tokens']
        )

        self.usage_db.insert(usage)

    def get_usage_report(self, period):
        """Generate usage and cost report."""
        data = self.usage_db.query(period)

        return {
            'total_tokens': sum(d['total_tokens'] for d in data),
            'total_cost': sum(d['estimated_cost'] for d in data),
            'by_user': self.aggregate_by_user(data),
            'by_model': self.aggregate_by_model(data),
            'trends': self.calculate_trends(data)
        }
```

#### Cost Optimization
```python
class CostOptimizer:
    """Optimize LLM usage costs."""

    def __init__(self):
        self.cache = ResponseCache()
        self.model_selector = ModelSelector()

    def optimize_request(self, request):
        """Optimize request for cost."""
        # Check cache first
        cached = self.cache.get(request)
        if cached:
            return cached

        # Select most cost-effective model
        model = self.model_selector.select_for_task(
            task_complexity=self.assess_complexity(request),
            quality_requirement=request.quality_requirement,
            budget=request.budget
        )

        # Optimize prompt
        optimized_prompt = self.compress_prompt(request.prompt)

        return {
            'model': model,
            'prompt': optimized_prompt,
            'max_tokens': self.calculate_optimal_max_tokens(request)
        }
```

## Best Practices

### 1. Health Checks
```python
class HealthCheck:
    """Comprehensive health checking."""

    async def check_health(self):
        """Perform health checks."""
        checks = {
            'model_loaded': await self.check_model(),
            'database': await self.check_database(),
            'cache': await self.check_cache(),
            'dependencies': await self.check_dependencies()
        }

        status = 'healthy' if all(checks.values()) else 'unhealthy'

        return {
            'status': status,
            'checks': checks,
            'timestamp': datetime.utcnow().isoformat()
        }
```

### 2. Graceful Degradation
```python
class GracefulDegradation:
    """Fallback strategies for failures."""

    def __init__(self):
        self.fallback_models = ['gpt-4', 'gpt-3.5', 'cached_response']

    async def execute_with_fallback(self, request):
        """Try multiple strategies."""
        for model in self.fallback_models:
            try:
                if model == 'cached_response':
                    return self.get_cached_similar(request)
                else:
                    return await self.execute_with_model(request, model)
            except Exception as e:
                logger.warning(f"Failed with {model}: {e}")
                continue

        return self.get_default_response()
```

### 3. Performance Testing
```python
class LoadTester:
    """Load testing for LLM services."""

    async def run_load_test(self, endpoint, scenarios):
        """Execute load test scenarios."""
        results = []

        for scenario in scenarios:
            # Generate load
            responses = await self.generate_load(
                endpoint,
                scenario['rps'],
                scenario['duration']
            )

            # Analyze results
            analysis = {
                'scenario': scenario['name'],
                'success_rate': self.calculate_success_rate(responses),
                'p50_latency': self.calculate_percentile(responses, 50),
                'p95_latency': self.calculate_percentile(responses, 95),
                'p99_latency': self.calculate_percentile(responses, 99),
                'error_rate': self.calculate_error_rate(responses)
            }

            results.append(analysis)

        return results
```

## Maintenance & Operations

### 1. Model Updates
```python
class ModelManager:
    """Manage model versions and updates."""

    async def update_model(self, new_version):
        """Update model with validation."""
        # Download new model
        model_path = await self.download_model(new_version)

        # Validate model
        if not await self.validate_model(model_path):
            raise InvalidModelError()

        # Run regression tests
        if not await self.run_regression_tests(model_path):
            raise RegressionFailure()

        # Deploy with rollback capability
        await self.deploy_with_rollback(model_path)
```

### 2. Incident Response
```python
class IncidentManager:
    """Handle production incidents."""

    async def handle_incident(self, alert):
        """Respond to production incident."""
        # Classify severity
        severity = self.classify_severity(alert)

        # Page on-call if critical
        if severity == 'CRITICAL':
            await self.page_oncall(alert)

        # Gather context
        context = await self.gather_context(alert)

        # Attempt auto-remediation
        if self.can_auto_remediate(alert):
            await self.auto_remediate(alert)

        # Create incident report
        self.create_incident_report(alert, context)
```

## Exercises Overview

1. **Production Pipeline**: Build complete deployment pipeline
2. **Monitoring System**: Implement comprehensive monitoring
3. **Cost Optimizer**: Create cost optimization system
4. **Security Layer**: Build security and compliance layer
5. **Incident Response**: Design incident management system

## Success Metrics
- **Availability**: >99.9% uptime
- **Latency**: <500ms p95
- **Error Rate**: <0.1%
- **Cost Efficiency**: 30% reduction
- **MTTR**: <30 minutes

## Course Completion
Congratulations! You've completed the comprehensive prompt engineering curriculum. You now have the knowledge to build, deploy, and maintain production-grade LLM applications. Continue practicing with real-world projects and stay updated with the rapidly evolving field of AI engineering.