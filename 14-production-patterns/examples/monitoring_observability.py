"""
Module 14: Production Patterns
Monitoring and Observability Examples

Learn to implement comprehensive monitoring for production LLM applications.
"""

import os
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import logging
from enum import Enum
import traceback
import numpy as np
from dotenv import load_dotenv

load_dotenv()


# Example 1: Metrics Collection System
print("=" * 50)
print("Example 1: Metrics Collection System")
print("=" * 50)


class MetricType(Enum):
    """Types of metrics we collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Represents a single metric."""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self, namespace: str = "llm_app"):
        self.namespace = namespace
        self.metrics = defaultdict(list)
        self.aggregation_interval = 60  # seconds

    def record(self, name: str, value: float,
               metric_type: MetricType = MetricType.GAUGE,
               labels: Dict[str, str] = None, unit: str = ""):
        """Record a metric value."""
        metric = Metric(
            name=f"{self.namespace}.{name}",
            type=metric_type,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            unit=unit
        )
        self.metrics[metric.name].append(metric)

    def record_latency(self, operation: str, duration: float):
        """Record operation latency."""
        self.record(
            f"latency.{operation}",
            duration,
            MetricType.HISTOGRAM,
            unit="seconds"
        )

    def record_token_usage(self, model: str, input_tokens: int,
                          output_tokens: int):
        """Record token usage metrics."""
        self.record(
            "tokens.input",
            input_tokens,
            MetricType.COUNTER,
            {"model": model}
        )
        self.record(
            "tokens.output",
            output_tokens,
            MetricType.COUNTER,
            {"model": model}
        )

    def record_error(self, error_type: str, operation: str):
        """Record error occurrence."""
        self.record(
            "errors",
            1,
            MetricType.COUNTER,
            {"type": error_type, "operation": operation}
        )

    def get_aggregated_metrics(self,
                               time_window: timedelta = None) -> Dict:
        """Get aggregated metrics for time window."""
        if not time_window:
            time_window = timedelta(seconds=self.aggregation_interval)

        cutoff_time = datetime.now() - time_window
        aggregated = {}

        for metric_name, metric_list in self.metrics.items():
            recent_metrics = [
                m for m in metric_list
                if m.timestamp > cutoff_time
            ]

            if not recent_metrics:
                continue

            values = [m.value for m in recent_metrics]
            metric_type = recent_metrics[0].type

            if metric_type == MetricType.COUNTER:
                aggregated[metric_name] = sum(values)
            elif metric_type == MetricType.GAUGE:
                aggregated[metric_name] = values[-1]  # Latest value
            elif metric_type == MetricType.HISTOGRAM:
                aggregated[metric_name] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p50": np.percentile(values, 50),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99),
                    "count": len(values)
                }

        return aggregated


# Test metrics collection
collector = MetricsCollector()

# Simulate some operations
for i in range(10):
    # Record latency
    collector.record_latency("api_call", 0.1 + i * 0.02)

    # Record token usage
    collector.record_token_usage("gpt-4", 100 + i * 10, 50 + i * 5)

    # Occasionally record errors
    if i % 3 == 0:
        collector.record_error("RateLimitError", "api_call")

# Get aggregated metrics
metrics = collector.get_aggregated_metrics()
print("Aggregated Metrics:")
for name, value in metrics.items():
    print(f"  {name}: {value}")


# Example 2: Distributed Tracing
print("\n" + "=" * 50)
print("Example 2: Distributed Tracing")
print("=" * 50)


@dataclass
class Span:
    """Represents a trace span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "in_progress"
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict] = field(default_factory=list)


class DistributedTracer:
    """Implements distributed tracing for LLM operations."""

    def __init__(self):
        self.traces = {}
        self.active_spans = {}

    def start_trace(self, operation_name: str) -> str:
        """Start a new trace."""
        import uuid
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            operation_name=operation_name,
            start_time=datetime.now()
        )

        self.traces[trace_id] = [span]
        self.active_spans[trace_id] = span

        return trace_id

    def start_span(self, trace_id: str, operation_name: str,
                   parent_span_id: str = None) -> str:
        """Start a new span within a trace."""
        import uuid
        span_id = str(uuid.uuid4())

        if not parent_span_id and trace_id in self.active_spans:
            parent_span_id = self.active_spans[trace_id].span_id

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now()
        )

        self.traces[trace_id].append(span)
        self.active_spans[f"{trace_id}.{span_id}"] = span

        return span_id

    def end_span(self, trace_id: str, span_id: str,
                 status: str = "success"):
        """End a span."""
        key = f"{trace_id}.{span_id}"
        if key in self.active_spans:
            span = self.active_spans[key]
            span.end_time = datetime.now()
            span.status = status
            del self.active_spans[key]

    def add_tag(self, trace_id: str, span_id: str,
                key: str, value: Any):
        """Add tag to a span."""
        for span in self.traces.get(trace_id, []):
            if span.span_id == span_id:
                span.tags[key] = value
                break

    def add_log(self, trace_id: str, span_id: str,
                message: str, level: str = "info"):
        """Add log to a span."""
        for span in self.traces.get(trace_id, []):
            if span.span_id == span_id:
                span.logs.append({
                    "timestamp": datetime.now().isoformat(),
                    "level": level,
                    "message": message
                })
                break

    def get_trace_timeline(self, trace_id: str) -> List[Dict]:
        """Get timeline view of a trace."""
        if trace_id not in self.traces:
            return []

        timeline = []
        for span in self.traces[trace_id]:
            duration = None
            if span.end_time:
                duration = (span.end_time - span.start_time).total_seconds()

            timeline.append({
                "operation": span.operation_name,
                "start": span.start_time.isoformat(),
                "duration": duration,
                "status": span.status,
                "tags": span.tags
            })

        return timeline


# Test distributed tracing
tracer = DistributedTracer()

# Simulate a complex operation
trace_id = tracer.start_trace("process_user_request")
tracer.add_tag(trace_id, tracer.active_spans[trace_id].span_id,
               "user_id", "user_123")

# Preprocessing span
preprocess_span = tracer.start_span(trace_id, "preprocess_input")
tracer.add_log(trace_id, preprocess_span, "Cleaning input text")
time.sleep(0.1)
tracer.end_span(trace_id, preprocess_span)

# API call span
api_span = tracer.start_span(trace_id, "llm_api_call")
tracer.add_tag(trace_id, api_span, "model", "gpt-4")
tracer.add_tag(trace_id, api_span, "tokens", 150)
time.sleep(0.2)
tracer.end_span(trace_id, api_span)

# Postprocessing span
post_span = tracer.start_span(trace_id, "postprocess_output")
time.sleep(0.05)
tracer.end_span(trace_id, post_span)

# Get trace timeline
timeline = tracer.get_trace_timeline(trace_id)
print("Trace Timeline:")
for event in timeline:
    print(f"  {event['operation']}: {event['duration']:.3f}s - {event['status']}")


# Example 3: Structured Logging
print("\n" + "=" * 50)
print("Example 3: Structured Logging")
print("=" * 50)


class StructuredLogger:
    """Structured logging for production systems."""

    def __init__(self, service_name: str, environment: str = "production"):
        self.service_name = service_name
        self.environment = environment
        self.context = {}

        # Configure logger
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(self._get_json_formatter())
        self.logger.addHandler(handler)

    def _get_json_formatter(self):
        """Create JSON log formatter."""
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_obj = {
                    "timestamp": datetime.now().isoformat(),
                    "level": record.levelname,
                    "service": self.service_name,
                    "environment": self.environment,
                    "message": record.getMessage(),
                    "logger": record.name,
                    "thread": record.thread,
                    "process": record.process
                }

                # Add extra fields
                if hasattr(record, 'extra_fields'):
                    log_obj.update(record.extra_fields)

                # Add context
                log_obj.update(self.context)

                # Add exception info if present
                if record.exc_info:
                    log_obj["exception"] = {
                        "type": record.exc_info[0].__name__,
                        "message": str(record.exc_info[1]),
                        "traceback": traceback.format_exception(*record.exc_info)
                    }

                return json.dumps(log_obj)

        return JsonFormatter()

    def set_context(self, **kwargs):
        """Set context that will be included in all logs."""
        self.context.update(kwargs)

    def clear_context(self):
        """Clear context."""
        self.context = {}

    def info(self, message: str, **fields):
        """Log info message."""
        self.logger.info(message, extra={"extra_fields": fields})

    def error(self, message: str, exception: Exception = None, **fields):
        """Log error message."""
        if exception:
            self.logger.error(message, exc_info=True,
                            extra={"extra_fields": fields})
        else:
            self.logger.error(message, extra={"extra_fields": fields})

    def warning(self, message: str, **fields):
        """Log warning message."""
        self.logger.warning(message, extra={"extra_fields": fields})

    def debug(self, message: str, **fields):
        """Log debug message."""
        self.logger.debug(message, extra={"extra_fields": fields})


# Test structured logging
logger = StructuredLogger("llm_service", "development")

# Set request context
logger.set_context(
    request_id="req_abc123",
    user_id="user_456",
    session_id="sess_789"
)

# Log various events
logger.info("Request received",
           endpoint="/api/generate",
           method="POST")

logger.info("LLM call started",
           model="gpt-4",
           max_tokens=500,
           temperature=0.7)

try:
    # Simulate an error
    raise ValueError("Invalid prompt format")
except Exception as e:
    logger.error("LLM call failed",
                exception=e,
                retry_count=2,
                fallback_enabled=True)

logger.warning("Rate limit approaching",
              current_rate=95,
              limit=100,
              reset_in_seconds=30)

logger.clear_context()


# Example 4: Alerting System
print("\n" + "=" * 50)
print("Example 4: Alerting System")
print("=" * 50)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents an alert."""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class AlertingSystem:
    """Production alerting system."""

    def __init__(self):
        self.alerts = {}
        self.alert_rules = []
        self.notification_channels = []
        self.alert_history = deque(maxlen=1000)

    def add_rule(self, name: str, condition_fn,
                 severity: AlertSeverity, message_template: str):
        """Add an alert rule."""
        self.alert_rules.append({
            "name": name,
            "condition": condition_fn,
            "severity": severity,
            "message_template": message_template
        })

    def add_notification_channel(self, channel_fn):
        """Add notification channel."""
        self.notification_channels.append(channel_fn)

    def check_metrics(self, metrics: Dict) -> List[Alert]:
        """Check metrics against alert rules."""
        triggered_alerts = []

        for rule in self.alert_rules:
            if rule["condition"](metrics):
                import uuid
                alert = Alert(
                    id=str(uuid.uuid4()),
                    name=rule["name"],
                    severity=rule["severity"],
                    message=rule["message_template"].format(**metrics),
                    timestamp=datetime.now(),
                    metadata=metrics
                )

                # Check if alert already exists
                existing = None
                for alert_id, existing_alert in self.alerts.items():
                    if (existing_alert.name == alert.name and
                        not existing_alert.resolved):
                        existing = existing_alert
                        break

                if not existing:
                    self.alerts[alert.id] = alert
                    triggered_alerts.append(alert)
                    self._send_notifications(alert)

        return triggered_alerts

    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            self.alert_history.append(alert)

    def _send_notifications(self, alert: Alert):
        """Send alert notifications."""
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                print(f"Failed to send notification: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [a for a in self.alerts.values() if not a.resolved]

    def get_alert_summary(self) -> Dict:
        """Get alert summary statistics."""
        active_alerts = self.get_active_alerts()

        summary = {
            "total_active": len(active_alerts),
            "by_severity": defaultdict(int),
            "oldest_alert": None,
            "most_frequent": None
        }

        for alert in active_alerts:
            summary["by_severity"][alert.severity.value] += 1

        if active_alerts:
            summary["oldest_alert"] = min(
                active_alerts,
                key=lambda a: a.timestamp
            ).name

        # Find most frequent alert type
        alert_counts = defaultdict(int)
        for alert in self.alert_history:
            alert_counts[alert.name] += 1

        if alert_counts:
            summary["most_frequent"] = max(
                alert_counts.items(),
                key=lambda x: x[1]
            )[0]

        return summary


# Test alerting system
alerting = AlertingSystem()

# Define alert rules
alerting.add_rule(
    "high_error_rate",
    lambda m: m.get("error_rate", 0) > 0.05,
    AlertSeverity.ERROR,
    "Error rate is {error_rate:.1%}"
)

alerting.add_rule(
    "high_latency",
    lambda m: m.get("p95_latency", 0) > 2.0,
    AlertSeverity.WARNING,
    "P95 latency is {p95_latency:.2f}s"
)

alerting.add_rule(
    "token_limit",
    lambda m: m.get("daily_tokens", 0) > 900000,
    AlertSeverity.CRITICAL,
    "Daily token usage at {daily_tokens:,} tokens"
)

# Add notification channel
def console_notifier(alert: Alert):
    print(f"🚨 [{alert.severity.value.upper()}] {alert.name}: {alert.message}")

alerting.add_notification_channel(console_notifier)

# Check metrics
test_metrics = {
    "error_rate": 0.08,
    "p95_latency": 2.5,
    "daily_tokens": 950000
}

triggered = alerting.check_metrics(test_metrics)
print(f"\nTriggered {len(triggered)} alerts")

# Get summary
summary = alerting.get_alert_summary()
print(f"Alert Summary: {summary}")


# Example 5: Performance Monitoring
print("\n" + "=" * 50)
print("Example 5: Performance Monitoring")
print("=" * 50)


class PerformanceMonitor:
    """Monitor LLM application performance."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latency_windows = defaultdict(lambda: deque(maxlen=window_size))
        self.token_usage = defaultdict(lambda: deque(maxlen=window_size))
        self.cache_metrics = {"hits": 0, "misses": 0}
        self.model_performance = defaultdict(dict)

    def record_request(self, operation: str, latency: float,
                      input_tokens: int, output_tokens: int,
                      model: str, cache_hit: bool = False):
        """Record a request's performance metrics."""
        # Record latency
        self.latency_windows[operation].append(latency)

        # Record token usage
        self.token_usage[model].append({
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
            "timestamp": datetime.now()
        })

        # Update cache metrics
        if cache_hit:
            self.cache_metrics["hits"] += 1
        else:
            self.cache_metrics["misses"] += 1

        # Update model performance
        if model not in self.model_performance:
            self.model_performance[model] = {
                "total_requests": 0,
                "total_latency": 0,
                "total_tokens": 0,
                "errors": 0
            }

        self.model_performance[model]["total_requests"] += 1
        self.model_performance[model]["total_latency"] += latency
        self.model_performance[model]["total_tokens"] += input_tokens + output_tokens

    def get_latency_stats(self, operation: str) -> Dict:
        """Get latency statistics for an operation."""
        if operation not in self.latency_windows:
            return {}

        latencies = list(self.latency_windows[operation])
        if not latencies:
            return {}

        return {
            "min": min(latencies),
            "max": max(latencies),
            "avg": sum(latencies) / len(latencies),
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "count": len(latencies)
        }

    def get_token_stats(self, model: str = None) -> Dict:
        """Get token usage statistics."""
        if model:
            usage = list(self.token_usage[model])
        else:
            usage = []
            for model_usage in self.token_usage.values():
                usage.extend(list(model_usage))

        if not usage:
            return {}

        input_tokens = [u["input"] for u in usage]
        output_tokens = [u["output"] for u in usage]
        total_tokens = [u["total"] for u in usage]

        return {
            "input": {
                "total": sum(input_tokens),
                "avg": sum(input_tokens) / len(input_tokens)
            },
            "output": {
                "total": sum(output_tokens),
                "avg": sum(output_tokens) / len(output_tokens)
            },
            "total": {
                "sum": sum(total_tokens),
                "avg": sum(total_tokens) / len(total_tokens)
            }
        }

    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.cache_metrics["hits"] + self.cache_metrics["misses"]
        if total == 0:
            return 0.0
        return self.cache_metrics["hits"] / total

    def get_model_comparison(self) -> Dict:
        """Compare performance across models."""
        comparison = {}

        for model, stats in self.model_performance.items():
            if stats["total_requests"] > 0:
                comparison[model] = {
                    "avg_latency": stats["total_latency"] / stats["total_requests"],
                    "avg_tokens": stats["total_tokens"] / stats["total_requests"],
                    "error_rate": stats["errors"] / stats["total_requests"],
                    "total_requests": stats["total_requests"]
                }

        return comparison

    def detect_anomalies(self, threshold_factor: float = 2.0) -> List[str]:
        """Detect performance anomalies."""
        anomalies = []

        for operation, latencies in self.latency_windows.items():
            if len(latencies) < 10:
                continue

            recent = list(latencies)[-10:]
            historical = list(latencies)[:-10]

            if historical:
                historical_avg = sum(historical) / len(historical)
                recent_avg = sum(recent) / len(recent)

                if recent_avg > historical_avg * threshold_factor:
                    anomalies.append(
                        f"{operation} latency increased {recent_avg/historical_avg:.1f}x"
                    )

        return anomalies


# Test performance monitoring
monitor = PerformanceMonitor()

# Simulate requests
import random

models = ["gpt-4", "gpt-3.5-turbo", "claude-3"]
operations = ["generate", "summarize", "translate"]

for i in range(50):
    model = random.choice(models)
    operation = random.choice(operations)

    # Simulate varying performance
    base_latency = {"gpt-4": 1.0, "gpt-3.5-turbo": 0.5, "claude-3": 0.8}
    latency = base_latency.get(model, 1.0) + random.uniform(-0.2, 0.5)

    # Introduce anomaly
    if i > 40 and operation == "generate":
        latency *= 3  # Spike in latency

    monitor.record_request(
        operation=operation,
        latency=latency,
        input_tokens=random.randint(50, 200),
        output_tokens=random.randint(100, 500),
        model=model,
        cache_hit=random.random() < 0.3
    )

# Get performance stats
print("\nLatency Stats (generate):")
print(monitor.get_latency_stats("generate"))

print("\nToken Stats (gpt-4):")
print(monitor.get_token_stats("gpt-4"))

print(f"\nCache Hit Rate: {monitor.get_cache_hit_rate():.1%}")

print("\nModel Comparison:")
for model, stats in monitor.get_model_comparison().items():
    print(f"  {model}: {stats}")

# Detect anomalies
anomalies = monitor.detect_anomalies()
if anomalies:
    print("\n⚠️ Anomalies Detected:")
    for anomaly in anomalies:
        print(f"  - {anomaly}")


# Example 6: Cost Tracking
print("\n" + "=" * 50)
print("Example 6: Cost Tracking")
print("=" * 50)


class CostTracker:
    """Track and optimize LLM usage costs."""

    # Cost per 1K tokens (example rates)
    MODEL_COSTS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015}
    }

    def __init__(self):
        self.usage_by_model = defaultdict(lambda: {"input": 0, "output": 0})
        self.usage_by_user = defaultdict(lambda: defaultdict(lambda: {"input": 0, "output": 0}))
        self.usage_by_operation = defaultdict(lambda: defaultdict(lambda: {"input": 0, "output": 0}))
        self.daily_usage = defaultdict(lambda: defaultdict(lambda: {"input": 0, "output": 0}))

    def track_usage(self, model: str, input_tokens: int,
                   output_tokens: int, user_id: str = None,
                   operation: str = None):
        """Track token usage."""
        # Track by model
        self.usage_by_model[model]["input"] += input_tokens
        self.usage_by_model[model]["output"] += output_tokens

        # Track by user
        if user_id:
            self.usage_by_user[user_id][model]["input"] += input_tokens
            self.usage_by_user[user_id][model]["output"] += output_tokens

        # Track by operation
        if operation:
            self.usage_by_operation[operation][model]["input"] += input_tokens
            self.usage_by_operation[operation][model]["output"] += output_tokens

        # Track daily
        today = datetime.now().date().isoformat()
        self.daily_usage[today][model]["input"] += input_tokens
        self.daily_usage[today][model]["output"] += output_tokens

    def calculate_cost(self, model: str, input_tokens: int,
                      output_tokens: int) -> float:
        """Calculate cost for token usage."""
        if model not in self.MODEL_COSTS:
            return 0.0

        costs = self.MODEL_COSTS[model]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]

        return input_cost + output_cost

    def get_total_cost(self) -> Dict[str, float]:
        """Get total cost by model."""
        costs = {}

        for model, usage in self.usage_by_model.items():
            costs[model] = self.calculate_cost(
                model,
                usage["input"],
                usage["output"]
            )

        costs["total"] = sum(costs.values())
        return costs

    def get_user_costs(self, user_id: str) -> Dict[str, float]:
        """Get costs for a specific user."""
        if user_id not in self.usage_by_user:
            return {"total": 0.0}

        costs = {}
        for model, usage in self.usage_by_user[user_id].items():
            costs[model] = self.calculate_cost(
                model,
                usage["input"],
                usage["output"]
            )

        costs["total"] = sum(costs.values())
        return costs

    def get_operation_costs(self) -> Dict[str, Dict[str, float]]:
        """Get costs by operation."""
        operation_costs = {}

        for operation, model_usage in self.usage_by_operation.items():
            operation_costs[operation] = {}

            for model, usage in model_usage.items():
                operation_costs[operation][model] = self.calculate_cost(
                    model,
                    usage["input"],
                    usage["output"]
                )

            operation_costs[operation]["total"] = sum(
                operation_costs[operation].values()
            )

        return operation_costs

    def get_daily_trend(self, days: int = 7) -> List[Dict]:
        """Get daily cost trend."""
        trend = []

        for i in range(days):
            date = (datetime.now().date() - timedelta(days=i)).isoformat()

            if date in self.daily_usage:
                daily_cost = 0
                for model, usage in self.daily_usage[date].items():
                    daily_cost += self.calculate_cost(
                        model,
                        usage["input"],
                        usage["output"]
                    )

                trend.append({
                    "date": date,
                    "cost": daily_cost,
                    "models": list(self.daily_usage[date].keys())
                })

        return list(reversed(trend))

    def get_cost_optimization_suggestions(self) -> List[str]:
        """Get cost optimization suggestions."""
        suggestions = []

        # Check for expensive model overuse
        total_costs = self.get_total_cost()
        if "gpt-4" in total_costs and total_costs["gpt-4"] > total_costs.get("total", 0) * 0.8:
            suggestions.append(
                "Consider using gpt-3.5-turbo for simpler tasks - GPT-4 is 80% of costs"
            )

        # Check operation costs
        op_costs = self.get_operation_costs()
        for operation, costs in op_costs.items():
            if "gpt-4" in costs and costs["gpt-4"] > costs["total"] * 0.5:
                suggestions.append(
                    f"Operation '{operation}' uses expensive models - consider optimization"
                )

        # Check for high-volume users
        for user_id, models in self.usage_by_user.items():
            user_cost = self.get_user_costs(user_id)
            if user_cost["total"] > total_costs.get("total", 0) * 0.3:
                suggestions.append(
                    f"User {user_id} is 30%+ of costs - consider rate limiting"
                )

        return suggestions


# Test cost tracking
tracker = CostTracker()

# Simulate usage patterns
users = ["user_1", "user_2", "user_3"]
operations = ["chat", "summarize", "translate", "code_generation"]

for _ in range(100):
    model = random.choices(
        ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"],
        weights=[0.2, 0.6, 0.2]
    )[0]

    user = random.choice(users)
    operation = random.choice(operations)

    # Different token usage by operation
    if operation == "code_generation":
        input_tokens = random.randint(200, 500)
        output_tokens = random.randint(500, 1500)
    else:
        input_tokens = random.randint(50, 200)
        output_tokens = random.randint(100, 400)

    tracker.track_usage(model, input_tokens, output_tokens, user, operation)

# Display cost analysis
print("\nTotal Costs by Model:")
for model, cost in tracker.get_total_cost().items():
    print(f"  {model}: ${cost:.2f}")

print("\nCosts by Operation:")
for operation, costs in tracker.get_operation_costs().items():
    print(f"  {operation}: ${costs['total']:.2f}")

print("\nOptimization Suggestions:")
for suggestion in tracker.get_cost_optimization_suggestions():
    print(f"  - {suggestion}")


# Example 7: Complete Observability Stack
print("\n" + "=" * 50)
print("Example 7: Complete Observability Stack")
print("=" * 50)


class ObservabilityStack:
    """Complete observability solution for LLM applications."""

    def __init__(self, app_name: str):
        self.app_name = app_name
        self.metrics = MetricsCollector(app_name)
        self.tracer = DistributedTracer()
        self.logger = StructuredLogger(app_name)
        self.alerting = AlertingSystem()
        self.performance = PerformanceMonitor()
        self.costs = CostTracker()

        self._setup_default_alerts()

    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # Error rate alert
        self.alerting.add_rule(
            "high_error_rate",
            lambda m: m.get("error_rate", 0) > 0.05,
            AlertSeverity.ERROR,
            "Error rate exceeded 5%: {error_rate:.1%}"
        )

        # Latency alert
        self.alerting.add_rule(
            "high_latency",
            lambda m: m.get("p95_latency", 0) > 3.0,
            AlertSeverity.WARNING,
            "P95 latency > 3s: {p95_latency:.2f}s"
        )

        # Cost alert
        self.alerting.add_rule(
            "high_daily_cost",
            lambda m: m.get("daily_cost", 0) > 100,
            AlertSeverity.WARNING,
            "Daily cost exceeded $100: ${daily_cost:.2f}"
        )

    async def process_request(self, request_id: str,
                             operation: str,
                             model: str,
                             prompt: str,
                             user_id: str = None):
        """Process request with full observability."""
        # Start trace
        trace_id = self.tracer.start_trace(f"{operation}_request")

        # Set logging context
        self.logger.set_context(
            request_id=request_id,
            trace_id=trace_id,
            user_id=user_id,
            operation=operation,
            model=model
        )

        self.logger.info("Request started",
                        prompt_length=len(prompt))

        try:
            # Preprocessing
            preprocess_span = self.tracer.start_span(trace_id, "preprocess")
            start_time = time.time()

            # Simulate preprocessing
            await asyncio.sleep(0.1)

            preprocess_time = time.time() - start_time
            self.tracer.end_span(trace_id, preprocess_span)

            # API call
            api_span = self.tracer.start_span(trace_id, "api_call")
            self.tracer.add_tag(trace_id, api_span, "model", model)

            api_start = time.time()

            # Simulate API call
            await asyncio.sleep(0.5)
            input_tokens = len(prompt.split()) * 2  # Rough estimate
            output_tokens = random.randint(100, 300)

            api_time = time.time() - api_start
            self.tracer.end_span(trace_id, api_span)

            # Record metrics
            self.metrics.record_latency(operation, api_time)
            self.metrics.record_token_usage(model, input_tokens, output_tokens)

            # Record performance
            self.performance.record_request(
                operation, api_time,
                input_tokens, output_tokens,
                model, cache_hit=False
            )

            # Track costs
            self.costs.track_usage(
                model, input_tokens, output_tokens,
                user_id, operation
            )

            # Postprocessing
            post_span = self.tracer.start_span(trace_id, "postprocess")
            await asyncio.sleep(0.05)
            self.tracer.end_span(trace_id, post_span)

            total_time = time.time() - start_time

            self.logger.info("Request completed",
                           duration=total_time,
                           input_tokens=input_tokens,
                           output_tokens=output_tokens)

            return {
                "success": True,
                "trace_id": trace_id,
                "duration": total_time,
                "tokens": input_tokens + output_tokens
            }

        except Exception as e:
            self.logger.error("Request failed", exception=e)
            self.metrics.record_error(type(e).__name__, operation)

            return {
                "success": False,
                "trace_id": trace_id,
                "error": str(e)
            }

        finally:
            self.logger.clear_context()

    def get_health_status(self) -> Dict:
        """Get overall system health status."""
        # Calculate metrics
        metrics = self.metrics.get_aggregated_metrics()

        # Calculate error rate
        total_requests = metrics.get(f"{self.app_name}.latency.generate.count", 0)
        total_errors = metrics.get(f"{self.app_name}.errors", 0)
        error_rate = total_errors / max(total_requests, 1)

        # Get latency stats
        latency_stats = self.performance.get_latency_stats("generate")
        p95_latency = latency_stats.get("p95", 0)

        # Get daily cost
        daily_costs = self.costs.get_total_cost()
        daily_cost = daily_costs.get("total", 0)

        # Check alerts
        alert_metrics = {
            "error_rate": error_rate,
            "p95_latency": p95_latency,
            "daily_cost": daily_cost
        }

        self.alerting.check_metrics(alert_metrics)

        # Determine health status
        active_alerts = self.alerting.get_active_alerts()

        if any(a.severity == AlertSeverity.CRITICAL for a in active_alerts):
            health = "critical"
        elif any(a.severity == AlertSeverity.ERROR for a in active_alerts):
            health = "degraded"
        elif any(a.severity == AlertSeverity.WARNING for a in active_alerts):
            health = "warning"
        else:
            health = "healthy"

        return {
            "status": health,
            "metrics": {
                "error_rate": f"{error_rate:.1%}",
                "p95_latency": f"{p95_latency:.2f}s",
                "daily_cost": f"${daily_cost:.2f}",
                "cache_hit_rate": f"{self.performance.get_cache_hit_rate():.1%}"
            },
            "active_alerts": len(active_alerts),
            "anomalies": self.performance.detect_anomalies()
        }


# Test complete observability stack
async def test_observability():
    stack = ObservabilityStack("llm_production_app")

    # Simulate multiple requests
    requests = [
        ("req_001", "generate", "gpt-4", "Write a story about...", "user_1"),
        ("req_002", "summarize", "gpt-3.5-turbo", "Summarize this text...", "user_2"),
        ("req_003", "translate", "claude-3-opus", "Translate to Spanish...", "user_1"),
    ]

    results = []
    for request_id, operation, model, prompt, user_id in requests:
        result = await stack.process_request(
            request_id, operation, model, prompt, user_id
        )
        results.append(result)

    # Get health status
    health = stack.get_health_status()
    print("\nSystem Health Status:")
    print(f"  Status: {health['status']}")
    print(f"  Metrics: {health['metrics']}")
    if health['anomalies']:
        print(f"  Anomalies: {health['anomalies']}")

    # Get cost analysis
    costs = stack.costs.get_total_cost()
    print(f"\nTotal Costs: ${costs['total']:.2f}")

    return stack

# Run async test
stack = asyncio.run(test_observability())


print("\n" + "=" * 50)
print("Monitoring and Observability Examples Complete!")
print("=" * 50)
print("""
These examples demonstrated:
1. Metrics collection and aggregation
2. Distributed tracing
3. Structured logging
4. Alerting systems
5. Performance monitoring
6. Cost tracking and optimization
7. Complete observability stack

Key concepts for production monitoring:
- Collect comprehensive metrics
- Implement distributed tracing
- Use structured logging
- Set up proactive alerting
- Monitor performance trends
- Track and optimize costs
- Build integrated observability
""")