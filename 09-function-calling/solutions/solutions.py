"""
Module 09: Function Calling - Solutions

Complete solutions for all function calling exercises.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
import time
from datetime import datetime, timedelta
import threading
import hashlib
from collections import defaultdict, deque
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum


# ===== Solution 1: Tool Designer =====

def solution_1_tool_designer():
    """
    Solution 1: Design comprehensive tool interfaces.
    """
    print("Solution 1: Tool Designer")
    print("=" * 50)

    class WeatherService:
        """Comprehensive weather service with multiple endpoints."""

        def __init__(self):
            self.functions = []
            self._init_functions()

        def _init_functions(self):
            """Initialize all function definitions."""
            self.functions = [
                self.get_current_weather_definition(),
                self.get_forecast_definition(),
                self.get_alerts_definition(),
                self.get_historical_definition()
            ]

        def get_current_weather_definition(self) -> Dict:
            """Define current weather function."""
            return {
                "name": "get_current_weather",
                "description": "Get current weather conditions for a specific location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location in format: 'City, State', ZIP code, or coordinates",
                            "pattern": "^([a-zA-Z\\s]+,\\s*[A-Z]{2}|\\d{5}|[-]?\\d+\\.\\d+,[-]?\\d+\\.\\d+)$"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit", "kelvin"],
                            "default": "fahrenheit",
                            "description": "Temperature unit system"
                        },
                        "include_details": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include detailed metrics (humidity, wind, pressure)"
                        }
                    },
                    "required": ["location"]
                },
                "examples": [
                    {
                        "location": "San Francisco, CA",
                        "units": "celsius"
                    },
                    {
                        "location": "10001",
                        "units": "fahrenheit",
                        "include_details": True
                    }
                ]
            }

        def get_forecast_definition(self) -> Dict:
            """Define forecast function."""
            return {
                "name": "get_weather_forecast",
                "description": "Get weather forecast for upcoming days",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location in format: 'City, State', ZIP code, or coordinates"
                        },
                        "days": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5,
                            "description": "Number of days to forecast"
                        },
                        "include_hourly": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include hourly breakdown for each day"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit", "kelvin"],
                            "default": "fahrenheit"
                        }
                    },
                    "required": ["location"]
                }
            }

        def get_alerts_definition(self) -> Dict:
            """Define weather alerts function."""
            return {
                "name": "get_weather_alerts",
                "description": "Get active weather alerts and warnings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location or region"
                        },
                        "coordinates": {
                            "type": "object",
                            "properties": {
                                "lat": {"type": "number", "minimum": -90, "maximum": 90},
                                "lon": {"type": "number", "minimum": -180, "maximum": 180}
                            }
                        },
                        "severity": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["minor", "moderate", "severe", "extreme"]
                            },
                            "description": "Filter by severity levels"
                        },
                        "types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["tornado", "hurricane", "flood", "winter", "heat", "other"]
                            },
                            "description": "Filter by alert types"
                        }
                    },
                    "oneOf": [
                        {"required": ["location"]},
                        {"required": ["coordinates"]}
                    ]
                }
            }

        def get_historical_definition(self) -> Dict:
            """Define historical weather function."""
            return {
                "name": "get_historical_weather",
                "description": "Get historical weather data for analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location for historical data"
                        },
                        "start_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Start date (YYYY-MM-DD)"
                        },
                        "end_date": {
                            "type": "string",
                            "format": "date",
                            "description": "End date (YYYY-MM-DD)"
                        },
                        "metrics": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["temperature", "precipitation", "wind", "humidity", "pressure"]
                            },
                            "default": ["temperature", "precipitation"],
                            "description": "Metrics to include"
                        },
                        "aggregation": {
                            "type": "string",
                            "enum": ["daily", "weekly", "monthly"],
                            "default": "daily",
                            "description": "Data aggregation level"
                        }
                    },
                    "required": ["location", "start_date", "end_date"]
                }
            }

        def validate_location(self, location: str) -> tuple[bool, str]:
            """Validate location format."""
            # City, State format
            if re.match(r'^[a-zA-Z\s]+,\s*[A-Z]{2}$', location):
                return True, "city_state"
            # ZIP code
            elif re.match(r'^\d{5}(-\d{4})?$', location):
                return True, "zip"
            # Coordinates
            elif re.match(r'^[-]?\d+\.?\d*,[-]?\d+\.?\d*$', location):
                coords = location.split(',')
                lat, lon = float(coords[0]), float(coords[1])
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return True, "coordinates"
            return False, "invalid"

        def execute_function(self, function_name: str, arguments: Dict) -> Dict:
            """Execute weather function with validation."""
            # Validate function exists
            function_map = {
                "get_current_weather": self._get_current_weather,
                "get_weather_forecast": self._get_forecast,
                "get_weather_alerts": self._get_alerts,
                "get_historical_weather": self._get_historical
            }

            if function_name not in function_map:
                return {"error": f"Unknown function: {function_name}"}

            # Validate location if present
            if "location" in arguments:
                is_valid, location_type = self.validate_location(arguments["location"])
                if not is_valid:
                    return {"error": "Invalid location format"}
                arguments["_location_type"] = location_type

            try:
                # Execute function
                result = function_map[function_name](**arguments)
                return {"status": "success", "data": result}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        def _get_current_weather(self, location: str, units: str = "fahrenheit",
                                include_details: bool = False, **kwargs) -> Dict:
            """Get current weather (mock implementation)."""
            base_data = {
                "location": location,
                "temperature": 72 if units == "fahrenheit" else 22,
                "units": units,
                "conditions": "Partly cloudy",
                "timestamp": datetime.now().isoformat()
            }

            if include_details:
                base_data.update({
                    "humidity": 65,
                    "wind_speed": 12,
                    "wind_direction": "NW",
                    "pressure": 1013.25,
                    "visibility": 10
                })

            return base_data

        def _get_forecast(self, location: str, days: int = 5,
                         include_hourly: bool = False, units: str = "fahrenheit", **kwargs) -> Dict:
            """Get weather forecast (mock implementation)."""
            forecast_data = {
                "location": location,
                "days": days,
                "units": units,
                "forecast": []
            }

            for day in range(days):
                day_data = {
                    "date": (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d"),
                    "high": 75 + day if units == "fahrenheit" else 24 + day,
                    "low": 60 + day if units == "fahrenheit" else 15 + day,
                    "conditions": ["Sunny", "Partly cloudy", "Cloudy", "Rainy"][day % 4],
                    "precipitation_chance": (day * 10) % 60
                }

                if include_hourly:
                    day_data["hourly"] = [
                        {"hour": h, "temp": 70 + h % 10} for h in range(0, 24, 3)
                    ]

                forecast_data["forecast"].append(day_data)

            return forecast_data

        def _get_alerts(self, location: Optional[str] = None,
                       coordinates: Optional[Dict] = None,
                       severity: Optional[List[str]] = None,
                       types: Optional[List[str]] = None, **kwargs) -> Dict:
            """Get weather alerts (mock implementation)."""
            return {
                "location": location or f"{coordinates['lat']},{coordinates['lon']}",
                "alerts": [
                    {
                        "type": "winter",
                        "severity": "moderate",
                        "title": "Winter Storm Warning",
                        "description": "Heavy snow expected",
                        "expires": (datetime.now() + timedelta(hours=12)).isoformat()
                    }
                ] if severity is None or "moderate" in severity else []
            }

        def _get_historical(self, location: str, start_date: str, end_date: str,
                          metrics: List[str] = None, aggregation: str = "daily", **kwargs) -> Dict:
            """Get historical weather (mock implementation)."""
            metrics = metrics or ["temperature", "precipitation"]
            return {
                "location": location,
                "period": f"{start_date} to {end_date}",
                "aggregation": aggregation,
                "data": {
                    metric: [{"date": start_date, "value": 70 if metric == "temperature" else 0.5}]
                    for metric in metrics
                }
            }

    # Create service and demonstrate
    service = WeatherService()

    # Display function definitions
    print("\nRegistered Functions:")
    print("-" * 30)
    for func_def in service.functions:
        print(f"\n{func_def['name']}:")
        print(f"  Description: {func_def['description']}")
        print(f"  Required params: {func_def['parameters'].get('required', [])}")

    # Test executions
    print("\n" + "-" * 30)
    print("Function Executions:")
    print("-" * 30)

    test_cases = [
        ("get_current_weather", {"location": "New York, NY", "units": "celsius", "include_details": True}),
        ("get_weather_forecast", {"location": "90210", "days": 3}),
        ("get_weather_alerts", {"location": "Miami, FL", "severity": ["severe", "extreme"]})
    ]

    for func_name, args in test_cases:
        print(f"\n{func_name}({json.dumps(args, indent=2)})")
        result = service.execute_function(func_name, args)
        print(f"Result: {json.dumps(result, indent=2)}")


# ===== Solution 2: Error Handler =====

def solution_2_error_handler():
    """
    Solution 2: Build robust error handling system.
    """
    print("\nSolution 2: Error Handler")
    print("=" * 50)

    class CircuitState(Enum):
        CLOSED = "closed"  # Normal operation
        OPEN = "open"      # Blocking calls
        HALF_OPEN = "half_open"  # Testing recovery

    class CircuitBreaker:
        """Circuit breaker pattern implementation."""

        def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                    success_threshold: int = 2):
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.success_threshold = success_threshold
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.last_error = None

        def call(self, func: Callable, *args, **kwargs):
            """Execute function with circuit breaker."""
            # Check if circuit should attempt reset
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise Exception(f"Circuit breaker is OPEN. Last error: {self.last_error}")

            try:
                # Execute function
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure(e)
                raise

        def _record_success(self):
            """Record successful call."""
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    print(f"Circuit breaker: HALF_OPEN -> CLOSED")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

        def _record_failure(self, error: Exception):
            """Record failed call."""
            self.last_error = str(error)
            self.last_failure_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                print(f"Circuit breaker: HALF_OPEN -> OPEN")
            elif self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    print(f"Circuit breaker: CLOSED -> OPEN (failures: {self.failure_count})")

        def _should_attempt_reset(self) -> bool:
            """Check if should try half-open state."""
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                return elapsed >= self.recovery_timeout
            return False

        def get_state(self) -> Dict:
            """Get circuit breaker state."""
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_error": self.last_error
            }

    class RobustExecutor:
        """Execute functions with comprehensive error handling."""

        def __init__(self):
            self.circuit_breakers = {}
            self.retry_config = {
                "max_retries": 3,
                "base_delay": 1,
                "max_delay": 30,
                "exponential_base": 2,
                "jitter": 0.1
            }
            self.fallback_functions = {}
            self.error_counts = defaultdict(int)

        def get_circuit_breaker(self, func_name: str) -> CircuitBreaker:
            """Get or create circuit breaker for function."""
            if func_name not in self.circuit_breakers:
                self.circuit_breakers[func_name] = CircuitBreaker()
            return self.circuit_breakers[func_name]

        def execute_with_retry(self, func: Callable, func_name: str = None, *args, **kwargs):
            """Execute with exponential backoff retry."""
            func_name = func_name or func.__name__
            last_error = None

            for attempt in range(self.retry_config["max_retries"]):
                try:
                    # Use circuit breaker if available
                    if func_name in self.circuit_breakers:
                        return self.circuit_breakers[func_name].call(func, *args, **kwargs)
                    else:
                        return func(*args, **kwargs)

                except Exception as e:
                    last_error = e
                    error_type = self.categorize_error(e)

                    # Don't retry non-retryable errors
                    if error_type == "non_retryable":
                        raise

                    # Calculate backoff with jitter
                    if attempt < self.retry_config["max_retries"] - 1:
                        delay = self._calculate_backoff(attempt)
                        print(f"Retry {attempt + 1}/{self.retry_config['max_retries']} "
                              f"after {delay:.1f}s (error: {error_type})")
                        time.sleep(delay)

            # Max retries exceeded
            self.error_counts[func_name] += 1
            raise last_error

        def _calculate_backoff(self, attempt: int) -> float:
            """Calculate exponential backoff with jitter."""
            base_delay = self.retry_config["base_delay"]
            exponential = self.retry_config["exponential_base"] ** attempt
            delay = min(base_delay * exponential, self.retry_config["max_delay"])

            # Add jitter
            jitter_range = delay * self.retry_config["jitter"]
            import random
            jitter = random.uniform(-jitter_range, jitter_range)

            return max(0, delay + jitter)

        def execute_with_fallback(self, primary: Callable, fallback: Callable,
                                 *args, **kwargs):
            """Execute with fallback strategy."""
            try:
                return {
                    "status": "primary",
                    "result": self.execute_with_retry(primary, *args, **kwargs)
                }
            except Exception as e:
                print(f"Primary failed: {e}, trying fallback")
                try:
                    return {
                        "status": "fallback",
                        "result": fallback(*args, **kwargs),
                        "degraded": True
                    }
                except Exception as fallback_error:
                    return {
                        "status": "error",
                        "primary_error": str(e),
                        "fallback_error": str(fallback_error)
                    }

        def categorize_error(self, error: Exception) -> str:
            """Categorize error for appropriate handling."""
            error_str = str(error)
            error_type = type(error).__name__

            # Non-retryable errors
            non_retryable = [
                "ValueError", "TypeError", "AttributeError",
                "KeyError", "IndexError", "NotImplementedError"
            ]
            if error_type in non_retryable:
                return "non_retryable"

            # Rate limiting
            if "rate limit" in error_str.lower() or "429" in error_str:
                return "rate_limit"

            # Authentication
            if "auth" in error_str.lower() or "401" in error_str or "403" in error_str:
                return "auth"

            # Network/timeout
            if "timeout" in error_str.lower() or "connection" in error_str.lower():
                return "network"

            # Default to retryable
            return "retryable"

        def register_fallback(self, func_name: str, fallback: Callable):
            """Register fallback function."""
            self.fallback_functions[func_name] = fallback

        def get_statistics(self) -> Dict:
            """Get error handling statistics."""
            return {
                "error_counts": dict(self.error_counts),
                "circuit_breakers": {
                    name: cb.get_state()
                    for name, cb in self.circuit_breakers.items()
                }
            }

    # Test error handling
    executor = RobustExecutor()

    # Test functions
    call_count = {"flaky": 0, "broken": 0}

    def flaky_function(x: int) -> int:
        """Sometimes fails."""
        call_count["flaky"] += 1
        if call_count["flaky"] <= 2:
            raise ConnectionError("Network timeout")
        return x * 2

    def always_broken(x: int) -> int:
        """Always fails."""
        call_count["broken"] += 1
        raise ValueError("This function is broken")

    def reliable_fallback(x: int) -> int:
        """Reliable fallback."""
        return x + 10

    # Test retry with transient errors
    print("\nTesting retry logic:")
    print("-" * 30)
    try:
        result = executor.execute_with_retry(flaky_function, "flaky", 5)
        print(f"Success after retries: {result}")
    except Exception as e:
        print(f"Failed: {e}")

    # Test circuit breaker
    print("\n" + "-" * 30)
    print("Testing circuit breaker:")
    print("-" * 30)

    cb_func = lambda: always_broken(1)
    cb = executor.get_circuit_breaker("broken_func")

    for i in range(8):
        try:
            result = cb.call(cb_func)
            print(f"Call {i+1}: Success")
        except Exception as e:
            print(f"Call {i+1}: Failed - {cb.state.value}")
        time.sleep(0.1)

    # Test fallback
    print("\n" + "-" * 30)
    print("Testing fallback strategy:")
    print("-" * 30)

    result = executor.execute_with_fallback(
        lambda x: always_broken(x),
        reliable_fallback,
        10
    )
    print(f"Fallback result: {json.dumps(result, indent=2)}")

    # Show statistics
    print("\n" + "-" * 30)
    print("Error Handling Statistics:")
    print("-" * 30)
    print(json.dumps(executor.get_statistics(), indent=2))


# ===== Solution 3: Security Sandbox =====

def solution_3_security_sandbox():
    """
    Solution 3: Implement secure function execution.
    """
    print("\nSolution 3: Security Sandbox")
    print("=" * 50)

    @dataclass
    class SecurityContext:
        """Security context for execution."""
        user_id: str
        roles: List[str]
        permissions: List[str]
        rate_limit_remaining: int = 100
        session_id: str = ""
        ip_address: str = "127.0.0.1"
        metadata: Dict = field(default_factory=dict)

    class RateLimiter:
        """Token bucket rate limiter."""

        def __init__(self):
            self.limits = {}  # function -> (calls, period)
            self.buckets = {}  # (user, function) -> (tokens, last_refill)

        def set_limit(self, function: str, calls: int, period: int):
            """Set rate limit for function."""
            self.limits[function] = (calls, period)

        def check_limit(self, user_id: str, function: str) -> tuple[bool, int]:
            """Check if user can call function."""
            if function not in self.limits:
                return True, -1  # No limit

            calls_limit, period = self.limits[function]
            key = (user_id, function)

            # Initialize bucket if needed
            if key not in self.buckets:
                self.buckets[key] = [calls_limit, datetime.now()]

            tokens, last_refill = self.buckets[key]
            now = datetime.now()

            # Refill tokens based on elapsed time
            elapsed = (now - last_refill).total_seconds()
            refill_rate = calls_limit / period
            new_tokens = min(calls_limit, tokens + elapsed * refill_rate)

            self.buckets[key] = [new_tokens, now]

            if new_tokens >= 1:
                return True, int(new_tokens)
            else:
                return False, 0

        def record_call(self, user_id: str, function: str):
            """Record a function call."""
            key = (user_id, function)
            if key in self.buckets:
                self.buckets[key][0] -= 1

    class InputSanitizer:
        """Sanitize and validate inputs."""

        def __init__(self):
            self.patterns = {
                "sql_injection": r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b|;|--)",
                "script_injection": r"<script|javascript:|on\w+=",
                "path_traversal": r"\.\./|\.\\/",
                "command_injection": r"[;&|`$()]"
            }

        def sanitize(self, value: Any) -> Any:
            """Sanitize input value."""
            if isinstance(value, str):
                return self._sanitize_string(value)
            elif isinstance(value, dict):
                return {k: self.sanitize(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [self.sanitize(v) for v in value]
            else:
                return value

        def _sanitize_string(self, text: str) -> str:
            """Sanitize string input."""
            # Check for injection patterns
            for pattern_name, pattern in self.patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    # Log potential injection attempt
                    print(f"Warning: Potential {pattern_name} detected")
                    # Remove dangerous characters
                    text = re.sub(pattern, "", text, flags=re.IGNORECASE)

            # HTML escape
            text = text.replace("<", "&lt;").replace(">", "&gt;")

            # Limit length
            max_length = 10000
            if len(text) > max_length:
                text = text[:max_length]

            return text

        def validate_type(self, value: Any, expected_type: type) -> bool:
            """Validate value type."""
            if expected_type == int:
                return isinstance(value, int) and -2**31 <= value <= 2**31
            elif expected_type == float:
                return isinstance(value, (int, float)) and not math.isnan(value)
            elif expected_type == str:
                return isinstance(value, str) and len(value) <= 10000
            elif expected_type == bool:
                return isinstance(value, bool)
            elif expected_type == list:
                return isinstance(value, list) and len(value) <= 1000
            elif expected_type == dict:
                return isinstance(value, dict) and len(value) <= 100
            return True

    class SecureSandbox:
        """Secure execution environment."""

        def __init__(self):
            self.rate_limiter = RateLimiter()
            self.sanitizer = InputSanitizer()
            self.audit_log = []
            self.permissions = {}
            self.resource_limits = {
                "max_execution_time": 30,  # seconds
                "max_memory": 100 * 1024 * 1024,  # 100MB
                "max_output_size": 1024 * 1024  # 1MB
            }

        def register_function_permissions(self, function: str, required: List[str]):
            """Register required permissions for function."""
            self.permissions[function] = set(required)

        def check_permissions(self, context: SecurityContext, function: str) -> bool:
            """Check if user has required permissions."""
            if function not in self.permissions:
                return True  # No permissions required

            required = self.permissions[function]

            # Check if user has admin role
            if "admin" in context.roles:
                return True

            # Check specific permissions
            user_perms = set(context.permissions)
            return required.issubset(user_perms)

        def sanitize_input(self, function: str, arguments: Dict) -> Dict:
            """Sanitize and validate input."""
            # Sanitize all string inputs
            sanitized = self.sanitizer.sanitize(arguments)

            # Validate types based on function
            # (In production, would use function schema)

            return sanitized

        def execute_sandboxed(self, context: SecurityContext,
                            function: str, arguments: Dict,
                            func_impl: Callable) -> Dict:
            """Execute function in sandbox."""
            start_time = datetime.now()

            # 1. Check permissions
            if not self.check_permissions(context, function):
                self.audit_execution(context, function, False, "permission_denied")
                return {
                    "error": "Permission denied",
                    "required": list(self.permissions.get(function, set()))
                }

            # 2. Check rate limits
            can_call, remaining = self.rate_limiter.check_limit(context.user_id, function)
            if not can_call:
                self.audit_execution(context, function, False, "rate_limited")
                return {
                    "error": "Rate limit exceeded",
                    "retry_after": 60
                }

            # 3. Sanitize inputs
            try:
                sanitized_args = self.sanitize_input(function, arguments)
            except Exception as e:
                self.audit_execution(context, function, False, f"sanitization_failed: {e}")
                return {"error": f"Input validation failed: {e}"}

            # 4. Execute with resource limits
            try:
                # Record the call
                self.rate_limiter.record_call(context.user_id, function)

                # Execute with timeout (simplified - in production use subprocess/container)
                result = func_impl(**sanitized_args)

                # Check output size
                import sys
                result_size = sys.getsizeof(result)
                if result_size > self.resource_limits["max_output_size"]:
                    raise Exception("Output size exceeds limit")

                # Success
                execution_time = (datetime.now() - start_time).total_seconds()
                self.audit_execution(context, function, True, None, execution_time)

                return {
                    "status": "success",
                    "result": result,
                    "rate_limit_remaining": remaining - 1,
                    "execution_time": execution_time
                }

            except Exception as e:
                self.audit_execution(context, function, False, str(e))
                return {
                    "status": "error",
                    "error": f"Execution failed: {e}"
                }

        def audit_execution(self, context: SecurityContext, function: str,
                          success: bool, error: Optional[str] = None,
                          execution_time: Optional[float] = None):
            """Log execution for audit."""
            self.audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "user_id": context.user_id,
                "session_id": context.session_id,
                "ip_address": context.ip_address,
                "function": function,
                "success": success,
                "error": error,
                "execution_time": execution_time,
                "roles": context.roles
            })

        def get_audit_log(self, filters: Optional[Dict] = None) -> List[Dict]:
            """Get filtered audit log."""
            if not filters:
                return self.audit_log

            filtered = self.audit_log
            if "user_id" in filters:
                filtered = [l for l in filtered if l["user_id"] == filters["user_id"]]
            if "function" in filters:
                filtered = [l for l in filtered if l["function"] == filters["function"]]
            if "success" in filters:
                filtered = [l for l in filtered if l["success"] == filters["success"]]

            return filtered

    # Test security features
    sandbox = SecureSandbox()

    # Set up permissions and rate limits
    sandbox.register_function_permissions("delete_user", ["admin:delete", "user:write"])
    sandbox.register_function_permissions("read_user", ["user:read"])
    sandbox.register_function_permissions("export_data", ["data:export"])

    sandbox.rate_limiter.set_limit("export_data", calls=5, period=60)
    sandbox.rate_limiter.set_limit("delete_user", calls=2, period=60)

    # Test functions
    def delete_user(user_id: str) -> Dict:
        return {"deleted": user_id}

    def read_user(user_id: str) -> Dict:
        return {"user_id": user_id, "name": "Test User"}

    def export_data(format: str) -> Dict:
        return {"exported": True, "format": format}

    # Create test contexts
    admin_context = SecurityContext(
        user_id="admin1",
        roles=["admin"],
        permissions=["admin:delete", "user:write", "user:read", "data:export"],
        session_id="sess_123"
    )

    user_context = SecurityContext(
        user_id="user1",
        roles=["user"],
        permissions=["user:read"],
        session_id="sess_456"
    )

    # Test executions
    print("\nSecurity Tests:")
    print("-" * 30)

    test_cases = [
        (admin_context, "delete_user", {"user_id": "target123"}, delete_user),
        (user_context, "delete_user", {"user_id": "target456"}, delete_user),
        (user_context, "read_user", {"user_id": "self"}, read_user),
        (admin_context, "export_data", {"format": "csv"}, export_data)
    ]

    for context, func_name, args, impl in test_cases:
        print(f"\n{context.user_id} calling {func_name}:")
        result = sandbox.execute_sandboxed(context, func_name, args, impl)
        if "error" in result:
            print(f"  ❌ {result['error']}")
        else:
            print(f"  ✅ Success: {result['result']}")

    # Test input sanitization
    print("\n" + "-" * 30)
    print("Input Sanitization Tests:")
    print("-" * 30)

    dangerous_inputs = [
        {"query": "'; DROP TABLE users; --"},
        {"script": "<script>alert('xss')</script>"},
        {"path": "../../etc/passwd"}
    ]

    for inp in dangerous_inputs:
        sanitized = sandbox.sanitizer.sanitize(inp)
        print(f"Original: {inp}")
        print(f"Sanitized: {sanitized}")

    # Test rate limiting
    print("\n" + "-" * 30)
    print("Rate Limiting Tests:")
    print("-" * 30)

    for i in range(7):
        result = sandbox.execute_sandboxed(
            admin_context, "export_data", {"format": "json"}, export_data
        )
        status = "✅" if result.get("status") == "success" else "❌"
        remaining = result.get("rate_limit_remaining", "N/A")
        print(f"Call {i+1}: {status} (remaining: {remaining})")

    # Show audit log
    print("\n" + "-" * 30)
    print("Audit Log Summary:")
    print("-" * 30)

    recent_logs = sandbox.get_audit_log()[-5:]
    for log in recent_logs:
        status = "✅" if log["success"] else "❌"
        print(f"{log['timestamp']}: {log['user_id']} -> {log['function']}: {status}")


# ===== Solution 4: Chain Builder =====

def solution_4_chain_builder():
    """
    Solution 4: Create complex function chains.
    """
    print("\nSolution 4: Chain Builder")
    print("=" * 50)

    @dataclass
    class ChainStep:
        """Step in a function chain."""
        name: str
        function: str
        arguments: Dict[str, Any]
        output_mapping: Dict[str, str] = field(default_factory=dict)
        condition: Optional[str] = None
        on_error: Optional[str] = "stop"  # stop, continue, retry

    class ChainBuilder:
        """Build and execute function chains."""

        def __init__(self):
            self.chains = {}
            self.functions = {}
            self.context = {}

        def register_function(self, name: str, func: Callable):
            """Register a function for use in chains."""
            self.functions[name] = func

        def create_chain(self, chain_name: str) -> 'ChainDefinition':
            """Create a new chain definition."""
            if chain_name not in self.chains:
                self.chains[chain_name] = []
            return ChainDefinition(self, chain_name)

        def add_step(self, chain_name: str, step: ChainStep):
            """Add a step to a chain."""
            if chain_name not in self.chains:
                self.chains[chain_name] = []
            self.chains[chain_name].append(step)

        def add_conditional(self, chain_name: str, condition: str,
                          true_branch: List[ChainStep],
                          false_branch: List[ChainStep]):
            """Add conditional branching."""
            conditional_step = {
                "type": "conditional",
                "condition": condition,
                "true_branch": true_branch,
                "false_branch": false_branch
            }
            self.chains[chain_name].append(conditional_step)

        def add_loop(self, chain_name: str, items_key: str,
                    loop_steps: List[ChainStep], max_iterations: int = 100):
            """Add loop construct."""
            loop_step = {
                "type": "loop",
                "items_key": items_key,
                "steps": loop_steps,
                "max_iterations": max_iterations
            }
            self.chains[chain_name].append(loop_step)

        def execute_chain(self, chain_name: str, initial_context: Dict) -> Dict:
            """Execute a complete chain."""
            if chain_name not in self.chains:
                return {"error": f"Chain {chain_name} not found"}

            context = initial_context.copy()
            results = []

            print(f"\nExecuting chain: {chain_name}")
            print("-" * 30)

            for step in self.chains[chain_name]:
                if isinstance(step, ChainStep):
                    result = self._execute_step(step, context)
                    results.append(result)
                elif isinstance(step, dict):
                    if step["type"] == "conditional":
                        result = self._execute_conditional(step, context)
                        results.append(result)
                    elif step["type"] == "loop":
                        result = self._execute_loop(step, context)
                        results.extend(result)

            return {
                "status": "completed",
                "context": context,
                "results": results
            }

        def _execute_step(self, step: ChainStep, context: Dict) -> Dict:
            """Execute a single step."""
            print(f"\nStep: {step.name}")

            # Check condition
            if step.condition:
                if not self._evaluate_condition(step.condition, context):
                    print(f"  Skipped (condition false)")
                    return {"step": step.name, "status": "skipped"}

            # Prepare arguments
            args = self._resolve_arguments(step.arguments, context)
            print(f"  Function: {step.function}")
            print(f"  Arguments: {json.dumps(args, indent=2)}")

            # Execute function
            if step.function not in self.functions:
                return {"step": step.name, "status": "error", "error": "Function not found"}

            try:
                result = self.functions[step.function](**args)
                print(f"  Result: {json.dumps(result, indent=2)}")

                # Update context with output mapping
                for output_key, context_key in step.output_mapping.items():
                    if isinstance(result, dict) and output_key in result:
                        context[context_key] = result[output_key]
                    else:
                        context[context_key] = result

                return {"step": step.name, "status": "success", "result": result}

            except Exception as e:
                print(f"  Error: {e}")
                if step.on_error == "continue":
                    return {"step": step.name, "status": "error", "error": str(e)}
                elif step.on_error == "retry":
                    # Simple retry once
                    time.sleep(1)
                    try:
                        result = self.functions[step.function](**args)
                        return {"step": step.name, "status": "success_retry", "result": result}
                    except:
                        return {"step": step.name, "status": "error", "error": str(e)}
                else:  # stop
                    raise

        def _execute_conditional(self, conditional: Dict, context: Dict) -> Dict:
            """Execute conditional branch."""
            condition = conditional["condition"]
            print(f"\nConditional: {condition}")

            if self._evaluate_condition(condition, context):
                print("  Branch: TRUE")
                results = []
                for step in conditional["true_branch"]:
                    results.append(self._execute_step(step, context))
                return {"type": "conditional", "branch": "true", "results": results}
            else:
                print("  Branch: FALSE")
                results = []
                for step in conditional["false_branch"]:
                    results.append(self._execute_step(step, context))
                return {"type": "conditional", "branch": "false", "results": results}

        def _execute_loop(self, loop: Dict, context: Dict) -> List[Dict]:
            """Execute loop construct."""
            items_key = loop["items_key"]
            items = self._resolve_value(items_key, context)

            if not isinstance(items, list):
                return [{"type": "loop", "error": "Items is not a list"}]

            print(f"\nLoop: Processing {len(items)} items")
            results = []

            for i, item in enumerate(items[:loop["max_iterations"]]):
                print(f"\n  Iteration {i+1}:")
                loop_context = context.copy()
                loop_context["_item"] = item
                loop_context["_index"] = i

                for step in loop["steps"]:
                    result = self._execute_step(step, loop_context)
                    results.append(result)

                # Update main context with any changes
                context.update({k: v for k, v in loop_context.items()
                              if not k.startswith("_")})

            return results

        def _resolve_arguments(self, args: Dict, context: Dict) -> Dict:
            """Resolve arguments from context."""
            resolved = {}
            for key, value in args.items():
                resolved[key] = self._resolve_value(value, context)
            return resolved

        def _resolve_value(self, value: Any, context: Dict) -> Any:
            """Resolve a value from context."""
            if isinstance(value, str) and value.startswith("$"):
                key = value[1:]
                return context.get(key, value)
            elif isinstance(value, dict):
                return {k: self._resolve_value(v, context) for k, v in value.items()}
            elif isinstance(value, list):
                return [self._resolve_value(v, context) for v in value]
            return value

        def _evaluate_condition(self, condition: str, context: Dict) -> bool:
            """Evaluate a condition."""
            try:
                # Simple evaluation (in production use safe eval)
                return eval(condition, {"__builtins__": {}}, context)
            except:
                return False

    class ChainDefinition:
        """Fluent interface for chain building."""

        def __init__(self, builder: ChainBuilder, name: str):
            self.builder = builder
            self.name = name

        def step(self, name: str, function: str, **kwargs) -> 'ChainDefinition':
            """Add a step to the chain."""
            step = ChainStep(
                name=name,
                function=function,
                arguments=kwargs.get("arguments", {}),
                output_mapping=kwargs.get("output_mapping", {}),
                condition=kwargs.get("condition"),
                on_error=kwargs.get("on_error", "stop")
            )
            self.builder.add_step(self.name, step)
            return self

        def conditional(self, condition: str, true_steps: List[ChainStep],
                       false_steps: List[ChainStep]) -> 'ChainDefinition':
            """Add conditional branch."""
            self.builder.add_conditional(self.name, condition, true_steps, false_steps)
            return self

        def loop(self, over: str, steps: List[ChainStep]) -> 'ChainDefinition':
            """Add loop construct."""
            self.builder.add_loop(self.name, over, steps)
            return self

        def build(self) -> str:
            """Finalize and return chain name."""
            return self.name

    # Create sample data processing pipeline
    builder = ChainBuilder()

    # Register functions
    def fetch_data(source: str) -> Dict:
        """Fetch data from source."""
        data = [
            {"id": i, "value": i * 10, "category": ["A", "B", "C"][i % 3]}
            for i in range(10)
        ]
        return {"data": data, "source": source, "count": len(data)}

    def filter_data(data: List[Dict], threshold: int) -> Dict:
        """Filter data by threshold."""
        filtered = [d for d in data if d["value"] > threshold]
        return {"filtered": filtered, "original_count": len(data), "filtered_count": len(filtered)}

    def transform_data(data: List[Dict]) -> List[Dict]:
        """Transform data format."""
        return [{"key": d["id"], "score": d["value"] / 10} for d in data]

    def aggregate_data(data: List[Dict]) -> Dict:
        """Aggregate data statistics."""
        if not data:
            return {"count": 0, "sum": 0, "avg": 0}

        values = [d.get("score", 0) for d in data]
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values) if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0
        }

    def save_results(data: Any, format: str) -> Dict:
        """Save results to storage."""
        return {"saved": True, "format": format, "records": len(data) if isinstance(data, list) else 1}

    builder.register_function("fetch_data", fetch_data)
    builder.register_function("filter_data", filter_data)
    builder.register_function("transform_data", transform_data)
    builder.register_function("aggregate_data", aggregate_data)
    builder.register_function("save_results", save_results)

    # Build complex pipeline
    chain = (builder.create_chain("data_pipeline")
             .step("fetch", "fetch_data",
                   arguments={"source": "database"},
                   output_mapping={"data": "raw_data", "count": "total_count"})
             .step("filter", "filter_data",
                   arguments={"data": "$raw_data", "threshold": 30},
                   output_mapping={"filtered": "filtered_data"})
             .step("transform", "transform_data",
                   arguments={"data": "$filtered_data"},
                   output_mapping={})  # Result goes to context as 'transform'
             .build())

    # Add conditional processing
    builder.add_conditional(
        "data_pipeline",
        "total_count > 5",
        true_branch=[
            ChainStep("aggregate", "aggregate_data",
                     {"data": "$transform"},
                     {"avg": "average_score"})
        ],
        false_branch=[
            ChainStep("skip_aggregate", "save_results",
                     {"data": [], "format": "empty"},
                     {})
        ]
    )

    # Execute pipeline
    result = builder.execute_chain("data_pipeline", {"threshold": 30})

    print("\n" + "=" * 50)
    print("Pipeline Results:")
    print("=" * 50)
    print(f"Status: {result['status']}")
    print(f"Final context keys: {list(result['context'].keys())}")
    if "average_score" in result["context"]:
        print(f"Average score: {result['context']['average_score']:.2f}")


# ===== Solution 5: Performance Optimizer =====

def solution_5_performance_optimizer():
    """
    Solution 5: Optimize function call performance.
    """
    print("\nSolution 5: Performance Optimizer")
    print("=" * 50)

    import math

    class CacheManager:
        """Manage function result caching with TTL."""

        def __init__(self):
            self.cache = {}
            self.ttls = {}
            self.hits = 0
            self.misses = 0

        def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
            """Create cache key from function call."""
            key_data = f"{func_name}:{args}:{sorted(kwargs.items())}"
            return hashlib.md5(key_data.encode()).hexdigest()

        def get(self, key: str) -> Optional[Any]:
            """Get cached result if valid."""
            if key in self.cache:
                if key in self.ttls:
                    if datetime.now() > self.ttls[key]:
                        # Expired
                        del self.cache[key]
                        del self.ttls[key]
                        self.misses += 1
                        return None
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

        def set(self, key: str, value: Any, ttl: int = 300):
            """Cache a result with TTL."""
            self.cache[key] = value
            self.ttls[key] = datetime.now() + timedelta(seconds=ttl)

        def invalidate(self, pattern: str):
            """Invalidate cache entries matching pattern."""
            keys_to_remove = []
            for key in self.cache:
                if pattern in key:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.cache[key]
                if key in self.ttls:
                    del self.ttls[key]

        def get_stats(self) -> Dict:
            """Get cache statistics."""
            total = self.hits + self.misses
            return {
                "size": len(self.cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total > 0 else 0,
                "expired": sum(1 for k, t in self.ttls.items() if datetime.now() > t)
            }

    class BatchProcessor:
        """Process multiple function calls efficiently."""

        def __init__(self, batch_size: int = 10):
            self.batch_size = batch_size
            self.pending = []
            self.results = {}

        def add_call(self, call_id: str, function: Callable, arguments: Dict):
            """Add a call to the batch."""
            self.pending.append({
                "id": call_id,
                "function": function,
                "arguments": arguments
            })

        def should_process(self) -> bool:
            """Check if batch should be processed."""
            return len(self.pending) >= self.batch_size

        def process_batch(self) -> Dict[str, Any]:
            """Process all pending calls."""
            if not self.pending:
                return {}

            # Group by function for efficient processing
            grouped = defaultdict(list)
            for call in self.pending:
                func_name = call["function"].__name__
                grouped[func_name].append(call)

            results = {}

            # Process each group
            for func_name, calls in grouped.items():
                print(f"  Batch processing {len(calls)} calls to {func_name}")

                # Execute in parallel using threads
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {}
                    for call in calls:
                        future = executor.submit(call["function"], **call["arguments"])
                        futures[future] = call["id"]

                    for future in futures:
                        call_id = futures[future]
                        try:
                            results[call_id] = future.result(timeout=5)
                        except Exception as e:
                            results[call_id] = {"error": str(e)}

            self.pending.clear()
            return results

    class PerformanceOptimizer:
        """Optimize function execution performance."""

        def __init__(self):
            self.cache = CacheManager()
            self.batch_processor = BatchProcessor()
            self.lazy_loaded = {}
            self.metrics = {
                "total_calls": 0,
                "cached_calls": 0,
                "batch_calls": 0,
                "parallel_calls": 0,
                "total_time": 0
            }

        def execute_with_cache(self, function: Callable, *args,
                             cache_ttl: int = 300, **kwargs):
            """Execute with caching."""
            self.metrics["total_calls"] += 1

            # Check cache
            cache_key = self.cache._make_key(function.__name__, args, kwargs)
            cached = self.cache.get(cache_key)

            if cached is not None:
                self.metrics["cached_calls"] += 1
                print(f"  Cache HIT for {function.__name__}")
                return cached

            print(f"  Cache MISS for {function.__name__}")

            # Execute function
            start_time = time.time()
            result = function(*args, **kwargs)
            execution_time = time.time() - start_time
            self.metrics["total_time"] += execution_time

            # Cache result
            self.cache.set(cache_key, result, cache_ttl)

            return result

        def execute_batch(self, calls: List[Dict]) -> List[Any]:
            """Execute multiple calls efficiently."""
            self.metrics["batch_calls"] += len(calls)

            # Add all calls to batch processor
            for i, call in enumerate(calls):
                self.batch_processor.add_call(
                    f"call_{i}",
                    call["function"],
                    call.get("arguments", {})
                )

            # Process batch
            results = self.batch_processor.process_batch()

            # Return results in order
            return [results.get(f"call_{i}") for i in range(len(calls))]

        async def execute_parallel_async(self, calls: List[Callable]) -> List[Any]:
            """Execute functions in parallel using asyncio."""
            self.metrics["parallel_calls"] += len(calls)

            async def wrap_sync(func):
                """Wrap sync function for async execution."""
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func)

            # Execute all in parallel
            tasks = [wrap_sync(call) for call in calls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            return results

        def execute_parallel(self, calls: List[Callable]) -> List[Any]:
            """Execute functions in parallel using threads."""
            self.metrics["parallel_calls"] += len(calls)

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(call) for call in calls]
                results = []

                for future in futures:
                    try:
                        results.append(future.result(timeout=10))
                    except Exception as e:
                        results.append({"error": str(e)})

            return results

        def lazy_load(self, name: str, loader: Callable) -> Callable:
            """Create lazy-loaded function."""
            def lazy_wrapper(*args, **kwargs):
                if name not in self.lazy_loaded:
                    print(f"  Lazy loading {name}...")
                    self.lazy_loaded[name] = loader()

                return self.lazy_loaded[name](*args, **kwargs)

            return lazy_wrapper

        def get_performance_report(self) -> Dict:
            """Get performance metrics."""
            cache_stats = self.cache.get_stats()

            return {
                "metrics": self.metrics,
                "cache": cache_stats,
                "avg_execution_time": self.metrics["total_time"] / self.metrics["total_calls"]
                                      if self.metrics["total_calls"] > 0 else 0,
                "cache_benefit": self.metrics["cached_calls"] / self.metrics["total_calls"]
                                if self.metrics["total_calls"] > 0 else 0
            }

    # Test optimization techniques
    optimizer = PerformanceOptimizer()

    # Test functions
    def slow_function(x: int) -> int:
        """Simulate slow function."""
        time.sleep(0.5)
        return x * x

    def data_processor(data: List[int]) -> int:
        """Process data."""
        time.sleep(0.2)
        return sum(data)

    # Test caching
    print("\nTesting Caching:")
    print("-" * 30)

    for i in range(5):
        result = optimizer.execute_with_cache(slow_function, 10, cache_ttl=10)
        print(f"  Call {i+1}: Result = {result}")

    # Test batch processing
    print("\n" + "-" * 30)
    print("Testing Batch Processing:")
    print("-" * 30)

    batch_calls = [
        {"function": data_processor, "arguments": {"data": [1, 2, 3]}},
        {"function": data_processor, "arguments": {"data": [4, 5, 6]}},
        {"function": data_processor, "arguments": {"data": [7, 8, 9]}},
        {"function": slow_function, "arguments": {"x": 5}},
        {"function": slow_function, "arguments": {"x": 7}}
    ]

    start = time.time()
    results = optimizer.execute_batch(batch_calls)
    batch_time = time.time() - start
    print(f"  Batch results: {results}")
    print(f"  Batch time: {batch_time:.2f}s")

    # Test parallel execution
    print("\n" + "-" * 30)
    print("Testing Parallel Execution:")
    print("-" * 30)

    parallel_calls = [
        lambda: slow_function(i) for i in range(5)
    ]

    start = time.time()
    results = optimizer.execute_parallel(parallel_calls)
    parallel_time = time.time() - start
    print(f"  Parallel results: {results}")
    print(f"  Parallel time: {parallel_time:.2f}s")
    print(f"  Speedup vs sequential: {(0.5 * 5) / parallel_time:.2f}x")

    # Test lazy loading
    print("\n" + "-" * 30)
    print("Testing Lazy Loading:")
    print("-" * 30)

    def expensive_loader():
        """Simulate expensive initialization."""
        print("    Loading expensive resource...")
        time.sleep(1)
        return lambda x: x * 100

    lazy_func = optimizer.lazy_load("expensive", expensive_loader)

    for i in range(3):
        result = lazy_func(i)
        print(f"  Call {i+1}: {result}")

    # Performance report
    print("\n" + "=" * 50)
    print("Performance Report:")
    print("=" * 50)
    report = optimizer.get_performance_report()
    print(json.dumps(report, indent=2, default=str))


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 09: Function Calling Solutions")
    parser.add_argument("--solution", type=int, help="Run specific solution (1-5)")
    parser.add_argument("--all", action="store_true", help="Run all solutions")

    args = parser.parse_args()

    solutions = {
        1: solution_1_tool_designer,
        2: solution_2_error_handler,
        3: solution_3_security_sandbox,
        4: solution_4_chain_builder,
        5: solution_5_performance_optimizer
    }

    if args.all:
        for sol_num, sol_func in solutions.items():
            print(f"\n{'='*60}")
            sol_func()
            print(f"\n{'='*60}\n")
    elif args.solution and args.solution in solutions:
        solutions[args.solution]()
    else:
        print("Module 09: Function Calling - Solutions")
        print("\nUsage:")
        print("  python solutions.py --solution N  # Run solution N")
        print("  python solutions.py --all          # Run all solutions")
        print("\nAvailable solutions:")
        print("  1: Tool Designer")
        print("  2: Error Handler")
        print("  3: Security Sandbox")
        print("  4: Chain Builder")
        print("  5: Performance Optimizer")