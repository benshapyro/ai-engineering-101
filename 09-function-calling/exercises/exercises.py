"""
Module 09: Function Calling - Exercises

Practice exercises for mastering function calling and tool use with LLMs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import time
from datetime import datetime


# ===== Exercise 1: Tool Designer =====

def exercise_1_tool_designer():
    """
    Exercise 1: Design comprehensive tool interfaces.

    TODO:
    1. Create a weather service with multiple endpoints
    2. Define proper JSON schemas for each function
    3. Add detailed descriptions and examples
    4. Implement parameter validation
    5. Create response formatters
    """
    print("Exercise 1: Tool Designer")
    print("=" * 50)

    class WeatherService:
        """TODO: Implement a comprehensive weather service."""

        def __init__(self):
            self.functions = []

        def get_current_weather_definition(self) -> Dict:
            """TODO: Define current weather function."""
            # return {
            #     "name": "get_current_weather",
            #     "description": "...",
            #     "parameters": {
            #         "type": "object",
            #         "properties": {
            #             "location": {...},
            #             "units": {...}
            #         },
            #         "required": [...]
            #     }
            # }
            pass

        def get_forecast_definition(self) -> Dict:
            """TODO: Define forecast function."""
            # Include:
            # - location (with validation)
            # - days (1-10)
            # - include_hourly (boolean)
            # - units
            pass

        def get_alerts_definition(self) -> Dict:
            """TODO: Define weather alerts function."""
            # Include:
            # - location or coordinates
            # - severity levels
            # - alert types
            pass

        def validate_location(self, location: str) -> bool:
            """TODO: Validate location format."""
            # Check for:
            # - City, State format
            # - ZIP code
            # - Coordinates
            pass

        def execute_function(self, function_name: str, arguments: Dict) -> Dict:
            """TODO: Execute weather function with validation."""
            # 1. Validate function exists
            # 2. Validate parameters
            # 3. Execute appropriate function
            # 4. Format response
            pass

    # TODO: Create service instance
    service = WeatherService()

    # TODO: Test function definitions
    print("TODO: Display function definitions")
    print("TODO: Test parameter validation")
    print("TODO: Execute sample functions")


# ===== Exercise 2: Error Handler =====

def exercise_2_error_handler():
    """
    Exercise 2: Build robust error handling system.

    TODO:
    1. Implement retry logic with exponential backoff
    2. Create circuit breaker pattern
    3. Add fallback strategies
    4. Implement error categorization
    5. Create recovery mechanisms
    """
    print("\nExercise 2: Error Handler")
    print("=" * 50)

    class CircuitBreaker:
        """TODO: Implement circuit breaker pattern."""

        def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            # TODO: Add state tracking
            # self.state = "closed"  # closed, open, half-open
            # self.failure_count = 0
            # self.last_failure_time = None

        def call(self, func: Callable, *args, **kwargs):
            """TODO: Execute function with circuit breaker."""
            # Check circuit state
            # If open, return cached result or error
            # If half-open, try once
            # If closed, execute normally
            # Track failures and successes
            pass

        def record_success(self):
            """TODO: Record successful call."""
            pass

        def record_failure(self):
            """TODO: Record failed call."""
            pass

        def should_attempt_reset(self) -> bool:
            """TODO: Check if should try half-open state."""
            pass

    class RobustExecutor:
        """TODO: Execute functions with comprehensive error handling."""

        def __init__(self):
            self.circuit_breakers = {}
            self.retry_config = {
                "max_retries": 3,
                "base_delay": 1,
                "max_delay": 30,
                "exponential_base": 2
            }

        def execute_with_retry(self, func: Callable, *args, **kwargs):
            """TODO: Execute with exponential backoff retry."""
            # for attempt in range(max_retries):
            #     try:
            #         result = func(*args, **kwargs)
            #         return result
            #     except RetryableError as e:
            #         delay = calculate_backoff(attempt)
            #         time.sleep(delay)
            #     except NonRetryableError as e:
            #         raise
            pass

        def execute_with_fallback(self, primary: Callable, fallback: Callable):
            """TODO: Execute with fallback strategy."""
            # Try primary function
            # On failure, try fallback
            # Log degraded operation
            pass

        def categorize_error(self, error: Exception) -> str:
            """TODO: Categorize error for appropriate handling."""
            # Return: "retryable", "non_retryable", "rate_limit", "auth", etc.
            pass

    # TODO: Test error handling
    executor = RobustExecutor()

    print("TODO: Test retry logic with transient errors")
    print("TODO: Test circuit breaker with persistent failures")
    print("TODO: Test fallback strategies")
    print("TODO: Test error categorization")


# ===== Exercise 3: Security Sandbox =====

def exercise_3_security_sandbox():
    """
    Exercise 3: Implement secure function execution.

    TODO:
    1. Create permission system for functions
    2. Implement input sanitization
    3. Add rate limiting per user/function
    4. Create audit logging
    5. Implement resource limits (timeout, memory)
    """
    print("\nExercise 3: Security Sandbox")
    print("=" * 50)

    @dataclass
    class SecurityContext:
        """TODO: Define security context for execution."""
        user_id: str
        roles: List[str]
        permissions: List[str]
        # TODO: Add more security attributes
        # rate_limit_remaining: int
        # session_id: str
        # ip_address: str

    class RateLimiter:
        """TODO: Implement rate limiting."""

        def __init__(self):
            self.limits = {}  # function -> limit config
            self.calls = {}   # (user, function) -> call timestamps

        def set_limit(self, function: str, calls: int, period: int):
            """TODO: Set rate limit for function."""
            # Store limit configuration
            pass

        def check_limit(self, user_id: str, function: str) -> bool:
            """TODO: Check if user can call function."""
            # Check call history
            # Remove old calls outside window
            # Return True if under limit
            pass

        def record_call(self, user_id: str, function: str):
            """TODO: Record a function call."""
            pass

    class SecureSandbox:
        """TODO: Secure execution environment."""

        def __init__(self):
            self.rate_limiter = RateLimiter()
            self.audit_log = []
            self.permissions = {}

        def register_function_permissions(self, function: str, required: List[str]):
            """TODO: Register required permissions for function."""
            pass

        def check_permissions(self, context: SecurityContext, function: str) -> bool:
            """TODO: Check if user has required permissions."""
            pass

        def sanitize_input(self, function: str, arguments: Dict) -> Dict:
            """TODO: Sanitize and validate input."""
            # Remove dangerous characters
            # Validate data types
            # Check for injection attempts
            pass

        def execute_sandboxed(self, context: SecurityContext,
                            function: str, arguments: Dict) -> Dict:
            """TODO: Execute function in sandbox."""
            # 1. Check permissions
            # 2. Check rate limits
            # 3. Sanitize inputs
            # 4. Execute with resource limits
            # 5. Audit log the execution
            # 6. Return result or error
            pass

        def audit_execution(self, context: SecurityContext, function: str,
                          success: bool, error: Optional[str] = None):
            """TODO: Log execution for audit."""
            pass

    # TODO: Test security features
    sandbox = SecureSandbox()
    context = SecurityContext(user_id="user123", roles=["developer"], permissions=[])

    print("TODO: Test permission checking")
    print("TODO: Test rate limiting")
    print("TODO: Test input sanitization")
    print("TODO: Test audit logging")


# ===== Exercise 4: Chain Builder =====

def exercise_4_chain_builder():
    """
    Exercise 4: Create complex function chains.

    TODO:
    1. Build a data pipeline with multiple steps
    2. Implement conditional branching
    3. Add loop constructs
    4. Create parallel execution branches
    5. Implement state management between steps
    """
    print("\nExercise 4: Chain Builder")
    print("=" * 50)

    @dataclass
    class ChainStep:
        """TODO: Define a step in the chain."""
        name: str
        function: str
        arguments: Dict[str, Any]
        # TODO: Add more attributes
        # output_mapping: Dict[str, str]  # Map output to context
        # condition: Optional[str]  # Condition to execute
        # on_error: Optional[str]  # Error handling strategy

    class ChainBuilder:
        """TODO: Build and execute function chains."""

        def __init__(self):
            self.chains = {}
            self.functions = {}
            self.context = {}

        def register_function(self, name: str, func: Callable):
            """TODO: Register a function for use in chains."""
            pass

        def create_chain(self, chain_name: str) -> 'ChainDefinition':
            """TODO: Create a new chain definition."""
            # Return a fluent interface for building chains
            pass

        def add_step(self, chain_name: str, step: ChainStep):
            """TODO: Add a step to a chain."""
            pass

        def add_conditional(self, chain_name: str, condition: str,
                          true_branch: List[ChainStep],
                          false_branch: List[ChainStep]):
            """TODO: Add conditional branching."""
            pass

        def add_loop(self, chain_name: str, items_key: str,
                    loop_steps: List[ChainStep]):
            """TODO: Add loop construct."""
            pass

        def execute_chain(self, chain_name: str, initial_context: Dict) -> Dict:
            """TODO: Execute a complete chain."""
            # 1. Initialize context
            # 2. Execute each step in order
            # 3. Handle conditionals
            # 4. Process loops
            # 5. Manage state between steps
            # 6. Return final result
            pass

    class ChainDefinition:
        """TODO: Fluent interface for chain building."""

        def __init__(self, builder: ChainBuilder, name: str):
            self.builder = builder
            self.name = name

        def step(self, function: str, **kwargs) -> 'ChainDefinition':
            """TODO: Add a step to the chain."""
            # Create step and add to chain
            # Return self for chaining
            pass

        def conditional(self, condition: str) -> 'ConditionalBuilder':
            """TODO: Start conditional branch."""
            pass

        def loop(self, over: str) -> 'LoopBuilder':
            """TODO: Start loop construct."""
            pass

        def build(self) -> str:
            """TODO: Finalize and return chain name."""
            pass

    # TODO: Create sample data processing pipeline
    builder = ChainBuilder()

    # Register functions
    def fetch_data(source: str) -> List[Dict]:
        """Fetch data from source."""
        return [{"id": i, "value": i * 10} for i in range(5)]

    def filter_data(data: List[Dict], threshold: int) -> List[Dict]:
        """Filter data by threshold."""
        return [d for d in data if d["value"] > threshold]

    def transform_data(data: List[Dict]) -> List[Dict]:
        """Transform data format."""
        return [{"key": d["id"], "score": d["value"] / 10} for d in data]

    # TODO: Build and execute chain
    print("TODO: Build data processing pipeline")
    print("TODO: Add conditional processing")
    print("TODO: Add loop for batch processing")
    print("TODO: Execute and show results")


# ===== Exercise 5: Performance Optimizer =====

def exercise_5_performance_optimizer():
    """
    Exercise 5: Optimize function call performance.

    TODO:
    1. Implement result caching with TTL
    2. Create batch processing for multiple calls
    3. Implement parallel execution
    4. Add lazy loading for expensive functions
    5. Create performance monitoring
    """
    print("\nExercise 5: Performance Optimizer")
    print("=" * 50)

    class CacheManager:
        """TODO: Manage function result caching."""

        def __init__(self):
            self.cache = {}
            self.ttls = {}

        def get(self, key: str) -> Optional[Any]:
            """TODO: Get cached result if valid."""
            # Check if exists and not expired
            pass

        def set(self, key: str, value: Any, ttl: int = 300):
            """TODO: Cache a result with TTL."""
            pass

        def invalidate(self, pattern: str):
            """TODO: Invalidate cache entries matching pattern."""
            pass

        def get_stats(self) -> Dict:
            """TODO: Get cache statistics."""
            # Return hit rate, size, etc.
            pass

    class BatchProcessor:
        """TODO: Process multiple function calls efficiently."""

        def __init__(self, batch_size: int = 10):
            self.batch_size = batch_size
            self.pending = []

        def add_call(self, function: str, arguments: Dict):
            """TODO: Add a call to the batch."""
            pass

        def should_process(self) -> bool:
            """TODO: Check if batch should be processed."""
            pass

        def process_batch(self) -> List[Any]:
            """TODO: Process all pending calls."""
            # Group similar calls
            # Execute in parallel where possible
            # Return results in order
            pass

    class PerformanceOptimizer:
        """TODO: Optimize function execution performance."""

        def __init__(self):
            self.cache = CacheManager()
            self.batch_processor = BatchProcessor()
            self.metrics = {
                "cache_hits": 0,
                "cache_misses": 0,
                "total_calls": 0,
                "total_time": 0
            }

        def execute_with_cache(self, function: Callable, cache_key: str,
                              ttl: int = 300):
            """TODO: Execute with caching."""
            # Check cache first
            # Execute if miss
            # Cache result
            # Update metrics
            pass

        def execute_batch(self, calls: List[Dict]) -> List[Any]:
            """TODO: Execute multiple calls efficiently."""
            # Group by function
            # Execute in parallel
            # Combine results
            pass

        def execute_parallel(self, calls: List[Callable]) -> List[Any]:
            """TODO: Execute functions in parallel."""
            # Use threading or asyncio
            # Handle errors
            # Return results in order
            pass

        def lazy_load(self, loader: Callable) -> Callable:
            """TODO: Create lazy-loaded function."""
            # Return wrapper that loads on first call
            pass

        def get_performance_report(self) -> Dict:
            """TODO: Get performance metrics."""
            pass

    # TODO: Test optimization techniques
    optimizer = PerformanceOptimizer()

    print("TODO: Test caching with repeated calls")
    print("TODO: Test batch processing")
    print("TODO: Test parallel execution")
    print("TODO: Test lazy loading")
    print("TODO: Show performance metrics")


# ===== Challenge: Build Complete AI Assistant =====

def challenge_ai_assistant():
    """
    Challenge: Build a complete AI assistant with 20+ tools.

    Requirements:
    1. Implement tools across multiple categories:
       - Data processing (5 tools)
       - Communication (4 tools)
       - Analytics (4 tools)
       - Automation (4 tools)
       - Utilities (3+ tools)

    2. Features to implement:
       - Dynamic tool discovery
       - Context-aware tool selection
       - Multi-step task planning
       - Error recovery
       - User preference learning
       - Performance optimization
       - Security and permissions
       - Usage analytics

    TODO: Complete the implementation
    """
    print("\nChallenge: Complete AI Assistant")
    print("=" * 50)

    class AIAssistant:
        """TODO: Implement complete AI assistant."""

        def __init__(self, llm_client: LLMClient):
            self.llm_client = llm_client
            self.tools = {}
            self.user_preferences = {}
            self.task_planner = None
            self.security_manager = None
            self.performance_optimizer = None
            self.analytics = None

        def register_all_tools(self):
            """TODO: Register 20+ tools across categories."""
            # Data Processing Tools
            # - csv_reader
            # - json_parser
            # - data_cleaner
            # - data_transformer
            # - data_validator

            # Communication Tools
            # - send_email
            # - send_sms
            # - slack_message
            # - schedule_meeting

            # Analytics Tools
            # - calculate_statistics
            # - generate_report
            # - create_visualization
            # - trend_analysis

            # Automation Tools
            # - schedule_task
            # - trigger_workflow
            # - monitor_system
            # - auto_respond

            # Utility Tools
            # - file_operations
            # - time_converter
            # - text_formatter
            pass

        def process_request(self, user_request: str, user_context: Dict) -> Dict:
            """TODO: Process user request with intelligent tool selection."""
            # 1. Understand intent
            # 2. Select appropriate tools
            # 3. Plan execution steps
            # 4. Execute with error handling
            # 5. Optimize performance
            # 6. Format response
            pass

        def plan_task(self, request: str) -> List[Dict]:
            """TODO: Create execution plan for complex tasks."""
            # Analyze request
            # Break into steps
            # Identify required tools
            # Determine dependencies
            # Return execution plan
            pass

        def select_tools(self, task_description: str) -> List[str]:
            """TODO: Intelligently select tools for task."""
            # Use LLM to understand task
            # Match with available tools
            # Consider user preferences
            # Return ranked tool list
            pass

        def execute_with_recovery(self, plan: List[Dict]) -> Dict:
            """TODO: Execute plan with error recovery."""
            # Execute each step
            # Handle failures
            # Try alternatives
            # Graceful degradation
            pass

        def learn_preferences(self, user_id: str, feedback: Dict):
            """TODO: Learn from user feedback."""
            # Update preference model
            # Adjust tool rankings
            # Improve response format
            pass

        def get_assistant_stats(self) -> Dict:
            """TODO: Get comprehensive assistant statistics."""
            # Tool usage stats
            # Success rates
            # Performance metrics
            # User satisfaction
            pass

    # TODO: Create and test assistant
    client = LLMClient("openai")
    assistant = AIAssistant(client)

    print("TODO: Register all 20+ tools")
    print("TODO: Process sample requests")
    print("TODO: Test multi-step planning")
    print("TODO: Test error recovery")
    print("TODO: Show usage analytics")

    # Test scenarios
    test_requests = [
        "Analyze this month's sales data and send a report to the team",
        "Schedule a meeting with available participants and send reminders",
        "Monitor system health and alert on anomalies",
        "Process customer feedback and generate insights"
    ]

    print("\nTODO: Process each test request")
    for request in test_requests:
        print(f"  - {request}")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 09: Function Calling Exercises")
    parser.add_argument("--exercise", type=int, help="Run specific exercise (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge")
    parser.add_argument("--all", action="store_true", help="Run all exercises")

    args = parser.parse_args()

    exercises = {
        1: exercise_1_tool_designer,
        2: exercise_2_error_handler,
        3: exercise_3_security_sandbox,
        4: exercise_4_chain_builder,
        5: exercise_5_performance_optimizer
    }

    if args.all:
        for ex_num, ex_func in exercises.items():
            print(f"\n{'='*60}")
            ex_func()
            print(f"\n{'='*60}\n")
        challenge_ai_assistant()
    elif args.challenge:
        challenge_ai_assistant()
    elif args.exercise and args.exercise in exercises:
        exercises[args.exercise]()
    else:
        print("Module 09: Function Calling - Exercises")
        print("\nUsage:")
        print("  python exercises.py --exercise N  # Run exercise N")
        print("  python exercises.py --challenge    # Run challenge")
        print("  python exercises.py --all          # Run all exercises")
        print("\nAvailable exercises:")
        print("  1: Tool Designer")
        print("  2: Error Handler")
        print("  3: Security Sandbox")
        print("  4: Chain Builder")
        print("  5: Performance Optimizer")
        print("  Challenge: Complete AI Assistant")