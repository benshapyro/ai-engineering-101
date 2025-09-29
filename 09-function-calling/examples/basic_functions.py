"""
Module 09: Function Calling - Basic Functions

Learn the fundamentals of function calling with LLMs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import random


# ===== Example 1: Simple Function Definitions =====

def example_1_simple_functions():
    """Define and execute simple functions."""
    print("Example 1: Simple Function Definitions")
    print("=" * 50)

    # Define functions for OpenAI format
    openai_functions = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit"
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "calculate",
            "description": "Perform basic arithmetic calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The arithmetic operation"
                    },
                    "a": {
                        "type": "number",
                        "description": "First operand"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second operand"
                    }
                },
                "required": ["operation", "a", "b"]
            }
        }
    ]

    # Define functions for Anthropic/Claude format
    anthropic_tools = [
        {
            "name": "search_database",
            "description": "Search customer database",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "filters": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["active", "inactive", "pending"]
                            },
                            "created_after": {
                                "type": "string",
                                "format": "date"
                            }
                        }
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["query"]
            }
        }
    ]

    # Function implementations
    def get_weather(location: str, unit: str = "fahrenheit") -> Dict:
        """Mock weather function."""
        temp = random.randint(60, 85) if unit == "fahrenheit" else random.randint(15, 30)
        return {
            "location": location,
            "temperature": temp,
            "unit": unit,
            "conditions": random.choice(["sunny", "cloudy", "rainy"]),
            "humidity": random.randint(30, 70)
        }

    def calculate(operation: str, a: float, b: float) -> float:
        """Basic calculator function."""
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else None
        }
        return operations.get(operation, lambda x, y: None)(a, b)

    def search_database(query: str, filters: Optional[Dict] = None, limit: int = 10) -> Dict:
        """Mock database search."""
        results = []
        for i in range(min(limit, 5)):
            results.append({
                "id": f"CUST{1000 + i}",
                "name": f"Customer {query} {i}",
                "status": filters.get("status", "active") if filters else "active",
                "created": "2024-01-01"
            })
        return {"count": len(results), "results": results}

    # Function registry
    function_registry = {
        "get_weather": get_weather,
        "calculate": calculate,
        "search_database": search_database
    }

    # Execute function based on LLM response
    def execute_function_call(function_name: str, arguments: Dict) -> Any:
        """Execute a function call with arguments."""
        if function_name not in function_registry:
            return {"error": f"Function {function_name} not found"}

        try:
            result = function_registry[function_name](**arguments)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # Test function execution
    print("\n1. Weather Function:")
    weather_result = execute_function_call("get_weather", {
        "location": "San Francisco, CA",
        "unit": "celsius"
    })
    print(json.dumps(weather_result, indent=2))

    print("\n2. Calculator Function:")
    calc_result = execute_function_call("calculate", {
        "operation": "multiply",
        "a": 15,
        "b": 3.5
    })
    print(json.dumps(calc_result, indent=2))

    print("\n3. Database Search:")
    search_result = execute_function_call("search_database", {
        "query": "tech companies",
        "filters": {"status": "active"},
        "limit": 3
    })
    print(json.dumps(search_result, indent=2))


# ===== Example 2: Function Call Flow with LLM =====

def example_2_function_call_flow():
    """Demonstrate complete function calling flow with LLM."""
    print("\nExample 2: Function Call Flow with LLM")
    print("=" * 50)

    client = LLMClient("openai")

    # Define available functions
    functions = [
        {
            "name": "get_stock_price",
            "description": "Get current stock price for a ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g., AAPL"
                    },
                    "exchange": {
                        "type": "string",
                        "enum": ["NYSE", "NASDAQ", "LSE"],
                        "description": "Stock exchange"
                    }
                },
                "required": ["ticker"]
            }
        },
        {
            "name": "get_company_info",
            "description": "Get basic company information",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Company name"
                    }
                },
                "required": ["company_name"]
            }
        }
    ]

    # Mock function implementations
    def get_stock_price(ticker: str, exchange: str = "NASDAQ") -> Dict:
        """Mock stock price fetcher."""
        prices = {
            "AAPL": 185.50,
            "GOOGL": 140.25,
            "MSFT": 380.75,
            "AMZN": 155.30
        }
        return {
            "ticker": ticker,
            "price": prices.get(ticker, random.uniform(50, 500)),
            "exchange": exchange,
            "currency": "USD",
            "timestamp": datetime.now().isoformat()
        }

    def get_company_info(company_name: str) -> Dict:
        """Mock company info fetcher."""
        companies = {
            "Apple": {
                "ticker": "AAPL",
                "sector": "Technology",
                "employees": 150000,
                "founded": 1976
            },
            "Microsoft": {
                "ticker": "MSFT",
                "sector": "Technology",
                "employees": 220000,
                "founded": 1975
            }
        }
        return companies.get(company_name, {
            "ticker": "UNKNOWN",
            "sector": "Unknown",
            "employees": 0,
            "founded": 0
        })

    function_registry = {
        "get_stock_price": get_stock_price,
        "get_company_info": get_company_info
    }

    # Simulate function calling flow
    def function_calling_flow(user_query: str):
        """Complete function calling flow."""
        print(f"\nUser Query: {user_query}")

        # Step 1: Send query with function definitions
        prompt = f"""You have access to the following functions:
{json.dumps(functions, indent=2)}

User query: {user_query}

If you need to use a function, respond with:
FUNCTION_CALL: function_name
ARGUMENTS: {{arguments}}

Otherwise, respond normally."""

        response = client.complete(prompt, max_tokens=200)
        print(f"\nLLM Response:\n{response}")

        # Step 2: Parse function call if present
        if "FUNCTION_CALL:" in response:
            lines = response.split("\n")
            function_name = None
            arguments = None

            for i, line in enumerate(lines):
                if "FUNCTION_CALL:" in line:
                    function_name = line.split("FUNCTION_CALL:")[1].strip()
                if "ARGUMENTS:" in line:
                    # Extract JSON arguments
                    arg_start = line.index("{")
                    arguments = json.loads(line[arg_start:])

            if function_name and arguments:
                print(f"\nExecuting function: {function_name}")
                print(f"Arguments: {json.dumps(arguments, indent=2)}")

                # Step 3: Execute function
                if function_name in function_registry:
                    result = function_registry[function_name](**arguments)
                    print(f"\nFunction Result:\n{json.dumps(result, indent=2)}")

                    # Step 4: Send result back to LLM
                    follow_up = f"""Function {function_name} returned:
{json.dumps(result, indent=2)}

Please provide a natural language response to the user's original query: {user_query}"""

                    final_response = client.complete(follow_up, max_tokens=150)
                    print(f"\nFinal Response:\n{final_response}")

    # Test different queries
    queries = [
        "What's the current stock price of Apple?",
        "Tell me about Microsoft as a company",
        "Compare the stock prices of AAPL and GOOGL"
    ]

    for query in queries[:1]:  # Test first query to save API calls
        function_calling_flow(query)


# ===== Example 3: Parameter Validation =====

def example_3_parameter_validation():
    """Validate function parameters before execution."""
    print("\nExample 3: Parameter Validation")
    print("=" * 50)

    from jsonschema import validate, ValidationError as JsonValidationError

    class FunctionValidator:
        """Validate function calls against schemas."""

        def __init__(self):
            self.schemas = {}

        def register_function(self, name: str, schema: Dict):
            """Register function with its schema."""
            self.schemas[name] = schema

        def validate_call(self, function_name: str, arguments: Dict) -> tuple[bool, Optional[str]]:
            """Validate function call arguments."""
            if function_name not in self.schemas:
                return False, f"Unknown function: {function_name}"

            schema = self.schemas[function_name]["parameters"]

            try:
                validate(instance=arguments, schema=schema)
                return True, None
            except JsonValidationError as e:
                return False, str(e.message)

        def coerce_types(self, function_name: str, arguments: Dict) -> Dict:
            """Attempt to coerce argument types to match schema."""
            if function_name not in self.schemas:
                return arguments

            schema = self.schemas[function_name]["parameters"]
            properties = schema.get("properties", {})

            coerced = {}
            for key, value in arguments.items():
                if key in properties:
                    expected_type = properties[key].get("type")
                    coerced[key] = self._coerce_value(value, expected_type)
                else:
                    coerced[key] = value

            return coerced

        def _coerce_value(self, value: Any, expected_type: str) -> Any:
            """Coerce a value to expected type."""
            if expected_type == "string":
                return str(value)
            elif expected_type == "number":
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return value
            elif expected_type == "integer":
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return value
            elif expected_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ["true", "yes", "1"]
                return bool(value)
            else:
                return value

    # Create validator
    validator = FunctionValidator()

    # Register functions with schemas
    validator.register_function("send_email", {
        "name": "send_email",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "format": "email",
                    "description": "Recipient email"
                },
                "subject": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100
                },
                "body": {
                    "type": "string",
                    "maxLength": 5000
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high"]
                },
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["to", "subject", "body"]
        }
    })

    # Test cases
    test_cases = [
        {
            "name": "Valid email",
            "function": "send_email",
            "arguments": {
                "to": "user@example.com",
                "subject": "Test Email",
                "body": "This is a test message",
                "priority": "normal"
            }
        },
        {
            "name": "Missing required field",
            "function": "send_email",
            "arguments": {
                "to": "user@example.com",
                "body": "Missing subject"
            }
        },
        {
            "name": "Invalid enum value",
            "function": "send_email",
            "arguments": {
                "to": "user@example.com",
                "subject": "Test",
                "body": "Test",
                "priority": "urgent"  # Invalid
            }
        },
        {
            "name": "Type coercion needed",
            "function": "send_email",
            "arguments": {
                "to": "user@example.com",
                "subject": 123,  # Should be string
                "body": "Test"
            }
        }
    ]

    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"Arguments: {json.dumps(test['arguments'], indent=2)}")

        # Validate
        is_valid, error = validator.validate_call(test["function"], test["arguments"])

        if is_valid:
            print("✓ Validation passed")
        else:
            print(f"✗ Validation failed: {error}")

            # Try type coercion
            coerced = validator.coerce_types(test["function"], test["arguments"])
            if coerced != test["arguments"]:
                print(f"Attempting coercion: {json.dumps(coerced, indent=2)}")
                is_valid, error = validator.validate_call(test["function"], coerced)
                if is_valid:
                    print("✓ Validation passed after coercion")
                else:
                    print(f"✗ Still failed: {error}")


# ===== Example 4: Error Handling =====

def example_4_error_handling():
    """Implement comprehensive error handling for function calls."""
    print("\nExample 4: Error Handling")
    print("=" * 50)

    import time
    import traceback
    from enum import Enum

    class ErrorType(Enum):
        VALIDATION_ERROR = "validation_error"
        EXECUTION_ERROR = "execution_error"
        TIMEOUT_ERROR = "timeout_error"
        PERMISSION_ERROR = "permission_error"
        RATE_LIMIT_ERROR = "rate_limit_error"
        NOT_FOUND_ERROR = "not_found_error"

    class FunctionError(Exception):
        """Custom exception for function errors."""

        def __init__(self, error_type: ErrorType, message: str, details: Optional[Dict] = None):
            self.error_type = error_type
            self.message = message
            self.details = details or {}
            super().__init__(message)

    class SafeFunctionExecutor:
        """Execute functions with comprehensive error handling."""

        def __init__(self):
            self.functions = {}
            self.error_handlers = {}
            self.retry_config = {
                "max_retries": 3,
                "backoff_factor": 2,
                "max_backoff": 30
            }

        def register_function(self, name: str, func, timeout: float = 30.0):
            """Register a function with timeout."""
            self.functions[name] = {
                "func": func,
                "timeout": timeout,
                "call_count": 0,
                "error_count": 0
            }

        def register_error_handler(self, error_type: ErrorType, handler):
            """Register custom error handler."""
            self.error_handlers[error_type] = handler

        def execute(self, function_name: str, arguments: Dict) -> Dict:
            """Execute function with error handling and retries."""
            if function_name not in self.functions:
                return self._handle_error(
                    FunctionError(
                        ErrorType.NOT_FOUND_ERROR,
                        f"Function {function_name} not found",
                        {"available": list(self.functions.keys())}
                    )
                )

            func_info = self.functions[function_name]
            func_info["call_count"] += 1

            retry_count = 0
            last_error = None

            while retry_count <= self.retry_config["max_retries"]:
                try:
                    # Execute with timeout simulation
                    start_time = time.time()
                    result = func_info["func"](**arguments)

                    # Simulate timeout check
                    execution_time = time.time() - start_time
                    if execution_time > func_info["timeout"]:
                        raise FunctionError(
                            ErrorType.TIMEOUT_ERROR,
                            f"Function exceeded timeout of {func_info['timeout']}s",
                            {"execution_time": execution_time}
                        )

                    return {
                        "status": "success",
                        "result": result,
                        "execution_time": execution_time,
                        "retry_count": retry_count
                    }

                except FunctionError as e:
                    last_error = e
                    if not self._should_retry(e.error_type):
                        func_info["error_count"] += 1
                        return self._handle_error(e)

                except Exception as e:
                    last_error = FunctionError(
                        ErrorType.EXECUTION_ERROR,
                        str(e),
                        {"traceback": traceback.format_exc()}
                    )

                # Exponential backoff
                if retry_count < self.retry_config["max_retries"]:
                    backoff = min(
                        self.retry_config["backoff_factor"] ** retry_count,
                        self.retry_config["max_backoff"]
                    )
                    print(f"Retry {retry_count + 1} after {backoff}s...")
                    time.sleep(backoff)

                retry_count += 1

            # Max retries exceeded
            func_info["error_count"] += 1
            return self._handle_error(last_error)

        def _should_retry(self, error_type: ErrorType) -> bool:
            """Determine if error type is retryable."""
            retryable_errors = [
                ErrorType.TIMEOUT_ERROR,
                ErrorType.RATE_LIMIT_ERROR,
                ErrorType.EXECUTION_ERROR
            ]
            return error_type in retryable_errors

        def _handle_error(self, error: FunctionError) -> Dict:
            """Handle error with custom handler if available."""
            if error.error_type in self.error_handlers:
                return self.error_handlers[error.error_type](error)

            return {
                "status": "error",
                "error_type": error.error_type.value,
                "message": error.message,
                "details": error.details
            }

        def get_statistics(self) -> Dict:
            """Get execution statistics."""
            total_calls = sum(f["call_count"] for f in self.functions.values())
            total_errors = sum(f["error_count"] for f in self.functions.values())

            return {
                "total_calls": total_calls,
                "total_errors": total_errors,
                "error_rate": total_errors / total_calls if total_calls > 0 else 0,
                "functions": {
                    name: {
                        "calls": info["call_count"],
                        "errors": info["error_count"],
                        "error_rate": info["error_count"] / info["call_count"]
                        if info["call_count"] > 0 else 0
                    }
                    for name, info in self.functions.items()
                }
            }

    # Create executor
    executor = SafeFunctionExecutor()

    # Define test functions
    def reliable_function(x: int) -> int:
        """Always succeeds."""
        return x * 2

    def flaky_function(x: int) -> int:
        """Sometimes fails."""
        if random.random() < 0.3:
            raise Exception("Random failure")
        return x * 3

    def slow_function(x: int) -> int:
        """Simulates slow execution."""
        time.sleep(0.5)
        return x * 4

    def restricted_function(x: int) -> int:
        """Requires permission."""
        raise FunctionError(
            ErrorType.PERMISSION_ERROR,
            "Insufficient permissions",
            {"required_role": "admin"}
        )

    # Register functions
    executor.register_function("reliable", reliable_function)
    executor.register_function("flaky", flaky_function)
    executor.register_function("slow", slow_function, timeout=0.3)
    executor.register_function("restricted", restricted_function)

    # Custom error handler
    def permission_error_handler(error: FunctionError) -> Dict:
        """Handle permission errors specially."""
        return {
            "status": "permission_denied",
            "message": "Please contact admin for access",
            "required_role": error.details.get("required_role")
        }

    executor.register_error_handler(ErrorType.PERMISSION_ERROR, permission_error_handler)

    # Test executions
    test_functions = [
        ("reliable", {"x": 5}),
        ("flaky", {"x": 10}),
        ("slow", {"x": 3}),
        ("restricted", {"x": 7}),
        ("nonexistent", {"x": 1})
    ]

    for func_name, args in test_functions:
        print(f"\nExecuting: {func_name}({args})")
        result = executor.execute(func_name, args)
        print(f"Result: {json.dumps(result, indent=2)}")

    # Show statistics
    print("\nExecution Statistics:")
    print(json.dumps(executor.get_statistics(), indent=2))


# ===== Example 5: Response Formatting =====

def example_5_response_formatting():
    """Format function responses for different use cases."""
    print("\nExample 5: Response Formatting")
    print("=" * 50)

    class ResponseFormatter:
        """Format function responses for different contexts."""

        @staticmethod
        def format_for_user(function_name: str, result: Any) -> str:
            """Format result for end user display."""
            formatters = {
                "get_weather": lambda r: f"The weather in {r['location']} is {r['temperature']}° "
                                        f"{r['unit']} and {r['conditions']}.",
                "search_database": lambda r: f"Found {r['count']} results for your search.",
                "calculate": lambda r: f"The result is {r}",
                "get_stock_price": lambda r: f"{r['ticker']} is currently trading at "
                                             f"${r['price']:.2f} on {r['exchange']}."
            }

            formatter = formatters.get(function_name, lambda r: str(r))
            return formatter(result)

        @staticmethod
        def format_for_llm(function_name: str, result: Any) -> str:
            """Format result for LLM consumption."""
            return f"""Function: {function_name}
Status: Success
Result:
{json.dumps(result, indent=2)}

You can now use this information to answer the user's question."""

        @staticmethod
        def format_for_logging(function_name: str, arguments: Dict, result: Any,
                             execution_time: float) -> Dict:
            """Format for structured logging."""
            return {
                "timestamp": datetime.now().isoformat(),
                "function": function_name,
                "arguments": arguments,
                "result": result,
                "execution_time_ms": execution_time * 1000,
                "success": True
            }

        @staticmethod
        def format_error(function_name: str, error: Exception) -> Dict:
            """Format error response."""
            return {
                "function": function_name,
                "success": False,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "timestamp": datetime.now().isoformat()
                }
            }

    formatter = ResponseFormatter()

    # Test different response formats
    test_results = {
        "get_weather": {
            "location": "New York, NY",
            "temperature": 72,
            "unit": "fahrenheit",
            "conditions": "partly cloudy",
            "humidity": 65
        },
        "search_database": {
            "count": 42,
            "results": [{"id": 1, "name": "Result 1"}]
        },
        "calculate": 150.75,
        "get_stock_price": {
            "ticker": "AAPL",
            "price": 185.50,
            "exchange": "NASDAQ",
            "currency": "USD"
        }
    }

    for func_name, result in test_results.items():
        print(f"\nFunction: {func_name}")
        print(f"Raw Result: {result}")
        print(f"User Format: {formatter.format_for_user(func_name, result)}")
        print(f"LLM Format:\n{formatter.format_for_llm(func_name, result)}")
        print(f"Log Format: {json.dumps(formatter.format_for_logging(func_name, {}, result, 0.025), indent=2)}")


# ===== Example 6: Multi-Provider Support =====

def example_6_multi_provider_support():
    """Support function calling across different LLM providers."""
    print("\nExample 6: Multi-Provider Support")
    print("=" * 50)

    class UniversalFunctionCaller:
        """Unified interface for function calling across providers."""

        def __init__(self):
            self.providers = {}

        def register_provider(self, name: str, adapter):
            """Register a provider adapter."""
            self.providers[name] = adapter

        def convert_schema(self, schema: Dict, target_format: str) -> Dict:
            """Convert function schema between formats."""
            if target_format == "openai":
                return self._to_openai_format(schema)
            elif target_format == "anthropic":
                return self._to_anthropic_format(schema)
            elif target_format == "universal":
                return schema
            else:
                raise ValueError(f"Unknown format: {target_format}")

        def _to_openai_format(self, schema: Dict) -> Dict:
            """Convert to OpenAI function format."""
            return {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema.get("input_schema", schema.get("parameters", {}))
            }

        def _to_anthropic_format(self, schema: Dict) -> Dict:
            """Convert to Anthropic tool format."""
            return {
                "name": schema["name"],
                "description": schema["description"],
                "input_schema": schema.get("parameters", schema.get("input_schema", {}))
            }

        def call_function(self, provider: str, user_input: str, functions: List[Dict]) -> Dict:
            """Call function using specified provider."""
            if provider not in self.providers:
                raise ValueError(f"Unknown provider: {provider}")

            adapter = self.providers[provider]
            return adapter.call(user_input, functions)

    # Provider adapters
    class OpenAIAdapter:
        """Adapter for OpenAI function calling."""

        def call(self, user_input: str, functions: List[Dict]) -> Dict:
            """Simulate OpenAI function call."""
            return {
                "provider": "openai",
                "response": f"OpenAI would process: {user_input}",
                "function_call": {
                    "name": functions[0]["name"] if functions else None,
                    "arguments": "{}"
                }
            }

    class AnthropicAdapter:
        """Adapter for Anthropic function calling."""

        def call(self, user_input: str, tools: List[Dict]) -> Dict:
            """Simulate Anthropic tool use."""
            return {
                "provider": "anthropic",
                "response": f"Claude would process: {user_input}",
                "tool_use": {
                    "name": tools[0]["name"] if tools else None,
                    "input": {}
                }
            }

    # Create universal caller
    universal_caller = UniversalFunctionCaller()
    universal_caller.register_provider("openai", OpenAIAdapter())
    universal_caller.register_provider("anthropic", AnthropicAdapter())

    # Universal schema
    universal_function = {
        "name": "universal_search",
        "description": "Search across multiple data sources",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "sources": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["query"]
        }
    }

    # Test conversion
    print("Universal Schema:")
    print(json.dumps(universal_function, indent=2))

    print("\nOpenAI Format:")
    openai_format = universal_caller.convert_schema(universal_function, "openai")
    print(json.dumps(openai_format, indent=2))

    print("\nAnthropic Format:")
    anthropic_format = universal_caller.convert_schema(universal_function, "anthropic")
    print(json.dumps(anthropic_format, indent=2))

    # Test function calling
    user_input = "Search for customer data"

    print(f"\nUser Input: {user_input}")
    for provider in ["openai", "anthropic"]:
        result = universal_caller.call_function(provider, user_input, [universal_function])
        print(f"\n{provider} Result:")
        print(json.dumps(result, indent=2))


# ===== Example 7: Function Discovery =====

def example_7_function_discovery():
    """Automatic function discovery and registration."""
    print("\nExample 7: Function Discovery")
    print("=" * 50)

    import inspect
    from typing import get_type_hints

    class FunctionDiscovery:
        """Automatically discover and register functions."""

        def __init__(self):
            self.discovered_functions = {}

        def discover_module_functions(self, module):
            """Discover all decorated functions in a module."""
            discovered = []

            for name, obj in inspect.getmembers(module):
                if hasattr(obj, '__llm_callable__'):
                    func_info = self._extract_function_info(obj)
                    self.discovered_functions[name] = func_info
                    discovered.append(name)

            return discovered

        def _extract_function_info(self, func) -> Dict:
            """Extract function information from decorated function."""
            # Get function signature
            sig = inspect.signature(func)
            params = {}

            # Get type hints
            type_hints = get_type_hints(func)

            for param_name, param in sig.parameters.items():
                param_type = type_hints.get(param_name, Any)
                param_info = {
                    "type": self._python_type_to_json_type(param_type),
                    "required": param.default == inspect.Parameter.empty
                }

                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default

                params[param_name] = param_info

            return {
                "name": func.__name__,
                "description": func.__doc__ or "No description",
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": [p for p, info in params.items() if info["required"]]
                },
                "function": func
            }

        def _python_type_to_json_type(self, python_type) -> str:
            """Convert Python type to JSON schema type."""
            type_mapping = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object"
            }
            return type_mapping.get(python_type, "string")

        def get_functions_for_context(self, context: str) -> List[Dict]:
            """Get relevant functions based on context."""
            # Simple keyword matching (could be ML-based)
            relevant = []
            keywords = context.lower().split()

            for name, info in self.discovered_functions.items():
                description = info["description"].lower()
                if any(keyword in description for keyword in keywords):
                    relevant.append(info)

            return relevant

    # Decorator for marking functions as LLM-callable
    def llm_callable(func):
        """Mark a function as callable by LLM."""
        func.__llm_callable__ = True
        return func

    # Example module with decorated functions
    class DataProcessingModule:
        """Module with data processing functions."""

        @staticmethod
        @llm_callable
        def clean_text(text: str, remove_punctuation: bool = False) -> str:
            """Clean and normalize text data."""
            result = text.strip().lower()
            if remove_punctuation:
                import string
                result = result.translate(str.maketrans("", "", string.punctuation))
            return result

        @staticmethod
        @llm_callable
        def extract_numbers(text: str) -> list:
            """Extract all numbers from text."""
            import re
            numbers = re.findall(r"\d+\.?\d*", text)
            return [float(n) if "." in n else int(n) for n in numbers]

        @staticmethod
        @llm_callable
        def format_currency(amount: float, currency: str = "USD") -> str:
            """Format number as currency."""
            symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
            symbol = symbols.get(currency, currency)
            return f"{symbol}{amount:,.2f}"

    # Discover functions
    discovery = FunctionDiscovery()
    discovered = discovery.discover_module_functions(DataProcessingModule)

    print(f"Discovered functions: {discovered}")
    print("\nFunction definitions:")
    for name, info in discovery.discovered_functions.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Parameters: {json.dumps(info['parameters'], indent=4)}")

    # Get relevant functions for context
    contexts = [
        "I need to process some text data",
        "Extract numerical values",
        "Format monetary amounts"
    ]

    for context in contexts:
        print(f"\nContext: {context}")
        relevant = discovery.get_functions_for_context(context)
        print(f"Relevant functions: {[f['name'] for f in relevant]}")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 09: Function Calling Examples")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    examples = {
        1: example_1_simple_functions,
        2: example_2_function_call_flow,
        3: example_3_parameter_validation,
        4: example_4_error_handling,
        5: example_5_response_formatting,
        6: example_6_multi_provider_support,
        7: example_7_function_discovery
    }

    if args.all:
        for example in examples.values():
            example()
            print("\n" + "=" * 70 + "\n")
    elif args.example and args.example in examples:
        examples[args.example]()
    else:
        print("Module 09: Function Calling - Examples")
        print("\nUsage:")
        print("  python basic_functions.py --example N  # Run example N")
        print("  python basic_functions.py --all         # Run all examples")
        print("\nAvailable examples:")
        print("  1: Simple Function Definitions")
        print("  2: Function Call Flow with LLM")
        print("  3: Parameter Validation")
        print("  4: Error Handling")
        print("  5: Response Formatting")
        print("  6: Multi-Provider Support")
        print("  7: Function Discovery")