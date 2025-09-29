# Module 09: Function Calling

## Learning Objectives
By the end of this module, you will:
- Understand function calling capabilities in modern LLMs
- Design and implement tool interfaces for AI systems
- Handle complex multi-step function orchestration
- Build reliable error handling for function execution
- Create extensible tool ecosystems for AI agents

## Key Concepts

### 1. What is Function Calling?
Function calling enables LLMs to interact with external systems, APIs, and tools by generating structured function calls with appropriate parameters. This transforms LLMs from text processors into action-taking agents.

### 2. Function Definition Structure

#### OpenAI Format
```python
functions = [{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and state, e.g., San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        "required": ["location"]
    }
}]
```

#### Anthropic/Claude Format
```python
tools = [{
    "name": "calculate_discount",
    "description": "Calculate discounted price",
    "input_schema": {
        "type": "object",
        "properties": {
            "original_price": {"type": "number"},
            "discount_percentage": {"type": "number"}
        },
        "required": ["original_price", "discount_percentage"]
    }
}]
```

### 3. Function Execution Flow

```python
def function_calling_flow(user_input):
    # 1. Send input with function definitions
    response = llm.complete(
        messages=[{"role": "user", "content": user_input}],
        functions=function_definitions
    )

    # 2. Check if function call requested
    if response.function_call:
        # 3. Execute function
        result = execute_function(
            response.function_call.name,
            response.function_call.arguments
        )

        # 4. Send result back to model
        final_response = llm.complete(
            messages=[
                {"role": "function", "name": response.function_call.name,
                 "content": json.dumps(result)}
            ]
        )

    return final_response
```

### 4. Common Challenges
- **Parameter Validation**: Ensuring correct types and formats
- **Error Handling**: Gracefully handling function failures
- **Security**: Preventing malicious function calls
- **Chaining**: Managing multi-step function sequences
- **State Management**: Maintaining context across calls

## Module Structure

### Examples
1. `basic_functions.py` - Simple function definitions and calls
2. `function_orchestration.py` - Complex multi-function workflows
3. `tool_ecosystem.py` - Building extensible tool systems

### Exercises
Practice problems focusing on:
- Designing effective function interfaces
- Parameter validation and error handling
- Multi-step function orchestration
- Security and sandboxing
- Performance optimization

### Project: Tool Registry System
Build a system that:
- Manages a registry of available tools
- Handles dynamic tool loading
- Validates and executes function calls
- Provides security sandboxing
- Tracks usage and performance metrics

## Best Practices

### 1. Function Design
```python
class WellDesignedFunction:
    """Example of a well-designed function for LLM use."""

    @staticmethod
    def definition():
        return {
            "name": "search_database",
            "description": "Search customer database with filters",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "filters": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["active", "inactive"]},
                            "created_after": {"type": "string", "format": "date"}
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

    @staticmethod
    def execute(query, filters=None, limit=10):
        """Actual function implementation."""
        # Validate inputs
        if not query:
            raise ValueError("Query cannot be empty")

        # Execute search
        results = database.search(query, filters, limit)

        # Return structured result
        return {
            "status": "success",
            "count": len(results),
            "results": results
        }
```

### 2. Error Handling
```python
def safe_function_execution(func_name, args):
    try:
        # Validate function exists
        if func_name not in registered_functions:
            return {"error": f"Function {func_name} not found"}

        # Validate arguments
        validated_args = validate_arguments(func_name, args)

        # Execute with timeout
        result = execute_with_timeout(
            registered_functions[func_name],
            validated_args,
            timeout=30
        )

        return {"status": "success", "result": result}

    except ValidationError as e:
        return {"error": f"Invalid arguments: {e}"}
    except TimeoutError:
        return {"error": "Function execution timed out"}
    except Exception as e:
        logger.error(f"Function {func_name} failed: {e}")
        return {"error": "Function execution failed"}
```

### 3. Security Considerations
```python
class SecureFunctionRegistry:
    def __init__(self):
        self.allowed_functions = {}
        self.permission_levels = {}
        self.rate_limits = {}

    def register_function(self, func, permission_level="user"):
        """Register function with security constraints."""
        func_name = func.__name__
        self.allowed_functions[func_name] = func
        self.permission_levels[func_name] = permission_level
        self.rate_limits[func_name] = RateLimiter(calls=10, period=60)

    def execute(self, func_name, args, user_context):
        """Execute with security checks."""
        # Check permissions
        if not self.has_permission(func_name, user_context):
            raise PermissionError(f"Insufficient permissions for {func_name}")

        # Check rate limits
        if not self.rate_limits[func_name].allow():
            raise RateLimitError(f"Rate limit exceeded for {func_name}")

        # Sandbox execution
        return self.sandboxed_execute(func_name, args)
```

## Production Considerations

### Function Registry Management
```python
class FunctionRegistry:
    def __init__(self):
        self.functions = {}
        self.metadata = {}
        self.usage_stats = defaultdict(int)

    def discover_functions(self, module):
        """Auto-discover functions from module."""
        for name, obj in inspect.getmembers(module):
            if hasattr(obj, 'llm_callable'):
                self.register(obj)

    def get_definitions_for_context(self, context):
        """Get relevant function definitions for context."""
        relevant = []
        for func_name, metadata in self.metadata.items():
            if self.is_relevant(metadata, context):
                relevant.append(metadata['definition'])
        return relevant
```

### Performance Optimization
- **Lazy Loading**: Load functions only when needed
- **Caching**: Cache function results when appropriate
- **Batching**: Execute multiple functions in parallel
- **Async Execution**: Use async functions for I/O operations

### Monitoring
```python
class FunctionCallMonitor:
    def track_call(self, function_name, args, result, duration):
        self.metrics.add({
            "function": function_name,
            "success": "error" not in result,
            "duration": duration,
            "timestamp": datetime.now()
        })

    def get_analytics(self):
        return {
            "most_used": self.get_top_functions(),
            "error_rate": self.calculate_error_rate(),
            "avg_duration": self.calculate_avg_duration(),
            "usage_pattern": self.analyze_usage_pattern()
        }
```

## Advanced Techniques

### 1. Function Chaining
```python
def execute_function_chain(chain_definition, initial_input):
    """Execute a chain of functions."""
    result = initial_input

    for step in chain_definition:
        function_name = step['function']
        # Map previous result to function parameters
        args = map_result_to_args(result, step['mapping'])
        result = execute_function(function_name, args)

        if step.get('condition'):
            if not evaluate_condition(result, step['condition']):
                break

    return result
```

### 2. Dynamic Function Generation
```python
def generate_function_from_api(api_endpoint):
    """Generate function definition from API spec."""
    spec = fetch_api_spec(api_endpoint)

    return {
        "name": spec['operation_id'],
        "description": spec['description'],
        "parameters": convert_openapi_to_function_params(spec['parameters']),
        "executor": create_api_executor(api_endpoint)
    }
```

### 3. Parallel Function Execution
```python
async def execute_parallel_functions(function_calls):
    """Execute multiple functions in parallel."""
    tasks = []

    for call in function_calls:
        task = asyncio.create_task(
            execute_async_function(call.name, call.args)
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    return process_parallel_results(results)
```

## Common Patterns

### 1. Tool Selection
```python
def select_best_tool(task_description, available_tools):
    """Let LLM select appropriate tool."""
    prompt = f"""Task: {task_description}

Available tools:
{format_tool_descriptions(available_tools)}

Which tool is best suited for this task?"""

    return llm.select_tool(prompt)
```

### 2. Progressive Disclosure
```python
# Start with basic tools, add advanced as needed
def get_tools_for_user(user_level):
    if user_level == "beginner":
        return basic_tools
    elif user_level == "intermediate":
        return basic_tools + intermediate_tools
    else:
        return all_tools
```

### 3. Function Composition
```python
def compose_functions(functions):
    """Create new function from composition."""
    def composed(*args, **kwargs):
        result = functions[0](*args, **kwargs)
        for func in functions[1:]:
            result = func(result)
        return result

    return composed
```

## Real-World Examples

### 1. Data Pipeline Tools
```python
data_tools = [
    {"name": "fetch_data", "description": "Fetch from database"},
    {"name": "clean_data", "description": "Clean and validate"},
    {"name": "transform_data", "description": "Apply transformations"},
    {"name": "export_data", "description": "Export to format"}
]
```

### 2. DevOps Automation
```python
devops_tools = [
    {"name": "deploy_service", "description": "Deploy to kubernetes"},
    {"name": "check_health", "description": "Verify service health"},
    {"name": "rollback", "description": "Rollback deployment"},
    {"name": "scale_service", "description": "Adjust replicas"}
]
```

### 3. Business Intelligence
```python
bi_tools = [
    {"name": "run_query", "description": "Execute SQL query"},
    {"name": "generate_chart", "description": "Create visualization"},
    {"name": "calculate_metrics", "description": "Compute KPIs"},
    {"name": "send_report", "description": "Email results"}
]
```

## Exercises Overview

1. **Tool Designer**: Design comprehensive tool interfaces
2. **Error Handler**: Build robust error handling system
3. **Security Sandbox**: Implement secure function execution
4. **Chain Builder**: Create complex function chains
5. **Performance Optimizer**: Optimize function call efficiency

## Success Metrics
- **Execution Success Rate**: >95% successful function calls
- **Parameter Accuracy**: >99% correct parameter types
- **Security**: Zero unauthorized function executions
- **Performance**: <50ms overhead per function call
- **Extensibility**: Support 50+ tools without degradation

## Next Steps
After mastering function calling, you'll move to Module 10: RAG Basics, where you'll learn to augment LLMs with external knowledge through retrieval systems - combining function calling with document retrieval for powerful knowledge-enhanced applications.