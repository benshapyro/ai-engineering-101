# Module 05: Prompt Chaining

## Learning Objectives
By the end of this module, you will:
- Understand when and how to chain multiple prompts together
- Master sequential, conditional, and parallel chaining patterns
- Learn state management and context accumulation strategies
- Implement error handling and recovery in prompt chains
- Build production-ready workflow orchestration systems

## Key Concepts

### 1. What is Prompt Chaining?
Prompt chaining involves connecting multiple LLM calls in sequence, where outputs from one prompt become inputs to the next. This enables complex, multi-step reasoning and processing that would be difficult or impossible in a single prompt.

### 2. Core Benefits
- **Modularity**: Break complex tasks into manageable pieces
- **Reliability**: Each step can be validated independently
- **Flexibility**: Dynamic routing based on intermediate results
- **Cost Efficiency**: Only run necessary steps
- **Debugging**: Easier to identify where issues occur

### 3. Chaining Patterns

#### Sequential Chains
```python
# Step 1: Extract information
extracted_data = llm("Extract key facts from: {text}")

# Step 2: Analyze
analysis = llm(f"Analyze this data: {extracted_data}")

# Step 3: Generate report
report = llm(f"Create report from: {analysis}")
```

#### Conditional Chains
```python
# Initial classification
category = llm("Classify this request: {input}")

# Route based on category
if category == "technical":
    response = llm("Provide technical solution: {input}")
elif category == "business":
    response = llm("Provide business analysis: {input}")
else:
    response = llm("Provide general help: {input}")
```

#### Parallel Chains
```python
# Run multiple analyses concurrently
results = parallel_execute([
    lambda: llm("Sentiment analysis: {text}"),
    lambda: llm("Extract entities: {text}"),
    lambda: llm("Summarize: {text}")
])
combined = llm(f"Synthesize insights: {results}")
```

### 4. State Management

**Context Accumulation:**
```python
class ChainState:
    def __init__(self):
        self.context = []
        self.results = {}
        self.metadata = {}

    def add_step(self, name, result):
        self.context.append(result)
        self.results[name] = result
        self.metadata[name] = {"timestamp": now()}
```

**Memory Strategies:**
- **Full History**: Keep all intermediate results
- **Sliding Window**: Keep last N results
- **Selective Memory**: Keep only important results
- **Compressed Memory**: Summarize as you go

### 5. Error Handling

**Retry Logic:**
```python
def resilient_chain(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                return fallback_response()
            time.sleep(2 ** attempt)  # Exponential backoff
```

**Validation Checkpoints:**
```python
def validated_chain(steps):
    for step in steps:
        result = step.execute()
        if not step.validate(result):
            result = step.correct(result)
        yield result
```

### 6. Common Pitfalls
- **Context Bloat**: Accumulating too much context
- **Error Propagation**: Errors compounding through chain
- **Token Limits**: Exceeding context windows
- **Cost Explosion**: Unnecessary API calls
- **Latency Issues**: Sequential bottlenecks

## Module Structure

### Examples
1. `sequential_chains.py` - Linear workflow patterns
2. `conditional_flows.py` - Branching and routing logic
3. `parallel_processing.py` - Concurrent execution strategies

### Exercises
Practice problems focusing on:
- Building multi-step data pipelines
- Implementing decision trees
- Error recovery mechanisms
- Optimization for cost and speed

### Project: Workflow Orchestrator
Build a system that:
- Defines reusable workflow components
- Manages state across chain execution
- Handles errors and retries gracefully
- Provides monitoring and debugging
- Optimizes execution paths

## Best Practices

### 1. Chain Design
```python
# Good: Clear, focused steps
chain = [
    ("extract", "Extract data from document"),
    ("validate", "Validate extracted data"),
    ("transform", "Transform to target format"),
    ("store", "Store in database")
]

# Bad: Monolithic, hard to debug
chain = [("do_everything", "Extract, validate, transform, and store")]
```

### 2. State Management
```python
# Track intermediate results
state = {
    "original_input": input_data,
    "step_results": [],
    "errors": [],
    "metadata": {}
}

# Update after each step
for step in chain:
    result = execute(step, state)
    state["step_results"].append(result)
```

### 3. Error Recovery
```python
# Implement circuit breakers
class CircuitBreaker:
    def __init__(self, failure_threshold=5):
        self.failures = 0
        self.threshold = failure_threshold
        self.is_open = False

    def call(self, func):
        if self.is_open:
            raise CircuitOpen("Circuit breaker is open")
        try:
            result = func()
            self.failures = 0
            return result
        except Exception:
            self.failures += 1
            if self.failures >= self.threshold:
                self.is_open = True
            raise
```

## Production Considerations

### Performance Optimization
- **Caching**: Store and reuse intermediate results
- **Batching**: Process multiple items together
- **Parallelization**: Execute independent steps concurrently
- **Early Termination**: Stop chains when conditions are met

### Cost Management
```python
class CostAwareChain:
    def __init__(self, budget=10.0):
        self.budget = budget
        self.spent = 0.0

    def execute_step(self, step):
        estimated_cost = step.estimate_cost()
        if self.spent + estimated_cost > self.budget:
            raise BudgetExceeded()
        result = step.execute()
        self.spent += step.actual_cost()
        return result
```

### Monitoring and Debugging
```python
# Comprehensive logging
logger.info(f"Starting chain: {chain_id}")
logger.debug(f"Step {step_num}: Input={input[:100]}")
logger.info(f"Step {step_num}: Output={output[:100]}")
logger.error(f"Step {step_num} failed: {error}")

# Metrics collection
metrics = {
    "total_tokens": sum(step.tokens for step in chain),
    "total_time": sum(step.duration for step in chain),
    "success_rate": successful / total,
    "average_latency": total_time / step_count
}
```

## Common Patterns

### 1. Map-Reduce
```python
# Map: Process each item independently
mapped = [llm(f"Process: {item}") for item in items]

# Reduce: Combine results
reduced = llm(f"Combine these results: {mapped}")
```

### 2. Pipeline
```python
# Data flows through transformations
pipeline = [
    lambda x: llm(f"Clean: {x}"),
    lambda x: llm(f"Enrich: {x}"),
    lambda x: llm(f"Format: {x}")
]

result = input_data
for transform in pipeline:
    result = transform(result)
```

### 3. Router
```python
# Route to specialized handlers
def route_request(request):
    intent = llm(f"Classify intent: {request}")

    handlers = {
        "question": handle_question,
        "command": handle_command,
        "conversation": handle_chat
    }

    return handlers.get(intent, handle_default)(request)
```

### 4. Feedback Loop
```python
# Iterative refinement
def refine_until_good(initial):
    current = initial
    for iteration in range(max_iterations):
        feedback = llm(f"Evaluate: {current}")
        if "good" in feedback:
            return current
        current = llm(f"Improve based on feedback: {feedback}")
    return current
```

## Exercises Overview

1. **Sequential Pipeline**: Build a data processing pipeline
2. **Decision Tree**: Implement complex conditional logic
3. **Parallel Analysis**: Concurrent processing with aggregation
4. **Error Recovery**: Robust chains with fallbacks
5. **Optimization**: Balance speed, cost, and quality

## Success Metrics
- **Reliability**: 95%+ success rate for chains
- **Performance**: <2s latency for typical chains
- **Cost**: 30% reduction vs monolithic prompts
- **Maintainability**: Modular, testable components

## Next Steps
After mastering prompt chaining, you'll move to Module 06: Role-Based Prompting, where you'll learn to create specialized personas and expertise-driven interactions that can be combined with chaining for even more powerful workflows.