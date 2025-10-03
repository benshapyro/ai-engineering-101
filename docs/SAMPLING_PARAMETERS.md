# Sampling Parameters Reference Guide

This guide provides standardized conventions for LLM sampling parameters used throughout the curriculum.

## Core Parameters

### temperature

**Type:** `float`
**Range:** 0.0 to 2.0
**Default:** 0.7

Controls randomness in model outputs. Lower values make outputs more focused and deterministic, higher values increase creativity and diversity.

**Guidelines:**
- **0.0 - 0.3**: Deterministic, factual tasks
  - Code generation
  - Data extraction
  - Structured outputs
  - Testing/reproducibility
  - Mathematical calculations

- **0.4 - 0.7**: Balanced (recommended default)
  - General Q&A
  - Content summarization
  - Translation
  - Most production use cases

- **0.8 - 1.2**: Creative tasks
  - Content writing
  - Idea generation
  - Story creation
  - Marketing copy

- **1.3 - 2.0**: Maximum creativity
  - Experimental writing
  - Brainstorming
  - Poetry
  - Artistic outputs

**Example:**
```python
from llm.client import LLMClient

client = LLMClient()

# For code generation (deterministic)
code = client.generate(
    "Write a Python function to sort a list",
    temperature=0.0
)

# For creative writing (diverse)
story = client.generate(
    "Write a short story about a robot",
    temperature=1.2
)
```

### max_tokens

**Type:** `int`
**Default:** 1000

Maximum number of tokens in the model's response. Does not include input tokens.

**Guidelines:**
- **50 - 100**: Short answers, classifications, single-word responses
- **100 - 500**: Paragraph responses, summaries, explanations
- **500 - 2000**: Long-form content, detailed analysis, documentation
- **2000 - 4000**: Articles, comprehensive reports, code modules
- **4000+**: Books, extensive documentation (check model limits)

**Cost Impact:**
Higher max_tokens = higher maximum cost per request. Set conservatively based on actual needs.

**Example:**
```python
# Short classification
label = client.generate(
    "Classify sentiment: 'Great product!'",
    max_tokens=10
)

# Detailed explanation
explanation = client.generate(
    "Explain quantum computing",
    max_tokens=1000
)
```

### top_p (Nucleus Sampling)

**Type:** `float`
**Range:** 0.0 to 1.0
**Default:** 1.0

Controls diversity via nucleus sampling. Model considers tokens whose cumulative probability reaches `top_p`.

**Important:** Use temperature **OR** top_p, not both. Adjusting both simultaneously can lead to unpredictable behavior.

**Guidelines:**
- **1.0**: No filtering (default, recommended with temperature)
- **0.9 - 0.95**: Slight filtering, removes very low-probability tokens
- **0.5 - 0.8**: Moderate filtering, more focused outputs
- **0.1 - 0.4**: Strong filtering, highly focused (similar to low temperature)

**When to Use:**
- Prefer `temperature` for most use cases (more intuitive)
- Use `top_p` when you need fine-grained control over token filtering
- Use `top_p < 1.0` with `temperature=1.0` for focused but creative outputs

**Example:**
```python
# Using temperature (recommended)
response = client.generate(
    "Write a summary",
    temperature=0.7,
    top_p=1.0  # Default
)

# Using top_p (alternative approach)
response = client.generate(
    "Write a summary",
    temperature=1.0,  # Default
    top_p=0.9  # Filter low-probability tokens
)
```

## Other Parameters

### seed (OpenAI only)

**Type:** `int` or `None`
**Default:** `None`

Random seed for reproducible outputs. Only available on some OpenAI models.

**Example:**
```python
# Reproducible outputs
response1 = client.generate("Test", temperature=0.0, seed=42)
response2 = client.generate("Test", temperature=0.0, seed=42)
# response1 == response2 (likely, but not guaranteed)
```

### presence_penalty

**Type:** `float`
**Range:** -2.0 to 2.0
**Default:** 0.0

Penalizes tokens based on whether they appear in the text so far. Positive values encourage the model to talk about new topics.

### frequency_penalty

**Type:** `float`
**Range:** -2.0 to 2.0
**Default:** 0.0

Penalizes tokens based on their frequency in the text so far. Positive values reduce repetition.

## Best Practices

### 1. Default to Sensible Defaults

```python
# Good: Use library defaults
response = client.generate("Your prompt")

# Also good: Explicit sensible defaults
response = client.generate(
    "Your prompt",
    temperature=0.7,
    max_tokens=1000
)
```

### 2. Adjust Temperature Based on Task

```python
# Factual/deterministic tasks
facts = client.generate(
    "Extract dates from this text",
    temperature=0.0
)

# Creative tasks
story = client.generate(
    "Write a creative story",
    temperature=1.2
)
```

### 3. Set max_tokens Appropriately

```python
# Don't waste tokens
label = client.generate(
    "Classify: positive or negative?",
    max_tokens=5  # Single word response
)

# Allow enough for complete response
analysis = client.generate(
    "Provide detailed analysis",
    max_tokens=2000  # Comprehensive response
)
```

### 4. Use Deterministic Settings for Testing

```python
from shared.repro import get_deterministic_params

# Reproducible testing
params = get_deterministic_params()
response = client.generate("Test input", **params)
```

### 5. Don't Mix temperature and top_p

```python
# Bad: Adjusting both
response = client.generate(
    "Prompt",
    temperature=0.3,
    top_p=0.5  # Unpredictable interaction
)

# Good: Adjust one or the other
response = client.generate(
    "Prompt",
    temperature=0.3,
    top_p=1.0  # Default
)
```

## Parameter Selection Matrix

| Task Type | temperature | max_tokens | top_p |
|-----------|-------------|------------|-------|
| Code generation | 0.0 - 0.2 | 500 - 2000 | 1.0 |
| Data extraction | 0.0 - 0.3 | 100 - 500 | 1.0 |
| Classification | 0.0 - 0.3 | 5 - 50 | 1.0 |
| Summarization | 0.3 - 0.7 | 200 - 1000 | 1.0 |
| Q&A (factual) | 0.3 - 0.7 | 100 - 500 | 1.0 |
| Translation | 0.3 - 0.7 | Similar to input | 1.0 |
| Content writing | 0.7 - 1.0 | 500 - 2000 | 1.0 |
| Creative writing | 1.0 - 1.5 | 1000 - 4000 | 1.0 |
| Brainstorming | 1.2 - 1.8 | 500 - 2000 | 1.0 |
| Poetry/Art | 1.5 - 2.0 | 200 - 1000 | 1.0 |

## Provider-Specific Notes

### OpenAI (GPT-5, GPT-4, etc.)

- Supports `seed` parameter for reproducibility
- Temperature range: 0.0 - 2.0
- Recommended default: 0.7
- `top_p` and `temperature` can be used together but not recommended

### Anthropic (Claude)

- No `seed` parameter
- Temperature range: 0.0 - 1.0 (different from OpenAI!)
- Recommended default: 1.0
- Use `top_p` sparingly

**Important:** When using Claude, scale temperature values:
```python
# For Claude, use lower temperature values
if provider == "anthropic":
    temperature = min(temperature, 1.0)
```

## Common Mistakes

### ❌ Setting max_tokens Too High

```python
# Bad: Wastes money on simple tasks
label = client.generate(
    "Yes or no?",
    max_tokens=4000  # Way too high
)
```

### ❌ Using High Temperature for Factual Tasks

```python
# Bad: Factual task with creative temperature
facts = client.generate(
    "Extract all dates from this document",
    temperature=1.5  # Will produce inconsistent results
)
```

### ❌ Mixing Sampling Methods

```python
# Bad: Confusing interaction between parameters
response = client.generate(
    "Write something",
    temperature=0.2,  # Want focused
    top_p=0.3  # Also want focused (redundant/conflicting)
)
```

## References

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference/chat/create)
- [Anthropic Claude Documentation](https://docs.anthropic.com/claude/reference/messages_post)
- Module 01: Fundamentals - Temperature experiments
- Module 12: Prompt Optimization - Parameter tuning

---

**See Also:**
- [shared/utils.py](../shared/utils.py) - LLMClient implementation
- [shared/repro.py](../shared/repro.py) - Deterministic utilities
- [llm/client.py](../llm/client.py) - Modern Responses API client
