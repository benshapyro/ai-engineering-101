# OpenAI Structured Outputs & Responses API Summary (2025)

## Overview

OpenAI's Structured Outputs feature ensures that model responses strictly adhere to developer-supplied JSON schemas. This feature is available for models starting with `gpt-4o-mini`, `gpt-4o-mini-2024-07-18`, and `gpt-4o-2024-08-06`.

## Key Features

### 1. JSON Schema Support
- Guarantees responses match your defined JSON schema
- Supports complex nested objects and arrays
- Available for both Chat Completions API and Assistants API
- Works with tool calls and direct response content

### 2. Response Format Configuration

#### Python Example
```python
from pydantic import BaseModel
from typing import List
from openai import OpenAI

class Step(BaseModel):
    explanation: str
    output: str

class MathResponse(BaseModel):
    steps: List[Step]
    final_answer: str

client = OpenAI()
completion = client.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "solve 8x + 31 = 2"},
    ],
    response_format=MathResponse,
)
```

#### Direct JSON Schema Example
```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "math_reasoning",
        "schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "explanation": {"type": "string"},
                            "output": {"type": "string"}
                        },
                        "required": ["explanation", "output"],
                        "additionalProperties": false
                    }
                },
                "final_answer": {"type": "string"}
            },
            "required": ["steps", "final_answer"],
            "additionalProperties": false
        },
        "strict": true
    }
}
```

## API Methods

### Chat Completions API

#### Standard Method: `create()`
```python
completion = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format=response_format
)
```

#### Enhanced Method: `parse()`
- Wrapper over `create()` with richer type integration
- Returns `ParsedChatCompletion` object
- Auto-converts Pydantic models to JSON schemas
- Parses responses back into Python objects

```python
completion = client.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format=MathResponse,  # Pydantic model
)

# Access parsed content
if completion.choices[0].message.parsed:
    result = completion.choices[0].message.parsed
    print(result.final_answer)
```

## Function Calling with Structured Outputs

### Tool Definition with Strict Mode
```python
from pydantic import BaseModel
import openai

class Query(BaseModel):
    table_name: str
    columns: List[str]
    conditions: List[dict]
    order_by: str

# Using pydantic_function_tool helper
tools = [
    openai.pydantic_function_tool(Query),
]

completion = client.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[...],
    tools=tools,
)

# Access parsed function arguments
tool_call = completion.choices[0].message.tool_calls[0]
parsed_args = tool_call.function.parsed_arguments  # Instance of Query
```

## Streaming Support

### With Chat Completions
```python
# Stream responses while accumulating for final parsing
from openai.lib.streaming import ChatCompletionAccumulator

accumulator = ChatCompletionAccumulator()

for chunk in client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format=response_format,
    stream=True
):
    accumulator.add_chunk(chunk)
    # Process chunks as they arrive

# Convert to structured completion
structured = accumulator.chatCompletion(YourClass)
```

## Schema Restrictions

### Supported JSON Schema Features
- Basic types: `string`, `number`, `boolean`, `integer`
- Arrays with `items` definition
- Objects with `properties`
- `required` fields (all properties must be marked required)
- `enum` for restricted values
- Nested objects and arrays

### Key Limitations
- All properties must be required
- `additionalProperties` must be `false`
- Maximum nesting depth restrictions apply
- No `$ref` references
- No `anyOf`, `oneOf`, `allOf`

## Error Handling

### Parse Method Specific
```python
try:
    completion = client.chat.completions.parse(...)
except LengthFinishReasonError:
    # Response was truncated
    pass
except ContentFilterFinishReasonError:
    # Content was filtered
    pass
```

### Refusal Handling
```python
message = completion.choices[0].message
if message.parsed:
    # Use the parsed response
    result = message.parsed
else:
    # Handle refusal
    print(message.refusal)
```

## Best Practices

### 1. Use Pydantic Models
- Automatic schema generation
- Type safety
- Built-in validation
- IDE support

### 2. Schema Validation
- Test schemas locally before API calls
- Use `strict: true` for guaranteed compliance
- Keep schemas as simple as needed

### 3. Optional Fields
```python
from typing import Optional

class Response(BaseModel):
    required_field: str
    optional_field: Optional[str] = None
```

### 4. Complex Structures
```python
class Author(BaseModel):
    name: str
    birth_year: int

class Book(BaseModel):
    title: str
    author: Author
    publication_year: int

class BookList(BaseModel):
    books: List[Book]
```

## Language-Specific Implementations

### Python
- Use `client.chat.completions.parse()` method
- Pydantic model support
- Automatic parsing with `message.parsed`

### Java
- `StructuredChatCompletionCreateParams` for type safety
- Automatic JSON schema derivation from classes
- Jackson/Swagger annotation support

### .NET
- `ChatResponseFormat.CreateJsonSchemaFormat()` method
- Strong typing with C# classes
- Built-in JSON parsing

### Node.js/TypeScript
- TypeScript interface support
- Zod schema integration
- Runtime type checking

## Common Use Cases

### 1. Data Extraction
```python
class ExtractedData(BaseModel):
    entities: List[str]
    sentiment: str
    key_points: List[str]
```

### 2. Classification
```python
class Classification(BaseModel):
    category: str
    confidence: float
    reasoning: str
```

### 3. Structured Analysis
```python
class Analysis(BaseModel):
    summary: str
    findings: List[Finding]
    recommendations: List[str]
    risk_level: str
```

## Migration from Legacy Formats

### From JSON Mode
```python
# Old way
response_format = {"type": "json_object"}

# New way with structured outputs
response_format = YourPydanticModel
```

### From Manual Parsing
```python
# Old way
response = completion.choices[0].message.content
data = json.loads(response)

# New way
data = completion.choices[0].message.parsed
```

## Performance Considerations

- Structured outputs have minimal latency overhead
- Schema complexity affects token usage
- Simpler schemas generally perform better
- Use streaming for real-time applications

## Version Compatibility

- Minimum model versions:
  - `gpt-4o-mini-2024-07-18` or later
  - `gpt-4o-2024-08-06` or later
- SDK version requirements vary by language
- Check SDK documentation for latest features