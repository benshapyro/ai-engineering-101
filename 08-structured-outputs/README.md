# Module 08: Structured Outputs

## Learning Objectives
By the end of this module, you will:
- Master JSON mode and structured data generation
- Implement schema validation for LLM outputs
- Handle complex nested data structures reliably
- Build parsers for various output formats
- Create systems with guaranteed output structure

## Key Concepts

### 1. Why Structured Outputs Matter
Structured outputs enable reliable integration of LLMs into production systems by ensuring responses conform to expected formats, enabling direct consumption by downstream applications without complex parsing.

### 2. Output Format Types

#### JSON Outputs
```python
# Direct JSON generation
prompt = """Return the analysis as JSON:
{
  "sentiment": "positive|negative|neutral",
  "confidence": 0.0-1.0,
  "key_phrases": ["phrase1", "phrase2"],
  "entities": {
    "people": [],
    "organizations": [],
    "locations": []
  }
}"""
```

#### Schema-Driven Generation
```python
from pydantic import BaseModel
from typing import List, Optional

class AnalysisOutput(BaseModel):
    sentiment: str
    confidence: float
    key_phrases: List[str]
    entities: dict
    metadata: Optional[dict] = None
```

#### XML/YAML/Custom Formats
```python
# XML output
prompt = """Return as XML:
<analysis>
  <sentiment>positive</sentiment>
  <confidence>0.95</confidence>
</analysis>"""
```

### 3. Enforcement Strategies

#### Prompt Engineering
```python
def enforce_json_output(prompt, schema):
    return f"""{prompt}

You MUST return valid JSON matching this exact structure:
{json.dumps(schema, indent=2)}

Rules:
- Use double quotes for strings
- No trailing commas
- All required fields must be present
- Types must match exactly"""
```

#### Validation & Retry
```python
def get_structured_output(prompt, validator, max_retries=3):
    for attempt in range(max_retries):
        response = llm.complete(prompt)
        try:
            parsed = json.loads(response)
            validated = validator(parsed)
            return validated
        except (json.JSONDecodeError, ValidationError) as e:
            prompt = f"{prompt}\n\nError: {e}\nPlease fix and return valid JSON:"
    raise ValueError("Failed to get valid structured output")
```

#### Guided Generation
```python
# Using guidance libraries for guaranteed structure
def guided_generation(template, model):
    # Template with constraints
    template = """
    {
        "name": "{{gen 'name' pattern='[A-Za-z ]+' max_tokens=20}}",
        "age": {{gen 'age' pattern='[0-9]{1,3}'}},
        "valid": {{select 'valid' options=[true, false]}}
    }
    """
    return model.generate(template)
```

### 4. Common Challenges
- **Format Violations**: Model deviates from specified format
- **Type Mismatches**: Wrong data types in outputs
- **Missing Fields**: Required fields omitted
- **Extra Fields**: Unexpected fields added
- **Escaping Issues**: Special characters breaking format

## Module Structure

### Examples
1. `json_generation.py` - Reliable JSON output generation
2. `schema_validation.py` - Schema enforcement with Pydantic
3. `format_parsers.py` - Parsers for various output formats

### Exercises
Practice problems focusing on:
- Complex nested structure generation
- Schema design and validation
- Error recovery for malformed outputs
- Format conversion pipelines
- Performance optimization for structured outputs

### Project: Schema Registry System
Build a system that:
- Manages reusable output schemas
- Validates LLM outputs against schemas
- Handles version control for schemas
- Provides automatic retry and correction
- Generates documentation from schemas

## Best Practices

### 1. Schema Design
```python
class WellDesignedSchema(BaseModel):
    # Use clear, specific field names
    transaction_id: str
    amount_usd: float

    # Provide defaults for optional fields
    description: Optional[str] = None

    # Use enums for constrained values
    status: Literal["pending", "completed", "failed"]

    # Add validation
    @validator('amount_usd')
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError('Amount must be positive')
        return round(v, 2)
```

### 2. Progressive Enhancement
```python
def progressive_structure(data, complexity_level):
    """Start simple, add complexity gradually."""
    if complexity_level == 1:
        # Simple flat structure
        return {"summary": data}
    elif complexity_level == 2:
        # Add categories
        return {
            "summary": data,
            "categories": ["cat1", "cat2"]
        }
    else:
        # Full complex structure
        return {
            "summary": data,
            "categories": [],
            "metadata": {},
            "relationships": []
        }
```

### 3. Fallback Strategies
```python
def structured_with_fallback(prompt, schema):
    try:
        # Try structured generation
        return get_structured_json(prompt, schema)
    except StructuredOutputError:
        # Fall back to text + parsing
        text = get_text_response(prompt)
        return parse_to_structure(text, schema)
```

## Production Considerations

### Reliability Patterns
```python
class ReliableStructuredOutput:
    def __init__(self, schema, model):
        self.schema = schema
        self.model = model
        self.success_rate = 0.95

    def generate(self, prompt, confidence_threshold=0.9):
        # Multiple generation strategies
        strategies = [
            self.json_mode_generation,
            self.template_generation,
            self.guided_generation
        ]

        for strategy in strategies:
            result = strategy(prompt)
            if self.validate_output(result):
                return result

        raise StructuredOutputError("All strategies failed")
```

### Performance Optimization
- **Caching**: Cache validated schemas and templates
- **Batching**: Process multiple structured outputs together
- **Streaming**: Parse structured output incrementally
- **Prevalidation**: Validate likely outputs before full generation

### Monitoring
```python
class StructuredOutputMetrics:
    def __init__(self):
        self.attempts = 0
        self.successes = 0
        self.validation_errors = []
        self.retry_counts = []

    def track_generation(self, prompt, output, retries):
        self.attempts += 1
        if output:
            self.successes += 1
        self.retry_counts.append(retries)

    def get_metrics(self):
        return {
            "success_rate": self.successes / self.attempts,
            "avg_retries": sum(self.retry_counts) / len(self.retry_counts),
            "common_errors": self.get_common_errors()
        }
```

## Advanced Techniques

### 1. Hierarchical Schemas
```python
class NestedSchema(BaseModel):
    class PersonSchema(BaseModel):
        name: str
        role: str

    class ProjectSchema(BaseModel):
        title: str
        status: str
        team: List[PersonSchema]

    projects: List[ProjectSchema]
    summary: str
```

### 2. Dynamic Schema Generation
```python
def generate_schema_from_example(example_data):
    """Infer schema from example data."""
    schema = {}

    for key, value in example_data.items():
        if isinstance(value, str):
            schema[key] = "string"
        elif isinstance(value, int):
            schema[key] = "integer"
        elif isinstance(value, list):
            schema[key] = f"array of {type(value[0]).__name__}"

    return schema
```

### 3. Multi-Format Support
```python
class FormatConverter:
    @staticmethod
    def json_to_xml(json_data):
        """Convert JSON to XML."""
        pass

    @staticmethod
    def xml_to_yaml(xml_data):
        """Convert XML to YAML."""
        pass

    @staticmethod
    def ensure_format(data, target_format):
        """Ensure data is in target format."""
        pass
```

## Common Patterns

### 1. Extraction Pipeline
```python
# Text → Structured Data
pipeline = [
    extract_entities,
    classify_sentiment,
    identify_relationships,
    format_as_json
]
```

### 2. Validation Chain
```python
# Multi-level validation
validators = [
    validate_json_syntax,
    validate_schema_compliance,
    validate_business_rules,
    validate_consistency
]
```

### 3. Format Negotiation
```python
# Choose format based on use case
def select_output_format(use_case):
    formats = {
        "api": "json",
        "config": "yaml",
        "document": "xml",
        "database": "sql"
    }
    return formats.get(use_case, "json")
```

## Integration Examples

### 1. Database Integration
```python
def llm_to_database(prompt, table_schema):
    # Generate structured data
    data = get_structured_output(prompt, table_schema)

    # Validate against database schema
    validated = validate_for_db(data, table_schema)

    # Insert into database
    return db.insert(validated)
```

### 2. API Response Generation
```python
def generate_api_response(request):
    # Define response schema
    schema = APIResponseSchema()

    # Generate structured response
    response = llm.generate(request, schema=schema)

    # Return as API response
    return jsonify(response)
```

### 3. Configuration Generation
```python
def generate_config(requirements):
    # Generate YAML config
    config = llm.generate_yaml(requirements)

    # Validate configuration
    validated = validate_config(config)

    # Write to file
    save_config(validated)
```

## Exercises Overview

1. **Schema Designer**: Create schemas for complex business objects
2. **Format Converter**: Build converters between different formats
3. **Validation Pipeline**: Implement multi-stage validation
4. **Error Recovery**: Handle and fix malformed outputs
5. **Performance Challenge**: Optimize structured generation speed

## Success Metrics
- **Validation Rate**: >99% outputs pass schema validation
- **Retry Rate**: <5% require retry
- **Parse Errors**: <0.1% unparseable outputs
- **Performance**: <100ms overhead for structure enforcement
- **Flexibility**: Support 5+ output formats

## Next Steps
After mastering structured outputs, you'll move to Module 09: Function Calling, where you'll learn to enable LLMs to interact with external tools and APIs - building on structured outputs to ensure reliable function invocation and parameter passing.