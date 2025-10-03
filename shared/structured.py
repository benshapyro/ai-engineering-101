"""
Structured output utilities for JSON schema validation and parsing.

This module provides helpers for working with structured outputs from LLMs,
including JSON schema creation, validation, and extraction.
"""

import json
import re
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, ValidationError


def create_json_schema(
    name: str,
    description: str,
    properties: Dict[str, Dict[str, Any]],
    required: Optional[List[str]] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Create a JSON schema for structured outputs.

    Args:
        name: Schema name
        description: Schema description
        properties: Property definitions (name -> {type, description, ...})
        required: List of required property names
        strict: Whether to use strict mode (recommended)

    Returns:
        JSON schema dictionary ready for OpenAI structured outputs

    Example:
        schema = create_json_schema(
            name="PersonInfo",
            description="Information about a person",
            properties={
                "name": {"type": "string", "description": "Full name"},
                "age": {"type": "integer", "description": "Age in years"},
                "email": {"type": "string", "description": "Email address"}
            },
            required=["name", "email"]
        )
    """
    schema = {
        "name": name,
        "description": description,
        "strict": strict,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": required or list(properties.keys()),
            "additionalProperties": False
        }
    }
    return schema


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from mixed text content.

    Handles common cases:
    - Plain JSON objects
    - JSON in markdown code blocks (```json ... ```)
    - JSON with surrounding text

    Args:
        text: Text containing JSON

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If no valid JSON found

    Example:
        text = "Here's the result: ```json\\n{\"key\": \"value\"}\\n```"
        data = extract_json_from_text(text)
        # Returns: {"key": "value"}
    """
    text = text.strip()

    # Try to extract from code blocks first
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)

    if matches:
        # Try each code block
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Try to find JSON object boundaries
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        # Try the largest match first (most likely to be complete)
        for match in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Last resort: try parsing the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("No valid JSON found in text")


def validate_with_pydantic(data: Dict[str, Any], model: Type[BaseModel]) -> BaseModel:
    """
    Validate JSON data against a Pydantic model.

    Args:
        data: JSON data to validate
        model: Pydantic model class

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If validation fails

    Example:
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        data = {"name": "Alice", "age": 30}
        person = validate_with_pydantic(data, Person)
    """
    try:
        return model(**data)
    except ValidationError as e:
        raise ValidationError(f"Validation failed: {e}")


def ask_json(
    client,
    input_text: str,
    schema: Dict[str, Any],
    instructions: str = None,
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    extract_fallback: bool = True
) -> Dict[str, Any]:
    """
    Request structured JSON output from LLM with validation.

    This is a high-level helper that combines structured output requests
    with automatic extraction and validation.

    Args:
        client: LLMClient instance
        input_text: User input/prompt
        schema: JSON schema (from create_json_schema)
        instructions: Optional system instructions
        model: Model name (uses client default if None)
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        extract_fallback: If True, try to extract JSON from text on failure

    Returns:
        Parsed and validated JSON dictionary

    Raises:
        ValueError: If JSON extraction/parsing fails

    Example:
        from llm.client import LLMClient

        client = LLMClient()
        schema = create_json_schema(
            name="Analysis",
            description="Sentiment analysis result",
            properties={
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            }
        )

        result = ask_json(
            client,
            "The product is amazing!",
            schema,
            instructions="Analyze the sentiment"
        )
        # Returns: {"sentiment": "positive", "confidence": 0.95}
    """
    try:
        # Try structured output first
        response = client.structured(
            input_text=input_text,
            schema=schema,
            instructions=instructions,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Get output text
        output = client.get_output_text(response)

        # Parse JSON
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            if extract_fallback:
                # Try to extract JSON from mixed content
                return extract_json_from_text(output)
            raise

    except Exception as e:
        raise ValueError(f"Failed to get structured JSON: {e}")


def pydantic_to_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Convert Pydantic model to OpenAI JSON schema format.

    Args:
        model: Pydantic model class

    Returns:
        JSON schema dictionary

    Example:
        from pydantic import BaseModel, Field

        class Product(BaseModel):
            name: str = Field(description="Product name")
            price: float = Field(description="Price in USD")
            in_stock: bool = Field(description="Availability")

        schema = pydantic_to_json_schema(Product)
    """
    # Get Pydantic's JSON schema
    pydantic_schema = model.model_json_schema()

    # Convert to OpenAI format
    return {
        "name": model.__name__,
        "description": model.__doc__ or f"{model.__name__} schema",
        "strict": True,
        "schema": pydantic_schema
    }


# Example usage
if __name__ == "__main__":
    # Example 1: Create a simple schema
    person_schema = create_json_schema(
        name="Person",
        description="Information about a person",
        properties={
            "name": {"type": "string", "description": "Full name"},
            "age": {"type": "integer", "description": "Age in years"},
            "email": {"type": "string", "description": "Email address"}
        },
        required=["name", "email"]
    )
    print("Person Schema:", json.dumps(person_schema, indent=2))

    # Example 2: Extract JSON from text
    mixed_text = """
    Here's the analysis result:
    ```json
    {
        "sentiment": "positive",
        "confidence": 0.95,
        "keywords": ["amazing", "great", "excellent"]
    }
    ```
    """
    extracted = extract_json_from_text(mixed_text)
    print("\nExtracted JSON:", extracted)

    # Example 3: Pydantic validation
    from pydantic import BaseModel, Field

    class Product(BaseModel):
        """Product information"""
        name: str = Field(description="Product name")
        price: float = Field(description="Price in USD", gt=0)
        in_stock: bool = Field(description="Availability")

    # Valid data
    valid_data = {"name": "Widget", "price": 19.99, "in_stock": True}
    product = validate_with_pydantic(valid_data, Product)
    print("\nValidated Product:", product)

    # Convert Pydantic to JSON schema
    product_schema = pydantic_to_json_schema(Product)
    print("\nProduct Schema:", json.dumps(product_schema, indent=2))
