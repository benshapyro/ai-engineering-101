"""
Module 08: JSON Generation

Master reliable JSON output generation from LLMs with validation and error handling.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from typing import Dict, List, Optional, Any
import re


def example_1_basic_json_generation():
    """Generate basic JSON outputs from LLMs."""
    print("=" * 60)
    print("Example 1: Basic JSON Generation")
    print("=" * 60)

    client = LLMClient("openai")

    # Simple JSON request
    prompt = """
Analyze the sentiment of this text and return the result as JSON:

Text: "I absolutely love this product! It exceeded all my expectations and the customer service was fantastic."

Return JSON in this exact format:
{
    "sentiment": "positive/negative/neutral",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}
"""

    response = client.complete(prompt, temperature=0.1, max_tokens=200)
    print("RAW RESPONSE:")
    print(response)

    # Try to parse JSON
    try:
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            print("\nPARSED JSON:")
            print(json.dumps(parsed, indent=2))

            # Validate structure
            required_keys = ["sentiment", "confidence", "reasoning"]
            missing = [k for k in required_keys if k not in parsed]
            if missing:
                print(f"\n⚠️ Missing keys: {missing}")
            else:
                print("\n✅ All required keys present")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON parsing error: {e}")

    print("\n💡 Basic JSON generation requires careful prompt formatting")


def example_2_enforced_json_structure():
    """Enforce specific JSON structure through prompting."""
    print("\n" + "=" * 60)
    print("Example 2: Enforced JSON Structure")
    print("=" * 60)

    client = LLMClient("openai")

    def enforce_json_output(task, schema_example):
        """Create a prompt that strongly enforces JSON output."""
        prompt = f"""{task}

You MUST respond with ONLY valid JSON matching this EXACT structure:
{json.dumps(schema_example, indent=2)}

Rules:
1. Use ONLY double quotes for strings
2. NO trailing commas
3. NO comments or text outside the JSON
4. All fields are REQUIRED
5. Types must match exactly (string, number, boolean, array, object)

JSON Output:"""
        return prompt

    # Complex schema example
    schema_example = {
        "product": {
            "name": "string",
            "category": "string",
            "price": 0.00,
            "in_stock": True,
            "tags": ["tag1", "tag2"],
            "ratings": {
                "average": 0.0,
                "count": 0
            }
        }
    }

    task = """
Extract product information from this description:
"The UltraBook Pro laptop is a high-performance computer in our electronics category,
priced at $1299.99. Currently in stock with tags: portable, powerful, business.
It has an average rating of 4.5 stars from 328 reviews."
"""

    prompt = enforce_json_output(task, schema_example)
    response = client.complete(prompt, temperature=0.1, max_tokens=300)

    print("ENFORCED JSON RESPONSE:")
    try:
        # Clean response (remove any non-JSON text)
        json_str = response.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]

        parsed = json.loads(json_str.strip())
        print(json.dumps(parsed, indent=2))
        print("\n✅ Successfully parsed enforced structure")
    except json.JSONDecodeError as e:
        print(f"❌ Parse error: {e}")
        print(f"Raw response: {response}")

    print("\n💡 Strong enforcement reduces parsing errors")


def example_3_nested_json_handling():
    """Handle complex nested JSON structures."""
    print("\n" + "=" * 60)
    print("Example 3: Nested JSON Structures")
    print("=" * 60)

    client = LLMClient("openai")

    # Complex nested structure
    prompt = """
Parse this company information into nested JSON:

"TechCorp Inc. was founded in 2010 by Jane Smith (CEO) and John Doe (CTO).
They have offices in San Francisco (HQ, 500 employees), New York (200 employees),
and London (150 employees). Their main products are CloudSync (SaaS, $99/mo)
and DataAnalyzer (Enterprise, $999/mo)."

Return as nested JSON:
{
    "company": {
        "name": "string",
        "founded": number,
        "founders": [
            {
                "name": "string",
                "role": "string"
            }
        ],
        "offices": [
            {
                "location": "string",
                "type": "string",
                "employees": number
            }
        ],
        "products": [
            {
                "name": "string",
                "type": "string",
                "price": "string"
            }
        ]
    }
}

JSON Output:"""

    response = client.complete(prompt, temperature=0.1, max_tokens=500)

    print("NESTED JSON RESPONSE:")
    try:
        # Extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            print(json.dumps(parsed, indent=2))

            # Validate nested structure
            def validate_nested(obj, path=""):
                """Recursively validate nested structure."""
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        validate_nested(value, f"{path}.{key}" if path else key)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        validate_nested(item, f"{path}[{i}]")
                return True

            if validate_nested(parsed):
                print("\n✅ Valid nested structure")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ Error: {e}")

    print("\n💡 Nested structures require clear schema examples")


def example_4_json_with_retry():
    """Implement retry logic for JSON generation."""
    print("\n" + "=" * 60)
    print("Example 4: JSON Generation with Retry")
    print("=" * 60)

    client = LLMClient("openai")

    def get_json_with_retry(prompt, expected_keys, max_retries=3):
        """Get JSON with automatic retry on failure."""
        attempt = 0
        errors = []

        while attempt < max_retries:
            attempt += 1
            print(f"\nAttempt {attempt}/{max_retries}:")

            # Modify prompt based on previous errors
            current_prompt = prompt
            if errors:
                error_msg = "\n".join([f"- {e}" for e in errors])
                current_prompt += f"\n\nPrevious errors to avoid:\n{error_msg}\n\nCorrected JSON:"

            response = client.complete(current_prompt, temperature=0.1, max_tokens=300)

            try:
                # Extract and parse JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON object found in response")

                parsed = json.loads(json_match.group())

                # Validate expected keys
                missing_keys = [k for k in expected_keys if k not in parsed]
                if missing_keys:
                    raise KeyError(f"Missing required keys: {missing_keys}")

                print(f"✅ Success on attempt {attempt}")
                return parsed

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                error_msg = str(e)
                errors.append(error_msg)
                print(f"❌ Error: {error_msg}")

                if attempt == max_retries:
                    print("\n❌ Max retries reached")
                    return None

        return None

    # Test with potentially problematic prompt
    prompt = """
Analyze this tweet and return JSON:
"Just deployed my app! 🚀 #coding #success"

Required JSON format:
{
    "text": "the tweet text",
    "hashtags": ["list", "of", "hashtags"],
    "emoji_count": number,
    "language": "detected language",
    "tone": "emotional tone"
}"""

    result = get_json_with_retry(prompt, ["text", "hashtags", "emoji_count", "language", "tone"])

    if result:
        print("\nFINAL RESULT:")
        print(json.dumps(result, indent=2))

    print("\n💡 Retry logic improves reliability significantly")


def example_5_streaming_json_parsing():
    """Parse JSON from streaming responses."""
    print("\n" + "=" * 60)
    print("Example 5: Streaming JSON Parsing")
    print("=" * 60)

    class StreamingJSONParser:
        """Parse JSON as it streams in."""
        def __init__(self):
            self.buffer = ""
            self.brace_count = 0
            self.in_string = False
            self.escape_next = False

        def add_chunk(self, chunk):
            """Add a chunk and try to parse if complete."""
            self.buffer += chunk

            # Track brace depth
            for char in chunk:
                if self.escape_next:
                    self.escape_next = False
                    continue

                if char == '\\':
                    self.escape_next = True
                elif char == '"' and not self.escape_next:
                    self.in_string = not self.in_string
                elif not self.in_string:
                    if char == '{':
                        self.brace_count += 1
                    elif char == '}':
                        self.brace_count -= 1

            # Try to parse if braces are balanced
            if self.brace_count == 0 and '{' in self.buffer:
                try:
                    # Extract JSON object
                    start = self.buffer.index('{')
                    end = self.buffer.rindex('}') + 1
                    json_str = self.buffer[start:end]
                    return json.loads(json_str)
                except (ValueError, json.JSONDecodeError):
                    return None

            return None

    # Simulate streaming response
    streaming_chunks = [
        "Here is the JSON",
        " output:\n{",
        '\n  "status"',
        ': "success",',
        '\n  "data": {',
        '\n    "id": 123,',
        '\n    "values"',
        ': [1, 2, 3]',
        '\n  }',
        '\n}',
        '\nThat\'s the response.'
    ]

    parser = StreamingJSONParser()
    print("STREAMING JSON CHUNKS:")

    for i, chunk in enumerate(streaming_chunks):
        print(f"Chunk {i+1}: {repr(chunk)}")
        result = parser.add_chunk(chunk)

        if result:
            print(f"\n✅ Complete JSON parsed at chunk {i+1}:")
            print(json.dumps(result, indent=2))
            break

    print("\n💡 Streaming parsing enables real-time processing")


def example_6_json_array_generation():
    """Generate and handle JSON arrays."""
    print("\n" + "=" * 60)
    print("Example 6: JSON Array Generation")
    print("=" * 60)

    client = LLMClient("openai")

    prompt = """
Extract all action items from this meeting transcript as a JSON array:

"In today's meeting, we discussed several important points:
- John will complete the API documentation by Friday
- Sarah needs to review the security audit report
- The team should fix the login bug before the release
- Marketing will prepare the launch campaign materials"

Return as JSON array:
[
    {
        "assignee": "person responsible",
        "task": "description",
        "deadline": "date or null",
        "priority": "high/medium/low"
    }
]

JSON Array:"""

    response = client.complete(prompt, temperature=0.1, max_tokens=400)

    print("ARRAY GENERATION RESPONSE:")
    try:
        # Extract array
        array_match = re.search(r'\[.*\]', response, re.DOTALL)
        if array_match:
            parsed_array = json.loads(array_match.group())
            print(json.dumps(parsed_array, indent=2))

            # Validate array items
            print(f"\n✅ Generated array with {len(parsed_array)} items")

            # Check consistency
            if parsed_array:
                first_keys = set(parsed_array[0].keys())
                consistent = all(set(item.keys()) == first_keys for item in parsed_array)
                if consistent:
                    print("✅ All array items have consistent structure")
                else:
                    print("⚠️ Inconsistent structure across array items")

    except json.JSONDecodeError as e:
        print(f"❌ Parse error: {e}")

    print("\n💡 Arrays need consistent item structure")


def example_7_json_validation_and_fixing():
    """Validate and automatically fix malformed JSON."""
    print("\n" + "=" * 60)
    print("Example 7: JSON Validation and Auto-Fixing")
    print("=" * 60)

    client = LLMClient("openai")

    class JSONValidator:
        """Validate and fix common JSON errors."""

        @staticmethod
        def fix_common_errors(json_str):
            """Fix common JSON formatting errors."""
            # Remove trailing commas
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            # Fix single quotes (convert to double quotes)
            # Be careful not to replace apostrophes in values
            json_str = re.sub(r"(?<=[{\[,:]\s)'([^']*)'(?=\s*[,:}\]])", r'"\1"', json_str)

            # Remove comments
            json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

            # Fix unquoted keys (simple cases)
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)

            # Remove BOM if present
            if json_str.startswith('\ufeff'):
                json_str = json_str[1:]

            return json_str

        @staticmethod
        def validate_and_fix(json_str, required_schema=None):
            """Validate JSON and attempt fixes."""
            attempts = [
                ("Original", json_str),
                ("Basic fixes", JSONValidator.fix_common_errors(json_str)),
                ("Extracted object", re.search(r'\{.*\}', json_str, re.DOTALL).group()
                 if re.search(r'\{.*\}', json_str, re.DOTALL) else json_str)
            ]

            for attempt_name, attempt_str in attempts:
                try:
                    parsed = json.loads(attempt_str)

                    # Validate against schema if provided
                    if required_schema:
                        for key in required_schema:
                            if key not in parsed:
                                raise KeyError(f"Missing required key: {key}")

                    print(f"✅ Fixed with: {attempt_name}")
                    return parsed

                except (json.JSONDecodeError, KeyError, AttributeError) as e:
                    print(f"❌ {attempt_name} failed: {e}")
                    continue

            return None

    # Test with malformed JSON
    malformed_responses = [
        '{"name": "John", "age": 30,}',  # Trailing comma
        "{'name': 'John', 'age': 30}",   # Single quotes
        '{"name": "John" /* comment */, "age": 30}',  # Comment
        'Here is the JSON: {name: "John", age: 30}',  # Unquoted keys
    ]

    validator = JSONValidator()

    print("TESTING JSON VALIDATION AND FIXING:\n")

    for i, malformed in enumerate(malformed_responses, 1):
        print(f"Test {i}: {repr(malformed[:50])}...")
        result = validator.validate_and_fix(malformed, required_schema=["name", "age"])

        if result:
            print(f"Fixed JSON: {json.dumps(result)}\n")
        else:
            print("Could not fix JSON\n")

    # Also test with LLM fixing
    print("LLM-ASSISTED FIXING:")

    broken_json = '{"users": [{"name": "John", "age": 30,}, {"name": "Jane" "age": 25}]'

    fix_prompt = f"""
Fix this malformed JSON:
{broken_json}

Common issues to check:
- Trailing commas
- Missing commas
- Unclosed brackets
- Quote issues

Return ONLY the fixed valid JSON:"""

    response = client.complete(fix_prompt, temperature=0.1, max_tokens=200)

    try:
        fixed = json.loads(response.strip())
        print("✅ LLM successfully fixed JSON:")
        print(json.dumps(fixed, indent=2))
    except json.JSONDecodeError as e:
        print(f"❌ LLM fix failed: {e}")

    print("\n💡 Multiple validation strategies improve robustness")


def run_all_examples():
    """Run all JSON generation examples."""
    examples = [
        example_1_basic_json_generation,
        example_2_enforced_json_structure,
        example_3_nested_json_handling,
        example_4_json_with_retry,
        example_5_streaming_json_parsing,
        example_6_json_array_generation,
        example_7_json_validation_and_fixing
    ]

    for example in examples:
        try:
            example()
            print("\n" + "=" * 60 + "\n")
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 08: JSON Generation")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_basic_json_generation,
            2: example_2_enforced_json_structure,
            3: example_3_nested_json_handling,
            4: example_4_json_with_retry,
            5: example_5_streaming_json_parsing,
            6: example_6_json_array_generation,
            7: example_7_json_validation_and_fixing
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 08: JSON Generation Examples")
        print("\nUsage:")
        print("  python json_generation.py --all        # Run all examples")
        print("  python json_generation.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: Basic JSON Generation")
        print("  2: Enforced JSON Structure")
        print("  3: Nested JSON Handling")
        print("  4: JSON with Retry Logic")
        print("  5: Streaming JSON Parsing")
        print("  6: JSON Array Generation")
        print("  7: JSON Validation and Fixing")