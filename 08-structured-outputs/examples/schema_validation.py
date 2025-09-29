"""
Module 08: Schema Validation

Implement robust schema validation using Pydantic and custom validators.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
import re


def example_1_basic_pydantic_schemas():
    """Define and use basic Pydantic schemas."""
    print("=" * 60)
    print("Example 1: Basic Pydantic Schemas")
    print("=" * 60)

    # Define a schema
    class ProductSchema(BaseModel):
        name: str = Field(..., min_length=1, max_length=100)
        price: float = Field(..., gt=0, description="Price in USD")
        quantity: int = Field(ge=0, description="Available quantity")
        in_stock: bool
        tags: List[str] = Field(default_factory=list)
        description: Optional[str] = None

        # Custom validator
        @validator('price')
        def round_price(cls, v):
            return round(v, 2)

        @validator('tags')
        def validate_tags(cls, v):
            # Remove duplicates and empty strings
            return list(set(tag for tag in v if tag.strip()))

    client = LLMClient("openai")

    prompt = """
Extract product information from this description:

"The Professional Coffee Maker is priced at $149.999 with 25 units in stock.
Tags include: kitchen, appliances, coffee, coffee, premium.
This is a high-end coffee maker with advanced brewing technology."

Return as JSON matching this schema:
{
    "name": "string",
    "price": number,
    "quantity": integer,
    "in_stock": boolean,
    "tags": ["array", "of", "strings"],
    "description": "optional string"
}
"""

    response = client.complete(prompt, temperature=0.1, max_tokens=200)

    print("LLM RESPONSE:")
    print(response)

    # Parse and validate
    try:
        # Extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            raw_json = json.loads(json_match.group())

            # Validate with Pydantic
            product = ProductSchema(**raw_json)

            print("\nVALIDATED SCHEMA:")
            print(product.json(indent=2))

            print("\nVALIDATION RESULTS:")
            print(f"✅ Price rounded to: ${product.price}")
            print(f"✅ Tags deduplicated: {product.tags}")
            print(f"✅ All fields validated")

    except ValidationError as e:
        print(f"\n❌ Validation error:")
        for error in e.errors():
            print(f"  - {error['loc'][0]}: {error['msg']}")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON parse error: {e}")

    print("\n💡 Pydantic provides automatic validation and type conversion")


def example_2_nested_schema_validation():
    """Handle complex nested schemas."""
    print("\n" + "=" * 60)
    print("Example 2: Nested Schema Validation")
    print("=" * 60)

    # Define nested schemas
    class AddressSchema(BaseModel):
        street: str
        city: str
        state: str = Field(..., regex="^[A-Z]{2}$")
        zip_code: str = Field(..., regex="^\\d{5}(-\\d{4})?$")
        country: str = "USA"

    class ContactSchema(BaseModel):
        email: str = Field(..., regex="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$")
        phone: Optional[str] = Field(None, regex="^\\+?1?\\d{10,14}$")
        website: Optional[str] = None

    class CompanySchema(BaseModel):
        name: str
        founded: int = Field(..., ge=1800, le=2024)
        employees: int = Field(..., ge=1)
        address: AddressSchema
        contact: ContactSchema
        products: List[str]
        revenue: Optional[float] = None

        @validator('founded')
        def validate_founded_year(cls, v):
            current_year = datetime.now().year
            if v > current_year:
                raise ValueError(f'Founded year cannot be in the future')
            return v

    client = LLMClient("openai")

    prompt = """
Parse this company information into structured JSON:

TechStart Inc. was founded in 2019 and has 50 employees.
Located at 123 Innovation Drive, San Francisco, CA 94105.
Contact: info@techstart.com, phone +14155551234, website: www.techstart.com
Main products: CloudAPI, DataSync, Analytics Pro
Annual revenue: $5.2 million

Return as nested JSON with all required fields.
"""

    response = client.complete(prompt, temperature=0.1, max_tokens=400)

    # Manually construct expected structure for testing
    test_data = {
        "name": "TechStart Inc.",
        "founded": 2019,
        "employees": 50,
        "address": {
            "street": "123 Innovation Drive",
            "city": "San Francisco",
            "state": "CA",
            "zip_code": "94105"
        },
        "contact": {
            "email": "info@techstart.com",
            "phone": "+14155551234",
            "website": "www.techstart.com"
        },
        "products": ["CloudAPI", "DataSync", "Analytics Pro"],
        "revenue": 5200000.0
    }

    print("VALIDATING NESTED SCHEMA:")

    try:
        company = CompanySchema(**test_data)
        print("✅ Nested schema validation successful!")
        print("\nValidated data:")
        print(company.json(indent=2))

        # Access nested fields
        print(f"\nAccessing nested fields:")
        print(f"Company email: {company.contact.email}")
        print(f"City: {company.address.city}")
        print(f"First product: {company.products[0]}")

    except ValidationError as e:
        print("❌ Validation errors:")
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            print(f"  {field_path}: {error['msg']}")

    print("\n💡 Nested schemas enable complex data validation")


def example_3_enum_and_union_types():
    """Use enums and union types for constrained values."""
    print("\n" + "=" * 60)
    print("Example 3: Enum and Union Types")
    print("=" * 60)

    class Status(str, Enum):
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        FAILED = "failed"

    class Priority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

    class TaskSchema(BaseModel):
        id: int
        title: str
        status: Status
        priority: Priority
        assignee: Optional[str] = None
        due_date: Optional[str] = None
        metadata: Union[Dict[str, Any], List[str], str, None] = None

        @validator('due_date')
        def validate_date(cls, v):
            if v:
                try:
                    datetime.strptime(v, "%Y-%m-%d")
                except ValueError:
                    raise ValueError('Date must be in YYYY-MM-DD format')
            return v

    client = LLMClient("openai")

    prompt = """
Extract task information and return as JSON:

"Task #42: Fix the login bug (CRITICAL)
Currently in progress, assigned to John Smith.
Due by 2024-03-15.
Notes: Check authentication middleware, users report intermittent failures"

Use these exact values:
- status: pending/processing/completed/failed
- priority: low/medium/high/critical
"""

    # Simulate parsed response
    task_data = {
        "id": 42,
        "title": "Fix the login bug",
        "status": "processing",
        "priority": "critical",
        "assignee": "John Smith",
        "due_date": "2024-03-15",
        "metadata": {
            "notes": "Check authentication middleware",
            "issue": "users report intermittent failures"
        }
    }

    print("VALIDATING WITH ENUMS AND UNIONS:")

    try:
        task = TaskSchema(**task_data)
        print("✅ Valid task with enum constraints")
        print(f"\nTask details:")
        print(f"  Status: {task.status.value} (Enum)")
        print(f"  Priority: {task.priority.value} (Enum)")
        print(f"  Metadata type: {type(task.metadata).__name__}")

        # Test invalid enum value
        print("\nTesting invalid enum value:")
        invalid_data = task_data.copy()
        invalid_data['status'] = 'in_progress'  # Invalid enum value

        try:
            TaskSchema(**invalid_data)
        except ValidationError as e:
            print(f"✅ Correctly rejected invalid enum: {e.errors()[0]['msg']}")

    except ValidationError as e:
        print(f"❌ Validation error: {e}")

    print("\n💡 Enums enforce specific value sets")


def example_4_custom_validators():
    """Implement complex custom validation logic."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Validators")
    print("=" * 60)

    class UserSchema(BaseModel):
        username: str = Field(..., min_length=3, max_length=20)
        email: str
        age: int = Field(..., ge=13, le=120)
        password: str = Field(..., min_length=8)
        confirm_password: str
        interests: List[str] = Field(..., min_items=1, max_items=10)
        bio: Optional[str] = Field(None, max_length=500)

        @validator('username')
        def username_alphanumeric(cls, v):
            if not re.match("^[a-zA-Z0-9_]+$", v):
                raise ValueError('Username must be alphanumeric with underscores only')
            if v.lower() in ['admin', 'root', 'system']:
                raise ValueError('Reserved username')
            return v.lower()

        @validator('email')
        def email_valid(cls, v):
            email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_regex, v):
                raise ValueError('Invalid email format')
            # Check for disposable email domains
            disposable_domains = ['tempmail.com', '10minutemail.com', 'throwaway.email']
            domain = v.split('@')[1]
            if domain in disposable_domains:
                raise ValueError('Disposable email addresses not allowed')
            return v.lower()

        @validator('password')
        def password_strength(cls, v):
            if not any(char.isdigit() for char in v):
                raise ValueError('Password must contain at least one number')
            if not any(char.isupper() for char in v):
                raise ValueError('Password must contain at least one uppercase letter')
            if not any(char in '!@#$%^&*()_+-=' for char in v):
                raise ValueError('Password must contain at least one special character')
            return v

        @validator('confirm_password')
        def passwords_match(cls, v, values):
            if 'password' in values and v != values['password']:
                raise ValueError('Passwords do not match')
            return v

        @validator('interests')
        def validate_interests(cls, v):
            # Normalize and validate interests
            normalized = []
            valid_interests = [
                'technology', 'sports', 'music', 'art', 'travel',
                'cooking', 'reading', 'gaming', 'fitness', 'photography'
            ]
            for interest in v:
                normalized_interest = interest.lower().strip()
                if normalized_interest not in valid_interests:
                    raise ValueError(f'Invalid interest: {interest}')
                if normalized_interest not in normalized:
                    normalized.append(normalized_interest)
            return normalized

    # Test various user inputs
    test_cases = [
        {
            "name": "Valid User",
            "data": {
                "username": "john_doe123",
                "email": "john@example.com",
                "age": 25,
                "password": "SecureP@ss123",
                "confirm_password": "SecureP@ss123",
                "interests": ["Technology", "Gaming", "MUSIC"],
                "bio": "Software developer and gamer"
            }
        },
        {
            "name": "Invalid Password",
            "data": {
                "username": "jane_smith",
                "email": "jane@example.com",
                "age": 30,
                "password": "weakpass",
                "confirm_password": "weakpass",
                "interests": ["Art"]
            }
        },
        {
            "name": "Reserved Username",
            "data": {
                "username": "admin",
                "email": "admin@example.com",
                "age": 25,
                "password": "Admin@123",
                "confirm_password": "Admin@123",
                "interests": ["Technology"]
            }
        }
    ]

    print("TESTING CUSTOM VALIDATORS:\n")

    for test_case in test_cases:
        print(f"Test: {test_case['name']}")
        print("-" * 40)

        try:
            user = UserSchema(**test_case['data'])
            print("✅ Validation passed")
            print(f"  Normalized username: {user.username}")
            print(f"  Normalized interests: {user.interests}")
        except ValidationError as e:
            print("❌ Validation failed:")
            for error in e.errors()[:3]:  # Show first 3 errors
                print(f"  - {error['loc'][0]}: {error['msg']}")
        print()

    print("💡 Custom validators enforce business logic")


def example_5_schema_inheritance():
    """Use schema inheritance for reusable components."""
    print("\n" + "=" * 60)
    print("Example 5: Schema Inheritance")
    print("=" * 60)

    # Base schemas
    class TimestampMixin(BaseModel):
        created_at: datetime = Field(default_factory=datetime.now)
        updated_at: Optional[datetime] = None

    class IdentifiableMixin(BaseModel):
        id: str = Field(..., regex="^[A-Za-z0-9]{8,}$")
        version: int = Field(default=1, ge=1)

    # Inherited schemas
    class BaseDocument(TimestampMixin, IdentifiableMixin):
        """Base class for all documents."""
        title: str
        tags: List[str] = Field(default_factory=list)

    class Article(BaseDocument):
        """Article extends BaseDocument."""
        content: str = Field(..., min_length=100)
        author: str
        published: bool = False
        word_count: Optional[int] = None

        @validator('word_count', always=True)
        def calculate_word_count(cls, v, values):
            if 'content' in values:
                return len(values['content'].split())
            return v

    class VideoContent(BaseDocument):
        """Video extends BaseDocument."""
        video_url: str
        duration_seconds: int = Field(..., gt=0)
        thumbnail_url: Optional[str] = None
        transcript: Optional[str] = None

        @validator('video_url')
        def validate_video_url(cls, v):
            if not (v.startswith('http://') or v.startswith('https://')):
                raise ValueError('Video URL must start with http:// or https://')
            return v

    # Test inheritance
    print("TESTING SCHEMA INHERITANCE:\n")

    article_data = {
        "id": "ART12345678",
        "title": "Understanding Schema Inheritance",
        "content": "Schema inheritance in Pydantic allows you to create reusable components. " * 20,
        "author": "Jane Developer",
        "tags": ["python", "pydantic", "validation"],
        "published": True
    }

    video_data = {
        "id": "VID98765432",
        "title": "Pydantic Tutorial",
        "video_url": "https://example.com/video.mp4",
        "duration_seconds": 600,
        "tags": ["tutorial", "python"]
    }

    try:
        article = Article(**article_data)
        print("✅ Article validated with inheritance")
        print(f"  Word count (auto-calculated): {article.word_count}")
        print(f"  Created at: {article.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Version: {article.version}")

        video = VideoContent(**video_data)
        print("\n✅ Video validated with inheritance")
        print(f"  Duration: {video.duration_seconds}s")
        print(f"  Has same base fields: id={video.id}, version={video.version}")

    except ValidationError as e:
        print(f"❌ Validation error: {e}")

    print("\n💡 Inheritance promotes code reuse and consistency")


def example_6_dynamic_schema_generation():
    """Generate schemas dynamically from data."""
    print("\n" + "=" * 60)
    print("Example 6: Dynamic Schema Generation")
    print("=" * 60)

    def create_schema_from_dict(data: Dict[str, Any], schema_name: str = "DynamicSchema"):
        """Create a Pydantic schema from a dictionary."""
        fields = {}

        for key, value in data.items():
            # Determine field type
            if isinstance(value, str):
                fields[key] = (str, Field(...))
            elif isinstance(value, int):
                fields[key] = (int, Field(...))
            elif isinstance(value, float):
                fields[key] = (float, Field(...))
            elif isinstance(value, bool):
                fields[key] = (bool, Field(...))
            elif isinstance(value, list):
                if value and isinstance(value[0], str):
                    fields[key] = (List[str], Field(...))
                else:
                    fields[key] = (List[Any], Field(...))
            elif isinstance(value, dict):
                fields[key] = (Dict[str, Any], Field(...))
            else:
                fields[key] = (Optional[Any], None)

        # Create dynamic model
        DynamicModel = type(schema_name, (BaseModel,), {
            '__annotations__': {k: v[0] for k, v in fields.items()},
            **{k: v[1] for k, v in fields.items() if v[1] is not None}
        })

        return DynamicModel

    # Example data
    sample_data = {
        "product_id": "PROD123",
        "name": "Widget",
        "price": 29.99,
        "available": True,
        "categories": ["tools", "hardware"],
        "specifications": {
            "weight": "500g",
            "dimensions": "10x10x5cm"
        }
    }

    print("GENERATING SCHEMA FROM DATA:\n")
    print("Sample data:")
    print(json.dumps(sample_data, indent=2))

    # Generate schema
    DynamicProductSchema = create_schema_from_dict(sample_data, "ProductSchema")

    print("\nGenerated schema fields:")
    for field_name, field_info in DynamicProductSchema.__fields__.items():
        print(f"  {field_name}: {field_info.type_}")

    # Validate data with generated schema
    try:
        product = DynamicProductSchema(**sample_data)
        print("\n✅ Data validates against generated schema")

        # Test with modified data
        modified_data = sample_data.copy()
        modified_data['price'] = "not_a_number"

        try:
            DynamicProductSchema(**modified_data)
        except ValidationError as e:
            print("\n✅ Generated schema correctly rejects invalid data:")
            print(f"  {e.errors()[0]['msg']}")

    except ValidationError as e:
        print(f"❌ Validation error: {e}")

    print("\n💡 Dynamic schemas useful for unknown data structures")


def example_7_schema_with_llm_integration():
    """Integrate schema validation with LLM outputs."""
    print("\n" + "=" * 60)
    print("Example 7: Schema + LLM Integration")
    print("=" * 60)

    client = LLMClient("openai")

    class ExtractedEntity(BaseModel):
        """Schema for extracted entities."""
        text: str
        type: str = Field(..., pattern="^(PERSON|ORGANIZATION|LOCATION|DATE|MONEY|PRODUCT)$")
        confidence: float = Field(..., ge=0.0, le=1.0)
        context: Optional[str] = None

    class TextAnalysis(BaseModel):
        """Complete text analysis schema."""
        entities: List[ExtractedEntity]
        sentiment: str = Field(..., pattern="^(positive|negative|neutral)$")
        key_topics: List[str] = Field(..., max_items=5)
        summary: str = Field(..., max_length=200)
        language: str = Field(default="en")

        @validator('entities')
        def validate_entities(cls, v):
            # Remove duplicates based on text
            seen = set()
            unique = []
            for entity in v:
                if entity.text not in seen:
                    seen.add(entity.text)
                    unique.append(entity)
            return unique

    def extract_with_schema(text: str, max_retries: int = 3) -> Optional[TextAnalysis]:
        """Extract structured data from text using LLM."""
        schema_example = {
            "entities": [
                {"text": "example", "type": "PERSON", "confidence": 0.95, "context": "optional"}
            ],
            "sentiment": "positive",
            "key_topics": ["topic1", "topic2"],
            "summary": "brief summary",
            "language": "en"
        }

        prompt = f"""
Analyze this text and return structured JSON:

Text: "{text}"

Return JSON exactly matching this schema:
{json.dumps(schema_example, indent=2)}

Entity types: PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PRODUCT
Sentiment values: positive, negative, neutral

JSON Output:"""

        for attempt in range(max_retries):
            try:
                response = client.complete(prompt, temperature=0.1, max_tokens=500)

                # Extract JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    raw_json = json.loads(json_match.group())

                    # Validate with schema
                    analysis = TextAnalysis(**raw_json)
                    return analysis

            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == max_retries - 1:
                    print(f"❌ Failed after {max_retries} attempts: {e}")
                    return None
                # Retry with error feedback
                prompt += f"\n\nError in attempt {attempt + 1}: {str(e)}\nPlease fix and return valid JSON:"

        return None

    # Test text
    test_text = """
    Apple Inc. announced that CEO Tim Cook will visit London next week to meet
    with Prime Minister. The new iPhone 15 Pro, priced at $999, has exceeded
    sales expectations with over 10 million units sold.
    """

    print("EXTRACTING WITH SCHEMA VALIDATION:\n")
    print(f"Input text: {test_text[:100]}...\n")

    # Simulate extraction (using mock data for reliability)
    mock_result = {
        "entities": [
            {"text": "Apple Inc.", "type": "ORGANIZATION", "confidence": 0.99},
            {"text": "Tim Cook", "type": "PERSON", "confidence": 0.98},
            {"text": "London", "type": "LOCATION", "confidence": 0.95},
            {"text": "iPhone 15 Pro", "type": "PRODUCT", "confidence": 0.97},
            {"text": "$999", "type": "MONEY", "confidence": 0.99}
        ],
        "sentiment": "positive",
        "key_topics": ["technology", "business", "product launch"],
        "summary": "Apple CEO to visit London; iPhone 15 Pro exceeds sales expectations",
        "language": "en"
    }

    try:
        analysis = TextAnalysis(**mock_result)
        print("✅ Structured extraction successful!\n")
        print("Extracted entities:")
        for entity in analysis.entities:
            print(f"  - {entity.text} ({entity.type}): {entity.confidence:.2f}")

        print(f"\nSentiment: {analysis.sentiment}")
        print(f"Topics: {', '.join(analysis.key_topics)}")
        print(f"Summary: {analysis.summary}")

    except ValidationError as e:
        print(f"❌ Validation failed: {e}")

    print("\n💡 Schema validation ensures reliable LLM outputs")


def run_all_examples():
    """Run all schema validation examples."""
    examples = [
        example_1_basic_pydantic_schemas,
        example_2_nested_schema_validation,
        example_3_enum_and_union_types,
        example_4_custom_validators,
        example_5_schema_inheritance,
        example_6_dynamic_schema_generation,
        example_7_schema_with_llm_integration
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

    parser = argparse.ArgumentParser(description="Module 08: Schema Validation")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_basic_pydantic_schemas,
            2: example_2_nested_schema_validation,
            3: example_3_enum_and_union_types,
            4: example_4_custom_validators,
            5: example_5_schema_inheritance,
            6: example_6_dynamic_schema_generation,
            7: example_7_schema_with_llm_integration
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 08: Schema Validation Examples")
        print("\nUsage:")
        print("  python schema_validation.py --all        # Run all examples")
        print("  python schema_validation.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: Basic Pydantic Schemas")
        print("  2: Nested Schema Validation")
        print("  3: Enum and Union Types")
        print("  4: Custom Validators")
        print("  5: Schema Inheritance")
        print("  6: Dynamic Schema Generation")
        print("  7: Schema + LLM Integration")