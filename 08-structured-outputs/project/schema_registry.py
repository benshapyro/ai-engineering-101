"""
Module 08: Structured Outputs - Project
Schema Registry and Validation Service

Build a production-ready schema registry that manages structured output schemas,
validates LLM outputs, and provides format conversion capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from pydantic import BaseModel, Field, validator, ValidationError
from typing import Dict, List, Optional, Any, Type, Union
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv
import io
from datetime import datetime
from enum import Enum
import hashlib
import pickle
from functools import lru_cache
import time


# ===== Schema Models =====

class DataType(str, Enum):
    """Supported data types for schema fields."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    DATETIME = "datetime"
    ENUM = "enum"


class FieldDefinition(BaseModel):
    """Definition of a single field in a schema."""
    name: str
    type: DataType
    required: bool = True
    description: Optional[str] = None
    default: Optional[Any] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    enum_values: Optional[List[str]] = None
    nested_schema: Optional[str] = None  # Reference to another schema


class SchemaDefinition(BaseModel):
    """Complete schema definition."""
    name: str
    version: str = "1.0.0"
    description: str
    fields: List[FieldDefinition]
    examples: Optional[List[Dict]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# ===== Schema Registry =====

class SchemaRegistry:
    """Central registry for managing schemas."""

    def __init__(self):
        self.schemas: Dict[str, Dict[str, SchemaDefinition]] = {}
        self.compiled_schemas: Dict[str, Type[BaseModel]] = {}
        self.validation_cache = {}

    def register_schema(self, schema: SchemaDefinition) -> bool:
        """Register a new schema or version."""
        if schema.name not in self.schemas:
            self.schemas[schema.name] = {}

        if schema.version in self.schemas[schema.name]:
            raise ValueError(f"Schema {schema.name} v{schema.version} already exists")

        self.schemas[schema.name][schema.version] = schema
        self._compile_schema(schema)
        return True

    def get_schema(self, name: str, version: Optional[str] = None) -> SchemaDefinition:
        """Get a schema by name and version."""
        if name not in self.schemas:
            raise KeyError(f"Schema {name} not found")

        if version is None:
            # Get latest version
            versions = sorted(self.schemas[name].keys(), reverse=True)
            version = versions[0]

        if version not in self.schemas[name]:
            raise KeyError(f"Schema {name} v{version} not found")

        return self.schemas[name][version]

    def list_schemas(self) -> List[Dict[str, str]]:
        """List all registered schemas."""
        result = []
        for name, versions in self.schemas.items():
            for version, schema in versions.items():
                result.append({
                    "name": name,
                    "version": version,
                    "description": schema.description,
                    "fields": len(schema.fields)
                })
        return result

    def _compile_schema(self, schema: SchemaDefinition):
        """Compile schema to Pydantic model."""
        fields = {}
        validators = {}

        for field in schema.fields:
            # Create field type
            field_type = self._get_field_type(field)

            # Add to fields
            if field.required:
                fields[field.name] = (field_type, Field(..., description=field.description))
            else:
                default = field.default if field.default is not None else None
                fields[field.name] = (field_type, Field(default, description=field.description))

            # Add validators
            if field.min_value is not None or field.max_value is not None:
                validators[f"validate_{field.name}_range"] = self._create_range_validator(
                    field.name, field.min_value, field.max_value
                )

            if field.min_length is not None or field.max_length is not None:
                validators[f"validate_{field.name}_length"] = self._create_length_validator(
                    field.name, field.min_length, field.max_length
                )

        # Create dynamic model
        model_name = f"{schema.name}_{schema.version.replace('.', '_')}"
        DynamicModel = type(
            model_name,
            (BaseModel,),
            {
                **fields,
                **validators,
                "__module__": __name__
            }
        )

        self.compiled_schemas[f"{schema.name}:{schema.version}"] = DynamicModel

    def _get_field_type(self, field: FieldDefinition):
        """Convert field definition to Python type."""
        type_mapping = {
            DataType.STRING: str,
            DataType.INTEGER: int,
            DataType.FLOAT: float,
            DataType.BOOLEAN: bool,
            DataType.DATETIME: datetime,
        }

        base_type = type_mapping.get(field.type, Any)

        if field.type == DataType.ARRAY:
            return List[Any]
        elif field.type == DataType.OBJECT:
            return Dict[str, Any]
        elif field.type == DataType.ENUM and field.enum_values:
            return Enum(f"{field.name}_enum", {v: v for v in field.enum_values})
        else:
            return base_type

    def _create_range_validator(self, field_name: str, min_val: Optional[float], max_val: Optional[float]):
        """Create a range validator for numeric fields."""
        def validator_func(cls, v):
            if min_val is not None and v < min_val:
                raise ValueError(f"{field_name} must be >= {min_val}")
            if max_val is not None and v > max_val:
                raise ValueError(f"{field_name} must be <= {max_val}")
            return v
        return validator(field_name, allow_reuse=True)(validator_func)

    def _create_length_validator(self, field_name: str, min_len: Optional[int], max_len: Optional[int]):
        """Create a length validator for string fields."""
        def validator_func(cls, v):
            if isinstance(v, str):
                if min_len is not None and len(v) < min_len:
                    raise ValueError(f"{field_name} must have length >= {min_len}")
                if max_len is not None and len(v) > max_len:
                    raise ValueError(f"{field_name} must have length <= {max_len}")
            return v
        return validator(field_name, allow_reuse=True)(validator_func)


# ===== Format Converters =====

class FormatConverter:
    """Handles conversion between different formats."""

    @staticmethod
    def dict_to_json(data: Dict) -> str:
        """Convert dictionary to JSON string."""
        return json.dumps(data, indent=2, default=str)

    @staticmethod
    def dict_to_xml(data: Dict, root_name: str = "root") -> str:
        """Convert dictionary to XML string."""
        def dict_to_element(d, parent):
            for key, value in d.items():
                if isinstance(value, dict):
                    child = ET.SubElement(parent, key)
                    dict_to_element(value, child)
                elif isinstance(value, list):
                    for item in value:
                        child = ET.SubElement(parent, key)
                        if isinstance(item, dict):
                            dict_to_element(item, child)
                        else:
                            child.text = str(item)
                else:
                    child = ET.SubElement(parent, key)
                    child.text = str(value)

        root = ET.Element(root_name)
        dict_to_element(data, root)

        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    @staticmethod
    def dict_to_yaml(data: Dict) -> str:
        """Convert dictionary to YAML string."""
        return yaml.dump(data, default_flow_style=False)

    @staticmethod
    def dict_to_csv(data: Union[List[Dict], Dict]) -> str:
        """Convert dictionary or list of dictionaries to CSV."""
        if isinstance(data, dict):
            data = [data]

        if not data:
            return ""

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()

    @staticmethod
    def json_to_dict(json_str: str) -> Dict:
        """Parse JSON string to dictionary."""
        return json.loads(json_str)

    @staticmethod
    def xml_to_dict(xml_str: str) -> Dict:
        """Parse XML string to dictionary."""
        def element_to_dict(element):
            result = {}

            # Handle attributes
            if element.attrib:
                result['@attributes'] = element.attrib

            # Handle text content
            if element.text and element.text.strip():
                if len(element) == 0:  # No children
                    return element.text.strip()
                else:
                    result['#text'] = element.text.strip()

            # Handle children
            children = {}
            for child in element:
                child_data = element_to_dict(child)
                if child.tag in children:
                    if not isinstance(children[child.tag], list):
                        children[child.tag] = [children[child.tag]]
                    children[child.tag].append(child_data)
                else:
                    children[child.tag] = child_data

            result.update(children)
            return result if result else None

        root = ET.fromstring(xml_str)
        return {root.tag: element_to_dict(root)}

    @staticmethod
    def yaml_to_dict(yaml_str: str) -> Dict:
        """Parse YAML string to dictionary."""
        return yaml.safe_load(yaml_str)

    @staticmethod
    def csv_to_dict(csv_str: str) -> List[Dict]:
        """Parse CSV string to list of dictionaries."""
        output = io.StringIO(csv_str)
        reader = csv.DictReader(output)
        return list(reader)


# ===== Validation Service =====

class ValidationService:
    """Service for validating data against schemas."""

    def __init__(self, registry: SchemaRegistry):
        self.registry = registry
        self.validation_history = []

    def validate(self,
                data: Union[Dict, str],
                schema_name: str,
                schema_version: Optional[str] = None,
                format: str = "json") -> Dict[str, Any]:
        """Validate data against a schema."""

        # Parse data if string
        if isinstance(data, str):
            data = self._parse_data(data, format)

        # Get schema
        schema = self.registry.get_schema(schema_name, schema_version)

        # Get compiled model
        schema_key = f"{schema_name}:{schema_version or schema.version}"
        if schema_key not in self.registry.compiled_schemas:
            raise ValueError(f"Schema {schema_key} not compiled")

        model_class = self.registry.compiled_schemas[schema_key]

        # Validate
        start_time = time.time()
        errors = []
        warnings = []

        try:
            validated = model_class(**data)
            is_valid = True
            validated_data = validated.dict()
        except ValidationError as e:
            is_valid = False
            validated_data = None
            for error in e.errors():
                errors.append({
                    "field": ".".join(str(x) for x in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"]
                })

        validation_time = time.time() - start_time

        # Create result
        result = {
            "valid": is_valid,
            "schema": schema_name,
            "version": schema_version or schema.version,
            "errors": errors,
            "warnings": warnings,
            "data": validated_data,
            "validation_time": validation_time
        }

        # Store in history
        self.validation_history.append({
            **result,
            "timestamp": datetime.now()
        })

        return result

    def _parse_data(self, data_str: str, format: str) -> Dict:
        """Parse string data based on format."""
        converter = FormatConverter()

        if format == "json":
            return converter.json_to_dict(data_str)
        elif format == "xml":
            return converter.xml_to_dict(data_str)
        elif format == "yaml":
            return converter.yaml_to_dict(data_str)
        elif format == "csv":
            results = converter.csv_to_dict(data_str)
            if len(results) == 1:
                return results[0]
            return {"items": results}
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_validation_report(self, limit: int = 10) -> Dict[str, Any]:
        """Get validation history report."""
        recent = self.validation_history[-limit:]

        total_validations = len(self.validation_history)
        successful = sum(1 for v in self.validation_history if v["valid"])
        failed = total_validations - successful

        avg_time = sum(v["validation_time"] for v in self.validation_history) / total_validations if total_validations > 0 else 0

        return {
            "total_validations": total_validations,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_validations if total_validations > 0 else 0,
            "average_validation_time": avg_time,
            "recent_validations": recent
        }


# ===== LLM Integration =====

class LLMStructuredGenerator:
    """Generate structured outputs using LLMs."""

    def __init__(self, client: LLMClient, registry: SchemaRegistry):
        self.client = client
        self.registry = registry
        self.converter = FormatConverter()

    def generate(self,
                prompt: str,
                schema_name: str,
                schema_version: Optional[str] = None,
                output_format: str = "json",
                max_retries: int = 3) -> Dict[str, Any]:
        """Generate structured output matching a schema."""

        # Get schema
        schema = self.registry.get_schema(schema_name, schema_version)

        # Build generation prompt
        format_instructions = self._get_format_instructions(schema, output_format)

        full_prompt = f"""{prompt}

{format_instructions}

Examples:
{self._format_examples(schema.examples or [], output_format)}

Output:"""

        # Generate with retries
        for attempt in range(max_retries):
            try:
                response = self.client.complete(full_prompt, max_tokens=1000)

                # Parse response
                parsed = self._parse_response(response, output_format)

                # Validate
                validation_service = ValidationService(self.registry)
                result = validation_service.validate(
                    parsed,
                    schema_name,
                    schema_version,
                    "json"
                )

                if result["valid"]:
                    return {
                        "success": True,
                        "data": result["data"],
                        "format": output_format,
                        "attempts": attempt + 1
                    }

            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "attempts": attempt + 1
                    }

        return {
            "success": False,
            "error": "Max retries exceeded",
            "attempts": max_retries
        }

    def _get_format_instructions(self, schema: SchemaDefinition, output_format: str) -> str:
        """Generate format instructions for the LLM."""

        field_descriptions = []
        for field in schema.fields:
            desc = f"- {field.name} ({field.type.value}): {field.description or 'No description'}"
            if not field.required:
                desc += " [optional]"
            if field.min_value or field.max_value:
                desc += f" [range: {field.min_value or 'any'} - {field.max_value or 'any'}]"
            if field.enum_values:
                desc += f" [values: {', '.join(field.enum_values)}]"
            field_descriptions.append(desc)

        instructions = f"""Generate {output_format.upper()} output with the following schema:

Schema: {schema.name} v{schema.version}
Description: {schema.description}

Fields:
{chr(10).join(field_descriptions)}

Format: {output_format.upper()}
"""

        if output_format == "json":
            instructions += "\nEnsure valid JSON syntax with proper quotes and commas."
        elif output_format == "xml":
            instructions += f"\nUse <{schema.name}> as the root element."
        elif output_format == "yaml":
            instructions += "\nUse proper YAML indentation and syntax."

        return instructions

    def _format_examples(self, examples: List[Dict], output_format: str) -> str:
        """Format examples in the requested format."""
        if not examples:
            return "No examples available"

        formatted = []
        for example in examples[:2]:  # Limit to 2 examples
            if output_format == "json":
                formatted.append(self.converter.dict_to_json(example))
            elif output_format == "xml":
                formatted.append(self.converter.dict_to_xml(example))
            elif output_format == "yaml":
                formatted.append(self.converter.dict_to_yaml(example))
            elif output_format == "csv":
                formatted.append(self.converter.dict_to_csv(example))

        return "\n\n".join(formatted)

    def _parse_response(self, response: str, output_format: str) -> Dict:
        """Parse LLM response based on format."""
        # Clean response
        response = response.strip()

        # Remove code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])

        if output_format == "json":
            return self.converter.json_to_dict(response)
        elif output_format == "xml":
            return self.converter.xml_to_dict(response)
        elif output_format == "yaml":
            return self.converter.yaml_to_dict(response)
        elif output_format == "csv":
            results = self.converter.csv_to_dict(response)
            if len(results) == 1:
                return results[0]
            return {"items": results}
        else:
            raise ValueError(f"Unsupported format: {output_format}")


# ===== Example Usage =====

def demo_schema_registry():
    """Demonstrate the schema registry system."""
    print("Schema Registry Demo")
    print("=" * 50)

    # Initialize
    registry = SchemaRegistry()
    client = LLMClient("openai")

    # Define schemas
    user_schema = SchemaDefinition(
        name="User",
        version="1.0.0",
        description="User profile schema",
        fields=[
            FieldDefinition(
                name="username",
                type=DataType.STRING,
                required=True,
                min_length=3,
                max_length=20,
                description="Unique username"
            ),
            FieldDefinition(
                name="email",
                type=DataType.STRING,
                required=True,
                description="Email address"
            ),
            FieldDefinition(
                name="age",
                type=DataType.INTEGER,
                required=False,
                min_value=13,
                max_value=120,
                description="User age"
            ),
            FieldDefinition(
                name="role",
                type=DataType.ENUM,
                required=True,
                enum_values=["admin", "user", "moderator"],
                description="User role"
            )
        ],
        examples=[
            {
                "username": "johndoe",
                "email": "john@example.com",
                "age": 25,
                "role": "user"
            }
        ]
    )

    product_schema = SchemaDefinition(
        name="Product",
        version="1.0.0",
        description="E-commerce product schema",
        fields=[
            FieldDefinition(
                name="name",
                type=DataType.STRING,
                required=True,
                max_length=100,
                description="Product name"
            ),
            FieldDefinition(
                name="price",
                type=DataType.FLOAT,
                required=True,
                min_value=0,
                description="Product price"
            ),
            FieldDefinition(
                name="stock",
                type=DataType.INTEGER,
                required=True,
                min_value=0,
                description="Stock quantity"
            ),
            FieldDefinition(
                name="categories",
                type=DataType.ARRAY,
                required=False,
                description="Product categories"
            )
        ]
    )

    # Register schemas
    registry.register_schema(user_schema)
    registry.register_schema(product_schema)

    print(f"Registered schemas: {len(registry.list_schemas())}")

    # Test validation
    validation_service = ValidationService(registry)

    test_user = {
        "username": "alice",
        "email": "alice@example.com",
        "age": 30,
        "role": "admin"
    }

    result = validation_service.validate(test_user, "User")
    print(f"\nValidation result: {result['valid']}")

    # Test LLM generation
    generator = LLMStructuredGenerator(client, registry)

    generated = generator.generate(
        prompt="Generate a product listing for a laptop",
        schema_name="Product",
        output_format="json"
    )

    if generated["success"]:
        print(f"\nGenerated product:")
        print(json.dumps(generated["data"], indent=2))

    # Test format conversion
    converter = FormatConverter()

    xml_output = converter.dict_to_xml(test_user, "user")
    print(f"\nXML format:")
    print(xml_output[:200] + "...")

    yaml_output = converter.dict_to_yaml(test_user)
    print(f"\nYAML format:")
    print(yaml_output)

    # Generate report
    report = validation_service.get_validation_report()
    print(f"\nValidation Report:")
    print(f"Total validations: {report['total_validations']}")
    print(f"Success rate: {report['success_rate']:.1%}")


def create_api_service():
    """Create a FastAPI service for the schema registry."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Schema Registry API")

    # Global instances
    registry = SchemaRegistry()
    validation_service = ValidationService(registry)

    @app.post("/schemas")
    async def register_schema(schema: SchemaDefinition):
        """Register a new schema."""
        try:
            registry.register_schema(schema)
            return {"message": f"Schema {schema.name} v{schema.version} registered"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/schemas")
    async def list_schemas():
        """List all schemas."""
        return registry.list_schemas()

    @app.get("/schemas/{name}")
    async def get_schema(name: str, version: Optional[str] = None):
        """Get a specific schema."""
        try:
            schema = registry.get_schema(name, version)
            return schema.dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/validate")
    async def validate_data(
        data: Dict,
        schema_name: str,
        schema_version: Optional[str] = None
    ):
        """Validate data against a schema."""
        result = validation_service.validate(data, schema_name, schema_version)
        return result

    @app.get("/validation/report")
    async def get_validation_report(limit: int = 10):
        """Get validation report."""
        return validation_service.get_validation_report(limit)

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Schema Registry System")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--server", action="store_true", help="Start API server")

    args = parser.parse_args()

    if args.demo:
        demo_schema_registry()
    elif args.server:
        import uvicorn
        app = create_api_service()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Schema Registry System")
        print("\nUsage:")
        print("  python schema_registry.py --demo    # Run demo")
        print("  python schema_registry.py --server  # Start API server")