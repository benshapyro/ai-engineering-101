"""
Module 08: Structured Outputs - Solutions

Complete solutions for all exercises demonstrating structured output mastery.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from pydantic import BaseModel, Field, validator, ValidationError
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv
import io
import re
import hashlib
import time
from functools import lru_cache


# ===== Exercise 1 Solution: Schema Designer =====

def solution_1_schema_designer():
    """
    Solution: Design complex business schemas for e-commerce.
    """
    print("Solution 1: Schema Designer")
    print("=" * 50)

    # Define comprehensive schemas
    class OrderItemSchema(BaseModel):
        """Order item with validation."""
        product_id: str = Field(..., regex="^PROD[0-9]{6}$")
        name: str = Field(..., min_length=1, max_length=100)
        quantity: int = Field(..., ge=1, le=100)
        unit_price: float = Field(..., gt=0)
        discount_percent: float = Field(default=0.0, ge=0, le=100)
        subtotal: float = Field(default=0.0)

        @validator('subtotal', always=True)
        def calculate_subtotal(cls, v, values):
            if 'quantity' in values and 'unit_price' in values and 'discount_percent' in values:
                base = values['quantity'] * values['unit_price']
                discount = base * (values['discount_percent'] / 100)
                return round(base - discount, 2)
            return v

    class CustomerSchema(BaseModel):
        """Customer information."""
        customer_id: str = Field(..., regex="^CUST[0-9]{8}$")
        name: str = Field(..., min_length=2, max_length=100)
        email: str = Field(..., regex="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$")
        phone: Optional[str] = Field(None, regex="^\\+?[1-9]\\d{1,14}$")
        loyalty_tier: str = Field(default="bronze")

        @validator('email')
        def lowercase_email(cls, v):
            return v.lower()

    class AddressSchema(BaseModel):
        """Shipping address."""
        street: str = Field(..., min_length=5)
        city: str = Field(..., min_length=2)
        state: str = Field(..., regex="^[A-Z]{2}$")
        zip_code: str = Field(..., regex="^\\d{5}(-\\d{4})?$")
        country: str = Field(default="USA")

    class ShippingMethod(str, Enum):
        STANDARD = "standard"
        EXPRESS = "express"
        OVERNIGHT = "overnight"

    class ShippingSchema(BaseModel):
        """Shipping information."""
        address: AddressSchema
        method: ShippingMethod
        cost: float = Field(..., ge=0)
        estimated_delivery: str = Field(...)

        @validator('estimated_delivery')
        def validate_delivery_date(cls, v):
            try:
                delivery = datetime.strptime(v, "%Y-%m-%d")
                if delivery < datetime.now():
                    raise ValueError("Delivery date cannot be in the past")
                return v
            except ValueError as e:
                if "time data" in str(e):
                    raise ValueError("Date must be in YYYY-MM-DD format")
                raise

    class OrderStatus(str, Enum):
        PENDING = "pending"
        PROCESSING = "processing"
        SHIPPED = "shipped"
        DELIVERED = "delivered"
        CANCELLED = "cancelled"

    class OrderSchema(BaseModel):
        """Complete order with all components."""
        order_id: str = Field(..., regex="^ORD[0-9]{10}$")
        customer: CustomerSchema
        items: List[OrderItemSchema] = Field(..., min_items=1)
        shipping: ShippingSchema
        subtotal: float = Field(default=0.0)
        tax_rate: float = Field(default=0.08, ge=0, le=0.15)
        tax_amount: float = Field(default=0.0)
        total: float = Field(default=0.0)
        status: OrderStatus = Field(default=OrderStatus.PENDING)
        created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
        notes: Optional[str] = None

        @validator('subtotal', always=True)
        def calculate_subtotal(cls, v, values):
            if 'items' in values:
                return sum(item.subtotal for item in values['items'])
            return v

        @validator('tax_amount', always=True)
        def calculate_tax(cls, v, values):
            if 'subtotal' in values and 'tax_rate' in values:
                return round(values['subtotal'] * values['tax_rate'], 2)
            return v

        @validator('total', always=True)
        def calculate_total(cls, v, values):
            if all(k in values for k in ['subtotal', 'tax_amount', 'shipping']):
                return round(
                    values['subtotal'] + values['tax_amount'] + values['shipping'].cost,
                    2
                )
            return v

    # Create sample order data
    sample_order = {
        "order_id": "ORD2024123456",
        "customer": {
            "customer_id": "CUST12345678",
            "name": "John Smith",
            "email": "JOHN.SMITH@EXAMPLE.COM",
            "phone": "+14155551234",
            "loyalty_tier": "gold"
        },
        "items": [
            {
                "product_id": "PROD000001",
                "name": "Laptop",
                "quantity": 1,
                "unit_price": 999.99,
                "discount_percent": 10.0
            },
            {
                "product_id": "PROD000002",
                "name": "Mouse",
                "quantity": 2,
                "unit_price": 29.99,
                "discount_percent": 0.0
            }
        ],
        "shipping": {
            "address": {
                "street": "123 Main Street",
                "city": "San Francisco",
                "state": "CA",
                "zip_code": "94105"
            },
            "method": "express",
            "cost": 19.99,
            "estimated_delivery": "2024-12-25"
        }
    }

    print("VALIDATING ORDER SCHEMA:\n")

    try:
        order = OrderSchema(**sample_order)
        print("✅ Order validated successfully!")
        print(f"\nOrder Summary:")
        print(f"  Order ID: {order.order_id}")
        print(f"  Customer: {order.customer.name} ({order.customer.email})")
        print(f"  Items: {len(order.items)}")
        print(f"  Subtotal: ${order.subtotal:.2f}")
        print(f"  Tax: ${order.tax_amount:.2f}")
        print(f"  Shipping: ${order.shipping.cost:.2f}")
        print(f"  Total: ${order.total:.2f}")
        print(f"  Status: {order.status.value}")

        # Generate JSON
        print("\nJSON OUTPUT:")
        print(order.json(indent=2)[:500] + "...")

    except ValidationError as e:
        print("❌ Validation errors:")
        for error in e.errors():
            print(f"  - {' -> '.join(str(x) for x in error['loc'])}: {error['msg']}")

    # Test with LLM
    client = LLMClient("openai")

    prompt = f"""
Generate a complete e-commerce order as JSON matching this exact schema:

{json.dumps(OrderSchema.schema(), indent=2)[:1000]}...

Include:
- Valid order ID (ORD + 10 digits)
- Customer with valid ID (CUST + 8 digits)
- At least 2 items with valid product IDs (PROD + 6 digits)
- Complete shipping address
- All calculated fields

JSON Output:"""

    print("\n💡 Complex schemas enforce business rules automatically")


# ===== Exercise 2 Solution: Format Converter =====

def solution_2_format_converter():
    """
    Solution: Universal format converter with robust handling.
    """
    print("\nSolution 2: Format Converter")
    print("=" * 50)

    class UniversalFormatConverter:
        """Convert between JSON, XML, YAML, and CSV."""

        @staticmethod
        def json_to_xml(json_data: Dict, root_name: str = "root") -> str:
            """Convert JSON to XML with proper escaping."""
            def dict_to_xml(tag, d):
                elem = ET.Element(tag)

                if isinstance(d, dict):
                    for key, val in d.items():
                        if isinstance(val, list):
                            for item in val:
                                child = dict_to_xml(key, item)
                                elem.append(child)
                        elif isinstance(val, dict):
                            child = dict_to_xml(key, val)
                            elem.append(child)
                        else:
                            child = ET.SubElement(elem, key)
                            child.text = str(val)
                elif isinstance(d, (list, tuple)):
                    for item in d:
                        child = dict_to_xml("item", item)
                        elem.append(child)
                else:
                    elem.text = str(d)

                return elem

            root = dict_to_xml(root_name, json_data)
            rough_string = ET.tostring(root, encoding='unicode')

            # Pretty print
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ")

        @staticmethod
        def xml_to_yaml(xml_string: str) -> str:
            """Convert XML to YAML."""
            root = ET.fromstring(xml_string)

            def xml_to_dict(element):
                result = {}

                # Add attributes
                if element.attrib:
                    result['@attributes'] = dict(element.attrib)

                # Add text content
                if element.text and element.text.strip():
                    if len(element) == 0:  # Leaf node
                        return element.text.strip()
                    else:
                        result['#text'] = element.text.strip()

                # Add children
                children = {}
                for child in element:
                    child_data = xml_to_dict(child)
                    if child.tag in children:
                        if not isinstance(children[child.tag], list):
                            children[child.tag] = [children[child.tag]]
                        children[child.tag].append(child_data)
                    else:
                        children[child.tag] = child_data

                result.update(children)
                return result if result else None

            data = {root.tag: xml_to_dict(root)}
            return yaml.dump(data, default_flow_style=False, sort_keys=False)

        @staticmethod
        def csv_to_json(csv_string: str, has_header: bool = True) -> List[Dict]:
            """Convert CSV to JSON with type inference."""
            reader = csv.reader(io.StringIO(csv_string))

            if has_header:
                headers = next(reader)
            else:
                # Generate headers
                first_row = next(reader)
                headers = [f"column_{i}" for i in range(len(first_row))]
                reader = csv.reader(io.StringIO(csv_string))

            result = []
            for row in reader:
                if has_header and reader.line_num == 1:
                    continue

                record = {}
                for i, value in enumerate(row):
                    if i < len(headers):
                        # Type inference
                        if value.lower() in ('true', 'false'):
                            record[headers[i]] = value.lower() == 'true'
                        elif value.replace('.', '').replace('-', '').isdigit():
                            try:
                                if '.' in value:
                                    record[headers[i]] = float(value)
                                else:
                                    record[headers[i]] = int(value)
                            except ValueError:
                                record[headers[i]] = value
                        else:
                            record[headers[i]] = value

                result.append(record)

            return result

        @staticmethod
        def detect_format(data_string: str) -> str:
            """Detect the format of input data."""
            data_string = data_string.strip()

            # JSON detection
            if (data_string.startswith('{') and data_string.endswith('}')) or \
               (data_string.startswith('[') and data_string.endswith(']')):
                try:
                    json.loads(data_string)
                    return "json"
                except:
                    pass

            # XML detection
            if data_string.startswith('<') and data_string.endswith('>'):
                try:
                    ET.fromstring(data_string)
                    return "xml"
                except:
                    pass

            # YAML detection
            if ':' in data_string and not data_string.startswith('{'):
                try:
                    yaml.safe_load(data_string)
                    return "yaml"
                except:
                    pass

            # CSV detection
            if ',' in data_string or '\t' in data_string:
                lines = data_string.split('\n')
                if len(lines) > 1:
                    # Check if consistent column count
                    counts = [len(line.split(',')) for line in lines if line]
                    if len(set(counts)) == 1:
                        return "csv"

            return "unknown"

    # Test data
    test_json = {
        "product": {
            "id": 123,
            "name": "Widget",
            "price": 29.99,
            "tags": ["new", "featured", "sale"],
            "specs": {
                "weight": "500g",
                "dimensions": "10x10x5"
            }
        }
    }

    test_csv = """id,name,price,in_stock
1,Product A,19.99,true
2,Product B,29.99,false
3,Product C,39.99,true"""

    test_xml = """<product>
  <id>123</id>
  <name>Widget</name>
  <price>29.99</price>
</product>"""

    converter = UniversalFormatConverter()

    print("JSON TO XML CONVERSION:")
    xml_output = converter.json_to_xml(test_json, "catalog")
    print(xml_output[:300])

    print("\nCSV TO JSON CONVERSION:")
    json_from_csv = converter.csv_to_json(test_csv)
    print(json.dumps(json_from_csv, indent=2))

    print("\nFORMAT DETECTION:")
    test_formats = [
        ('{"test": 123}', "JSON"),
        ('<root>test</root>', "XML"),
        ('key: value\nlist:\n  - item1', "YAML"),
        ('col1,col2\nval1,val2', "CSV")
    ]

    for data, expected in test_formats:
        detected = converter.detect_format(data)
        print(f"  {expected}: {detected} {'✅' if detected.lower() == expected.lower() else '❌'}")

    print("\n💡 Universal converter handles all major formats")


# ===== Exercise 3 Solution: Validation Pipeline =====

def solution_3_validation_pipeline():
    """
    Solution: Multi-stage validation pipeline with detailed reporting.
    """
    print("\nSolution 3: Validation Pipeline")
    print("=" * 50)

    class ValidationStage:
        """Individual validation stage."""
        def __init__(self, name: str, validator_func, critical: bool = False):
            self.name = name
            self.validator = validator_func
            self.critical = critical

    class ValidationPipeline:
        """Multi-stage validation with detailed error reporting."""

        def __init__(self):
            self.stages = []
            self.errors = []
            self.warnings = []
            self.passed_stages = []
            self.failed_stages = []

        def add_stage(self, name: str, validator_func, critical: bool = False):
            """Add validation stage."""
            stage = ValidationStage(name, validator_func, critical)
            self.stages.append(stage)

        def validate(self, data: Any) -> bool:
            """Run all validation stages."""
            self.errors = []
            self.warnings = []
            self.passed_stages = []
            self.failed_stages = []

            for stage in self.stages:
                try:
                    result = stage.validator(data)
                    if result:
                        self.passed_stages.append(stage.name)
                    else:
                        self.failed_stages.append(stage.name)
                        self.errors.append(f"{stage.name}: Validation failed")
                        if stage.critical:
                            return False
                except Exception as e:
                    self.failed_stages.append(stage.name)
                    self.errors.append(f"{stage.name}: {str(e)}")
                    if stage.critical:
                        return False

            return len(self.errors) == 0

        def syntax_validator(self, data: str) -> bool:
            """Validate JSON/XML/YAML syntax."""
            # Try JSON
            try:
                json.loads(data)
                return True
            except:
                pass

            # Try XML
            try:
                ET.fromstring(data)
                return True
            except:
                pass

            # Try YAML
            try:
                yaml.safe_load(data)
                return True
            except:
                pass

            raise ValueError("Invalid syntax: not valid JSON, XML, or YAML")

        def schema_validator(self, schema: BaseModel):
            """Create schema validator for Pydantic model."""
            def validator(data: Union[str, Dict]) -> bool:
                if isinstance(data, str):
                    data = json.loads(data)

                try:
                    schema(**data)
                    return True
                except ValidationError as e:
                    errors = []
                    for error in e.errors():
                        field = " -> ".join(str(x) for x in error['loc'])
                        errors.append(f"{field}: {error['msg']}")
                    raise ValueError(f"Schema validation failed: {'; '.join(errors)}")

            return validator

        def business_rules_validator(self, rules: List[callable]):
            """Create business rules validator."""
            def validator(data: Union[str, Dict]) -> bool:
                if isinstance(data, str):
                    data = json.loads(data)

                failed_rules = []
                for rule in rules:
                    try:
                        if not rule(data):
                            failed_rules.append(rule.__name__)
                    except Exception as e:
                        failed_rules.append(f"{rule.__name__}: {str(e)}")

                if failed_rules:
                    raise ValueError(f"Business rules failed: {', '.join(failed_rules)}")
                return True

            return validator

        def get_report(self) -> Dict:
            """Generate detailed validation report."""
            total_stages = len(self.stages)
            passed_count = len(self.passed_stages)
            failed_count = len(self.failed_stages)

            return {
                "summary": {
                    "total_stages": total_stages,
                    "passed": passed_count,
                    "failed": failed_count,
                    "success_rate": f"{(passed_count/total_stages*100):.1f}%" if total_stages > 0 else "0%"
                },
                "passed_stages": self.passed_stages,
                "failed_stages": self.failed_stages,
                "errors": self.errors,
                "warnings": self.warnings,
                "recommendations": self._generate_recommendations()
            }

        def _generate_recommendations(self) -> List[str]:
            """Generate fix recommendations based on errors."""
            recommendations = []

            for error in self.errors:
                if "syntax" in error.lower():
                    recommendations.append("Check for missing quotes, commas, or brackets")
                elif "schema" in error.lower():
                    recommendations.append("Ensure all required fields are present with correct types")
                elif "business" in error.lower():
                    recommendations.append("Review business logic constraints")

            return list(set(recommendations))

    # Create test schema
    class TestSchema(BaseModel):
        name: str = Field(..., min_length=1)
        value: int = Field(..., ge=0)
        active: bool

    # Business rules
    def value_range_rule(data):
        """Value must be between 0 and 1000."""
        return 0 <= data.get('value', 0) <= 1000

    def name_format_rule(data):
        """Name must be alphanumeric."""
        name = data.get('name', '')
        return bool(re.match(r'^[a-zA-Z0-9\s]+$', name))

    # Create pipeline
    pipeline = ValidationPipeline()
    pipeline.add_stage("syntax", pipeline.syntax_validator, critical=True)
    pipeline.add_stage("schema", pipeline.schema_validator(TestSchema), critical=True)
    pipeline.add_stage("business", pipeline.business_rules_validator([value_range_rule, name_format_rule]))

    # Test cases
    test_cases = [
        ('{"name": "Test Product", "value": 100, "active": true}', "Valid"),
        ('{"name": "Test", "value": 100,}', "Invalid JSON"),
        ('{"name": "Test"}', "Missing fields"),
        ('{"name": "Test@#$", "value": 100, "active": true}', "Invalid name format"),
        ('{"name": "Test", "value": 2000, "active": true}', "Value out of range")
    ]

    print("VALIDATION PIPELINE TESTS:\n")

    for data, description in test_cases:
        print(f"Test: {description}")
        print(f"Data: {data[:50]}...")

        is_valid = pipeline.validate(data)
        print(f"Result: {'✅ VALID' if is_valid else '❌ INVALID'}")

        if not is_valid:
            report = pipeline.get_report()
            print(f"Errors: {report['errors'][0] if report['errors'] else 'None'}")
            if report['recommendations']:
                print(f"Recommendation: {report['recommendations'][0]}")
        print()

    print("💡 Pipeline provides stage-by-stage validation with detailed feedback")


# ===== Exercise 4 Solution: Error Recovery =====

def solution_4_error_recovery():
    """
    Solution: Automatic error detection and recovery for structured outputs.
    """
    print("\nSolution 4: Error Recovery")
    print("=" * 50)

    client = LLMClient("openai")

    class StructuredOutputFixer:
        """Automatic error recovery for malformed outputs."""

        def __init__(self, llm_client):
            self.client = llm_client
            self.fix_strategies = [
                self._fix_trailing_commas,
                self._fix_quotes,
                self._fix_missing_commas,
                self._fix_unquoted_keys,
                self._fix_unclosed_brackets
            ]
            self.fix_stats = {
                "attempts": 0,
                "successes": 0,
                "llm_fixes": 0
            }

        def _fix_trailing_commas(self, text: str) -> str:
            """Remove trailing commas."""
            return re.sub(r',(\s*[}\]])', r'\1', text)

        def _fix_quotes(self, text: str) -> str:
            """Convert single quotes to double quotes."""
            # Be careful with apostrophes in values
            text = re.sub(r"(?<=[{\[,:])\s*'([^']*)'(?=\s*[:,}\]])", r'"\1"', text)
            text = re.sub(r"'([^']+)'\s*:", r'"\1":', text)
            return text

        def _fix_missing_commas(self, text: str) -> str:
            """Add missing commas between elements."""
            # Add comma between "}" and "{"
            text = re.sub(r'}\s*{', r'},{', text)
            # Add comma between "]" and "["
            text = re.sub(r']\s*\[', r'],[', text)
            # Add comma between string values
            text = re.sub(r'"\s+"', r'","', text)
            return text

        def _fix_unquoted_keys(self, text: str) -> str:
            """Quote unquoted object keys."""
            # Match unquoted keys before colons
            text = re.sub(r'(\w+)(\s*):', r'"\1"\2:', text)
            # Fix boolean values that got quoted
            text = re.sub(r'"(true|false)"', r'\1', text)
            text = re.sub(r'"(null)"', r'\1', text)
            # Fix numbers that got quoted
            text = re.sub(r'"(\d+\.?\d*)"', r'\1', text)
            return text

        def _fix_unclosed_brackets(self, text: str) -> str:
            """Balance brackets."""
            open_braces = text.count('{')
            close_braces = text.count('}')
            open_brackets = text.count('[')
            close_brackets = text.count(']')

            # Add missing closing brackets
            if open_braces > close_braces:
                text += '}' * (open_braces - close_braces)
            if open_brackets > close_brackets:
                text += ']' * (open_brackets - close_brackets)

            return text

        def fix_json(self, malformed_json: str) -> Optional[Dict]:
            """Fix malformed JSON using multiple strategies."""
            self.fix_stats["attempts"] += 1

            # Try each fix strategy
            current = malformed_json
            for strategy in self.fix_strategies:
                current = strategy(current)

                # Test if valid JSON now
                try:
                    result = json.loads(current)
                    self.fix_stats["successes"] += 1
                    return result
                except json.JSONDecodeError:
                    continue

            # If still broken, try LLM
            return self.fix_with_llm(malformed_json, "JSON")

        def fix_with_llm(self, malformed_data: str, expected_format: str) -> Optional[Dict]:
            """Use LLM to fix malformed data."""
            prompt = f"""Fix this malformed {expected_format} and return ONLY valid {expected_format}:

Malformed data:
{malformed_data}

Common issues to fix:
- Trailing commas
- Missing quotes or commas
- Unbalanced brackets
- Wrong quote types

Return ONLY the fixed {expected_format}:"""

            try:
                response = self.client.complete(prompt, temperature=0.1, max_tokens=500)

                # Extract JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    fixed = json.loads(json_match.group())
                    self.fix_stats["llm_fixes"] += 1
                    self.fix_stats["successes"] += 1
                    return fixed
            except:
                pass

            return None

        def validate_and_retry(self, data: str, validator, max_retries: int = 3):
            """Validate and retry with fixes."""
            for attempt in range(max_retries):
                try:
                    # Try to validate
                    parsed = json.loads(data) if isinstance(data, str) else data
                    if validator(parsed):
                        return parsed
                except (json.JSONDecodeError, ValidationError) as e:
                    if attempt < max_retries - 1:
                        # Try to fix
                        fixed = self.fix_json(data) if isinstance(data, str) else None
                        if fixed:
                            data = fixed
                        else:
                            break

            return None

        def get_stats(self) -> Dict:
            """Get fix statistics."""
            success_rate = (
                self.fix_stats["successes"] / self.fix_stats["attempts"] * 100
                if self.fix_stats["attempts"] > 0 else 0
            )
            return {
                **self.fix_stats,
                "success_rate": f"{success_rate:.1f}%",
                "llm_fix_rate": f"{(self.fix_stats['llm_fixes'] / max(self.fix_stats['attempts'], 1) * 100):.1f}%"
            }

    fixer = StructuredOutputFixer(client)

    # Test cases with various errors
    malformed_examples = [
        ('{"name": "Test", "value": 100,}', "Trailing comma"),
        ("{'name': 'Test', 'value': 100}", "Single quotes"),
        ('{"name": "Test" "value": 100}', "Missing comma"),
        ('{name: "Test", value: 100}', "Unquoted keys"),
        ('{"name": "Test", "nested": {"value": 100}', "Unclosed bracket"),
        ('{"items": [1, 2, 3,], "total": 6}', "Array trailing comma")
    ]

    print("TESTING ERROR RECOVERY:\n")

    for malformed, error_type in malformed_examples:
        print(f"Error type: {error_type}")
        print(f"Malformed: {malformed}")

        fixed = fixer.fix_json(malformed)
        if fixed:
            print(f"✅ Fixed: {json.dumps(fixed)}")
        else:
            print(f"❌ Could not fix")
        print()

    print("FIX STATISTICS:")
    stats = fixer.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n💡 Multi-strategy approach maximizes recovery success")


# ===== Exercise 5 Solution: Performance Optimizer =====

def solution_5_performance_optimizer():
    """
    Solution: Optimize structured output generation performance.
    """
    print("\nSolution 5: Performance Optimizer")
    print("=" * 50)

    class PerformanceOptimizer:
        """Performance optimizations for structured output."""

        def __init__(self):
            self.schema_cache = {}
            self.template_cache = {}
            self.validation_cache = {}
            self.metrics = []
            self.cache_hits = 0
            self.cache_misses = 0

        @lru_cache(maxsize=128)
        def get_cached_schema(self, schema_key: str):
            """Cache and retrieve schemas."""
            self.cache_hits += 1

            if schema_key not in self.schema_cache:
                self.cache_misses += 1
                # Generate schema (simulate expensive operation)
                time.sleep(0.01)  # Simulate generation time

                # Create schema based on key
                if schema_key == "user":
                    schema = {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "email": {"type": "string"}
                        }
                    }
                else:
                    schema = {"type": "object"}

                self.schema_cache[schema_key] = schema

            return self.schema_cache[schema_key]

        def create_template(self, schema: Union[Dict, BaseModel]) -> str:
            """Create reusable prompt template from schema."""
            # Generate cache key
            if isinstance(schema, BaseModel):
                cache_key = schema.__class__.__name__
                schema_dict = schema.schema()
            else:
                cache_key = hashlib.md5(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:8]
                schema_dict = schema

            if cache_key in self.template_cache:
                self.cache_hits += 1
                return self.template_cache[cache_key]

            self.cache_misses += 1

            # Generate template
            template = f"""Generate data matching this exact JSON schema:

{json.dumps(schema_dict, indent=2)}

Rules:
- All required fields must be present
- Types must match exactly
- Use realistic sample data

JSON Output:"""

            self.template_cache[cache_key] = template
            return template

        def fast_validate(self, data: Dict, schema_key: str) -> bool:
            """Optimized validation with caching."""
            # Create cache key from data + schema
            data_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]
            cache_key = f"{schema_key}:{data_hash}"

            if cache_key in self.validation_cache:
                self.cache_hits += 1
                return self.validation_cache[cache_key]

            self.cache_misses += 1

            # Perform validation (simulate)
            time.sleep(0.005)  # Simulate validation time

            # Simple validation
            is_valid = all(k in data for k in ['id', 'name']) if schema_key == "user" else True

            self.validation_cache[cache_key] = is_valid
            return is_valid

        def stream_structured_output(self, prompt: str, schema: Dict):
            """Stream and parse structured output incrementally."""
            # Simulate streaming chunks
            response_chunks = [
                '{"id": "123",',
                ' "name": "Test User",',
                ' "email": "test@example.com",',
                ' "metadata": {',
                '   "created": "2024-01-01",',
                '   "updated": "2024-01-02"',
                ' }}'
            ]

            buffer = ""
            partial_results = []

            for chunk in response_chunks:
                buffer += chunk

                # Try to parse partial JSON
                if buffer.count('{') == buffer.count('}') and buffer.strip():
                    try:
                        result = json.loads(buffer)
                        partial_results.append(result)
                        yield result
                        buffer = ""
                    except json.JSONDecodeError:
                        continue

        def benchmark(self, func, *args, **kwargs):
            """Benchmark function performance."""
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start

            self.metrics.append({
                "function": func.__name__,
                "time": elapsed,
                "timestamp": datetime.now()
            })

            return result, elapsed

        def get_performance_report(self) -> Dict:
            """Generate performance report."""
            if not self.metrics:
                return {"message": "No metrics collected"}

            # Calculate statistics
            function_times = {}
            for metric in self.metrics:
                func = metric["function"]
                if func not in function_times:
                    function_times[func] = []
                function_times[func].append(metric["time"])

            report = {
                "cache_performance": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate": f"{(self.cache_hits / max(self.cache_hits + self.cache_misses, 1) * 100):.1f}%"
                },
                "function_performance": {}
            }

            for func, times in function_times.items():
                report["function_performance"][func] = {
                    "calls": len(times),
                    "avg_time": f"{sum(times) / len(times):.4f}s",
                    "min_time": f"{min(times):.4f}s",
                    "max_time": f"{max(times):.4f}s"
                }

            return report

    optimizer = PerformanceOptimizer()

    print("PERFORMANCE OPTIMIZATION TESTS:\n")

    # Test schema caching
    print("1. SCHEMA CACHING:")
    for i in range(3):
        _, time_taken = optimizer.benchmark(optimizer.get_cached_schema, "user")
        print(f"  Call {i+1}: {time_taken:.4f}s")

    # Test template caching
    print("\n2. TEMPLATE CACHING:")
    test_schema = {"type": "object", "properties": {"id": {"type": "string"}}}
    for i in range(3):
        _, time_taken = optimizer.benchmark(optimizer.create_template, test_schema)
        print(f"  Call {i+1}: {time_taken:.4f}s")

    # Test validation caching
    print("\n3. VALIDATION CACHING:")
    test_data = {"id": "123", "name": "Test"}
    for i in range(3):
        _, time_taken = optimizer.benchmark(optimizer.fast_validate, test_data, "user")
        print(f"  Call {i+1}: {time_taken:.4f}s")

    # Test streaming
    print("\n4. STREAMING OUTPUT:")
    for partial in optimizer.stream_structured_output("test prompt", {}):
        print(f"  Received partial: {list(partial.keys())}")

    # Generate report
    print("\nPERFORMANCE REPORT:")
    report = optimizer.get_performance_report()
    print(json.dumps(report, indent=2))

    print("\n💡 Caching and streaming significantly improve performance")


# ===== Challenge Solution: Complete Structured Output System =====

def challenge_solution_structured_output_system():
    """
    Challenge Solution: Production-ready structured output system.
    """
    print("\nChallenge: Complete Structured Output System")
    print("=" * 50)

    client = LLMClient("openai")

    class StructuredOutputSystem:
        """Complete structured output system with all features."""

        def __init__(self, llm_client):
            self.client = llm_client
            self.schemas = {}
            self.format_handlers = {
                "json": self._handle_json,
                "xml": self._handle_xml,
                "yaml": self._handle_yaml,
                "csv": self._handle_csv
            }
            self.cache = {}
            self.metrics = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "retries": 0,
                "formats": {}
            }
            self.fixer = StructuredOutputFixer(llm_client)

        def register_schema(self, name: str, schema: BaseModel):
            """Register a reusable schema."""
            self.schemas[name] = schema
            print(f"✓ Registered schema: {name}")

        def register_format_handler(self, format_name: str, handler):
            """Register custom format handler."""
            self.format_handlers[format_name] = handler
            print(f"✓ Registered handler: {format_name}")

        def generate(self,
                    prompt: str,
                    schema_name: Optional[str] = None,
                    format: str = "json",
                    validate: bool = True,
                    retry: bool = True,
                    max_retries: int = 3) -> Any:
            """Generate structured output with all features."""
            self.metrics["requests"] += 1
            self.metrics["formats"][format] = self.metrics["formats"].get(format, 0) + 1

            # Get schema if specified
            schema = self.schemas.get(schema_name) if schema_name else None

            # Build enhanced prompt
            enhanced_prompt = self._build_prompt(prompt, schema, format)

            # Try generation with retries
            for attempt in range(max_retries if retry else 1):
                try:
                    # Generate response
                    response = self.client.complete(enhanced_prompt, temperature=0.1, max_tokens=500)

                    # Parse based on format
                    parsed = self._parse_response(response, format)

                    # Validate if requested
                    if validate and schema:
                        validated = self._validate_output(parsed, schema)
                        if not validated:
                            raise ValidationError("Schema validation failed")

                    self.metrics["successes"] += 1
                    return parsed

                except Exception as e:
                    self.metrics["retries"] += 1
                    if attempt == max_retries - 1:
                        self.metrics["failures"] += 1
                        print(f"❌ Generation failed: {e}")
                        return None

                    # Enhance prompt with error feedback
                    enhanced_prompt += f"\n\nError in attempt {attempt + 1}: {str(e)}\nPlease fix and try again:"

        def _build_prompt(self, base_prompt: str, schema: Optional[BaseModel], format: str) -> str:
            """Build enhanced prompt with format and schema."""
            prompt = base_prompt

            if schema:
                prompt += f"\n\nReturn data matching this schema:\n{json.dumps(schema.schema(), indent=2)[:500]}"

            format_instructions = {
                "json": "\n\nReturn as valid JSON with proper formatting.",
                "xml": "\n\nReturn as well-formed XML with proper tags.",
                "yaml": "\n\nReturn as valid YAML with proper indentation.",
                "csv": "\n\nReturn as CSV with headers."
            }

            prompt += format_instructions.get(format, "")
            return prompt

        def _parse_response(self, response: str, format: str) -> Any:
            """Parse response based on format."""
            handler = self.format_handlers.get(format)
            if handler:
                return handler(response)
            raise ValueError(f"Unsupported format: {format}")

        def _handle_json(self, response: str) -> Dict:
            """Handle JSON format."""
            # Try to extract JSON
            json_match = re.search(r'\{.*\}|\[.*\]', response, re.DOTALL)
            if json_match:
                # Try to fix if broken
                fixed = self.fixer.fix_json(json_match.group())
                if fixed:
                    return fixed
                return json.loads(json_match.group())
            raise ValueError("No valid JSON found")

        def _handle_xml(self, response: str) -> ET.Element:
            """Handle XML format."""
            xml_match = re.search(r'<[^>]+>.*</[^>]+>', response, re.DOTALL)
            if xml_match:
                return ET.fromstring(xml_match.group())
            raise ValueError("No valid XML found")

        def _handle_yaml(self, response: str) -> Dict:
            """Handle YAML format."""
            # Extract YAML section
            yaml_match = re.search(r'^---\n(.*?)\n---$|^((?:^\w+:.*$\n?)+)',
                                 response, re.MULTILINE | re.DOTALL)
            if yaml_match:
                yaml_str = yaml_match.group(1) or yaml_match.group(2)
                return yaml.safe_load(yaml_str)
            return yaml.safe_load(response)

        def _handle_csv(self, response: str) -> List[Dict]:
            """Handle CSV format."""
            csv_reader = csv.DictReader(io.StringIO(response))
            return list(csv_reader)

        def _validate_output(self, data: Any, schema: BaseModel) -> bool:
            """Validate output against schema."""
            try:
                if isinstance(data, dict):
                    schema(**data)
                elif isinstance(data, list) and data:
                    for item in data:
                        schema(**item)
                return True
            except ValidationError:
                return False

        def validate_output(self, data: Any, schema_name: str) -> bool:
            """Public validation method."""
            schema = self.schemas.get(schema_name)
            if not schema:
                return False
            return self._validate_output(data, schema)

        def convert_format(self, data: Any, from_format: str, to_format: str) -> Any:
            """Convert between formats."""
            # Convert to common format (dict)
            if from_format == "xml":
                # XML to dict conversion
                common = {}  # Simplified
            elif from_format == "yaml":
                common = data if isinstance(data, dict) else yaml.safe_load(str(data))
            elif from_format == "csv":
                common = data if isinstance(data, list) else [data]
            else:
                common = data

            # Convert to target format
            if to_format == "json":
                return json.dumps(common, indent=2)
            elif to_format == "xml":
                return UniversalFormatConverter.json_to_xml(common)
            elif to_format == "yaml":
                return yaml.dump(common)
            elif to_format == "csv":
                if isinstance(common, list) and common:
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=common[0].keys())
                    writer.writeheader()
                    writer.writerows(common)
                    return output.getvalue()

            return common

        def negotiate_format(self, requirements: Dict) -> str:
            """Select best format for requirements."""
            scores = {
                "json": 0,
                "xml": 0,
                "yaml": 0,
                "csv": 0
            }

            # Score based on requirements
            if requirements.get("nested_data"):
                scores["json"] += 2
                scores["xml"] += 1
                scores["yaml"] += 2
                scores["csv"] -= 2

            if requirements.get("human_readable"):
                scores["yaml"] += 2
                scores["csv"] += 1

            if requirements.get("api_compatible"):
                scores["json"] += 3

            if requirements.get("tabular"):
                scores["csv"] += 3

            return max(scores, key=scores.get)

        def get_analytics(self) -> Dict:
            """Return comprehensive analytics."""
            success_rate = (
                self.metrics["successes"] / self.metrics["requests"] * 100
                if self.metrics["requests"] > 0 else 0
            )

            retry_rate = (
                self.metrics["retries"] / self.metrics["requests"] * 100
                if self.metrics["requests"] > 0 else 0
            )

            return {
                **self.metrics,
                "success_rate": f"{success_rate:.1f}%",
                "retry_rate": f"{retry_rate:.1f}%",
                "most_used_format": max(self.metrics["formats"], key=self.metrics["formats"].get)
                                   if self.metrics["formats"] else "none"
            }

    # Create system
    system = StructuredOutputSystem(client)

    # Register schemas
    class UserSchema(BaseModel):
        id: str
        name: str
        email: str
        active: bool = True

    class ProductSchema(BaseModel):
        id: str
        name: str
        price: float
        in_stock: bool

    system.register_schema("user", UserSchema)
    system.register_schema("product", ProductSchema)

    print("\nSYSTEM DEMONSTRATION:\n")

    # Test JSON generation
    print("1. JSON GENERATION WITH VALIDATION:")
    user_data = system.generate(
        "Generate a user record for John Smith",
        schema_name="user",
        format="json",
        validate=True
    )
    if user_data:
        print(f"✅ Generated: {json.dumps(user_data, indent=2)}")

    # Test format negotiation
    print("\n2. FORMAT NEGOTIATION:")
    requirements = {
        "nested_data": True,
        "api_compatible": True
    }
    best_format = system.negotiate_format(requirements)
    print(f"Best format for API with nested data: {best_format}")

    # Test format conversion
    print("\n3. FORMAT CONVERSION:")
    if user_data:
        yaml_output = system.convert_format(user_data, "json", "yaml")
        print(f"Converted to YAML:\n{yaml_output[:100]}...")

    # Get analytics
    print("\n4. SYSTEM ANALYTICS:")
    analytics = system.get_analytics()
    for key, value in analytics.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")

    print("\n💡 Complete system handles all structured output needs")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 08: Solutions")
    parser.add_argument("--solution", type=int, help="Run specific solution (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge solution")
    parser.add_argument("--all", action="store_true", help="Run all solutions")

    args = parser.parse_args()

    solutions = {
        1: solution_1_schema_designer,
        2: solution_2_format_converter,
        3: solution_3_validation_pipeline,
        4: solution_4_error_recovery,
        5: solution_5_performance_optimizer
    }

    if args.all:
        for sol_num, sol_func in solutions.items():
            print(f"\n{'='*60}")
            sol_func()
            print(f"\n{'='*60}\n")
        challenge_solution_structured_output_system()
    elif args.challenge:
        challenge_solution_structured_output_system()
    elif args.solution and args.solution in solutions:
        solutions[args.solution]()
    else:
        print("Module 08: Structured Outputs - Solutions")
        print("\nUsage:")
        print("  python solutions.py --solution N  # Run solution N")
        print("  python solutions.py --challenge    # Run challenge solution")
        print("  python solutions.py --all          # Run all solutions")
        print("\nAvailable solutions:")
        print("  1: Schema Designer")
        print("  2: Format Converter")
        print("  3: Validation Pipeline")
        print("  4: Error Recovery")
        print("  5: Performance Optimizer")
        print("  Challenge: Structured Output System")