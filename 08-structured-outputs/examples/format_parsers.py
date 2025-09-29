"""
Module 08: Format Parsers

Parse and generate various structured output formats beyond JSON.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv
import io
from typing import Dict, List, Any, Optional
import re


def example_1_xml_generation_and_parsing():
    """Generate and parse XML outputs."""
    print("=" * 60)
    print("Example 1: XML Generation and Parsing")
    print("=" * 60)

    client = LLMClient("openai")

    prompt = """
Convert this product information to XML:

Product: Laptop Pro X1
Price: $1299.99
Specs:
- CPU: Intel i7
- RAM: 16GB
- Storage: 512GB SSD
In stock: Yes
Categories: Electronics, Computers

Return as well-formed XML with proper structure:
<product>
    <name>...</name>
    <price currency="USD">...</price>
    <specifications>
        <spec name="..." value="..."/>
    </specifications>
    <stock>...</stock>
    <categories>
        <category>...</category>
    </categories>
</product>
"""

    response = client.complete(prompt, temperature=0.1, max_tokens=300)

    print("LLM XML RESPONSE:")
    print(response)

    # Try to parse XML
    try:
        # Extract XML from response
        xml_match = re.search(r'<product>.*</product>', response, re.DOTALL)
        if xml_match:
            xml_str = xml_match.group()

            # Parse XML
            root = ET.fromstring(xml_str)

            print("\nPARSED XML STRUCTURE:")
            print(f"Product name: {root.find('name').text if root.find('name') is not None else 'N/A'}")

            # Extract specifications
            specs = root.find('specifications')
            if specs is not None:
                print("Specifications:")
                for spec in specs.findall('spec'):
                    name = spec.get('name')
                    value = spec.get('value')
                    print(f"  - {name}: {value}")

            # Pretty print XML
            print("\nFORMATTED XML:")
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
            print(pretty_xml[:500])

    except ET.ParseError as e:
        print(f"❌ XML parsing error: {e}")

    print("\n💡 XML provides hierarchical structure with attributes")


def example_2_yaml_generation():
    """Generate and parse YAML outputs."""
    print("\n" + "=" * 60)
    print("Example 2: YAML Generation and Parsing")
    print("=" * 60)

    client = LLMClient("openai")

    prompt = """
Convert this configuration to YAML format:

Application: WebServer
Environment: Production
Database:
  - Type: PostgreSQL
  - Host: db.example.com
  - Port: 5432
  - Credentials:
    - Username: app_user
    - Password: [ENCRYPTED]
Features:
  - Caching: enabled
  - Logging: verbose
  - SSL: required

Return as valid YAML with proper indentation:
"""

    # Simulate YAML response
    yaml_response = """
application: WebServer
environment: production

database:
  type: PostgreSQL
  host: db.example.com
  port: 5432
  credentials:
    username: app_user
    password: "[ENCRYPTED]"

features:
  caching: enabled
  logging: verbose
  ssl: required

settings:
  max_connections: 100
  timeout_seconds: 30
  retry_attempts: 3
"""

    print("YAML RESPONSE:")
    print(yaml_response)

    try:
        # Parse YAML
        config = yaml.safe_load(yaml_response)

        print("\nPARSED YAML TO PYTHON:")
        print(json.dumps(config, indent=2))

        # Access nested values
        print("\nACCESSING YAML VALUES:")
        print(f"Application: {config.get('application')}")
        print(f"Database type: {config.get('database', {}).get('type')}")
        print(f"SSL required: {config.get('features', {}).get('ssl')}")

        # Convert back to YAML
        print("\nCONVERT BACK TO YAML:")
        yaml_output = yaml.dump(config, default_flow_style=False, sort_keys=False)
        print(yaml_output[:200])

    except yaml.YAMLError as e:
        print(f"❌ YAML parsing error: {e}")

    print("\n💡 YAML is human-readable and great for configs")


def example_3_csv_generation():
    """Generate and parse CSV outputs."""
    print("\n" + "=" * 60)
    print("Example 3: CSV Generation and Parsing")
    print("=" * 60)

    client = LLMClient("openai")

    prompt = """
Extract the following data as CSV:

Sales Report:
- John Smith: Q1=$50000, Q2=$55000, Q3=$48000, Q4=$62000
- Jane Doe: Q1=$45000, Q2=$52000, Q3=$58000, Q4=$61000
- Bob Johnson: Q1=$42000, Q2=$44000, Q3=$46000, Q4=$50000

Return as CSV with headers:
Name,Q1,Q2,Q3,Q4,Total
"""

    # Simulate CSV response
    csv_response = """Name,Q1,Q2,Q3,Q4,Total
John Smith,50000,55000,48000,62000,215000
Jane Doe,45000,52000,58000,61000,216000
Bob Johnson,42000,44000,46000,50000,182000"""

    print("CSV RESPONSE:")
    print(csv_response)

    # Parse CSV
    csv_reader = csv.DictReader(io.StringIO(csv_response))

    print("\nPARSED CSV DATA:")
    data = []
    for row in csv_reader:
        data.append(row)
        print(f"  {row['Name']}: Total = ${row['Total']}")

    # Analyze data
    print("\nDATA ANALYSIS:")
    q1_total = sum(int(row['Q1']) for row in data)
    print(f"Q1 Total: ${q1_total:,}")

    # Generate new CSV
    print("\nGENERATING NEW CSV:")
    output = io.StringIO()
    fieldnames = ['Name', 'Average', 'Best_Quarter']
    writer = csv.DictWriter(output, fieldnames=fieldnames)

    writer.writeheader()
    for row in data:
        quarters = [int(row[f'Q{i}']) for i in range(1, 5)]
        avg = sum(quarters) / len(quarters)
        best = max(quarters)
        writer.writerow({
            'Name': row['Name'],
            'Average': f'{avg:.0f}',
            'Best_Quarter': f'{best}'
        })

    print(output.getvalue())

    print("💡 CSV is perfect for tabular data")


def example_4_markdown_tables():
    """Generate and parse Markdown table format."""
    print("\n" + "=" * 60)
    print("Example 4: Markdown Table Generation")
    print("=" * 60)

    client = LLMClient("openai")

    def parse_markdown_table(md_table: str) -> List[Dict]:
        """Parse a Markdown table into structured data."""
        lines = md_table.strip().split('\n')
        data = []

        # Find header line
        header_line = None
        for i, line in enumerate(lines):
            if '|' in line and not line.strip().startswith('|---'):
                header_line = i
                break

        if header_line is None:
            return data

        # Parse headers
        headers = [h.strip() for h in lines[header_line].split('|') if h.strip()]

        # Parse data rows
        for line in lines[header_line + 2:]:  # Skip header and separator
            if '|' in line and not line.strip().startswith('|---'):
                values = [v.strip() for v in line.split('|') if v.strip()]
                if len(values) == len(headers):
                    data.append(dict(zip(headers, values)))

        return data

    # Generate Markdown table
    prompt = """
Create a Markdown table of programming languages:

Include: Language, Year Created, Creator, Paradigm, Popular Use Case

Include at least 5 languages.

Return as a properly formatted Markdown table.
"""

    # Simulate Markdown response
    markdown_response = """
| Language | Year | Creator | Paradigm | Use Case |
|----------|------|---------|----------|----------|
| Python | 1991 | Guido van Rossum | Multi-paradigm | Data Science, Web |
| JavaScript | 1995 | Brendan Eich | Multi-paradigm | Web Development |
| Rust | 2010 | Graydon Hoare | Systems | Systems Programming |
| Go | 2009 | Google | Concurrent | Cloud Services |
| Swift | 2014 | Apple | Multi-paradigm | iOS Development |
"""

    print("MARKDOWN TABLE:")
    print(markdown_response)

    # Parse table
    parsed_data = parse_markdown_table(markdown_response)

    print("\nPARSED TABLE DATA:")
    for row in parsed_data:
        print(f"  {row.get('Language', 'N/A')}: {row.get('Use Case', 'N/A')}")

    # Convert to other formats
    print("\nCONVERT TO JSON:")
    print(json.dumps(parsed_data[:2], indent=2))

    print("\n💡 Markdown tables are readable and parseable")


def example_5_custom_format_dsl():
    """Create a custom Domain-Specific Language format."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Format DSL")
    print("=" * 60)

    client = LLMClient("openai")

    # Define custom format
    custom_format_spec = """
RECIPE FORMAT:
@recipe: [Recipe Name]
@serves: [number]
@time: [duration]
@difficulty: [easy|medium|hard]

#ingredients:
- [amount] [unit] [ingredient]

#steps:
1. [instruction]
2. [instruction]

#notes:
[optional notes]
"""

    prompt = f"""
Convert this recipe to the custom format:

Chocolate Chip Cookies
Makes 24 cookies, takes 30 minutes, easy difficulty

Ingredients:
- 2 cups flour
- 1 cup butter
- 3/4 cup sugar
- 2 eggs
- 1 tsp vanilla
- 2 cups chocolate chips

Steps:
1. Preheat oven to 375°F
2. Cream butter and sugar
3. Add eggs and vanilla
4. Mix in flour
5. Fold in chocolate chips
6. Bake for 10-12 minutes

Note: Best served warm

Use this format:
{custom_format_spec}
"""

    # Simulate custom format response
    custom_response = """@recipe: Chocolate Chip Cookies
@serves: 24
@time: 30 minutes
@difficulty: easy

#ingredients:
- 2 cups flour
- 1 cup butter
- 3/4 cup sugar
- 2 large eggs
- 1 tsp vanilla extract
- 2 cups chocolate chips

#steps:
1. Preheat oven to 375°F
2. Cream butter and sugar until fluffy
3. Beat in eggs and vanilla
4. Gradually mix in flour
5. Fold in chocolate chips
6. Drop by spoonfuls onto baking sheet
7. Bake for 10-12 minutes until golden

#notes:
Best served warm with milk"""

    print("CUSTOM FORMAT OUTPUT:")
    print(custom_response)

    # Parse custom format
    def parse_recipe_format(text: str) -> Dict:
        """Parse custom recipe format."""
        recipe = {
            'metadata': {},
            'ingredients': [],
            'steps': [],
            'notes': ''
        }

        lines = text.strip().split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith('@'):
                # Metadata
                key, value = line.split(':', 1)
                recipe['metadata'][key[1:]] = value.strip()

            elif line.startswith('#'):
                # Section header
                current_section = line[1:].rstrip(':')

            elif current_section == 'ingredients' and line.startswith('-'):
                recipe['ingredients'].append(line[1:].strip())

            elif current_section == 'steps' and re.match(r'^\d+\.', line):
                recipe['steps'].append(re.sub(r'^\d+\.\s*', '', line))

            elif current_section == 'notes' and line:
                recipe['notes'] += line + ' '

        return recipe

    parsed_recipe = parse_recipe_format(custom_response)

    print("\nPARSED CUSTOM FORMAT:")
    print(f"Recipe: {parsed_recipe['metadata'].get('recipe')}")
    print(f"Difficulty: {parsed_recipe['metadata'].get('difficulty')}")
    print(f"Ingredients: {len(parsed_recipe['ingredients'])} items")
    print(f"Steps: {len(parsed_recipe['steps'])} steps")

    print("\n💡 Custom formats can match domain requirements")


def example_6_format_conversion():
    """Convert between different formats."""
    print("\n" + "=" * 60)
    print("Example 6: Format Conversion")
    print("=" * 60)

    class FormatConverter:
        """Convert between different structured formats."""

        @staticmethod
        def json_to_xml(data: Dict, root_name: str = "root") -> str:
            """Convert JSON/dict to XML."""
            root = ET.Element(root_name)

            def build_xml(parent, obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, list):
                            list_elem = ET.SubElement(parent, key)
                            for item in value:
                                item_elem = ET.SubElement(list_elem, "item")
                                build_xml(item_elem, item)
                        elif isinstance(value, dict):
                            sub_elem = ET.SubElement(parent, key)
                            build_xml(sub_elem, value)
                        else:
                            elem = ET.SubElement(parent, key)
                            elem.text = str(value)
                else:
                    parent.text = str(obj)

            build_xml(root, data)
            return ET.tostring(root, encoding='unicode')

        @staticmethod
        def json_to_yaml(data: Dict) -> str:
            """Convert JSON/dict to YAML."""
            return yaml.dump(data, default_flow_style=False)

        @staticmethod
        def xml_to_json(xml_str: str) -> Dict:
            """Convert XML to JSON/dict."""
            root = ET.fromstring(xml_str)

            def parse_element(elem):
                result = {}
                # Add attributes
                if elem.attrib:
                    result['@attributes'] = elem.attrib

                # Add text content
                if elem.text and elem.text.strip():
                    if len(elem) == 0:  # Leaf node
                        return elem.text.strip()
                    else:
                        result['#text'] = elem.text.strip()

                # Add children
                for child in elem:
                    child_data = parse_element(child)
                    if child.tag in result:
                        # Convert to list if multiple elements
                        if not isinstance(result[child.tag], list):
                            result[child.tag] = [result[child.tag]]
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = child_data

                return result if result else None

            return {root.tag: parse_element(root)}

        @staticmethod
        def yaml_to_json(yaml_str: str) -> str:
            """Convert YAML to JSON."""
            data = yaml.safe_load(yaml_str)
            return json.dumps(data, indent=2)

    # Test data
    test_data = {
        "user": {
            "id": 123,
            "name": "John Doe",
            "email": "john@example.com",
            "roles": ["admin", "user"],
            "settings": {
                "theme": "dark",
                "notifications": True
            }
        }
    }

    converter = FormatConverter()

    print("ORIGINAL JSON:")
    print(json.dumps(test_data, indent=2))

    print("\nCONVERTED TO XML:")
    xml_output = converter.json_to_xml(test_data, "data")
    pretty_xml = minidom.parseString(xml_output).toprettyxml(indent="  ")
    print(pretty_xml[:300])

    print("\nCONVERTED TO YAML:")
    yaml_output = converter.json_to_yaml(test_data)
    print(yaml_output)

    # Convert back
    print("\nXML BACK TO JSON:")
    json_from_xml = converter.xml_to_json(xml_output)
    print(json.dumps(json_from_xml, indent=2)[:200])

    print("\n💡 Format conversion enables interoperability")


def example_7_format_negotiation():
    """Automatically select best format for use case."""
    print("\n" + "=" * 60)
    print("Example 7: Format Negotiation")
    print("=" * 60)

    client = LLMClient("openai")

    class FormatNegotiator:
        """Select optimal format based on use case."""

        FORMAT_CHARACTERISTICS = {
            "json": {
                "strengths": ["parsing", "apis", "nested data", "type support"],
                "weaknesses": ["human readability", "comments"],
                "use_cases": ["apis", "data exchange", "configuration"]
            },
            "yaml": {
                "strengths": ["readability", "configuration", "comments"],
                "weaknesses": ["parsing complexity", "ambiguity"],
                "use_cases": ["configuration", "docker", "kubernetes"]
            },
            "xml": {
                "strengths": ["validation", "namespaces", "attributes"],
                "weaknesses": ["verbosity", "parsing overhead"],
                "use_cases": ["documents", "soap", "enterprise"]
            },
            "csv": {
                "strengths": ["tabular data", "excel compatible", "simple"],
                "weaknesses": ["no nesting", "type ambiguity"],
                "use_cases": ["reports", "data export", "spreadsheets"]
            },
            "markdown": {
                "strengths": ["documentation", "readability", "version control"],
                "weaknesses": ["limited structure", "no validation"],
                "use_cases": ["documentation", "readme", "notes"]
            }
        }

        @classmethod
        def recommend_format(cls, requirements: Dict[str, Any]) -> str:
            """Recommend best format based on requirements."""
            scores = {}

            for format_name, chars in cls.FORMAT_CHARACTERISTICS.items():
                score = 0

                # Check use case match
                if requirements.get("use_case") in chars["use_cases"]:
                    score += 3

                # Check required features
                for feature in requirements.get("features", []):
                    if feature in chars["strengths"]:
                        score += 2
                    elif feature in chars["weaknesses"]:
                        score -= 1

                # Check constraints
                if requirements.get("human_readable") and "readability" in chars["strengths"]:
                    score += 2

                if requirements.get("machine_parseable") and "parsing" in chars["strengths"]:
                    score += 2

                scores[format_name] = score

            return max(scores, key=scores.get)

        @classmethod
        def generate_in_format(cls, data: Any, format_name: str) -> str:
            """Generate output in specified format."""
            if format_name == "json":
                return json.dumps(data, indent=2)
            elif format_name == "yaml":
                return yaml.dump(data, default_flow_style=False)
            elif format_name == "xml":
                # Simplified XML generation
                return f"<data>{str(data)}</data>"
            elif format_name == "csv":
                # Simplified CSV for flat data
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                    return output.getvalue()
            elif format_name == "markdown":
                # Simplified Markdown
                return f"# Data\n\n```json\n{json.dumps(data, indent=2)}\n```"
            return str(data)

    negotiator = FormatNegotiator()

    # Test different requirements
    test_cases = [
        {
            "name": "API Response",
            "requirements": {
                "use_case": "apis",
                "features": ["parsing", "nested data", "type support"],
                "machine_parseable": True
            }
        },
        {
            "name": "Configuration File",
            "requirements": {
                "use_case": "configuration",
                "features": ["readability", "comments"],
                "human_readable": True
            }
        },
        {
            "name": "Data Export",
            "requirements": {
                "use_case": "spreadsheets",
                "features": ["tabular data", "simple"],
                "human_readable": True
            }
        }
    ]

    print("FORMAT NEGOTIATION:\n")

    for test in test_cases:
        recommended = negotiator.recommend_format(test["requirements"])
        print(f"{test['name']}:")
        print(f"  Requirements: {test['requirements']['use_case']}")
        print(f"  Recommended format: {recommended}")
        print(f"  Reason: {FormatNegotiator.FORMAT_CHARACTERISTICS[recommended]['strengths'][:2]}")
        print()

    # Generate in recommended format
    sample_data = [
        {"id": 1, "name": "Item 1", "value": 100},
        {"id": 2, "name": "Item 2", "value": 200}
    ]

    print("SAMPLE OUTPUT IN RECOMMENDED FORMATS:\n")

    for format_name in ["json", "yaml", "csv"]:
        print(f"{format_name.upper()}:")
        output = negotiator.generate_in_format(sample_data, format_name)
        print(output[:200] if len(output) > 200 else output)
        print()

    print("💡 Format negotiation optimizes data exchange")


def run_all_examples():
    """Run all format parser examples."""
    examples = [
        example_1_xml_generation_and_parsing,
        example_2_yaml_generation,
        example_3_csv_generation,
        example_4_markdown_tables,
        example_5_custom_format_dsl,
        example_6_format_conversion,
        example_7_format_negotiation
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

    parser = argparse.ArgumentParser(description="Module 08: Format Parsers")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_xml_generation_and_parsing,
            2: example_2_yaml_generation,
            3: example_3_csv_generation,
            4: example_4_markdown_tables,
            5: example_5_custom_format_dsl,
            6: example_6_format_conversion,
            7: example_7_format_negotiation
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 08: Format Parsers Examples")
        print("\nUsage:")
        print("  python format_parsers.py --all        # Run all examples")
        print("  python format_parsers.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: XML Generation and Parsing")
        print("  2: YAML Generation")
        print("  3: CSV Generation")
        print("  4: Markdown Tables")
        print("  5: Custom Format DSL")
        print("  6: Format Conversion")
        print("  7: Format Negotiation")