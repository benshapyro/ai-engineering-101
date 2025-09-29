"""
Module 02: Instruction Engineering Examples

Advanced patterns for crafting effective zero-shot instructions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient, validate_json_response
import json


def example_1_classification_instructions():
    """Demonstrate different classification instruction patterns."""
    print("=" * 60)
    print("Example 1: Classification Instructions")
    print("=" * 60)

    client = LLMClient("openai")

    texts = [
        "I absolutely loved this movie! Best film I've seen all year.",
        "The service was okay, nothing special but not terrible either.",
        "This product broke after one day. Complete waste of money!",
        "Interesting concept but poor execution. Has potential though."
    ]

    # Pattern 1: Simple classification
    simple_pattern = """Classify this text as Positive, Negative, or Neutral:
    Text: {text}
    Classification:"""

    # Pattern 2: Criteria-based classification
    criteria_pattern = """Classify the sentiment of this text.

    Text: {text}

    Classification criteria:
    - Positive: Expresses satisfaction, happiness, or approval
    - Negative: Expresses dissatisfaction, disappointment, or criticism
    - Neutral: Balanced or factual without strong emotion
    - Mixed: Contains both positive and negative elements

    Classification:"""

    # Pattern 3: Confidence scoring
    confidence_pattern = """Classify this text and provide a confidence score.

    Text: {text}

    Output format:
    Classification: [Positive/Negative/Neutral/Mixed]
    Confidence: [0-100]%
    Key indicators: [List words/phrases that led to this classification]

    Analysis:"""

    patterns = {
        "Simple": simple_pattern,
        "Criteria-based": criteria_pattern,
        "With Confidence": confidence_pattern
    }

    for pattern_name, pattern in patterns.items():
        print(f"\n{pattern_name.upper()} PATTERN:")
        print("-" * 40)
        test_text = texts[0]  # Use first text for comparison
        prompt = pattern.format(text=test_text)
        response = client.complete(prompt, temperature=0.2, max_tokens=100)
        print(f"Input: {test_text}")
        print(f"Response: {response}")


def example_2_extraction_instructions():
    """Show various extraction instruction techniques."""
    print("\n" + "=" * 60)
    print("Example 2: Extraction Instructions")
    print("=" * 60)

    client = LLMClient("openai")

    text = """
    John Smith, CEO of TechCorp (john@techcorp.com), announced a $50M Series B
    funding round led by Venture Partners. The company, founded in 2019,
    has grown to 150 employees across offices in San Francisco, New York,
    and London. Their main product, CloudSync, serves over 10,000 customers.
    Contact: +1-415-555-0100
    """

    # Pattern 1: Simple extraction
    simple_extraction = f"""Extract the following information from the text:
    - Person name and title
    - Company name
    - Funding amount
    - Employee count

    Text: {text}

    Extracted information:"""

    print("SIMPLE EXTRACTION:")
    response = client.complete(simple_extraction, temperature=0.1)
    print(response)

    # Pattern 2: Structured extraction with fallbacks
    structured_extraction = f"""Extract information from the text below into the specified structure.
    Use "N/A" for any information not found.

    Text: {text}

    Extract into this structure:
    {{
        "people": [
            {{"name": "", "title": "", "email": ""}}
        ],
        "company": {{
            "name": "",
            "founded": "",
            "employees": "",
            "offices": []
        }},
        "funding": {{
            "amount": "",
            "round": "",
            "lead_investor": ""
        }},
        "contact": {{
            "phone": "",
            "email": "",
            "website": ""
        }}
    }}

    JSON output:"""

    print("\n" + "-" * 40)
    print("\nSTRUCTURED EXTRACTION:")
    response = client.complete(structured_extraction, temperature=0.1)
    print(response)

    # Pattern 3: Conditional extraction
    conditional_extraction = f"""Extract information based on these conditions:

    Text: {text}

    Extraction rules:
    1. IF a person is mentioned with a title, extract: name, title, contact info
    2. IF funding is mentioned, extract: amount, round type, investors
    3. IF company metrics are mentioned, extract: metric type and value
    4. ONLY extract email addresses that contain @
    5. ONLY extract numbers that have units or context

    Extracted data:"""

    print("\n" + "-" * 40)
    print("\nCONDITIONAL EXTRACTION:")
    response = client.complete(conditional_extraction, temperature=0.1)
    print(response)


def example_3_generation_instructions():
    """Demonstrate content generation instruction patterns."""
    print("\n" + "=" * 60)
    print("Example 3: Generation Instructions")
    print("=" * 60)

    client = LLMClient("openai")

    # Pattern 1: Template-based generation
    template_generation = """Generate a professional email using this information:

    Purpose: Follow up on job application
    Recipient: Hiring manager (unknown name)
    Position: Senior Data Engineer
    Company: DataTech Solutions
    Applied: 2 weeks ago

    Requirements:
    - Professional tone
    - 3 paragraphs maximum
    - Include availability for interview
    - Express continued interest
    - No more than 150 words

    Email:"""

    print("TEMPLATE-BASED GENERATION:")
    response = client.complete(template_generation, temperature=0.7)
    print(response)

    # Pattern 2: Constraint-based generation
    constraint_generation = """Create a product name for a fitness tracking app.

    Constraints:
    - Maximum 2 words
    - Must be memorable and unique
    - Should convey: health, progress, or achievement
    - Avoid: common words like "Fit", "Health", "Track"
    - Must work as a domain name (no spaces or special characters)

    Generate 3 options with brief explanations:"""

    print("\n" + "-" * 40)
    print("\nCONSTRAINT-BASED GENERATION:")
    response = client.complete(constraint_generation, temperature=0.9)
    print(response)

    # Pattern 3: Style-controlled generation
    style_generation = """Write a description of cloud computing.

    Style requirements:
    - Audience: 10-year-old children
    - Length: Exactly 2 sentences
    - Must use an analogy
    - Avoid technical jargon
    - Make it engaging and fun

    Description:"""

    print("\n" + "-" * 40)
    print("\nSTYLE-CONTROLLED GENERATION:")
    response = client.complete(style_generation, temperature=0.6)
    print(response)


def example_4_analysis_instructions():
    """Show different analysis instruction approaches."""
    print("\n" + "=" * 60)
    print("Example 4: Analysis Instructions")
    print("=" * 60)

    client = LLMClient("openai")

    code = """
    def calculate_average(numbers):
        total = 0
        for n in numbers:
            total += n
        return total / len(numbers)
    """

    # Pattern 1: Checklist analysis
    checklist_analysis = f"""Analyze this code against the following checklist:

    Code:
    ```python
    {code}
    ```

    Checklist:
    [ ] Handles empty input
    [ ] Handles non-numeric input
    [ ] Has appropriate variable names
    [ ] Follows Python conventions
    [ ] Is efficient
    [ ] Has potential bugs

    For each item, mark with ✓ (pass) or ✗ (fail) and provide explanation:"""

    print("CHECKLIST ANALYSIS:")
    response = client.complete(checklist_analysis, temperature=0.2)
    print(response)

    # Pattern 2: Multi-dimensional analysis
    dimensional_analysis = f"""Analyze this code across multiple dimensions:

    Code:
    ```python
    {code}
    ```

    Analyze these dimensions (rate 1-5 and explain):
    1. Correctness: Does it work as intended?
    2. Robustness: How well does it handle edge cases?
    3. Readability: How easy is it to understand?
    4. Efficiency: How well does it perform?
    5. Maintainability: How easy is it to modify?

    Analysis:"""

    print("\n" + "-" * 40)
    print("\nMULTI-DIMENSIONAL ANALYSIS:")
    response = client.complete(dimensional_analysis, temperature=0.3)
    print(response)

    # Pattern 3: Comparative analysis
    comparative_analysis = f"""Analyze this code and compare it to best practices:

    Code:
    ```python
    {code}
    ```

    Compare against these approaches:
    1. Current implementation
    2. Using built-in functions (sum, len)
    3. Using numpy
    4. Using statistics module

    For each approach provide:
    - Pros
    - Cons
    - When to use

    Comparison:"""

    print("\n" + "-" * 40)
    print("\nCOMPARATIVE ANALYSIS:")
    response = client.complete(comparative_analysis, temperature=0.3)
    print(response)


def example_5_transformation_instructions():
    """Demonstrate transformation instruction patterns."""
    print("\n" + "=" * 60)
    print("Example 5: Transformation Instructions")
    print("=" * 60)

    client = LLMClient("openai")

    # Pattern 1: Format transformation
    sql_query = "SELECT * FROM users WHERE age > 18 AND status = 'active'"

    format_transform = f"""Transform this SQL query into MongoDB query syntax:

    SQL: {sql_query}

    Requirements:
    - Use proper MongoDB operators
    - Maintain the same logical conditions
    - Use JavaScript object notation

    MongoDB query:"""

    print("FORMAT TRANSFORMATION:")
    response = client.complete(format_transform, temperature=0.2)
    print(response)

    # Pattern 2: Style transformation
    technical_text = """
    The API utilizes REST principles and returns JSON responses.
    Authentication is handled via OAuth 2.0 with JWT tokens.
    Rate limiting is set at 100 requests per minute per IP.
    """

    style_transform = f"""Transform this technical documentation into user-friendly language:

    Technical version:
    {technical_text}

    Requirements:
    - No technical jargon
    - Explain what it means for the user
    - Keep the same information
    - Target audience: non-technical users

    User-friendly version:"""

    print("\n" + "-" * 40)
    print("\nSTYLE TRANSFORMATION:")
    response = client.complete(style_transform, temperature=0.5)
    print(response)

    # Pattern 3: Structure transformation
    flat_data = """
    Name: John Doe
    Age: 30
    Email: john@example.com
    Address: 123 Main St
    City: San Francisco
    State: CA
    Zip: 94102
    Phone: 555-1234
    Emergency Contact: Jane Doe
    Emergency Phone: 555-5678
    """

    structure_transform = f"""Transform this flat data into a nested JSON structure:

    Data:
    {flat_data}

    Required structure:
    - Group personal info (name, age, email, phone)
    - Group address info (address, city, state, zip)
    - Group emergency contact info
    - Use appropriate data types (numbers for age, etc.)

    JSON output:"""

    print("\n" + "-" * 40)
    print("\nSTRUCTURE TRANSFORMATION:")
    response = client.complete(structure_transform, temperature=0.1)
    print(response)


def example_6_meta_instructions():
    """Show meta-prompting: instructions that generate instructions."""
    print("\n" + "=" * 60)
    print("Example 6: Meta-Instructions")
    print("=" * 60)

    client = LLMClient("openai")

    # Generate instructions for a new task
    meta_prompt = """Create a detailed zero-shot prompt for the following task:

    Task: Analyzing customer support tickets to identify common issues

    The prompt should include:
    1. Clear task description
    2. Input format specification
    3. Analysis criteria
    4. Output structure
    5. Edge case handling

    Generated prompt:"""

    print("META-PROMPTING (Prompt that writes prompts):")
    response = client.complete(meta_prompt, temperature=0.4)
    print(response)

    # Test the generated prompt
    print("\n" + "-" * 40)
    print("\nTESTING GENERATED PROMPT:")

    test_ticket = """
    Ticket #1234
    Customer: John Smith
    Issue: App crashes when uploading photos larger than 5MB
    Priority: High
    Status: Open
    Previous tickets: 2 similar issues in past month
    """

    # Use a simplified version for testing
    test_prompt = f"""Analyze this customer support ticket:

    {test_ticket}

    Identify:
    1. Main issue category
    2. Severity level
    3. Potential root cause
    4. Suggested resolution
    5. Pattern indicators (recurring issue?)

    Analysis:"""

    response = client.complete(test_prompt, temperature=0.3, max_tokens=200)
    print(response)


def run_all_examples():
    """Run all instruction engineering examples."""
    examples = [
        example_1_classification_instructions,
        example_2_extraction_instructions,
        example_3_generation_instructions,
        example_4_analysis_instructions,
        example_5_transformation_instructions,
        example_6_meta_instructions
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

    parser = argparse.ArgumentParser(description="Module 02: Instruction Engineering")
    parser.add_argument("--example", type=int, help="Run specific example (1-6)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_classification_instructions,
            2: example_2_extraction_instructions,
            3: example_3_generation_instructions,
            4: example_4_analysis_instructions,
            5: example_5_transformation_instructions,
            6: example_6_meta_instructions
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Run with --all for all examples or --example N for a specific example")
        print("\nAvailable examples:")
        print("1. Classification Instructions")
        print("2. Extraction Instructions")
        print("3. Generation Instructions")
        print("4. Analysis Instructions")
        print("5. Transformation Instructions")
        print("6. Meta-Instructions (Prompt Generation)")