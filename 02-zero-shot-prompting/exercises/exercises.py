"""
Module 02: Zero-Shot Prompting - Exercises

Practice exercises for mastering zero-shot prompting techniques.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json


# ===== Exercise 1: Instruction Clarity =====

def exercise_1_instruction_clarity():
    """
    Exercise 1: Transform vague instructions into clear zero-shot prompts.

    TODO:
    1. Take each vague instruction
    2. Identify what's missing
    3. Create a clear, specific version
    4. Test both versions and compare
    """
    client = LLMClient("openai")

    vague_instructions = [
        "Analyze this data",
        "Fix this text",
        "Make this better",
        "Process this information",
        "Check if this is good"
    ]

    # TODO: Create clear versions of each instruction
    clear_instructions = [
        # Example for first one:
        """Analyze this data and provide:
        1. Data type (numeric, text, mixed)
        2. Key patterns or trends
        3. Any anomalies or outliers
        4. Summary statistics if applicable
        Data:""",

        # TODO: Complete the rest
        "...",
        "...",
        "...",
        "..."
    ]

    print("Exercise 1: Instruction Clarity")
    print("=" * 50)

    # Test with sample data
    sample_data = "Sales increased by 15% in Q1, dropped 5% in Q2, and grew 20% in Q3."

    for i, (vague, clear) in enumerate(zip(vague_instructions, clear_instructions)):
        if clear != "...":
            print(f"\n{i+1}. VAGUE: {vague}")
            print(f"   CLEAR: {clear[:100]}...")

            # Test vague version
            vague_prompt = f"{vague}: {sample_data}"
            vague_response = client.complete(vague_prompt, temperature=0.3, max_tokens=100)

            # Test clear version
            clear_prompt = f"{clear} {sample_data}"
            clear_response = client.complete(clear_prompt, temperature=0.3, max_tokens=100)

            print(f"\n   Vague output: {vague_response[:100]}...")
            print(f"   Clear output: {clear_response[:100]}...")


# ===== Exercise 2: Format Control =====

def exercise_2_format_control():
    """
    Exercise 2: Practice specifying exact output formats without examples.

    TODO:
    1. Create prompts that generate specific formats
    2. Ensure consistency without showing examples
    3. Handle different format types
    """
    client = LLMClient("openai")

    text = """
    John Smith is 28 years old and works as a software engineer at TechCorp.
    He has 5 years of experience and specializes in Python and JavaScript.
    His email is john.smith@techcorp.com and he's based in San Francisco.
    """

    # TODO: Create format-specific prompts WITHOUT showing examples

    # Format 1: JSON
    json_prompt = """
    Extract person information into JSON format.

    Text: {text}

    Required fields: name, age, job_title, company, skills[], location, email

    JSON output:
    """  # TODO: Make this more specific without examples

    # Format 2: Markdown table
    table_prompt = """
    ...
    """  # TODO: Create prompt for markdown table

    # Format 3: Bullet points with specific structure
    bullet_prompt = """
    ...
    """  # TODO: Create prompt for structured bullets

    # Format 4: XML
    xml_prompt = """
    ...
    """  # TODO: Create prompt for XML output

    print("Exercise 2: Format Control")
    print("=" * 50)

    formats = {
        "JSON": json_prompt,
        "Table": table_prompt,
        "Bullets": bullet_prompt,
        "XML": xml_prompt
    }

    for format_name, prompt in formats.items():
        if prompt != "..." and "..." not in prompt:
            formatted_prompt = prompt.format(text=text)
            print(f"\n{format_name} Format:")
            response = client.complete(formatted_prompt, temperature=0.2, max_tokens=200)
            print(response)


# ===== Exercise 3: Edge Case Handling =====

def exercise_3_edge_case_handling():
    """
    Exercise 3: Write zero-shot prompts that gracefully handle edge cases.

    TODO:
    1. Create a robust prompt that handles various edge cases
    2. Test with problematic inputs
    3. Ensure graceful failure modes
    """
    client = LLMClient("openai")

    # TODO: Create a robust email validation prompt
    email_validation_prompt = """
    Validate if this is a proper email address.

    Input: {input}

    TODO: Add rules for:
    - Empty input
    - Missing @ symbol
    - Invalid domains
    - Special characters
    - Multiple @ symbols

    Response format:
    Valid: [Yes/No]
    Reason: [Explanation]

    Validation:
    """

    test_cases = [
        "john@example.com",  # Valid
        "",  # Empty
        "not_an_email",  # No @
        "@example.com",  # No local part
        "user@",  # No domain
        "user@@example.com",  # Double @
        "user@.com",  # Invalid domain
        "user name@example.com",  # Space in email
        "user@example",  # No TLD
        "123456789",  # Just numbers
    ]

    print("Exercise 3: Edge Case Handling")
    print("=" * 50)

    if "TODO" not in email_validation_prompt:
        for test in test_cases:
            prompt = email_validation_prompt.format(input=test if test else "[EMPTY]")
            response = client.complete(prompt, temperature=0.1, max_tokens=50)
            print(f"\nInput: '{test}'")
            print(f"Response: {response}")


# ===== Exercise 4: Task Decomposition =====

def exercise_4_task_decomposition():
    """
    Exercise 4: Break complex tasks into zero-shot promptable subtasks.

    TODO:
    1. Take a complex task
    2. Break it into smaller, focused prompts
    3. Combine results for final output
    """
    client = LLMClient("openai")

    complex_task = """
    Analyze this business description and create a complete investor pitch summary:

    TechStart is a 2-year-old SaaS company providing AI-powered customer service tools.
    They have 50 enterprise clients, $2M ARR, growing 20% month-over-month.
    The team has 15 employees, raised $500K seed funding, and is seeking $5M Series A.
    Main competitors are Zendesk and Intercom. Their unique value is 90% faster response time.
    """

    # TODO: Create subtask prompts
    subtasks = {
        "company_overview": """
        Extract basic company information:
        - Company name
        - Age
        - Industry
        - Product/Service

        Text: {text}

        Overview:
        """,

        "financial_metrics": """
        ...  # TODO: Create prompt for financial extraction
        """,

        "team_funding": """
        ...  # TODO: Create prompt for team and funding info
        """,

        "competitive_analysis": """
        ...  # TODO: Create prompt for competitive positioning
        """,

        "investment_ask": """
        ...  # TODO: Create prompt for investment details
        """
    }

    print("Exercise 4: Task Decomposition")
    print("=" * 50)

    results = {}
    for task_name, task_prompt in subtasks.items():
        if "..." not in task_prompt:
            prompt = task_prompt.format(text=complex_task)
            print(f"\nSubtask: {task_name}")
            response = client.complete(prompt, temperature=0.3, max_tokens=150)
            results[task_name] = response
            print(response)

    # TODO: Combine results into final pitch
    if len(results) > 0:
        print("\n" + "-" * 40)
        print("TODO: Combine subtask results into cohesive pitch summary")


# ===== Exercise 5: Reliability Testing =====

def exercise_5_reliability_testing():
    """
    Exercise 5: Test and improve zero-shot prompt consistency.

    TODO:
    1. Create a prompt
    2. Test it multiple times
    3. Identify inconsistencies
    4. Improve the prompt
    5. Re-test for better consistency
    """
    client = LLMClient("openai")

    # Initial prompt (intentionally vague for improvement)
    initial_prompt = """
    Rate how positive this review is: "The food was okay but service was great!"

    Rating:
    """

    # TODO: Create an improved version
    improved_prompt = """
    Rate the positivity of this review on a scale of 1-5.

    Review: "The food was okay but service was great!"

    TODO: Add more specific instructions:
    - Define what each rating means
    - Explain how to handle mixed sentiments
    - Specify output format

    Rating (number only):
    """

    print("Exercise 5: Reliability Testing")
    print("=" * 50)

    # Test initial prompt
    print("\nTesting INITIAL prompt (5 runs):")
    initial_responses = []
    for i in range(5):
        response = client.complete(initial_prompt, temperature=0.5, max_tokens=20)
        initial_responses.append(response.strip())
        print(f"  Run {i+1}: {response.strip()}")

    # Calculate consistency
    unique_initial = len(set(initial_responses))
    print(f"Unique responses: {unique_initial}/5")

    # Test improved prompt if completed
    if "TODO" not in improved_prompt:
        print("\nTesting IMPROVED prompt (5 runs):")
        improved_responses = []
        for i in range(5):
            response = client.complete(improved_prompt, temperature=0.5, max_tokens=20)
            improved_responses.append(response.strip())
            print(f"  Run {i+1}: {response.strip()}")

        unique_improved = len(set(improved_responses))
        print(f"Unique responses: {unique_improved}/5")

        if unique_improved < unique_initial:
            print("\n✓ SUCCESS: Improved prompt is more consistent!")
        else:
            print("\n✗ Need further improvements for consistency")


# ===== Challenge: Universal Code Analyzer =====

def challenge_universal_code_analyzer():
    """
    Challenge: Build a zero-shot prompt that can analyze code in ANY language.

    Requirements:
    1. Language agnostic
    2. Identifies potential bugs
    3. Suggests improvements
    4. Rates code quality
    5. Works without any code examples in the prompt

    TODO: Complete the analyzer prompt
    """
    client = LLMClient("openai")

    # TODO: Create a universal code analyzer prompt
    analyzer_prompt = """
    Analyze this code snippet for quality and potential issues.

    Code:
    ```
    {code}
    ```

    TODO: Add comprehensive analysis instructions that work for any language:
    1. Detect the programming language
    2. Check for common issues (without language-specific examples)
    3. Evaluate code structure and readability
    4. Identify potential bugs or security issues
    5. Suggest improvements

    Output format:
    - Language: [detected language]
    - Quality Score: [1-10]
    - Issues Found: [list]
    - Improvements: [list]
    - Security Concerns: [if any]

    Analysis:
    """

    # Test cases in different languages
    test_codes = {
        "Python": """
def divide(a, b):
    return a / b

result = divide(10, 0)
print(result)
        """,

        "JavaScript": """
function getData() {
    fetch('/api/data')
        .then(res => res.json())
        .then(data => console.log(data))
}
        """,

        "SQL": """
SELECT * FROM users WHERE password = '$user_input';
        """
    }

    print("Challenge: Universal Code Analyzer")
    print("=" * 50)

    if "TODO" not in analyzer_prompt:
        for lang, code in test_codes.items():
            print(f"\nTesting with {lang} code:")
            prompt = analyzer_prompt.format(code=code)
            response = client.complete(prompt, temperature=0.3, max_tokens=300)
            print(response)
    else:
        print("\nTODO: Complete the universal analyzer prompt first")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 02: Zero-Shot Exercises")
    parser.add_argument("--exercise", type=int, help="Run specific exercise (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge")
    parser.add_argument("--all", action="store_true", help="Run all exercises")

    args = parser.parse_args()

    exercises = {
        1: exercise_1_instruction_clarity,
        2: exercise_2_format_control,
        3: exercise_3_edge_case_handling,
        4: exercise_4_task_decomposition,
        5: exercise_5_reliability_testing
    }

    if args.all:
        for ex_num, ex_func in exercises.items():
            print(f"\n{'='*60}")
            ex_func()
            print(f"\n{'='*60}\n")
        challenge_universal_code_analyzer()
    elif args.challenge:
        challenge_universal_code_analyzer()
    elif args.exercise and args.exercise in exercises:
        exercises[args.exercise]()
    else:
        print("Module 02: Zero-Shot Prompting - Exercises")
        print("\nUsage:")
        print("  python exercises.py --exercise N  # Run exercise N")
        print("  python exercises.py --challenge    # Run challenge")
        print("  python exercises.py --all          # Run all exercises")
        print("\nAvailable exercises:")
        print("  1: Instruction Clarity")
        print("  2: Format Control")
        print("  3: Edge Case Handling")
        print("  4: Task Decomposition")
        print("  5: Reliability Testing")
        print("  Challenge: Universal Code Analyzer")