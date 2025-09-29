"""
Module 02: Zero-Shot Prompting - Solutions

Complete solutions for all zero-shot prompting exercises.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
import statistics


# ===== Solution 1: Instruction Clarity =====

def solution_1_instruction_clarity():
    """
    Solution 1: Transform vague instructions into clear zero-shot prompts.
    """
    client = LLMClient("openai")

    vague_instructions = [
        "Analyze this data",
        "Fix this text",
        "Make this better",
        "Process this information",
        "Check if this is good"
    ]

    # Clear, specific versions
    clear_instructions = [
        """Analyze this data and provide:
        1. Data type (numeric, text, time-series, categorical)
        2. Key patterns or trends observed
        3. Statistical summary (if numeric) or theme summary (if text)
        4. Any anomalies, outliers, or inconsistencies
        5. Actionable insights or recommendations

        Data:""",

        """Fix the grammar, spelling, and punctuation in this text.
        Rules:
        - Preserve the original meaning and tone
        - Fix only errors, don't rewrite style
        - Mark any ambiguous corrections with [?]
        - If no errors found, return "No corrections needed"

        Text:""",

        """Improve this content by:
        1. Enhancing clarity and readability
        2. Fixing any grammatical errors
        3. Improving structure and flow
        4. Strengthening key points
        5. Maintaining the original tone and purpose

        Content:""",

        """Process this information to extract:
        1. Main topics or entities mentioned
        2. Key facts and figures
        3. Relationships between elements
        4. Temporal information (dates, sequences)
        5. Action items or decisions mentioned

        Information:""",

        """Evaluate this submission against these criteria:
        1. Completeness (all requirements met)
        2. Accuracy (factually correct)
        3. Quality (well-structured and clear)
        4. Relevance (addresses the topic)
        5. Overall assessment (Pass/Needs Work/Excellent)

        Provide specific feedback for each criterion.

        Submission:"""
    ]

    print("Solution 1: Instruction Clarity")
    print("=" * 50)

    # Test with sample data
    sample_data = "Sales increased by 15% in Q1, dropped 5% in Q2, and grew 20% in Q3."

    for i, (vague, clear) in enumerate(zip(vague_instructions, clear_instructions)):
        print(f"\n{i+1}. Transformation:")
        print(f"   FROM (vague): {vague}")
        print(f"   TO (clear): {clear[:100]}...")

        # Compare responses
        vague_prompt = f"{vague}: {sample_data}"
        vague_response = client.complete(vague_prompt, temperature=0.3, max_tokens=100)

        clear_prompt = f"{clear} {sample_data}"
        clear_response = client.complete(clear_prompt, temperature=0.3, max_tokens=150)

        print(f"\n   Vague output: {vague_response[:150]}...")
        print(f"   Clear output: {clear_response[:200]}...")
        print(f"\n   Improvement: Clear version provides structured, actionable output")


# ===== Solution 2: Format Control =====

def solution_2_format_control():
    """
    Solution 2: Master output format specification without examples.
    """
    client = LLMClient("openai")

    text = """
    John Smith is 28 years old and works as a software engineer at TechCorp.
    He has 5 years of experience and specializes in Python and JavaScript.
    His email is john.smith@techcorp.com and he's based in San Francisco.
    """

    # Precise format specifications without examples

    json_prompt = """Extract person information from the text below into JSON format.

    Text: {text}

    Required JSON structure (all fields required, use null if not found):
    {{
        "personal_info": {{
            "full_name": "string",
            "age": number,
            "location": "string"
        }},
        "professional_info": {{
            "job_title": "string",
            "company": "string",
            "years_experience": number,
            "skills": ["array of strings"]
        }},
        "contact": {{
            "email": "string or null",
            "phone": "string or null"
        }}
    }}

    Rules:
    - Extract exactly as specified in the structure
    - Numbers should not be in quotes
    - Arrays should contain individual skill items
    - Use null for missing information, not empty strings

    JSON output:"""

    table_prompt = """Convert the information to a markdown table.

    Text: {text}

    Table requirements:
    - Column headers: Field | Value
    - One row per piece of information
    - Left-align all columns
    - Include horizontal separators
    - Order: Personal info first, then professional, then contact

    Markdown table:"""

    bullet_prompt = """Organize the information as structured bullet points.

    Text: {text}

    Format requirements:
    - Top-level bullets for categories (Personal, Professional, Contact)
    - Sub-bullets for specific information
    - Use "•" for main bullets, "◦" for sub-bullets
    - Include labels before values (e.g., "Age: 28")
    - Skip categories if no information available

    Bullet list:"""

    xml_prompt = """Structure the information as valid XML.

    Text: {text}

    XML requirements:
    - Root element: <person>
    - Use semantic tag names (camelCase)
    - Attributes for metadata (e.g., years as attribute)
    - Text content for values
    - Skills as separate <skill> elements
    - Close all tags properly

    XML output:"""

    print("Solution 2: Format Control")
    print("=" * 50)

    formats = {
        "JSON": json_prompt,
        "Markdown Table": table_prompt,
        "Structured Bullets": bullet_prompt,
        "XML": xml_prompt
    }

    for format_name, prompt in formats.items():
        formatted_prompt = prompt.format(text=text)
        print(f"\n{format_name} Format:")
        print("-" * 40)
        response = client.complete(formatted_prompt, temperature=0.1, max_tokens=300)
        print(response)


# ===== Solution 3: Edge Case Handling =====

def solution_3_edge_case_handling():
    """
    Solution 3: Robust zero-shot prompts with comprehensive edge case handling.
    """
    client = LLMClient("openai")

    # Robust email validation prompt
    email_validation_prompt = """Validate if the input is a properly formatted email address.

    Input: {input}

    Validation rules:
    1. Must contain exactly one @ symbol
    2. Local part (before @) requirements:
       - At least 1 character
       - Can contain letters, numbers, dots, hyphens, underscores
       - Cannot start or end with a dot
       - No consecutive dots
    3. Domain part (after @) requirements:
       - At least one dot
       - Valid domain name format
       - Top-level domain of at least 2 characters
    4. No spaces anywhere
    5. Total length reasonable (under 254 characters)

    Special cases:
    - Empty input → "Error: Empty input"
    - Only whitespace → "Error: Empty input"
    - Non-string input (just numbers) → "Error: Invalid format"

    Response format:
    Valid: [Yes/No]
    Reason: [Specific explanation]
    Fixed suggestion: [If fixable with minor change, suggest correction, otherwise "N/A"]

    Validation:"""

    test_cases = [
        ("john@example.com", True, "Standard valid email"),
        ("", False, "Empty input"),
        ("not_an_email", False, "Missing @ symbol"),
        ("@example.com", False, "Missing local part"),
        ("user@", False, "Missing domain"),
        ("user@@example.com", False, "Multiple @ symbols"),
        ("user@.com", False, "Invalid domain start"),
        ("user name@example.com", False, "Contains space"),
        ("user@example", False, "Missing TLD"),
        ("user.name+tag@example.co.uk", True, "Complex valid email"),
        ("a@b.co", True, "Minimal valid email"),
        (".user@example.com", False, "Starts with dot"),
        ("user..name@example.com", False, "Consecutive dots"),
    ]

    print("Solution 3: Edge Case Handling")
    print("=" * 50)
    print("\nEmail Validation with Edge Cases:")
    print("-" * 40)

    correct = 0
    for test_input, expected_valid, description in test_cases:
        prompt = email_validation_prompt.format(
            input=test_input if test_input else "[EMPTY]"
        )
        response = client.complete(prompt, temperature=0.0, max_tokens=100)

        # Parse response to check if it matches expected
        is_valid = "Yes" in response if expected_valid else "No" in response
        status = "✓" if is_valid else "✗"

        print(f"\n{status} Input: '{test_input}' ({description})")
        print(f"   Expected: {'Valid' if expected_valid else 'Invalid'}")
        print(f"   Response: {response[:100]}")

        if is_valid:
            correct += 1

    accuracy = (correct / len(test_cases)) * 100
    print(f"\nAccuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)")


# ===== Solution 4: Task Decomposition =====

def solution_4_task_decomposition():
    """
    Solution 4: Decompose complex tasks into focused subtasks.
    """
    client = LLMClient("openai")

    complex_task = """
    TechStart is a 2-year-old SaaS company providing AI-powered customer service tools.
    They have 50 enterprise clients, $2M ARR, growing 20% month-over-month.
    The team has 15 employees, raised $500K seed funding, and is seeking $5M Series A.
    Main competitors are Zendesk and Intercom. Their unique value is 90% faster response time.
    """

    # Well-designed subtask prompts
    subtasks = {
        "company_overview": """Extract company basics from the text.

        Text: {text}

        Extract:
        - Company name
        - Age/Founded
        - Industry/Sector
        - Product/Service description
        - Stage (Startup/Growth/Mature)

        Company Overview:""",

        "financial_metrics": """Extract all financial and growth metrics.

        Text: {text}

        Identify:
        - Revenue figures (ARR/MRR)
        - Growth rates
        - Customer count
        - Customer type (B2B/B2C)
        - Financial trajectory

        Financial Metrics:""",

        "team_funding": """Extract team and funding information.

        Text: {text}

        Extract:
        - Team size
        - Previous funding (amount and round)
        - Current funding seek (amount and round)
        - Funding stage
        - Use of funds (if mentioned)

        Team & Funding:""",

        "competitive_analysis": """Analyze competitive positioning.

        Text: {text}

        Identify:
        - Main competitors named
        - Unique value proposition
        - Competitive advantages
        - Market position
        - Differentiation factors

        Competitive Analysis:""",

        "investment_thesis": """Synthesize an investment thesis.

        Text: {text}

        Create brief thesis covering:
        - Growth potential (based on metrics)
        - Market opportunity
        - Competitive moat
        - Risk factors
        - Investment recommendation

        Investment Thesis:"""
    }

    print("Solution 4: Task Decomposition")
    print("=" * 50)

    # Execute subtasks and collect results
    results = {}
    for task_name, task_prompt in subtasks.items():
        prompt = task_prompt.format(text=complex_task)
        print(f"\nSubtask: {task_name}")
        print("-" * 40)
        response = client.complete(prompt, temperature=0.2, max_tokens=150)
        results[task_name] = response
        print(response)

    # Combine results into cohesive pitch
    print("\n" + "=" * 50)
    print("COMBINED INVESTOR PITCH SUMMARY:")
    print("=" * 50)

    combination_prompt = f"""Create a concise investor pitch summary from these components:

    Company Overview:
    {results['company_overview']}

    Financials:
    {results['financial_metrics']}

    Team & Funding:
    {results['team_funding']}

    Competitive Position:
    {results['competitive_analysis']}

    Investment Thesis:
    {results['investment_thesis']}

    Create a 3-paragraph executive summary suitable for investors:
    - Paragraph 1: Company and product
    - Paragraph 2: Traction and metrics
    - Paragraph 3: Investment opportunity

    Executive Summary:"""

    final_summary = client.complete(combination_prompt, temperature=0.4, max_tokens=300)
    print(final_summary)


# ===== Solution 5: Reliability Testing =====

def solution_5_reliability_improvement():
    """
    Solution 5: Systematically improve zero-shot prompt consistency.
    """
    client = LLMClient("openai")

    # Initial prompt (intentionally vague)
    initial_prompt = """Rate how positive this review is: "The food was okay but service was great!"

    Rating:"""

    # Improved version with clear specifications
    improved_prompt = """Rate the overall positivity of this review on a scale of 1-5.

    Review: "The food was okay but service was great!"

    Rating Scale:
    1 = Very Negative (mostly complaints, dissatisfied)
    2 = Negative (more negative than positive points)
    3 = Neutral (balanced positive and negative)
    4 = Positive (more positive than negative points)
    5 = Very Positive (mostly praise, highly satisfied)

    Instructions:
    - Consider all aspects mentioned
    - "Okay" = neutral (neither positive nor negative)
    - "Great" = strongly positive
    - Weight all aspects equally unless emphasis suggests otherwise
    - Output only the number (1-5), no explanation

    Rating:"""

    # Ultimate version for maximum consistency
    ultimate_prompt = """Sentiment Rating Task

    Input: "The food was okay but service was great!"

    Step 1: Identify sentiment of each aspect
    - Food: "okay" → Neutral (0)
    - Service: "great" → Positive (+2)

    Step 2: Calculate overall sentiment
    - Total aspects: 2
    - Sum of sentiments: 0 + 2 = +2
    - Average: +1 (lean positive)

    Step 3: Map to 5-point scale
    - Very Negative (-2) → 1
    - Negative (-1) → 2
    - Neutral (0) → 3
    - Positive (+1) → 4
    - Very Positive (+2) → 5

    Based on average sentiment of +1 (Positive), output rating.

    Rating (single digit 1-5):"""

    print("Solution 5: Reliability Improvement")
    print("=" * 50)

    # Test each version
    versions = [
        ("Initial", initial_prompt),
        ("Improved", improved_prompt),
        ("Ultimate", ultimate_prompt)
    ]

    for version_name, prompt in versions:
        print(f"\n{version_name} Version:")
        print("-" * 40)
        responses = []

        for i in range(10):
            response = client.complete(prompt, temperature=0.3, max_tokens=20)
            # Extract just the rating
            rating = ''.join(filter(str.isdigit, response.strip()[:5]))
            if rating:
                responses.append(int(rating[0]))

        if responses:
            mean = statistics.mean(responses)
            stdev = statistics.stdev(responses) if len(responses) > 1 else 0
            unique = len(set(responses))

            print(f"Responses: {responses}")
            print(f"Mean: {mean:.2f}, StdDev: {stdev:.2f}")
            print(f"Unique values: {unique}")
            print(f"Consistency: {(1 - stdev/2) * 100:.1f}%" if stdev < 2 else "Consistency: Low")


# ===== Challenge Solution: Universal Code Analyzer =====

def challenge_solution_universal_analyzer():
    """
    Challenge Solution: Universal code analyzer that works for any language.
    """
    client = LLMClient("openai")

    analyzer_prompt = """Analyze this code for quality, issues, and improvements.

    Code:
    ```
    {code}
    ```

    Analysis Instructions:

    1. Language Detection:
       - Identify programming language from syntax patterns
       - Note language version if determinable

    2. Structure Analysis:
       - Check indentation and formatting consistency
       - Evaluate function/method organization
       - Assess variable and function naming conventions
       - Review code organization and modularity

    3. Common Issues (language-agnostic):
       - Unhandled edge cases (null, zero, empty)
       - Missing error handling
       - Resource leaks (unclosed files, connections)
       - Infinite loops or recursion risks
       - Hard-coded values that should be configurable
       - Dead or unreachable code

    4. Security Concerns:
       - Input validation missing
       - Potential injection vulnerabilities
       - Unsafe data handling
       - Authentication/authorization issues
       - Sensitive data exposure

    5. Performance Considerations:
       - Inefficient algorithms (nested loops over large data)
       - Unnecessary repeated computations
       - Memory usage concerns
       - I/O operation efficiency

    6. Best Practices:
       - Code reusability
       - Single responsibility principle
       - Documentation presence
       - Test coverage implications

    Output Format:
    ==============
    Language: [detected language and version if known]
    Overall Quality: [score 1-10]/10

    ✓ Strengths:
    - [list positive aspects]

    ⚠ Issues Found:
    - [list problems with severity: LOW/MEDIUM/HIGH/CRITICAL]

    💡 Improvements:
    - [list specific suggestions]

    🔒 Security Concerns:
    - [list if any, otherwise "None identified"]

    📊 Metrics:
    - Complexity: [Low/Medium/High]
    - Maintainability: [Poor/Fair/Good/Excellent]
    - Robustness: [score 1-10]/10

    Analysis:"""

    # Test cases in different languages
    test_codes = {
        "Python (with bug)": """
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)

# Usage
scores = [85, 90, 78, 92, 88]
avg = calculate_average(scores)
print(f"Average: {avg}")
        """,

        "JavaScript (security issue)": """
app.get('/user', (req, res) => {
    const userId = req.query.id;
    const query = `SELECT * FROM users WHERE id = ${userId}`;

    db.query(query, (err, result) => {
        if (err) throw err;
        res.json(result);
    });
});
        """,

        "Java (resource leak)": """
public String readFile(String filename) {
    BufferedReader reader = new BufferedReader(new FileReader(filename));
    String line;
    StringBuilder content = new StringBuilder();

    while ((line = reader.readLine()) != null) {
        content.append(line).append("\\n");
    }

    return content.toString();
}
        """,

        "Go (concurrency issue)": """
func processItems(items []string) {
    var results []string

    for _, item := range items {
        go func() {
            processed := strings.ToUpper(item)
            results = append(results, processed)
        }()
    }

    time.Sleep(1 * time.Second)
    fmt.Println(results)
}
        """
    }

    print("Challenge Solution: Universal Code Analyzer")
    print("=" * 50)

    for description, code in test_codes.items():
        print(f"\n\nAnalyzing: {description}")
        print("=" * 50)
        prompt = analyzer_prompt.format(code=code)
        response = client.complete(prompt, temperature=0.2, max_tokens=500)
        print(response)


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 02: Zero-Shot Solutions")
    parser.add_argument("--solution", type=int, help="Run specific solution (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge solution")
    parser.add_argument("--all", action="store_true", help="Run all solutions")

    args = parser.parse_args()

    solutions = {
        1: solution_1_instruction_clarity,
        2: solution_2_format_control,
        3: solution_3_edge_case_handling,
        4: solution_4_task_decomposition,
        5: solution_5_reliability_improvement
    }

    if args.all:
        for sol_num, sol_func in solutions.items():
            print(f"\n{'='*60}")
            sol_func()
            print(f"\n{'='*60}\n")
        challenge_solution_universal_analyzer()
    elif args.challenge:
        challenge_solution_universal_analyzer()
    elif args.solution and args.solution in solutions:
        solutions[args.solution]()
    else:
        print("Module 02: Zero-Shot Prompting - Solutions")
        print("\nUsage:")
        print("  python solutions.py --solution N  # Run solution N")
        print("  python solutions.py --challenge    # Run challenge solution")
        print("  python solutions.py --all          # Run all solutions")
        print("\nAvailable solutions:")
        print("  1: Instruction Clarity")
        print("  2: Format Control")
        print("  3: Edge Case Handling")
        print("  4: Task Decomposition")
        print("  5: Reliability Improvement")
        print("  Challenge: Universal Code Analyzer")