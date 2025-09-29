"""
Module 01: Fundamentals - Exercise Solutions

Complete solutions for all exercises in Module 01.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient, count_tokens


# ===== Exercise 1 Solution: Prompt Refinement =====

def exercise_1_prompt_refinement_solution():
    """
    Exercise 1 Solution: Progressive prompt refinement.
    """
    client = LLMClient("openai")

    original_prompt = "Help me with my presentation"

    improved_prompts = [
        # Version 1: Add topic
        "Help me with my presentation about renewable energy",

        # Version 2: Add topic, audience, and purpose
        "Help me create a 10-minute presentation about renewable energy for high school students, focusing on solar and wind power",

        # Version 3: Add all details including specific needs
        """Help me create a 10-minute presentation about renewable energy for high school students.
        Requirements:
        - Focus on solar and wind power
        - Include 3 real-world examples
        - Explain environmental benefits
        - Make it engaging with visuals suggestions
        - Provide speaker notes for key slides"""
    ]

    print("Exercise 1 Solution: Prompt Refinement")
    print("=" * 50)
    print(f"Original: {original_prompt}\n")

    for i, prompt in enumerate(improved_prompts):
        print(f"\nVersion {i+1}:")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        response = client.complete(prompt, temperature=0.7, max_tokens=150)
        print(f"Response preview: {response[:200]}...")
        print(f"Token count: {count_tokens(response)}")
        print(f"Improvement: Version {i+1} is {'much ' if i == 2 else ''}more specific")


# ===== Exercise 2 Solution: Temperature Experimentation =====

def exercise_2_temperature_experiment_solution():
    """
    Exercise 2 Solution: Temperature effects on creativity.
    """
    client = LLMClient("openai")

    prompt = "Generate three creative names for a futuristic city on Mars that combines Earth nostalgia with Martian innovation"

    temperatures = [0.0, 0.5, 0.9, 1.2]
    results = []

    print("Exercise 2 Solution: Temperature Experimentation")
    print("=" * 50)
    print(f"Prompt: {prompt}\n")

    for temp in temperatures:
        response = client.complete(prompt, temperature=temp, max_tokens=60)
        results.append({
            "temperature": temp,
            "response": response
        })
        print(f"Temperature {temp}:")
        print(f"{response}\n")

    print("Observations:")
    print("- Temperature 0.0: Most predictable, conventional names")
    print("- Temperature 0.5: Balanced creativity with coherence")
    print("- Temperature 0.9: More creative and unexpected combinations")
    print("- Temperature 1.2: Highly creative but sometimes less coherent")


# ===== Exercise 3 Solution: Format Specification =====

def exercise_3_format_specification_solution():
    """
    Exercise 3 Solution: Generating different output formats.
    """
    client = LLMClient("openai")

    base_info = """
    Python: High-level, interpreted, general-purpose. Used for web, data science, AI.
    JavaScript: High-level, interpreted. Used for web development, Node.js.
    Rust: Systems programming, memory safe. Used for performance-critical applications.
    """

    format_prompts = {
        "json": f"""Convert this programming language information to valid JSON:
        {base_info}

        Required JSON structure:
        {{
            "languages": [
                {{
                    "name": "string",
                    "type": "string",
                    "interpreted": boolean,
                    "use_cases": ["array of strings"]
                }}
            ]
        }}

        JSON output:""",

        "markdown_table": f"""Convert this programming language information to a markdown table:
        {base_info}

        Table columns: Language | Type | Interpreted | Main Use Cases

        Markdown table:""",

        "numbered_list": f"""Convert this programming language information to a detailed numbered list:
        {base_info}

        Format each entry as:
        1. [Language Name]
           - Type: ...
           - Interpreted: Yes/No
           - Primary use cases: ...
           - Key strength: ...

        Numbered list:""",

        "python_dict": f"""Convert this programming language information to a Python dictionary:
        {base_info}

        Format as valid Python code that could be executed:
        languages = {{
            "language_name": {{
                "type": "...",
                "interpreted": True/False,
                "use_cases": [...]
            }},
            ...
        }}

        Python dictionary:"""
    }

    print("Exercise 3 Solution: Format Specification")
    print("=" * 50)

    for format_name, prompt in format_prompts.items():
        print(f"\n{format_name.upper()} Format:")
        response = client.complete(prompt, temperature=0.2)
        print(f"Response:\n{response}\n")
        print("-" * 40)


# ===== Exercise 4 Solution: System Message Design =====

def exercise_4_system_messages_solution():
    """
    Exercise 4 Solution: Effective system messages for different personas.
    """
    client = LLMClient("openai")

    question = "Explain how machine learning works"

    personas = {
        "data_scientist": """You are a senior data scientist with expertise in machine learning.
        Provide technical explanations using appropriate terminology, mathematical concepts when relevant,
        and real-world implementation considerations.""",

        "teacher": """You are an elementary school teacher who excels at explaining complex topics
        to young children. Use simple language, relatable analogies, and avoid technical jargon.
        Make learning fun and engaging.""",

        "business_executive": """You are a business consultant explaining technical concepts to C-suite executives.
        Focus on business value, ROI, practical applications, and strategic implications.
        Avoid technical details unless absolutely necessary.""",

        "philosopher": """You are a philosopher interested in the epistemological and ethical dimensions
        of technology. Explore the deeper implications, raise thought-provoking questions,
        and consider the human condition in relation to artificial intelligence."""
    }

    print("Exercise 4 Solution: System Message Design")
    print("=" * 50)
    print(f"Question: {question}\n")

    for persona_name, system_message in personas.items():
        print(f"\n{persona_name.replace('_', ' ').upper()} Persona:")
        print(f"System message: {system_message[:80]}...")
        response = client.complete(
            question,
            system_message=system_message,
            temperature=0.7,
            max_tokens=150
        )
        print(f"Response: {response}\n")
        print("-" * 40)


# ===== Exercise 5 Solution: Delimiter Usage =====

def exercise_5_delimiter_practice_solution():
    """
    Exercise 5 Solution: Effective use of delimiters.
    """
    client = LLMClient("openai")

    messy_prompt = """Analyze this customer review I love the product but the shipping
    was slow and the packaging was damaged slightly however the item works perfectly
    and I would recommend it and extract the sentiment and also list the pros and cons
    and give an overall rating from 1-5 and suggest improvements"""

    structured_prompt = """Analyze the customer review below and provide a structured analysis.

    CUSTOMER REVIEW:
    ===
    I love the product but the shipping was slow and the packaging was damaged slightly.
    However, the item works perfectly and I would recommend it.
    ===

    REQUIRED ANALYSIS:
    1. Overall Sentiment: [Positive/Negative/Mixed]
    2. Pros (bullet points)
    3. Cons (bullet points)
    4. Overall Rating: [1-5 stars]
    5. Suggested Improvements (brief recommendations)

    ANALYSIS:"""

    print("Exercise 5 Solution: Delimiter Usage")
    print("=" * 50)

    print("ORIGINAL MESSY PROMPT:")
    print(messy_prompt)
    response = client.complete(messy_prompt, temperature=0.3)
    print(f"\nResponse: {response}")

    print("\n" + "=" * 50)
    print("\nIMPROVED STRUCTURED PROMPT:")
    print(structured_prompt)
    response = client.complete(structured_prompt, temperature=0.3)
    print(f"\nResponse: {response}")


# ===== Challenge Project Solution =====

def challenge_project_solution():
    """
    Challenge Solution: Complete Prompt Template Library
    """

    class PromptTemplate:
        def __init__(self, template: str, variables: list, metadata: dict = None):
            """
            Initialize a prompt template.
            """
            self.template = template
            self.variables = variables
            self.metadata = metadata or {}
            self.usage_count = 0
            self.performance_metrics = []

        def validate(self, **kwargs):
            """
            Validate that all required variables are provided.
            """
            missing = [var for var in self.variables if var not in kwargs]
            if missing:
                raise ValueError(f"Missing required variables: {missing}")
            return True

        def format(self, **kwargs):
            """
            Format the template with provided values.
            """
            # Validate first
            self.validate(**kwargs)

            # Format the template
            formatted = self.template
            for var, value in kwargs.items():
                formatted = formatted.replace(f"{{{var}}}", str(value))

            # Track usage
            self.usage_count += 1

            return formatted

        def add_performance_metric(self, metric: dict):
            """
            Track performance metrics for this template.
            """
            self.performance_metrics.append({
                **metric,
                "usage_number": self.usage_count
            })

        def get_statistics(self):
            """
            Get usage statistics for this template.
            """
            return {
                "usage_count": self.usage_count,
                "variables": self.variables,
                "metadata": self.metadata,
                "performance_samples": len(self.performance_metrics)
            }

    # Create a library of templates
    class PromptLibrary:
        def __init__(self):
            self.templates = {}

        def add_template(self, name: str, template: PromptTemplate):
            """Add a template to the library."""
            self.templates[name] = template

        def get_template(self, name: str) -> PromptTemplate:
            """Retrieve a template by name."""
            if name not in self.templates:
                raise KeyError(f"Template '{name}' not found")
            return self.templates[name]

        def list_templates(self):
            """List all available templates."""
            return list(self.templates.keys())

    # Example usage
    library = PromptLibrary()

    # Email template
    email_template = PromptTemplate(
        template="""Write a professional email with the following details:
        Recipient: {recipient}
        Subject: {subject}
        Main point: {main_point}
        Tone: {tone}

        Email:""",
        variables=["recipient", "subject", "main_point", "tone"],
        metadata={
            "purpose": "Professional email generation",
            "optimal_temperature": 0.7,
            "category": "communication"
        }
    )

    # Code review template
    code_review_template = PromptTemplate(
        template="""Review the following {language} code:

        ```{language}
        {code}
        ```

        Focus on:
        - {focus_areas}

        Provide feedback in this format:
        1. Issues found
        2. Suggestions for improvement
        3. Positive aspects

        Review:""",
        variables=["language", "code", "focus_areas"],
        metadata={
            "purpose": "Code review",
            "optimal_temperature": 0.3,
            "category": "development"
        }
    )

    # Add templates to library
    library.add_template("email", email_template)
    library.add_template("code_review", code_review_template)

    print("Challenge Solution: Prompt Template Library")
    print("=" * 50)

    # Test email template
    print("\nEmail Template Test:")
    email_prompt = library.get_template("email").format(
        recipient="Sarah Johnson, VP of Engineering",
        subject="Q4 Development Roadmap Review",
        main_point="Request for feedback on proposed Q4 development priorities before Monday's planning meeting",
        tone="professional but collaborative"
    )
    print(email_prompt)

    # Test code review template
    print("\n" + "-" * 50)
    print("\nCode Review Template Test:")
    code_prompt = library.get_template("code_review").format(
        language="Python",
        code="""def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)""",
        focus_areas="error handling, edge cases, and Pythonic improvements"
    )
    print(code_prompt)

    # Show statistics
    print("\n" + "-" * 50)
    print("\nTemplate Statistics:")
    for name in library.list_templates():
        template = library.get_template(name)
        stats = template.get_statistics()
        print(f"\n{name}:")
        print(f"  Usage count: {stats['usage_count']}")
        print(f"  Variables: {', '.join(stats['variables'])}")
        print(f"  Category: {stats['metadata'].get('category', 'uncategorized')}")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 01: Exercise Solutions")
    parser.add_argument("--solution", type=int, help="Run specific solution (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge solution")
    parser.add_argument("--all", action="store_true", help="Run all solutions")

    args = parser.parse_args()

    solutions = {
        1: exercise_1_prompt_refinement_solution,
        2: exercise_2_temperature_experiment_solution,
        3: exercise_3_format_specification_solution,
        4: exercise_4_system_messages_solution,
        5: exercise_5_delimiter_practice_solution
    }

    if args.all:
        for sol_num, sol_func in solutions.items():
            print(f"\n{'='*60}")
            sol_func()
            print(f"\n{'='*60}\n")
        challenge_project_solution()
    elif args.challenge:
        challenge_project_solution()
    elif args.solution and args.solution in solutions:
        solutions[args.solution]()
    else:
        print("Module 01: Fundamentals - Solutions")
        print("\nUsage:")
        print("  python solutions.py --solution N  # Run solution N")
        print("  python solutions.py --challenge    # Run challenge solution")
        print("  python solutions.py --all          # Run all solutions")
        print("\nAvailable solutions:")
        print("  1: Prompt Refinement")
        print("  2: Temperature Experimentation")
        print("  3: Format Specification")
        print("  4: System Message Design")
        print("  5: Delimiter Usage")
        print("  Challenge: Complete Prompt Template Library")