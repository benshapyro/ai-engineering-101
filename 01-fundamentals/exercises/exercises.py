"""
Module 01: Fundamentals - Exercises

Complete these exercises to practice fundamental prompt engineering concepts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient, count_tokens


# ===== Exercise 1: Prompt Refinement =====

def exercise_1_prompt_refinement():
    """
    Exercise 1: Take a vague prompt and improve it step by step.

    Starting prompt: "Help me with my presentation"

    Your task:
    1. Identify what's missing from this prompt
    2. Create 3 progressively better versions
    3. Test each version and compare the outputs

    TODO: Complete the improved_prompts list below
    """
    client = LLMClient("openai")

    original_prompt = "Help me with my presentation"

    # TODO: Create 3 improved versions of the prompt
    # Each should be more specific than the last
    improved_prompts = [
        # Version 1: Add topic
        "Help me with my presentation about...",  # TODO: Complete this

        # Version 2: Add topic and specifics
        "...",  # TODO: Write a better version

        # Version 3: Add all details
        "..."  # TODO: Write the best version
    ]

    print("Exercise 1: Prompt Refinement")
    print("=" * 50)
    print(f"Original: {original_prompt}\n")

    # Test each prompt
    for i, prompt in enumerate(improved_prompts):
        if prompt != "..." and "..." not in prompt:  # Only run completed prompts
            print(f"\nVersion {i+1}: {prompt}")
            response = client.complete(prompt, temperature=0.7, max_tokens=100)
            print(f"Response preview: {response[:150]}...")
            print(f"Token count: {count_tokens(response)}")


# ===== Exercise 2: Temperature Experimentation =====

def exercise_2_temperature_experiment():
    """
    Exercise 2: Experiment with different temperatures.

    Test the same creative prompt with temperatures: 0.0, 0.5, 0.9, 1.2

    TODO:
    1. Complete the prompt below
    2. Run with each temperature
    3. Document your observations about how temperature affects creativity
    """
    client = LLMClient("openai")

    # TODO: Write a creative prompt that will show temperature differences
    # Hint: Ask for something creative like names, stories, or ideas
    prompt = "..."  # TODO: Replace with your creative prompt

    temperatures = [0.0, 0.5, 0.9, 1.2]
    results = []

    print("Exercise 2: Temperature Experimentation")
    print("=" * 50)
    print(f"Prompt: {prompt}\n")

    for temp in temperatures:
        if prompt != "...":  # Only run if prompt is completed
            response = client.complete(prompt, temperature=temp, max_tokens=50)
            results.append({
                "temperature": temp,
                "response": response
            })
            print(f"Temperature {temp}: {response}")

    # TODO: Add your observations here
    """
    Your observations:
    - Temperature 0.0:
    - Temperature 0.5:
    - Temperature 0.9:
    - Temperature 1.2:
    """


# ===== Exercise 3: Format Specification =====

def exercise_3_format_specification():
    """
    Exercise 3: Generate outputs in different formats.

    Given information about programming languages,
    create prompts that generate:
    1. JSON format
    2. Markdown table
    3. Numbered list
    4. Python dictionary

    TODO: Complete each format prompt
    """
    client = LLMClient("openai")

    base_info = """
    Python: High-level, interpreted, general-purpose. Used for web, data science, AI.
    JavaScript: High-level, interpreted. Used for web development, Node.js.
    Rust: Systems programming, memory safe. Used for performance-critical applications.
    """

    # TODO: Complete these format-specific prompts
    format_prompts = {
        "json": f"""Convert this information to JSON:
        {base_info}

        JSON format with 'languages' array:
        """,  # TODO: Add more specific format instructions

        "markdown_table": f"""...""",  # TODO: Write prompt for markdown table

        "numbered_list": f"""...""",  # TODO: Write prompt for numbered list

        "python_dict": f"""..."""  # TODO: Write prompt for Python dictionary
    }

    print("Exercise 3: Format Specification")
    print("=" * 50)

    for format_name, prompt in format_prompts.items():
        if "..." not in prompt:  # Only run completed prompts
            print(f"\n{format_name.upper()} Format:")
            print(f"Prompt preview: {prompt[:100]}...")
            response = client.complete(prompt, temperature=0.2)
            print(f"Response:\n{response}\n")


# ===== Exercise 4: System Message Design =====

def exercise_4_system_messages():
    """
    Exercise 4: Design effective system messages.

    Create system messages for different personas and test how they
    affect responses to the same question.

    TODO: Complete the system messages for each persona
    """
    client = LLMClient("openai")

    question = "Explain how machine learning works"

    # TODO: Design system messages for these personas
    personas = {
        "data_scientist": "You are a data scientist...",  # TODO: Complete

        "teacher": "...",  # TODO: Design for elementary school teacher

        "business_executive": "...",  # TODO: Design for non-technical exec

        "philosopher": "..."  # TODO: Design for philosophical perspective
    }

    print("Exercise 4: System Message Design")
    print("=" * 50)
    print(f"Question: {question}\n")

    for persona_name, system_message in personas.items():
        if system_message != "..." and "..." not in system_message:
            print(f"\n{persona_name.upper()} Persona:")
            print(f"System: {system_message[:100]}...")
            response = client.complete(
                question,
                system_message=system_message,
                temperature=0.7,
                max_tokens=150
            )
            print(f"Response: {response}\n")


# ===== Exercise 5: Delimiter Usage =====

def exercise_5_delimiter_practice():
    """
    Exercise 5: Practice using delimiters effectively.

    Take a complex prompt with multiple parts and restructure it
    using appropriate delimiters.

    TODO: Rewrite the messy prompt using clear delimiters
    """
    client = LLMClient("openai")

    # This prompt is messy and hard to parse
    messy_prompt = """Analyze this customer review I love the product but the shipping
    was slow and the packaging was damaged slightly however the item works perfectly
    and I would recommend it and extract the sentiment and also list the pros and cons
    and give an overall rating from 1-5 and suggest improvements"""

    # TODO: Rewrite the prompt with clear structure and delimiters
    structured_prompt = """
    ...
    """  # TODO: Your improved version here

    print("Exercise 5: Delimiter Usage")
    print("=" * 50)

    print("MESSY PROMPT:")
    print(messy_prompt)
    if messy_prompt:
        response = client.complete(messy_prompt, temperature=0.3)
        print(f"\nResponse: {response}")

    if structured_prompt != "\n    ...\n    ":
        print("\n" + "-" * 50)
        print("\nSTRUCTURED PROMPT:")
        print(structured_prompt)
        response = client.complete(structured_prompt, temperature=0.3)
        print(f"\nResponse: {response}")


# ===== Challenge Project =====

def challenge_project():
    """
    Challenge: Build a Prompt Template Library

    Create a reusable PromptTemplate class that:
    1. Stores templates with placeholders
    2. Validates required variables
    3. Formats prompts with provided values
    4. Tracks usage and performance

    TODO: Complete the PromptTemplate class implementation
    """

    class PromptTemplate:
        def __init__(self, template: str, variables: list, metadata: dict = None):
            """
            Initialize a prompt template.

            Args:
                template: Template string with {variable} placeholders
                variables: List of required variable names
                metadata: Optional metadata (purpose, optimal_temperature, etc.)
            """
            self.template = template
            self.variables = variables
            self.metadata = metadata or {}
            self.usage_count = 0

        def format(self, **kwargs):
            """
            Format the template with provided values.

            TODO: Implement this method
            - Validate all required variables are provided
            - Format the template
            - Increment usage count
            - Return formatted prompt
            """
            # TODO: Your implementation here
            pass

        def validate(self, **kwargs):
            """
            Validate that all required variables are provided.

            TODO: Implement this method
            - Check if all required variables are in kwargs
            - Return True if valid, False otherwise
            """
            # TODO: Your implementation here
            pass

    # Example usage (complete the implementation above first)
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
            "optimal_temperature": 0.7
        }
    )

    # Test your implementation
    if hasattr(email_template, 'format'):
        try:
            prompt = email_template.format(
                recipient="John Doe",
                subject="Project Update",
                main_point="The project is on track for Q2 delivery",
                tone="formal but friendly"
            )
            print("Challenge Project: Prompt Template Library")
            print("=" * 50)
            print("Formatted prompt:")
            print(prompt)
        except Exception as e:
            print(f"Error: {e}")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 01: Exercises")
    parser.add_argument("--exercise", type=int, help="Run specific exercise (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge project")
    parser.add_argument("--all", action="store_true", help="Run all exercises")

    args = parser.parse_args()

    exercises = {
        1: exercise_1_prompt_refinement,
        2: exercise_2_temperature_experiment,
        3: exercise_3_format_specification,
        4: exercise_4_system_messages,
        5: exercise_5_delimiter_practice
    }

    if args.all:
        for ex_num, ex_func in exercises.items():
            print(f"\n{'='*60}")
            ex_func()
            print(f"\n{'='*60}\n")
        challenge_project()
    elif args.challenge:
        challenge_project()
    elif args.exercise and args.exercise in exercises:
        exercises[args.exercise]()
    else:
        print("Module 01: Fundamentals - Exercises")
        print("\nUsage:")
        print("  python exercises.py --exercise N  # Run exercise N")
        print("  python exercises.py --challenge    # Run challenge project")
        print("  python exercises.py --all          # Run all exercises")
        print("\nAvailable exercises:")
        print("  1: Prompt Refinement")
        print("  2: Temperature Experimentation")
        print("  3: Format Specification")
        print("  4: System Message Design")
        print("  5: Delimiter Usage")
        print("  Challenge: Prompt Template Library")