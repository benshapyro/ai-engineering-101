"""
Common prompt templates for the Prompt Engineering curriculum.
"""

from typing import Optional, List, Dict


# ===== System Messages =====

SYSTEM_MESSAGES = {
    "helpful_assistant": "You are a helpful assistant that provides clear, accurate, and concise responses.",

    "python_expert": "You are an expert Python programmer. Provide clean, efficient, and well-commented code.",

    "teacher": "You are a patient teacher who explains concepts clearly with examples. Break down complex topics into simple steps.",

    "analyst": "You are a data analyst who thinks step-by-step through problems and provides detailed reasoning.",

    "creative_writer": "You are a creative writer who crafts engaging and imaginative content.",

    "technical_writer": "You are a technical writer who creates clear, structured documentation with proper formatting.",

    "code_reviewer": "You are a senior developer reviewing code. Provide constructive feedback on code quality, best practices, and potential improvements.",
}


# ===== Prompt Templates =====

def zero_shot_template(instruction: str, input_text: Optional[str] = None) -> str:
    """
    Template for zero-shot prompting.

    Args:
        instruction: The task instruction
        input_text: Optional input to process

    Returns:
        Formatted prompt
    """
    if input_text:
        return f"{instruction}\n\nInput: {input_text}"
    return instruction


def few_shot_template(
    instruction: str,
    examples: List[Dict[str, str]],
    input_text: str
) -> str:
    """
    Template for few-shot prompting.

    Args:
        instruction: The task instruction
        examples: List of example dictionaries with 'input' and 'output' keys
        input_text: The input to process

    Returns:
        Formatted prompt
    """
    prompt = f"{instruction}\n\n"

    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Input: {example['input']}\n"
        prompt += f"Output: {example['output']}\n\n"

    prompt += f"Now process this:\n"
    prompt += f"Input: {input_text}\n"
    prompt += f"Output:"

    return prompt


def chain_of_thought_template(problem: str) -> str:
    """
    Template for chain-of-thought prompting.

    Args:
        problem: The problem to solve

    Returns:
        Formatted prompt
    """
    return f"""Solve this problem step by step, showing your reasoning at each stage.

Problem: {problem}

Let's think through this step by step:"""


def structured_output_template(
    instruction: str,
    schema: Dict,
    input_text: Optional[str] = None
) -> str:
    """
    Template for structured output generation.

    Args:
        instruction: The task instruction
        schema: Dictionary describing the output structure
        input_text: Optional input to process

    Returns:
        Formatted prompt
    """
    import json

    prompt = f"{instruction}\n\n"
    prompt += f"Return your response as valid JSON matching this schema:\n"
    prompt += f"```json\n{json.dumps(schema, indent=2)}\n```\n\n"

    if input_text:
        prompt += f"Input: {input_text}\n\n"

    prompt += "JSON Response:"

    return prompt


def role_prompt_template(
    role: str,
    task: str,
    context: Optional[str] = None,
    constraints: Optional[List[str]] = None
) -> str:
    """
    Template for role-based prompting.

    Args:
        role: The role to assume
        task: The task to perform
        context: Optional context information
        constraints: Optional list of constraints

    Returns:
        Formatted prompt
    """
    prompt = f"You are {role}.\n\n"

    if context:
        prompt += f"Context: {context}\n\n"

    prompt += f"Task: {task}\n"

    if constraints:
        prompt += "\nConstraints:\n"
        for constraint in constraints:
            prompt += f"- {constraint}\n"

    return prompt


def refinement_template(
    original: str,
    feedback: str
) -> str:
    """
    Template for iterative refinement.

    Args:
        original: Original response
        feedback: Feedback for improvement

    Returns:
        Formatted prompt
    """
    return f"""Original response:
{original}

Feedback: {feedback}

Please provide an improved version addressing the feedback:"""


def comparison_template(
    instruction: str,
    option_a: str,
    option_b: str
) -> str:
    """
    Template for comparing options.

    Args:
        instruction: The comparison criteria
        option_a: First option
        option_b: Second option

    Returns:
        Formatted prompt
    """
    return f"""{instruction}

Option A:
{option_a}

Option B:
{option_b}

Analysis:"""


def extraction_template(
    text: str,
    fields: List[str]
) -> str:
    """
    Template for information extraction.

    Args:
        text: Text to extract from
        fields: List of fields to extract

    Returns:
        Formatted prompt
    """
    fields_str = "\n".join([f"- {field}" for field in fields])

    return f"""Extract the following information from the text:
{fields_str}

Text:
{text}

Extracted Information:"""


def summarization_template(
    text: str,
    style: str = "concise",
    max_length: Optional[int] = None
) -> str:
    """
    Template for text summarization.

    Args:
        text: Text to summarize
        style: Summarization style (concise, detailed, bullet_points)
        max_length: Optional maximum length

    Returns:
        Formatted prompt
    """
    instruction = f"Summarize the following text"

    if style == "bullet_points":
        instruction += " as bullet points"
    elif style == "detailed":
        instruction += " in detail, preserving key information"
    else:
        instruction += " concisely"

    if max_length:
        instruction += f" (maximum {max_length} words)"

    return f"""{instruction}:

Text:
{text}

Summary:"""


def classification_template(
    text: str,
    categories: List[str],
    descriptions: Optional[Dict[str, str]] = None
) -> str:
    """
    Template for text classification.

    Args:
        text: Text to classify
        categories: List of categories
        descriptions: Optional descriptions of categories

    Returns:
        Formatted prompt
    """
    prompt = "Classify the following text into one of these categories:\n"

    for category in categories:
        if descriptions and category in descriptions:
            prompt += f"- {category}: {descriptions[category]}\n"
        else:
            prompt += f"- {category}\n"

    prompt += f"\nText: {text}\n\n"
    prompt += "Category:"

    return prompt


def reasoning_template(question: str) -> str:
    """
    Template for step-by-step reasoning.

    Args:
        question: Question to answer

    Returns:
        Formatted prompt
    """
    return f"""Question: {question}

Let me work through this systematically:

1. First, I'll identify what we're looking for
2. Then, I'll consider the relevant information
3. Next, I'll apply logical reasoning
4. Finally, I'll arrive at the answer

Reasoning:"""


def code_generation_template(
    description: str,
    language: str = "Python",
    requirements: Optional[List[str]] = None
) -> str:
    """
    Template for code generation.

    Args:
        description: Description of what the code should do
        language: Programming language
        requirements: Optional list of requirements

    Returns:
        Formatted prompt
    """
    prompt = f"Write {language} code that {description}\n\n"

    if requirements:
        prompt += "Requirements:\n"
        for req in requirements:
            prompt += f"- {req}\n"
        prompt += "\n"

    prompt += "Code:"

    return prompt


def evaluation_template(
    content: str,
    criteria: List[str]
) -> str:
    """
    Template for content evaluation.

    Args:
        content: Content to evaluate
        criteria: List of evaluation criteria

    Returns:
        Formatted prompt
    """
    criteria_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(criteria)])

    return f"""Evaluate the following content based on these criteria:
{criteria_str}

Content:
{content}

Evaluation:"""


def translation_template(
    text: str,
    source_lang: str,
    target_lang: str,
    style: Optional[str] = None
) -> str:
    """
    Template for language translation.

    Args:
        text: Text to translate
        source_lang: Source language
        target_lang: Target language
        style: Optional translation style (formal, informal, technical)

    Returns:
        Formatted prompt
    """
    instruction = f"Translate the following {source_lang} text to {target_lang}"

    if style:
        instruction += f" using a {style} style"

    return f"""{instruction}:

{source_lang} Text:
{text}

{target_lang} Translation:"""


def debugging_template(
    code: str,
    error: Optional[str] = None,
    language: str = "Python"
) -> str:
    """
    Template for code debugging.

    Args:
        code: Code with potential issues
        error: Optional error message
        language: Programming language

    Returns:
        Formatted prompt
    """
    prompt = f"Debug the following {language} code:\n\n"
    prompt += f"```{language.lower()}\n{code}\n```\n\n"

    if error:
        prompt += f"Error message:\n{error}\n\n"

    prompt += "Issue and solution:"

    return prompt


# ===== Prompt Chains =====

class PromptChain:
    """Helper class for creating prompt chains."""

    def __init__(self):
        self.steps = []

    def add_step(self, prompt: str, processor=None):
        """
        Add a step to the chain.

        Args:
            prompt: Prompt template or function
            processor: Optional function to process the output
        """
        self.steps.append({"prompt": prompt, "processor": processor})
        return self

    def execute(self, llm_client, initial_input: str = None):
        """
        Execute the prompt chain.

        Args:
            llm_client: LLM client instance
            initial_input: Initial input for the chain

        Returns:
            Final output
        """
        current_output = initial_input

        for step in self.steps:
            if callable(step["prompt"]):
                prompt = step["prompt"](current_output)
            else:
                prompt = step["prompt"]

            current_output = llm_client.complete(prompt)

            if step["processor"]:
                current_output = step["processor"](current_output)

        return current_output


# Example usage
if __name__ == "__main__":
    # Example of few-shot template
    examples = [
        {"input": "The sky is blue.", "output": "Positive"},
        {"input": "I hate rainy days.", "output": "Negative"},
    ]

    prompt = few_shot_template(
        instruction="Classify the sentiment as Positive or Negative.",
        examples=examples,
        input_text="I love sunny weather!"
    )
    print("Few-shot prompt:")
    print(prompt)
    print("\n" + "="*50 + "\n")

    # Example of structured output template
    schema = {
        "name": "string",
        "age": "number",
        "skills": ["string"]
    }

    prompt = structured_output_template(
        instruction="Extract person information",
        schema=schema,
        input_text="John Doe is a 30-year-old developer skilled in Python and JavaScript."
    )
    print("Structured output prompt:")
    print(prompt)