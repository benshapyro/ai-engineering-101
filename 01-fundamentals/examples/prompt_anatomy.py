"""
Module 01: Fundamentals
Prompt Anatomy Examples

Learn the key components that make up an effective prompt.
"""

import os
from dotenv import load_dotenv

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.utils import LLMClient

load_dotenv()

# Initialize LLM client
llm = LLMClient()


# Example 1: Minimal vs Complete Prompts
print("=" * 50)
print("Example 1: Minimal vs Complete Prompts")
print("=" * 50)

# Minimal prompt - lacks structure
minimal_prompt = "Tell me about Python."

print("\nMinimal Prompt:")
print(f"Prompt: {minimal_prompt}")
print("\nResponse:")
response = llm.complete(minimal_prompt, max_tokens=150)
print(response)

# Complete prompt with all components
complete_prompt = """
Context: I'm a web developer learning backend development.

Task: Explain Python's role in backend web development.

Requirements:
- Focus on web frameworks (Django, Flask)
- Mention database integration
- Keep it concise (3-4 sentences)

Output Format: Brief explanation followed by one practical use case.
"""

print("\n" + "=" * 50)
print("\nComplete Prompt:")
print(f"Prompt: {complete_prompt}")
print("\nResponse:")
response = llm.complete(complete_prompt, max_tokens=200)
print(response)


# Example 2: The Five Components of a Prompt
print("\n" + "=" * 50)
print("Example 2: The Five Components Demonstrated")
print("=" * 50)

# Component 1: Context/Background
context = "You are analyzing customer feedback for an e-commerce platform."

# Component 2: Instruction
instruction = "Categorize the following review and extract key sentiments."

# Component 3: Input Data
input_data = """
Review: "I love the product quality, but shipping took way too long.
The packaging was excellent though. Customer service was helpful when I called."
"""

# Component 4: Output Format
output_format = """
Format your response as:
- Category: [positive/negative/mixed]
- Key Points: [bullet list]
- Sentiment Score: [1-5]
"""

# Component 5: Constraints
constraints = "Be objective and focus only on facts mentioned in the review."

# Combine all components
full_prompt = f"""
{context}

{instruction}

{input_data}

{output_format}

{constraints}
"""

print("Full Prompt with All Components:")
print(full_prompt)
print("\nResponse:")
response = llm.complete(full_prompt, max_tokens=200)
print(response)


# Example 3: Context - When and How to Provide It
print("\n" + "=" * 50)
print("Example 3: The Role of Context")
print("=" * 50)

# Without context
no_context_prompt = "Explain 'cold start'."

print("\nWithout Context:")
print(f"Prompt: {no_context_prompt}")
response = llm.complete(no_context_prompt, max_tokens=100)
print(f"Response: {response}")

# With specific context
with_context_prompt = """
Context: We're discussing cloud computing and serverless functions.

Explain what a 'cold start' means in the context of AWS Lambda functions
and its impact on application performance.
"""

print("\nWith Specific Context:")
print(f"Prompt: {with_context_prompt}")
response = llm.complete(with_context_prompt, max_tokens=150)
print(f"Response: {response}")


# Example 4: Instructions - Clear vs Vague
print("\n" + "=" * 50)
print("Example 4: Instruction Clarity Matters")
print("=" * 50)

# Vague instruction
vague_instruction = "Do something with this data: 45, 67, 23, 89, 12"

print("\nVague Instruction:")
print(f"Prompt: {vague_instruction}")
response = llm.complete(vague_instruction, max_tokens=100)
print(f"Response: {response}")

# Clear instruction
clear_instruction = """
Calculate the following statistics for this dataset: 45, 67, 23, 89, 12

Required calculations:
1. Mean (average)
2. Median (middle value)
3. Range (max - min)

Show your work for each calculation.
"""

print("\nClear Instruction:")
print(f"Prompt: {clear_instruction}")
response = llm.complete(clear_instruction, max_tokens=200)
print(f"Response: {response}")


# Example 5: Output Format Specification
print("\n" + "=" * 50)
print("Example 5: Specifying Output Format")
print("=" * 50)

# Without format specification
no_format_prompt = "List the benefits of daily exercise."

print("\nNo Format Specification:")
print(f"Prompt: {no_format_prompt}")
response = llm.complete(no_format_prompt, max_tokens=150)
print(f"Response: {response}")

# With detailed format
structured_format_prompt = """
List the benefits of daily exercise.

Format your response as:
1. Physical Benefits (3 items)
2. Mental Benefits (3 items)
3. Long-term Benefits (2 items)

Use bullet points (•) for each item.
Keep each benefit to one concise sentence.
"""

print("\nWith Detailed Format:")
print(f"Prompt: {structured_format_prompt}")
response = llm.complete(structured_format_prompt, max_tokens=250)
print(f"Response: {response}")


# Example 6: Constraints - Guiding the Response
print("\n" + "=" * 50)
print("Example 6: Using Constraints Effectively")
print("=" * 50)

# Without constraints
unconstrained_prompt = "Write a product description for wireless headphones."

print("\nWithout Constraints:")
print(f"Prompt: {unconstrained_prompt}")
response = llm.complete(unconstrained_prompt, max_tokens=150)
print(f"Response: {response}")

# With specific constraints
constrained_prompt = """
Write a product description for wireless headphones.

Constraints:
- Target audience: Fitness enthusiasts
- Length: Exactly 50 words
- Tone: Energetic and motivational
- Must mention: water resistance, battery life, secure fit
- Avoid: Technical jargon
"""

print("\nWith Specific Constraints:")
print(f"Prompt: {constrained_prompt}")
response = llm.complete(constrained_prompt, max_tokens=150)
print(f"Response: {response}")


# Example 7: Complete Prompt Template
print("\n" + "=" * 50)
print("Example 7: Complete Prompt Template")
print("=" * 50)

prompt_template = """
[CONTEXT]
You are a senior software architect reviewing code for security vulnerabilities.

[TASK]
Analyze the following code snippet and identify potential security issues.

[INPUT]
```python
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    return result
```

[OUTPUT FORMAT]
Provide your analysis in this structure:
1. Vulnerability Type: [name]
2. Risk Level: [High/Medium/Low]
3. Explanation: [2-3 sentences]
4. Recommendation: [specific fix]

[CONSTRAINTS]
- Focus on the most critical issue first
- Keep explanations non-technical for management review
- Provide actionable recommendations
"""

print("Complete Prompt Template:")
print(prompt_template)
print("\nResponse:")
response = llm.complete(prompt_template, max_tokens=300)
print(response)


# Example 8: Iterative Prompt Refinement
print("\n" + "=" * 50)
print("Example 8: Iterative Refinement")
print("=" * 50)

# Version 1: Basic
v1 = "Explain recursion."
print("\nVersion 1 (Basic):")
print(f"Prompt: {v1}")
response = llm.complete(v1, max_tokens=100)
print(f"Response: {response}")

# Version 2: Add context
v2 = """
Context: Teaching computer science to beginners.
Explain recursion in simple terms.
"""
print("\nVersion 2 (Add Context):")
print(f"Prompt: {v2}")
response = llm.complete(v2, max_tokens=120)
print(f"Response: {response}")

# Version 3: Add example requirement
v3 = """
Context: Teaching computer science to beginners.
Explain recursion in simple terms.
Include a real-world analogy (not code).
"""
print("\nVersion 3 (Add Example Requirement):")
print(f"Prompt: {v3}")
response = llm.complete(v3, max_tokens=150)
print(f"Response: {response}")

# Version 4: Complete with all components
v4 = """
Context: Teaching computer science to beginners who understand loops but not recursion.

Task: Explain what recursion is and when to use it.

Requirements:
- Start with a real-world analogy
- Then connect it to programming
- Mention one clear use case
- Avoid technical jargon

Format:
1. Real-world analogy (2 sentences)
2. Programming explanation (2 sentences)
3. When to use it (1 sentence)

Constraint: Keep total response under 100 words.
"""
print("\nVersion 4 (Complete Refinement):")
print(f"Prompt: {v4}")
response = llm.complete(v4, max_tokens=200)
print(f"Response: {response}")


print("\n" + "=" * 50)
print("Key Takeaways")
print("=" * 50)
print("""
Effective prompts have five key components:
1. Context - Background information for the model
2. Instruction - Clear directive about the task
3. Input Data - The specific content to process
4. Output Format - How you want the response structured
5. Constraints - Limitations and requirements

Remember:
- Start simple and iterate
- Be explicit about what you want
- Specify format and constraints
- Provide relevant context
- Test and refine based on outputs
""")