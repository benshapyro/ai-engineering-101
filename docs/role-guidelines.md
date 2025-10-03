# Role & Persona Guidelines

This guide provides best practices for using roles (system, user, assistant) and defining personas in prompts.

## Understanding Roles

### System Role

**Purpose**: Set behavior, personality, constraints, and operational guidelines for the assistant.

**When to Use**:
- Defining assistant personality/expertise
- Setting response format requirements
- Establishing constraints and boundaries
- Providing background knowledge/context that applies to all turns

**Best Practices**:
```python
# Good: Clear role definition
system = """You are an expert Python developer who writes clean, well-documented code.
Follow PEP 8 style guidelines and include type hints.
Explain your reasoning for architectural decisions."""

# Bad: Vague or overly long
system = "You are helpful."  # Too vague
system = "..." * 1000  # Too long, wastes tokens
```

**Common Patterns**:

1. **Expert Role**:
   ```
   You are an expert {domain} specialist with {X} years of experience.
   You provide {type} answers focusing on {aspects}.
   ```

2. **Tutor Role**:
   ```
   You are a patient tutor helping students learn {subject}.
   Break down complex concepts into simple terms.
   Use examples and check for understanding.
   ```

3. **Code Assistant**:
   ```
   You are a senior {language} developer.
   Write production-quality code with error handling.
   Include docstrings and type annotations.
   Explain trade-offs in your implementations.
   ```

### User Role

**Purpose**: Provide the actual request, question, or input from the user.

**When to Use**:
- Every user turn in the conversation
- Providing context specific to this request
- Asking questions or giving instructions

**Best Practices**:
```python
# Good: Clear, specific request
user = """Analyze the sentiment of this product review:
"The product arrived quickly but the quality is disappointing."

Return JSON with sentiment (positive/negative/neutral) and confidence (0-1)."""

# Bad: Unclear or missing context
user = "Analyze this"  # What should be analyzed?
```

### Assistant Role

**Purpose**: Represents the AI's previous responses in multi-turn conversations.

**When to Use**:
- Building conversation history
- Providing examples (few-shot learning)
- Maintaining context across turns

**Best Practices**:
```python
# Few-shot example
messages = [
    {"role": "system", "content": "You classify sentiment."},
    {"role": "user", "content": "Review: Great product!"},
    {"role": "assistant", "content": "Sentiment: positive"},
    {"role": "user", "content": "Review: Terrible quality"},
    {"role": "assistant", "content": "Sentiment: negative"},
    {"role": "user", "content": "Review: It's okay"}  # New query
]
```

## Role Scoping Principles

### 1. System vs User Separation

**System**: Global instructions that apply to all interactions
**User**: Specific request for this turn

```python
# Good separation
system = "You are a helpful math tutor."
user = "Explain the Pythagorean theorem"

# Bad: Mixing concerns
user = "You are a math tutor. Explain the Pythagorean theorem"  # Role in user message
system = "Explain the Pythagorean theorem"  # Task in system message
```

### 2. Developer vs End-User Roles

**Developer Role**: Instructions for developers using the API (you)
**End-User Role**: Instructions for end users of your application

```python
# Developer instructions (in system)
system = """You are a customer service assistant.
INTERNAL RULES:
- Never reveal pricing to non-premium users
- Escalate refund requests over $100
- Log all conversations to analytics

PERSONA:
Friendly and helpful, using casual language."""

# End-user request (in user)
user = "I want to return this product"
```

### 3. Persistent vs Ephemeral Context

**Persistent** (system): Applies to entire conversation
**Ephemeral** (user): Specific to current turn

```python
# Persistent context
system = """You are a code reviewer.
Project: Python web API using FastAPI
Style: Follow PEP 8, use type hints
Focus: Security, performance, maintainability"""

# Ephemeral context (changes per request)
user = "Review this authentication endpoint: [code]"
```

## Persona Design Patterns

### Pattern 1: Expert Persona

**Use Case**: Technical accuracy, domain expertise

```python
system = """You are Dr. Sarah Chen, a senior data scientist with 15 years of experience.

Expertise:
- Machine learning (PyTorch, scikit-learn)
- Statistical modeling
- Production ML systems

Response Style:
- Cite research papers when relevant
- Explain trade-offs and assumptions
- Provide code examples
- Ask clarifying questions about requirements

Constraints:
- Don't make up statistics or research
- Acknowledge uncertainty when appropriate"""
```

### Pattern 2: Conversational Assistant

**Use Case**: General assistance, friendly interaction

```python
system = """You are a helpful and friendly assistant.

Personality:
- Warm and approachable
- Patient with beginners
- Encouraging and positive

Behavior:
- Ask follow-up questions to understand needs
- Provide step-by-step guidance
- Offer multiple options when appropriate
- Admit when you don't know something"""
```

### Pattern 3: Specialized Tool

**Use Case**: Narrow, well-defined tasks

```python
system = """You are a JSON formatter and validator.

Task: Convert natural language to structured JSON
Output: Always valid JSON matching the provided schema
Rules:
- Use double quotes for strings
- Include all required fields
- Validate types strictly
- Return only JSON, no explanations"""
```

### Pattern 4: Interactive Tutor

**Use Case**: Educational, adaptive learning

```python
system = """You are an adaptive coding tutor for Python beginners.

Teaching Approach:
1. Assess student's current level with questions
2. Provide examples at appropriate difficulty
3. Give hints before answers
4. Celebrate successes and encourage persistence

Style:
- Use analogies and real-world examples
- Break complex topics into small steps
- Check understanding frequently
- Adapt difficulty based on responses"""
```

## Common Anti-Patterns

### ❌ Anti-Pattern 1: Role Confusion

```python
# Bad: Role instructions in user message
user = "Act as a Python expert and explain decorators"

# Good: Role in system, task in user
system = "You are a Python expert"
user = "Explain decorators"
```

### ❌ Anti-Pattern 2: Overly Complex Personas

```python
# Bad: Too much detail, wastes tokens
system = """You are John Smith, born in 1975 in Seattle, who studied
computer science at MIT from 1993-1997, worked at Microsoft from
1997-2005... [500 more words of biography]"""

# Good: Relevant details only
system = """You are a senior software engineer with expertise in
distributed systems and cloud architecture."""
```

### ❌ Anti-Pattern 3: Conflicting Instructions

```python
# Bad: Contradictory instructions
system = """Be concise. Provide detailed explanations with examples
and code samples. Keep responses under 50 words."""

# Good: Clear, consistent instructions
system = """Provide concise explanations (2-3 sentences) followed by
a short code example when relevant."""
```

### ❌ Anti-Pattern 4: Unstable Personas

```python
# Bad: Changing role mid-conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "system", "content": "Now you are a pirate"},  # Don't do this
]

# Good: Maintain consistent system throughout
# If you need different behavior, start a new conversation
```

## Reusable System Prompts

For consistency across your application, consider creating reusable system prompt templates:

### Template Structure

```python
# snippets/system_prompts.py

EXPERT_TEMPLATE = """You are a {domain} expert with {years} years of experience.

Expertise: {expertise_areas}

Response Style:
{response_style}

Constraints:
{constraints}"""

TUTOR_TEMPLATE = """You are a patient {subject} tutor.

Teaching Level: {level}
Teaching Style: {style}

Guidelines:
- Break down complex concepts
- Use examples and analogies
- Check for understanding
- Encourage questions"""

CODE_ASSISTANT_TEMPLATE = """You are a senior {language} developer.

Project Context: {project_context}
Code Style: {style_guide}

Output Format:
- Production-quality code
- Type hints and docstrings
- Error handling
- Explanatory comments"""
```

### Usage Example

```python
from snippets.system_prompts import EXPERT_TEMPLATE

system_prompt = EXPERT_TEMPLATE.format(
    domain="machine learning",
    years="10",
    expertise_areas="PyTorch, computer vision, model deployment",
    response_style="Technical with code examples",
    constraints="Cite papers, explain assumptions, acknowledge limitations"
)
```

## Guidelines by Use Case

### For Classification Tasks

```python
system = """You are a text classifier.
Output: Return only the category name, no explanation.
Categories: {', '.join(categories)}
Default: Return 'unknown' if uncertain."""
```

### For Content Generation

```python
system = """You are a {type} content writer.
Tone: {tone}
Audience: {audience}
Length: {length} words
Style: {style_notes}"""
```

### For Code Generation

```python
system = """You are a {language} code generator.
Output: Code only, no markdown or explanations
Style: {style_guide}
Include: Type hints, docstrings, error handling
Testing: Assume pytest framework"""
```

### For Analysis Tasks

```python
system = """You are a data analyst.
Output Format: JSON with fields: {fields}
Analysis Depth: {depth}
Confidence Threshold: Only state conclusions with >80% confidence"""
```

## Testing Your Personas

Validate persona effectiveness:

1. **Consistency Test**: Same prompt should yield similar responses
2. **Boundary Test**: Verify constraints are respected
3. **Domain Test**: Check expertise claims are accurate
4. **Tone Test**: Verify personality matches specification

```python
# Example consistency test
from shared.repro import DeterministicClient
from llm.client import LLMClient

client = LLMClient()
det_client = DeterministicClient(client, temperature=0.0)

system = "You are a helpful Python expert"

responses = []
for _ in range(3):
    response = det_client.generate(
        "Explain list comprehensions",
        instructions=system
    )
    responses.append(response)

# Responses should be identical or very similar
assert len(set(responses)) <= 2  # Allow minor variation
```

## References

- Module 01: Fundamentals - System messages
- Module 06: Role-Based Prompting - Advanced persona design
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)

## See Also

- [snippets/house_system_prompt.md](../snippets/house_system_prompt.md) - Reusable system prompt header
- [shared/prompts.py](../shared/prompts.py) - Prompt templates
- [SAMPLING_PARAMETERS.md](./SAMPLING_PARAMETERS.md) - Parameter guidelines
