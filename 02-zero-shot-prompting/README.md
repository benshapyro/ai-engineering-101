# Module 02: Zero-Shot Prompting

## 📚 Overview

Zero-shot prompting is the art of getting LLMs to perform tasks without providing any examples. This module teaches you how to craft effective instructions that leverage the model's pre-trained knowledge to solve new problems.

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand when and why to use zero-shot prompting
- Master instruction engineering techniques
- Learn to leverage model knowledge effectively
- Handle edge cases and ambiguity
- Optimize prompts for reliability without examples

## 📖 Key Concepts

### What is Zero-Shot Prompting?

Zero-shot prompting asks the model to perform a task based solely on instructions, without any examples. The model relies entirely on its pre-trained knowledge and your task description.

```python
# Zero-shot example
prompt = "Classify this text as positive, negative, or neutral: 'The product exceeded my expectations!'"
# No examples provided - model uses its understanding
```

### When to Use Zero-Shot

Zero-shot prompting is ideal when:
- The task is straightforward and well-defined
- You want to test the model's baseline capabilities
- You don't have good examples readily available
- The task leverages common knowledge or reasoning
- You need maximum flexibility without example constraints

### Core Principles

#### 1. Clear Task Definition
Be explicit about what you want the model to do:
```python
# Vague
"Analyze this text"

# Clear
"Identify the main argument, supporting evidence, and conclusion in this text"
```

#### 2. Specify Output Format
Tell the model exactly how to structure its response:
```python
# Without format specification
"List the key points"

# With format specification
"List the key points as:
- Main Point: [description]
- Supporting Detail 1: [description]
- Supporting Detail 2: [description]"
```

#### 3. Provide Context
Give necessary background information:
```python
# Without context
"Is this correct?"

# With context
"Given that we're following PEP 8 style guidelines for Python, is this function naming correct?"
```

#### 4. Set Constraints
Define boundaries and limitations:
```python
# Without constraints
"Summarize this article"

# With constraints
"Summarize this article in exactly 3 sentences, focusing on the economic implications"
```

## 🔧 Advanced Techniques

### Instruction Templates

Create reusable instruction patterns:

```python
def create_classification_prompt(text, categories):
    return f"""Classify the following text into one of these categories: {', '.join(categories)}

Text: {text}

Category:"""
```

### Chain-of-Thought Zero-Shot

Encourage reasoning without examples:

```python
prompt = """Solve this problem step by step:
If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours,
what is its average speed for the entire journey?

Let's think through this step by step:"""
```

### Role-Based Zero-Shot

Leverage personas for better responses:

```python
prompt = """You are a financial advisor. A client asks:
"Should I invest my emergency fund in cryptocurrency?"

Provide professional advice considering risk management principles:"""
```

### Negative Instructions

Sometimes telling the model what NOT to do helps:

```python
prompt = """Translate this to French: 'Hello, how are you?'

Do NOT:
- Add explanations
- Include pronunciation guides
- Provide alternative translations

Translation:"""
```

## 🛠️ Common Patterns

### Pattern 1: Classification
```python
"Classify [input] as [category1, category2, ...].
Consider [criteria].
Output only the category name."
```

### Pattern 2: Extraction
```python
"Extract [specific information] from the following text.
Format as: [desired structure].
If not found, respond with 'N/A'."
```

### Pattern 3: Generation
```python
"Generate [type of content] that [requirements].
Constraints: [limitations].
Style: [tone/format]."
```

### Pattern 4: Analysis
```python
"Analyze [input] for [specific aspects].
Provide:
1. [First analysis point]
2. [Second analysis point]
Conclusion: [summary requirement]"
```

### Pattern 5: Transformation
```python
"Convert [input] from [format A] to [format B].
Maintain [what to preserve].
Adjust [what to change]."
```

## ⚠️ Common Pitfalls

### 1. Assuming Too Much Context
**Problem**: The model doesn't know your specific situation
**Solution**: Provide necessary background information

### 2. Ambiguous Instructions
**Problem**: Multiple valid interpretations
**Solution**: Be specific and use examples of format (not task)

### 3. Overly Complex Tasks
**Problem**: Too many requirements in one prompt
**Solution**: Break down into steps or use few-shot instead

### 4. Inconsistent Outputs
**Problem**: Results vary significantly between runs
**Solution**: Lower temperature, add more constraints

## 🔬 Hands-On Exercises

### Exercise 1: Instruction Clarity
Transform vague instructions into clear, zero-shot prompts for various tasks.

### Exercise 2: Format Control
Practice specifying exact output formats without examples.

### Exercise 3: Edge Case Handling
Write zero-shot prompts that gracefully handle unusual inputs.

### Exercise 4: Task Decomposition
Break complex tasks into zero-shot promptable subtasks.

### Exercise 5: Reliability Testing
Test the same zero-shot prompt multiple times and improve consistency.

## 📝 Module Project

**Build a Zero-Shot Task Processor**

Create a system that:
1. Takes a task description from the user
2. Automatically generates an optimized zero-shot prompt
3. Handles various task types (classification, extraction, generation)
4. Includes error handling and validation
5. Tests reliability across multiple runs

## 🎯 Self-Assessment Checklist

- [ ] I can write clear, unambiguous zero-shot prompts
- [ ] I know when zero-shot is appropriate vs few-shot
- [ ] I can specify output formats without examples
- [ ] I can handle edge cases in zero-shot prompts
- [ ] I can improve consistency of zero-shot outputs
- [ ] I understand the limitations of zero-shot prompting

## 📊 Module Challenge

Create a zero-shot prompt that can:
1. Analyze any piece of code (language agnostic)
2. Identify potential bugs or issues
3. Suggest improvements
4. Rate code quality on multiple dimensions
5. Work consistently across different programming languages

All without providing any code examples in the prompt!

## 📚 Additional Resources

### Required Reading
- [Zero-Shot Learning in Modern NLP](https://joeddav.github.io/blog/2020/05/29/ZSL.html)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - Section on Zero-Shot

### Practical Guides
- OpenAI's Guide to Instruction Following
- Anthropic's Constitutional AI Principles

### Research Papers
- "Large Language Models are Zero-Shot Reasoners" (2022)
- "Instruction Tuning for Large Language Models" (2023)

## 🔜 Next Steps

After mastering zero-shot prompting, you'll be ready for:
- [Module 03: Few-Shot Learning →](../03-few-shot-learning/README.md)

Where you'll learn to provide examples for even better performance!

---

*Remember: Zero-shot prompting is powerful but has limits. Knowing when to switch to few-shot is as important as mastering zero-shot itself.*