# Module 01: Prompt Engineering Fundamentals

## 📚 Overview

Welcome to the foundation of prompt engineering! This module introduces core concepts and principles that will guide your entire journey through LLM interaction and application development.

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand what prompt engineering is and why it matters
- Master the anatomy of an effective prompt
- Learn the key principles of clarity, specificity, and context
- Recognize common pitfalls and how to avoid them
- Apply temperature and parameter tuning basics

## 📖 Key Concepts

### What is Prompt Engineering?

Prompt engineering is the practice of designing and optimizing text inputs (prompts) to effectively communicate with Large Language Models (LLMs) to achieve desired outputs. It's both an art and a science that requires:

- **Clarity**: Being explicit about what you want
- **Context**: Providing necessary background information
- **Constraints**: Setting boundaries and requirements
- **Examples**: Showing desired output format when helpful

### The Anatomy of a Prompt

A well-structured prompt typically includes:

1. **Context/Background**: Relevant information the model needs
2. **Instruction**: Clear directive about the task
3. **Input Data**: The specific content to process
4. **Output Format**: How you want the response structured
5. **Constraints**: Any limitations or requirements

### Core Principles

#### 1. Be Specific and Clear
```python
# Poor prompt
"Tell me about dogs"

# Better prompt
"Write a 200-word educational summary about Golden Retrievers, 
focusing on their temperament, exercise needs, and suitability 
for families with young children."
```

#### 2. Provide Context
```python
# Without context
"Translate this to French: Hello"

# With context
"You are a professional translator specializing in business communication. 
Translate this greeting for a formal business email to French: Hello"
```

#### 3. Use Delimiters
```python
# Using delimiters to separate instructions from content
prompt = """
Summarize the text below in 3 bullet points.

Text: ###
{your_text_here}
###
"""
```

## 🛠️ Essential Parameters

### Temperature
- **0.0**: Deterministic, focused, factual
- **0.7**: Balanced creativity and coherence
- **1.0+**: Creative, diverse, potentially chaotic

### Max Tokens
Controls the maximum length of the response. Plan for:
- Short answers: 50-100 tokens
- Paragraphs: 200-500 tokens  
- Essays: 1000+ tokens

### Top-p (Nucleus Sampling)
Alternative to temperature for controlling randomness:
- **0.1**: Very focused, only top tokens
- **0.9**: Diverse but coherent
- **1.0**: Consider all tokens

## 💡 Best Practices

### 1. Start Simple, Then Iterate
Begin with a basic prompt and refine based on outputs:
```python
# Iteration 1
"Write a product description"

# Iteration 2  
"Write a 100-word product description for a smartphone"

# Iteration 3
"Write a 100-word product description for the iPhone 15 Pro, 
highlighting camera capabilities and battery life, 
targeted at photography enthusiasts"
```

### 2. Use System Messages (When Available)
```python
import openai

messages = [
    {"role": "system", "content": "You are a helpful assistant specialized in Python programming."},
    {"role": "user", "content": "Explain list comprehensions"}
]
```

### 3. Handle Edge Cases
Always consider:
- Empty inputs
- Ambiguous requests
- Contradictory instructions
- Out-of-scope queries

## ⚠️ Common Pitfalls

### 1. Assuming Context
The model doesn't know your previous conversations (unless in same session) or your specific situation.

### 2. Overloading the Prompt
Trying to accomplish too many tasks in one prompt often leads to poor results.

### 3. Neglecting Output Format
Not specifying how you want the output structured leads to inconsistent results.

## 🔬 Hands-On Exercises

### Exercise 1: Prompt Refinement
Take this vague prompt and improve it step by step:
```
Initial: "Help me with my presentation"
Your improved version: ?
```

### Exercise 2: Temperature Experimentation
Run the same prompt with different temperatures:
```python
prompt = "Generate a creative name for a coffee shop"
# Try with temperature: 0.0, 0.5, 0.9, 1.2
```

### Exercise 3: Format Specification
Write prompts that generate outputs in:
- JSON format
- Markdown table
- Numbered list
- Python dictionary

## 📝 Practice Project

**Build a Prompt Library**

Create a Python module that:
1. Stores reusable prompt templates
2. Allows parameter substitution
3. Includes metadata (purpose, optimal temperature)
4. Provides examples of successful outputs

See `exercises/prompt_library_starter.py` for starter code.

## 🎯 Module Challenge

Create a prompt that:
1. Generates a technical tutorial
2. Includes code examples
3. Has consistent formatting
4. Handles error cases
5. Works reliably across multiple runs

Submit your solution in `solutions/module_challenge.py`

## 📊 Self-Assessment Checklist

- [ ] I can identify the key components of an effective prompt
- [ ] I understand how temperature affects output
- [ ] I can write clear, specific instructions
- [ ] I know when and how to provide context
- [ ] I can specify output formats consistently
- [ ] I recognize and avoid common pitfalls

## 📚 Additional Resources

### Required Reading
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Best Practices for Prompt Engineering](https://help.openai.com/en/articles/6654000)

### Optional Deep Dives
- Research Paper: "Large Language Models are Zero-Shot Reasoners"
- Blog: "The Art of Prompt Engineering" by Anthropic
- Video: "Prompt Engineering Fundamentals" - DeepLearning.AI

## 🔜 Next Steps

Once you're comfortable with these fundamentals, proceed to:
- [Module 02: Zero-Shot Prompting →](../02-zero-shot-prompting/README.md)

---

*Remember: Prompt engineering is iterative. Every interaction teaches you something new about how to communicate with LLMs more effectively.*