# Module 03: Few-Shot Learning

## 📚 Overview

Few-shot learning teaches LLMs to perform tasks by providing a small number of examples. This powerful technique bridges the gap between zero-shot prompting and fine-tuning, allowing you to guide model behavior with just a handful of demonstrations.

## 🎯 Learning Objectives

By the end of this module, you will:
- Master the art of crafting effective examples
- Understand when few-shot outperforms zero-shot
- Learn optimal example selection strategies
- Implement dynamic few-shot systems
- Handle format consistency and edge cases
- Build production-ready few-shot applications

## 📖 Key Concepts

### What is Few-Shot Learning?

Few-shot learning provides the model with a small number of input-output examples before asking it to process new inputs. The model learns the pattern from these examples and applies it to new cases.

```python
# Few-shot example
prompt = """
Classify the sentiment of restaurant reviews:

Review: "The food was amazing and service was excellent!"
Sentiment: Positive

Review: "Terrible experience, cold food and rude staff."
Sentiment: Negative

Review: "Decent meal, nothing special but not bad either."
Sentiment: Neutral

Review: "Best pizza I've ever had, will definitely return!"
Sentiment:"""
# Model learns pattern from examples and classifies the new review
```

### Zero-Shot vs Few-Shot: When to Use Each

| Use Zero-Shot When | Use Few-Shot When |
|-------------------|-------------------|
| Task is straightforward | Task has specific format requirements |
| General knowledge suffices | Domain-specific patterns needed |
| Maximum flexibility needed | Consistency is critical |
| No examples available | Quality examples exist |
| Testing baseline capability | Production accuracy required |

### The Power of Examples

Examples serve multiple purposes:
1. **Pattern Demonstration** - Show the task structure
2. **Format Specification** - Define output format implicitly
3. **Edge Case Coverage** - Handle special cases
4. **Style Guide** - Set tone and approach
5. **Constraint Enforcement** - Show boundaries

## 🔬 Core Principles

### 1. Example Quality > Quantity

```python
# Poor: Many low-quality examples
prompt = """
Translate to French:
hello -> salut
dog -> chien
run -> cours
eat -> mange
big -> grand
small -> petit
good -> bon
bad -> mal

translate: The cat is sleeping ->"""

# Better: Few high-quality examples
prompt = """
Translate English to French (formal register):

"Good morning, how are you?" -> "Bonjour, comment allez-vous ?"
"Thank you for your help." -> "Merci pour votre aide."
"The meeting is scheduled for tomorrow." -> "La réunion est prévue pour demain."

"Could you please send me the report?" ->"""
```

### 2. Format Consistency is Critical

```python
# Inconsistent format (problematic)
prompt = """
Extract information:

Text: John is 30 years old
Output: Age = 30

Text: Sarah works at Google
→ Company: Google

Text: Mike lives in New York
Result - Location: New York

Text: Lisa earned $50,000 last year
Output:"""  # Model confused by format variations

# Consistent format (effective)
prompt = """
Extract information:

Text: John is 30 years old
Info: {{"age": 30}}

Text: Sarah works at Google
Info: {{"company": "Google"}}

Text: Mike lives in New York
Info: {{"location": "New York"}}

Text: Lisa earned $50,000 last year
Info:"""  # Clear, consistent pattern
```

### 3. Example Selection Strategies

#### Similarity-Based Selection
Choose examples most similar to the current input:
```python
def select_similar_examples(query, example_bank, n=3):
    # Calculate similarity scores
    similarities = [
        calculate_similarity(query, ex) for ex in example_bank
    ]
    # Return top n most similar
    return top_n_examples(similarities, n)
```

#### Diversity-Based Selection
Cover different aspects of the task:
```python
examples = [
    {"input": "Short text", "output": "..."},      # Length variety
    {"input": "Technical content", "output": "..."}, # Domain variety
    {"input": "Question format", "output": "..."},  # Format variety
]
```

#### Stratified Selection
Ensure balanced representation:
```python
# For classification, include example of each class
examples = [
    ("Positive example", "Positive"),
    ("Negative example", "Negative"),
    ("Neutral example", "Neutral"),
]
```

### 4. Optimal Shot Count

Research and practice show:
- **1-shot**: Simple pattern matching, format demonstration
- **2-3 shots**: Most common, balances context usage and coverage
- **4-5 shots**: Complex patterns, multiple edge cases
- **5+ shots**: Diminishing returns, context window concerns

```python
def determine_shot_count(task_complexity, context_limit, example_size):
    if task_complexity == "simple":
        return 1
    elif task_complexity == "moderate":
        return min(3, context_limit // example_size)
    else:  # complex
        return min(5, context_limit // example_size)
```

### 5. Dynamic Few-Shot

Adapt examples based on the input:
```python
def create_dynamic_prompt(user_input, example_library):
    # Detect input characteristics
    input_type = classify_input(user_input)
    complexity = assess_complexity(user_input)

    # Select appropriate examples
    examples = example_library.get_examples(
        type=input_type,
        complexity=complexity,
        count=3
    )

    # Build prompt
    return format_prompt(examples, user_input)
```

## 🛠️ Advanced Techniques

### Chain-of-Thought Few-Shot

Include reasoning in examples:
```python
prompt = """
Solve these math problems:

Problem: If a train travels 120 miles in 2 hours, what's its speed?
Reasoning: Speed = Distance / Time = 120 miles / 2 hours = 60 mph
Answer: 60 mph

Problem: A shirt costs $20 after a 20% discount. What was the original price?
Reasoning: If $20 is 80% of original (100% - 20%), then original = $20 / 0.8 = $25
Answer: $25

Problem: If 5 workers complete a job in 10 days, how long for 2 workers?
Reasoning:"""
```

### Negative Examples

Show what NOT to do:
```python
prompt = """
Format phone numbers correctly:

Correct: (555) 123-4567
Incorrect: 555.123.4567 ❌ (don't use periods)

Correct: (555) 987-6543
Incorrect: 555-987-6543 ❌ (missing parentheses)

Correct: (555) 246-8135
Incorrect: 5552468135 ❌ (needs formatting)

Format: 5559876543
Answer:"""
```

### Progressive Complexity

Start simple, build up:
```python
prompt = """
Parse these expressions:

Simple: 2 + 3
Result: 5

Moderate: (4 + 5) * 2
Result: 18

Complex: ((10 - 3) * 2) + (15 / 3)
Result: 19

Parse: ((8 + 2) * (6 - 1)) / 2
Result:"""
```

## ⚠️ Common Pitfalls

### 1. Overfitting to Examples
**Problem**: Model copies examples too literally
**Solution**: Use diverse examples, test with varied inputs

### 2. Format Drift
**Problem**: Output format degrades with complex inputs
**Solution**: Strong format consistency, explicit format instructions

### 3. Example Bias
**Problem**: Biased examples lead to biased outputs
**Solution**: Carefully curate balanced, representative examples

### 4. Context Window Overflow
**Problem**: Too many examples exceed token limits
**Solution**: Dynamic example selection, example compression

## 🔬 Hands-On Exercises

### Exercise 1: Example Crafting
Create effective examples for classification, extraction, and generation tasks.

### Exercise 2: Shot Optimization
Find the optimal number of examples for different task types.

### Exercise 3: Format Matching
Ensure outputs consistently match example formats.

### Exercise 4: Example Debugging
Identify and fix issues in problematic few-shot prompts.

### Exercise 5: Dynamic Selection
Build a system that selects examples based on input characteristics.

## 📝 Module Project

**Build a Few-Shot Learning Manager**

Create a comprehensive system that:
1. Manages libraries of examples
2. Automatically selects relevant examples
3. Validates format consistency
4. Tracks performance metrics
5. A/B tests different example sets
6. Adapts to user feedback

## 🎯 Self-Assessment Checklist

- [ ] I can identify when few-shot is better than zero-shot
- [ ] I can craft effective, consistent examples
- [ ] I know how to select the optimal number of shots
- [ ] I can implement dynamic example selection
- [ ] I can debug and improve few-shot prompts
- [ ] I understand format consistency importance
- [ ] I can handle edge cases with examples

## 📊 Module Challenge

Create a few-shot system that can:
1. Automatically detect task type from user input
2. Select optimal examples from a large library
3. Adapt example count based on complexity
4. Maintain format consistency across diverse inputs
5. Learn from user feedback to improve example selection

## 📚 Additional Resources

### Research Papers
- "Language Models are Few-Shot Learners" (GPT-3 Paper)
- "What Makes Good In-Context Examples for GPT-3?"
- "Rethinking the Role of Demonstrations"

### Practical Guides
- OpenAI's Few-Shot Learning Best Practices
- Anthropic's Guide to Example Selection
- "The Art of Prompt Examples" - Community Guide

### Tools and Libraries
- LangChain's FewShotPromptTemplate
- Example selection algorithms
- Embedding-based similarity tools

## 🔜 Next Steps

After mastering few-shot learning, you're ready for:
- [Module 04: Chain-of-Thought →](../04-chain-of-thought/README.md)

Where you'll learn to make models "think" step-by-step!

---

*Remember: The best few-shot prompt is one where the examples teach the pattern so clearly that even a human could continue the sequence.*