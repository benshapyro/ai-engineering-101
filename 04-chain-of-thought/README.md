# Module 04: Chain-of-Thought Prompting

## Learning Objectives
By the end of this module, you will:
- Understand when and why to use chain-of-thought (CoT) reasoning
- Master different CoT patterns (standard, zero-shot, auto-CoT)
- Learn to decompose complex problems into reasoning steps
- Implement self-verification and error checking in reasoning chains
- Build production systems that leverage CoT for accuracy

## Key Concepts

### 1. What is Chain-of-Thought?
Chain-of-Thought prompting encourages models to show their reasoning process step-by-step before arriving at a final answer. This dramatically improves performance on tasks requiring multi-step reasoning, mathematical calculations, or logical deduction.

### 2. Core Principles
- **Explicit Reasoning**: Make the thinking process visible
- **Step Decomposition**: Break complex problems into manageable steps
- **Intermediate Verification**: Check work at each stage
- **Error Recovery**: Identify and correct mistakes in reasoning

### 3. CoT Patterns

#### Standard CoT
```python
prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?

A: Let's think step by step.
Roger started with 5 tennis balls.
He bought 2 cans of tennis balls.
Each can has 3 tennis balls, so 2 cans have 2 × 3 = 6 tennis balls.
In total, Roger has 5 + 6 = 11 tennis balls.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more,
how many apples do they have?

A: Let's think step by step.
"""
```

#### Zero-Shot CoT
```python
prompt = """
Let's think step by step to solve this problem:

If a store has a 25% off sale and an additional 10% off for members,
what's the final price of a $80 item for a member?
"""
```

#### Auto-CoT (Self-Generated Examples)
```python
# Model generates its own reasoning examples
prompt = """
Generate 3 examples of solving percentage problems with step-by-step reasoning,
then solve this new problem using the same approach:
What's 15% of 240?
"""
```

### 4. When to Use CoT

**Effective for:**
- Mathematical word problems
- Multi-step logical reasoning
- Complex decision trees
- Code generation with logic
- Scientific problem solving
- Business analysis calculations

**Less effective for:**
- Simple lookups or facts
- Creative writing
- Subjective opinions
- Pattern matching without reasoning

### 5. Quality Indicators

Good CoT reasoning shows:
- Clear step labels (Step 1, Step 2, etc.)
- Explicit calculations
- Intermediate conclusions
- Verification of results
- Error checking

### 6. Common Pitfalls
- **Reasoning Drift**: Steps become disconnected
- **Circular Logic**: Using conclusion to justify steps
- **Skip Steps**: Missing critical intermediate reasoning
- **Over-Reasoning**: Adding unnecessary complexity
- **Arithmetic Errors**: Calculation mistakes in steps

## Module Structure

### Examples
1. `basic_reasoning.py` - Fundamental CoT patterns
2. `step_by_step.py` - Structured step decomposition
3. `verification_patterns.py` - Self-checking and validation

### Exercises
Practice problems focusing on:
- Converting direct prompts to CoT
- Debugging faulty reasoning chains
- Optimizing step granularity
- Building domain-specific CoT templates

### Project: Reasoning Chain Analyzer
Build a system that:
- Parses CoT outputs into structured steps
- Validates logical consistency
- Identifies potential errors
- Suggests reasoning improvements
- Tracks reasoning patterns across queries

## Best Practices

### 1. Step Structure
```python
good_structure = """
Step 1: [Clear action or analysis]
Result: [Specific outcome]

Step 2: [Next action building on Step 1]
Result: [Specific outcome]

Conclusion: [Final answer based on all steps]
"""
```

### 2. Verification Patterns
```python
verification = """
Let me verify this answer:
- Check 1: [Validation method]
- Check 2: [Alternative approach]
- Check 3: [Boundary conditions]
The answer is confirmed: [result]
"""
```

### 3. Error Recovery
```python
error_recovery = """
Wait, let me reconsider Step 3...
Actually, [correction]
This changes the final answer to: [updated result]
"""
```

### 4. Domain-Specific CoT
Customize reasoning patterns for specific domains:
- **Legal**: Precedent → Statute → Application → Conclusion
- **Medical**: Symptoms → Differential → Tests → Diagnosis
- **Engineering**: Requirements → Constraints → Design → Validation

## Production Considerations

### Performance Impact
- **Token Usage**: CoT increases token consumption 2-5x
- **Latency**: Longer responses mean higher latency
- **Cost**: More tokens = higher API costs

### Optimization Strategies
1. **Selective CoT**: Only use for complex problems
2. **CoT Caching**: Store reasoning for similar problems
3. **Parallel Processing**: Run multiple CoT chains concurrently
4. **Progressive Refinement**: Start simple, add detail as needed

### Quality Assurance
```python
class CoTValidator:
    def validate_reasoning(self, cot_output):
        checks = {
            'has_steps': self.check_step_presence(cot_output),
            'logical_flow': self.check_logical_consistency(cot_output),
            'math_accurate': self.verify_calculations(cot_output),
            'conclusion_supported': self.check_conclusion(cot_output)
        }
        return all(checks.values()), checks
```

## Common Patterns

### 1. Mathematical Reasoning
```
Given: [problem statement]
Step 1: Identify knowns and unknowns
Step 2: Set up equations
Step 3: Solve step by step
Step 4: Verify answer
```

### 2. Logical Deduction
```
Premises: [list facts]
Step 1: Apply rule/principle
Step 2: Draw intermediate conclusion
Step 3: Combine with other facts
Conclusion: Therefore...
```

### 3. Algorithm Design
```
Problem: [description]
Step 1: Understand requirements
Step 2: Consider edge cases
Step 3: Design approach
Step 4: Trace through example
Step 5: Analyze complexity
```

## Exercises Overview

1. **Basic CoT Construction**: Convert direct prompts to CoT format
2. **Reasoning Debugging**: Fix broken reasoning chains
3. **Domain Adaptation**: Create CoT templates for specific fields
4. **Verification Systems**: Build self-checking mechanisms
5. **Optimization**: Balance reasoning depth with efficiency

## Success Metrics
- **Accuracy Improvement**: 20-50% on reasoning tasks
- **Error Detection**: Catch 80%+ of reasoning mistakes
- **User Trust**: Explainable outputs increase confidence
- **Cost Efficiency**: Optimal token usage for task complexity

## Next Steps
After mastering CoT, you'll move to Module 05: Prompt Chaining, where you'll learn to connect multiple prompts into complex workflows, building on the reasoning foundations established here.