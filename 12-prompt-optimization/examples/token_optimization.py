"""
Module 12: Prompt Optimization - Token Optimization Examples

This file demonstrates techniques for reducing token usage while
maintaining or improving prompt effectiveness.

Author: Claude
Date: 2024
"""

import os
import re
import tiktoken
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI()

# Initialize tokenizer for GPT models
encoding = tiktoken.encoding_for_model("gpt-5-mini")

# ================================
# Example 1: Token Counting and Analysis
# ================================
print("=" * 50)
print("Example 1: Token Counting and Analysis")
print("=" * 50)

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(encoding.encode(text))

def analyze_token_usage(prompt: str) -> Dict[str, Any]:
    """Analyze token distribution in prompt."""
    lines = prompt.split('\n')

    analysis = {
        "total_tokens": count_tokens(prompt),
        "lines": len(lines),
        "sections": [],
        "token_distribution": {}
    }

    # Analyze each section
    current_section = []
    for line in lines:
        if line.strip() == "":
            if current_section:
                section_text = '\n'.join(current_section)
                section_tokens = count_tokens(section_text)
                analysis["sections"].append({
                    "text": section_text[:50] + "...",
                    "tokens": section_tokens,
                    "percentage": (section_tokens / analysis["total_tokens"]) * 100
                })
                current_section = []
        else:
            current_section.append(line)

    # Add last section if exists
    if current_section:
        section_text = '\n'.join(current_section)
        section_tokens = count_tokens(section_text)
        analysis["sections"].append({
            "text": section_text[:50] + "...",
            "tokens": section_tokens,
            "percentage": (section_tokens / analysis["total_tokens"]) * 100
        })

    return analysis

# Test token analysis
verbose_prompt = """
I would like you to carefully analyze the following text and provide me with a comprehensive summary.
Please make sure to include all the important details and key points.
The summary should be well-structured and easy to understand.
It should cover the main ideas, supporting arguments, and any conclusions.
Please ensure that you maintain the original meaning while making it concise.

Text to summarize:
{text}

Please provide your summary below:
"""

concise_prompt = """
Summarize the key points:

{text}

Summary:
"""

print(f"Verbose prompt tokens: {count_tokens(verbose_prompt)}")
print(f"Concise prompt tokens: {count_tokens(concise_prompt)}")
print(f"Token reduction: {count_tokens(verbose_prompt) - count_tokens(concise_prompt)} tokens")
print(f"Reduction percentage: {((count_tokens(verbose_prompt) - count_tokens(concise_prompt)) / count_tokens(verbose_prompt)) * 100:.1f}%")

# ================================
# Example 2: Redundancy Elimination
# ================================
print("\n" + "=" * 50)
print("Example 2: Redundancy Elimination")
print("=" * 50)

class RedundancyEliminator:
    """Remove redundant words and phrases."""

    def __init__(self):
        self.redundant_phrases = {
            # Verbose -> Concise
            "please make sure to": "",
            "please ensure that": "ensure",
            "in order to": "to",
            "at this point in time": "now",
            "due to the fact that": "because",
            "in the event that": "if",
            "for the purpose of": "for",
            "with regard to": "about",
            "in spite of the fact that": "although",
            "it is important to note that": "note:",
            "it should be noted that": "note:",
            "as a matter of fact": "actually",
            "at the present time": "now",
            "by means of": "by",
            "in light of the fact that": "since",
            "in the near future": "soon",
            "prior to": "before",
            "subsequent to": "after",
            "a large number of": "many",
            "the majority of": "most"
        }

        self.filler_words = [
            "actually", "basically", "literally", "simply",
            "just", "really", "very", "quite", "rather",
            "somewhat", "certainly", "definitely", "surely"
        ]

    def eliminate_redundancy(self, text: str) -> str:
        """Remove redundant phrases and filler words."""
        optimized = text.lower()

        # Replace redundant phrases
        for verbose, concise in self.redundant_phrases.items():
            optimized = optimized.replace(verbose, concise)

        # Remove standalone filler words (careful not to break meaning)
        for filler in self.filler_words:
            # Only remove if it's a standalone word
            optimized = re.sub(rf'\b{filler}\b\s*', '', optimized)

        # Remove multiple spaces
        optimized = re.sub(r'\s+', ' ', optimized)

        # Capitalize first letter of sentences
        optimized = '. '.join(s.strip().capitalize() for s in optimized.split('. '))

        return optimized.strip()

# Test redundancy elimination
eliminator = RedundancyEliminator()

redundant_prompt = """
Please make sure to analyze the data very carefully and provide a comprehensive report.
It is important to note that you should include all relevant metrics.
In order to complete this task, please ensure that you examine each data point.
At this point in time, we need to understand the trends.
Due to the fact that the data is complex, take your time.
"""

optimized_prompt = eliminator.eliminate_redundancy(redundant_prompt)

print("Original prompt:")
print(f"  Tokens: {count_tokens(redundant_prompt)}")
print(f"  Text: {redundant_prompt[:100]}...")

print("\nOptimized prompt:")
print(f"  Tokens: {count_tokens(optimized_prompt)}")
print(f"  Text: {optimized_prompt[:100]}...")

print(f"\nToken savings: {count_tokens(redundant_prompt) - count_tokens(optimized_prompt)}")

# ================================
# Example 3: Instruction Compression
# ================================
print("\n" + "=" * 50)
print("Example 3: Instruction Compression")
print("=" * 50)

class InstructionCompressor:
    """Compress instructions while maintaining clarity."""

    def compress_instructions(self, instructions: str) -> str:
        """Compress verbose instructions to concise form."""
        # Common instruction patterns to compress
        compressions = {
            # Analysis instructions
            r"analyze.*?and provide.*?summary": "summarize",
            r"examine.*?and explain": "explain",
            r"review.*?and identify": "identify",

            # Format instructions
            r"format.*?as.*?list": "list",
            r"structure.*?as.*?table": "tabulate",
            r"organize.*?into.*?categories": "categorize",

            # Output instructions
            r"provide.*?comprehensive.*?analysis": "analyze",
            r"give.*?detailed.*?explanation": "explain",
            r"create.*?complete.*?summary": "summarize"
        }

        compressed = instructions
        for pattern, replacement in compressions.items():
            compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)

        return compressed

    def use_abbreviations(self, text: str) -> str:
        """Use common abbreviations to save tokens."""
        abbreviations = {
            "for example": "e.g.",
            "that is": "i.e.",
            "versus": "vs.",
            "etcetera": "etc.",
            "maximum": "max",
            "minimum": "min",
            "average": "avg",
            "standard deviation": "std",
            "approximately": "~",
            "greater than": ">",
            "less than": "<",
            "equals": "="
        }

        result = text
        for full, abbrev in abbreviations.items():
            result = re.sub(rf'\b{full}\b', abbrev, result, flags=re.IGNORECASE)

        return result

compressor = InstructionCompressor()

# Test instruction compression
verbose_instructions = """
Please analyze the following data and provide a comprehensive summary of your findings.
Examine the patterns and explain any anomalies you discover.
Review the metrics and identify the key performance indicators.
Format your response as a bulleted list for clarity.
For example, you might find seasonal trends, outliers, etcetera.
The maximum value should be compared versus the minimum value.
"""

compressed = compressor.compress_instructions(verbose_instructions)
compressed = compressor.use_abbreviations(compressed)

print("Verbose instructions:")
print(f"  Tokens: {count_tokens(verbose_instructions)}")
print(f"  Preview: {verbose_instructions[:150]}...")

print("\nCompressed instructions:")
print(f"  Tokens: {count_tokens(compressed)}")
print(f"  Preview: {compressed[:150]}...")

print(f"\nCompression ratio: {count_tokens(compressed) / count_tokens(verbose_instructions):.2f}")

# ================================
# Example 4: Context Window Management
# ================================
print("\n" + "=" * 50)
print("Example 4: Context Window Management")
print("=" * 50)

class ContextWindowOptimizer:
    """Optimize context usage within token limits."""

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.reserved_output = 500  # Reserve tokens for response

    def optimize_context(
        self,
        prompt: str,
        context_items: List[str],
        priorities: List[float] = None
    ) -> str:
        """Fit maximum context within token budget."""
        prompt_tokens = count_tokens(prompt)
        available_tokens = self.max_tokens - prompt_tokens - self.reserved_output

        if priorities is None:
            priorities = [1.0] * len(context_items)

        # Sort by priority
        sorted_items = sorted(
            zip(context_items, priorities),
            key=lambda x: x[1],
            reverse=True
        )

        selected_context = []
        tokens_used = 0

        for item, priority in sorted_items:
            item_tokens = count_tokens(item)

            if tokens_used + item_tokens <= available_tokens:
                selected_context.append(item)
                tokens_used += item_tokens
            else:
                # Try to fit compressed version
                compressed = self._compress_item(item, available_tokens - tokens_used)
                if compressed and count_tokens(compressed) <= available_tokens - tokens_used:
                    selected_context.append(compressed)
                    tokens_used += count_tokens(compressed)
                    break

        # Format final prompt
        context_str = "\n\n".join(selected_context)
        return prompt.format(context=context_str)

    def _compress_item(self, text: str, max_tokens: int) -> str:
        """Compress text to fit within token limit."""
        current_tokens = count_tokens(text)

        if current_tokens <= max_tokens:
            return text

        # Progressive compression strategies
        strategies = [
            lambda t: t[:int(len(t) * (max_tokens / current_tokens))],  # Truncate
            lambda t: " ".join(t.split()[:int(len(t.split()) * (max_tokens / current_tokens))]),  # Word limit
            lambda t: ". ".join(t.split(". ")[:2]) + "..."  # Keep first sentences
        ]

        for strategy in strategies:
            compressed = strategy(text)
            if count_tokens(compressed) <= max_tokens:
                return compressed

        return None

# Test context optimization
optimizer = ContextWindowOptimizer(max_tokens=500)

prompt_template = "Based on the context below, answer the question.\n\nContext:\n{context}\n\nQuestion: What are the main points?"

context_items = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
    "Deep learning is a subset of machine learning that uses neural networks with multiple layers. These networks are capable of learning unsupervised from data that is unstructured or unlabeled.",
    "Natural language processing (NLP) is the technology used to aid computers to understand human language. NLP techniques are used in many applications including translation, sentiment analysis, and chatbots.",
    "Computer vision enables machines to interpret and understand visual information from the world. Applications include facial recognition, object detection, and autonomous vehicles.",
    "Reinforcement learning is an area of machine learning where agents learn to make decisions by performing actions and receiving rewards or penalties."
]

priorities = [0.9, 0.8, 0.7, 0.6, 0.5]  # Priority scores

optimized = optimizer.optimize_context(prompt_template, context_items, priorities)
print(f"Optimized prompt tokens: {count_tokens(optimized)}")
print(f"Prompt preview:\n{optimized[:300]}...")

# ================================
# Example 5: Progressive Detail Loading
# ================================
print("\n" + "=" * 50)
print("Example 5: Progressive Detail Loading")
print("=" * 50)

class ProgressivePrompt:
    """Load details progressively based on needs."""

    def __init__(self):
        self.levels = {
            "minimal": {
                "tokens": 50,
                "template": "{task}"
            },
            "basic": {
                "tokens": 150,
                "template": "{task}\n\nRequirements:\n{requirements}"
            },
            "detailed": {
                "tokens": 300,
                "template": "{task}\n\nRequirements:\n{requirements}\n\nExamples:\n{examples}"
            },
            "comprehensive": {
                "tokens": 500,
                "template": "{task}\n\nRequirements:\n{requirements}\n\nExamples:\n{examples}\n\nConstraints:\n{constraints}\n\nFormat:\n{format}"
            }
        }

    def generate_progressive(
        self,
        task: str,
        requirements: str = "",
        examples: str = "",
        constraints: str = "",
        format_spec: str = "",
        max_tokens: int = 200
    ) -> str:
        """Generate prompt with appropriate detail level."""
        # Select appropriate level based on token budget
        selected_level = None

        for level_name in ["comprehensive", "detailed", "basic", "minimal"]:
            level = self.levels[level_name]
            if level["tokens"] <= max_tokens:
                selected_level = level_name
                break

        if not selected_level:
            selected_level = "minimal"

        # Build prompt based on level
        template = self.levels[selected_level]["template"]

        # Prepare components
        components = {
            "task": task,
            "requirements": requirements,
            "examples": examples,
            "constraints": constraints,
            "format": format_spec
        }

        # Only include components mentioned in template
        included_components = {}
        for key in components:
            if f"{{{key}}}" in template:
                included_components[key] = components[key]

        return template.format(**included_components), selected_level

# Test progressive loading
progressive = ProgressivePrompt()

task = "Analyze sales data"
requirements = "Include trends, anomalies, and recommendations"
examples = "Example: Q1 showed 20% growth, Q2 had unusual spike on March 15"
constraints = "Focus on actionable insights, limit to 3 recommendations"
format_spec = "Use bullet points for findings, paragraph for summary"

# Test with different token budgets
for max_tokens in [50, 150, 300, 500]:
    prompt, level = progressive.generate_progressive(
        task, requirements, examples, constraints, format_spec, max_tokens
    )
    actual_tokens = count_tokens(prompt)
    print(f"\nBudget: {max_tokens} tokens")
    print(f"  Level: {level}")
    print(f"  Actual tokens: {actual_tokens}")
    print(f"  Prompt: {prompt[:100]}...")

# ================================
# Example 6: Few-Shot Example Optimization
# ================================
print("\n" + "=" * 50)
print("Example 6: Few-Shot Example Optimization")
print("=" * 50)

class ExampleOptimizer:
    """Optimize few-shot examples for token efficiency."""

    def select_examples(
        self,
        examples: List[Dict[str, str]],
        query: str,
        max_examples: int = 3,
        max_tokens: int = 500
    ) -> List[Dict[str, str]]:
        """Select most relevant examples within token budget."""
        # Score examples by relevance to query
        scored_examples = []

        for example in examples:
            relevance = self._calculate_relevance(query, example)
            tokens = count_tokens(f"Input: {example['input']}\nOutput: {example['output']}")

            scored_examples.append({
                "example": example,
                "relevance": relevance,
                "tokens": tokens
            })

        # Sort by relevance
        scored_examples.sort(key=lambda x: x["relevance"], reverse=True)

        # Select examples within token budget
        selected = []
        total_tokens = 0

        for item in scored_examples[:max_examples]:
            if total_tokens + item["tokens"] <= max_tokens:
                selected.append(item["example"])
                total_tokens += item["tokens"]

        return selected

    def _calculate_relevance(self, query: str, example: Dict[str, str]) -> float:
        """Calculate relevance score between query and example."""
        # Simple word overlap similarity
        query_words = set(query.lower().split())
        example_words = set(example["input"].lower().split())

        if not query_words or not example_words:
            return 0.0

        intersection = query_words & example_words
        union = query_words | example_words

        return len(intersection) / len(union)

    def compress_examples(self, examples: List[Dict[str, str]]) -> str:
        """Compress examples into efficient format."""
        compressed = []

        for ex in examples:
            # Use compact format
            compressed.append(f"[{ex['input']}] → [{ex['output']}]")

        return "\n".join(compressed)

# Test example optimization
example_optimizer = ExampleOptimizer()

# Sample examples pool
examples_pool = [
    {"input": "What is machine learning?", "output": "ML is AI that learns from data"},
    {"input": "Explain deep learning", "output": "DL uses multi-layer neural networks"},
    {"input": "Define natural language processing", "output": "NLP helps computers understand human language"},
    {"input": "What is computer vision?", "output": "CV enables machines to interpret visual data"},
    {"input": "Describe reinforcement learning", "output": "RL trains agents through rewards/penalties"},
    {"input": "What is supervised learning?", "output": "SL uses labeled data for training"},
    {"input": "Explain unsupervised learning", "output": "UL finds patterns in unlabeled data"}
]

query = "What is deep learning and how does it work?"

# Select optimal examples
selected = example_optimizer.select_examples(examples_pool, query, max_examples=3, max_tokens=200)

print(f"Query: {query}")
print(f"\nSelected {len(selected)} examples:")
for ex in selected:
    print(f"  - {ex['input'][:50]}...")

# Compress examples
compressed = example_optimizer.compress_examples(selected)
print(f"\nCompressed format:")
print(compressed)
print(f"\nTotal tokens: {count_tokens(compressed)}")

# ================================
# Example 7: Dynamic Token Allocation
# ================================
print("\n" + "=" * 50)
print("Example 7: Dynamic Token Allocation")
print("=" * 50)

class DynamicTokenAllocator:
    """Dynamically allocate tokens across prompt components."""

    def __init__(self, total_budget: int = 2000):
        self.total_budget = total_budget
        self.min_allocations = {
            "instruction": 50,
            "context": 100,
            "examples": 50,
            "constraints": 30,
            "output_format": 20
        }

    def allocate_tokens(
        self,
        components: Dict[str, str],
        priorities: Dict[str, float]
    ) -> Dict[str, str]:
        """Allocate tokens to components based on priorities."""
        # Calculate current usage
        current_usage = {
            name: count_tokens(text)
            for name, text in components.items()
        }

        total_current = sum(current_usage.values())

        if total_current <= self.total_budget:
            return components  # No optimization needed

        # Calculate target allocations
        total_priority = sum(priorities.values())
        allocations = {}

        for name in components:
            # Ensure minimum allocation
            min_tokens = self.min_allocations.get(name, 10)

            # Calculate proportional allocation
            priority_ratio = priorities.get(name, 0.1) / total_priority
            target_tokens = int(self.total_budget * priority_ratio)

            # Use max of minimum and calculated
            allocations[name] = max(min_tokens, target_tokens)

        # Optimize components to fit allocations
        optimized = {}
        for name, text in components.items():
            target = allocations[name]
            current = current_usage[name]

            if current <= target:
                optimized[name] = text
            else:
                # Compress to fit
                optimized[name] = self._fit_to_tokens(text, target)

        return optimized

    def _fit_to_tokens(self, text: str, target_tokens: int) -> str:
        """Fit text within target token count."""
        current = count_tokens(text)

        if current <= target_tokens:
            return text

        # Calculate compression ratio
        ratio = target_tokens / current

        # Try word-level truncation
        words = text.split()
        target_words = int(len(words) * ratio)

        truncated = " ".join(words[:target_words])

        # Add ellipsis if truncated
        if len(truncated) < len(text):
            truncated += "..."

        return truncated

# Test dynamic allocation
allocator = DynamicTokenAllocator(total_budget=500)

components = {
    "instruction": "Analyze the provided data and identify key patterns, trends, and anomalies. Provide detailed explanations for each finding.",
    "context": "Sales data from Q1-Q4 2023 showing monthly revenue, customer acquisition costs, retention rates, and product performance metrics across different regions and customer segments.",
    "examples": "Example analysis: In Q1, we observed a 15% increase in revenue driven primarily by new product launches. Customer acquisition cost decreased by 8% due to improved targeting.",
    "constraints": "Focus on actionable insights. Limit recommendations to top 3 priorities. Consider seasonal effects.",
    "output_format": "Structure as: 1) Executive Summary 2) Key Findings 3) Recommendations"
}

priorities = {
    "instruction": 0.3,
    "context": 0.4,
    "examples": 0.1,
    "constraints": 0.1,
    "output_format": 0.1
}

print("Original token usage:")
for name, text in components.items():
    print(f"  {name}: {count_tokens(text)} tokens")
print(f"  Total: {sum(count_tokens(text) for text in components.values())} tokens")

optimized = allocator.allocate_tokens(components, priorities)

print("\nOptimized token usage:")
for name, text in optimized.items():
    print(f"  {name}: {count_tokens(text)} tokens")
print(f"  Total: {sum(count_tokens(text) for text in optimized.values())} tokens")

# ================================
# Performance Comparison
# ================================
print("\n" + "=" * 50)
print("Performance Comparison")
print("=" * 50)

def compare_prompts(verbose_prompt: str, optimized_prompt: str, test_input: str):
    """Compare performance of verbose vs optimized prompts."""
    results = {}

    # Test verbose prompt
    start_time = time.time()
    verbose_response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": verbose_prompt.format(input=test_input)}
        ],
        max_tokens=100
    )
    verbose_time = time.time() - start_time

    # Test optimized prompt
    start_time = time.time()
    optimized_response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": optimized_prompt.format(input=test_input)}
        ],
        max_tokens=100
    )
    optimized_time = time.time() - start_time

    results["verbose"] = {
        "tokens": count_tokens(verbose_prompt),
        "time": verbose_time,
        "response": verbose_response.choices[0].message.content[:100]
    }

    results["optimized"] = {
        "tokens": count_tokens(optimized_prompt),
        "time": optimized_time,
        "response": optimized_response.choices[0].message.content[:100]
    }

    # Calculate improvements
    token_reduction = (results["verbose"]["tokens"] - results["optimized"]["tokens"]) / results["verbose"]["tokens"] * 100
    time_reduction = (results["verbose"]["time"] - results["optimized"]["time"]) / results["verbose"]["time"] * 100

    results["improvements"] = {
        "token_reduction": token_reduction,
        "time_reduction": time_reduction
    }

    return results

# Prepare test prompts
verbose_test = """
I would really appreciate it if you could please help me understand the following concept.
Please make sure to provide a clear and comprehensive explanation that covers all the important aspects.
It would be great if you could include some examples to illustrate your points.

Concept: {input}

Please provide your detailed explanation below:
"""

optimized_test = """
Explain this concept with examples:

{input}

Explanation:
"""

# Run comparison
test_input = "gradient descent in machine learning"
comparison = compare_prompts(verbose_test, optimized_test, test_input)

print("Comparison Results:")
print(f"\nVerbose Prompt:")
print(f"  Tokens: {comparison['verbose']['tokens']}")
print(f"  Time: {comparison['verbose']['time']:.3f}s")

print(f"\nOptimized Prompt:")
print(f"  Tokens: {comparison['optimized']['tokens']}")
print(f"  Time: {comparison['optimized']['time']:.3f}s")

print(f"\nImprovements:")
print(f"  Token reduction: {comparison['improvements']['token_reduction']:.1f}%")
print(f"  Time reduction: {comparison['improvements']['time_reduction']:.1f}%")

print("\n" + "=" * 50)
print("Token Optimization Examples Complete!")
print("=" * 50)