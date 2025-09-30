"""
Module 07: Token Management

Learn to count, budget, and optimize token usage for cost-effective LLM applications.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import tiktoken
import json
from typing import Dict, List, Optional


def example_1_token_counting_basics():
    """Understand how tokens work and how to count them."""
    print("=" * 60)
    print("Example 1: Token Counting Basics")
    print("=" * 60)

    # Get encoder for GPT-4
    encoding = tiktoken.encoding_for_model("gpt-4")

    test_strings = [
        "Hello world",
        "Hello, world!",
        "The quick brown fox",
        "Antidisestablishmentarianism",
        "你好世界",  # Chinese
        "🚀🌟💻",  # Emojis
        "function calculateSum(a, b) { return a + b; }",
        "user@example.com",
        "http://www.example.com/path?query=value"
    ]

    print("TOKEN COUNTING EXAMPLES:\n")

    for text in test_strings:
        tokens = encoding.encode(text)
        token_count = len(tokens)

        # Show token breakdown
        token_strings = [encoding.decode([t]) for t in tokens]

        print(f"Text: '{text}'")
        print(f"Tokens: {token_count}")
        print(f"Breakdown: {token_strings}")
        print("-" * 40)

    print("\n💡 Key insights:")
    print("- Punctuation often creates separate tokens")
    print("- Long words may split into multiple tokens")
    print("- Special characters and emojis use more tokens")
    print("- URLs and emails often split unexpectedly")


def example_2_model_specific_counting():
    """Different models have different tokenization."""
    print("\n" + "=" * 60)
    print("Example 2: Model-Specific Token Counting")
    print("=" * 60)

    test_text = """
    The rapid advancement of artificial intelligence has transformed
    how we approach problem-solving in various domains.
    """

    models = [
        "gpt-5-mini",
        "gpt-4",
        "text-davinci-003"
    ]

    print(f"Text: {test_text[:50]}...\n")
    print("TOKEN COUNTS BY MODEL:")

    for model in models:
        try:
            encoding = tiktoken.encoding_for_model(model)
            token_count = len(encoding.encode(test_text))
            print(f"{model}: {token_count} tokens")
        except KeyError:
            print(f"{model}: Model not found in tiktoken")

    # Also show general encoding
    cl100k_base = tiktoken.get_encoding("cl100k_base")
    print(f"cl100k_base (general): {len(cl100k_base.encode(test_text))} tokens")

    print("\n💡 Different models may tokenize the same text differently")


def example_3_token_budget_management():
    """Manage token budgets for different parts of your prompt."""
    print("\n" + "=" * 60)
    print("Example 3: Token Budget Management")
    print("=" * 60)

    class TokenBudget:
        def __init__(self, total_budget=4000, model="gpt-4"):
            self.total_budget = total_budget
            self.allocations = {}
            self.used = {}
            self.encoding = tiktoken.encoding_for_model(model)

        def allocate(self, component, tokens):
            """Allocate tokens to a component."""
            if sum(self.allocations.values()) + tokens > self.total_budget:
                raise ValueError("Exceeds total budget")
            self.allocations[component] = tokens
            self.used[component] = 0

        def use(self, component, text):
            """Use tokens from a component's allocation."""
            tokens = len(self.encoding.encode(text))
            if component not in self.allocations:
                raise ValueError(f"No allocation for {component}")
            if self.used[component] + tokens > self.allocations[component]:
                raise ValueError(f"Exceeds allocation for {component}")
            self.used[component] += tokens
            return tokens

        def remaining(self, component=None):
            """Get remaining tokens."""
            if component:
                return self.allocations.get(component, 0) - self.used.get(component, 0)
            total_allocated = sum(self.allocations.values())
            total_used = sum(self.used.values())
            return total_allocated - total_used

        def report(self):
            """Generate usage report."""
            print("\nTOKEN BUDGET REPORT:")
            print("-" * 40)
            for component in self.allocations:
                allocated = self.allocations[component]
                used = self.used[component]
                remaining = allocated - used
                usage_pct = (used / allocated * 100) if allocated > 0 else 0
                print(f"{component}:")
                print(f"  Allocated: {allocated}")
                print(f"  Used: {used} ({usage_pct:.1f}%)")
                print(f"  Remaining: {remaining}")

            total_used = sum(self.used.values())
            print(f"\nTotal Used: {total_used}/{self.total_budget}")

    # Example usage
    budget = TokenBudget(total_budget=2000)

    # Allocate tokens
    budget.allocate("system_prompt", 200)
    budget.allocate("examples", 500)
    budget.allocate("user_context", 800)
    budget.allocate("query", 300)
    budget.allocate("response", 200)

    # Use tokens
    system_prompt = "You are a helpful AI assistant specializing in Python programming."
    budget.use("system_prompt", system_prompt)

    examples = """
    Example 1: def add(a, b): return a + b
    Example 2: def multiply(a, b): return a * b
    """
    budget.use("examples", examples)

    user_context = "I'm building a web application using Flask."
    budget.use("user_context", user_context)

    query = "How do I handle user authentication?"
    budget.use("query", query)

    budget.report()

    print(f"\n💡 Budget management prevents unexpected token overflow")


def example_4_cost_calculation():
    """Calculate actual costs based on token usage."""
    print("\n" + "=" * 60)
    print("Example 4: Cost Calculation")
    print("=" * 60)

    class CostCalculator:
        def __init__(self):
            # Prices per 1K tokens (as of 2024)
            self.pricing = {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-32k": {"input": 0.06, "output": 0.12},
                "gpt-5-mini": {"input": 0.0015, "output": 0.002},
                "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004}
            }

        def calculate(self, model, input_tokens, output_tokens):
            """Calculate cost for a request."""
            if model not in self.pricing:
                return None

            input_cost = (input_tokens / 1000) * self.pricing[model]["input"]
            output_cost = (output_tokens / 1000) * self.pricing[model]["output"]
            total_cost = input_cost + output_cost

            return {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }

        def estimate_conversation(self, model, messages):
            """Estimate cost for a conversation."""
            encoding = tiktoken.encoding_for_model(model.replace("-32k", "").replace("-16k", ""))

            total_tokens = 0
            for message in messages:
                tokens = len(encoding.encode(message))
                total_tokens += tokens

            # Estimate output tokens (rough estimate)
            estimated_output = total_tokens // 3

            return self.calculate(model, total_tokens, estimated_output)

    calculator = CostCalculator()

    # Example conversation
    conversation = [
        "System: You are a helpful assistant.",
        "User: Explain quantum computing in simple terms.",
        "Assistant: Quantum computing uses quantum bits or 'qubits'...",
        "User: What are the main applications?",
        "Assistant: Key applications include cryptography, drug discovery..."
    ]

    print("COST ANALYSIS:\n")

    for model in ["gpt-5-mini", "gpt-4"]:
        result = calculator.estimate_conversation(model, conversation)
        if result:
            print(f"{model}:")
            print(f"  Input tokens: {result['input_tokens']}")
            print(f"  Output tokens (est): {result['output_tokens']}")
            print(f"  Input cost: ${result['input_cost']:.4f}")
            print(f"  Output cost: ${result['output_cost']:.4f}")
            print(f"  Total cost: ${result['total_cost']:.4f}")
            print()

    # Monthly projection
    conversations_per_day = 100
    days_per_month = 30

    print("MONTHLY PROJECTION:")
    for model in ["gpt-5-mini", "gpt-4"]:
        result = calculator.estimate_conversation(model, conversation)
        if result:
            monthly_cost = result['total_cost'] * conversations_per_day * days_per_month
            print(f"{model}: ${monthly_cost:.2f}/month")

    print("\n💡 Understanding costs helps optimize model selection")


def example_5_token_optimization():
    """Techniques to reduce token usage without losing quality."""
    print("\n" + "=" * 60)
    print("Example 5: Token Optimization Techniques")
    print("=" * 60)

    encoding = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(text):
        return len(encoding.encode(text))

    # Original verbose prompt
    verbose_prompt = """
    I would really appreciate it if you could help me understand how to implement
    a binary search algorithm. Could you please provide a detailed explanation of
    how it works, along with a Python implementation? It would be great if you
    could also include some examples of how to use it and explain the time
    complexity. Thank you so much in advance for your help with this!
    """

    # Optimized concise prompt
    concise_prompt = """
    Explain binary search algorithm with Python implementation,
    usage examples, and time complexity.
    """

    print("PROMPT OPTIMIZATION:\n")
    print(f"Verbose prompt ({count_tokens(verbose_prompt)} tokens):")
    print(f"'{verbose_prompt[:80]}...'\n")

    print(f"Concise prompt ({count_tokens(concise_prompt)} tokens):")
    print(f"'{concise_prompt}'\n")

    savings = count_tokens(verbose_prompt) - count_tokens(concise_prompt)
    print(f"Token savings: {savings} tokens")
    print(f"Reduction: {savings/count_tokens(verbose_prompt)*100:.1f}%\n")

    # More optimization techniques
    optimization_examples = [
        {
            "technique": "Remove redundant words",
            "before": "Could you please help me to understand",
            "after": "Explain"
        },
        {
            "technique": "Use abbreviations",
            "before": "Application Programming Interface",
            "after": "API"
        },
        {
            "technique": "Compress lists",
            "before": "I need help with Python, JavaScript, and TypeScript",
            "after": "Help with Python/JS/TS"
        },
        {
            "technique": "Remove formatting",
            "before": "Dear AI,\n\nI hope this message finds you well.\n\nBest regards",
            "after": "Question:"
        }
    ]

    print("OPTIMIZATION TECHNIQUES:\n")
    for example in optimization_examples:
        before_tokens = count_tokens(example["before"])
        after_tokens = count_tokens(example["after"])
        savings = before_tokens - after_tokens

        print(f"{example['technique']}:")
        print(f"  Before ({before_tokens} tokens): {example['before']}")
        print(f"  After ({after_tokens} tokens): {example['after']}")
        print(f"  Saved: {savings} tokens\n")

    print("💡 Small optimizations add up to significant savings")


def example_6_streaming_token_tracking():
    """Track tokens in streaming responses."""
    print("\n" + "=" * 60)
    print("Example 6: Streaming Token Tracking")
    print("=" * 60)

    class StreamingTokenTracker:
        def __init__(self, model="gpt-4"):
            self.encoding = tiktoken.encoding_for_model(model)
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.chunks = []

        def add_prompt(self, prompt):
            """Count tokens in the prompt."""
            self.prompt_tokens = len(self.encoding.encode(prompt))
            return self.prompt_tokens

        def add_chunk(self, chunk):
            """Add a streaming chunk and count tokens."""
            self.chunks.append(chunk)
            # In real streaming, count incrementally
            full_text = "".join(self.chunks)
            self.completion_tokens = len(self.encoding.encode(full_text))
            return self.completion_tokens

        def get_stats(self):
            """Get current statistics."""
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "chunks_received": len(self.chunks)
            }

    # Simulate streaming response
    tracker = StreamingTokenTracker()

    prompt = "Explain machine learning in one paragraph"
    tracker.add_prompt(prompt)

    # Simulate streaming chunks
    response_chunks = [
        "Machine learning is ",
        "a subset of artificial intelligence ",
        "that enables computers to learn ",
        "from data without being explicitly ",
        "programmed. It uses algorithms to ",
        "identify patterns and make decisions."
    ]

    print(f"Prompt: '{prompt}'")
    print(f"Prompt tokens: {tracker.prompt_tokens}\n")

    print("STREAMING RESPONSE:")
    print("-" * 40)

    for i, chunk in enumerate(response_chunks):
        tracker.add_chunk(chunk)
        stats = tracker.get_stats()
        print(f"Chunk {i+1}: '{chunk}'")
        print(f"  Total completion tokens: {stats['completion_tokens']}")

    print("\nFINAL STATISTICS:")
    final_stats = tracker.get_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

    print("\n💡 Track tokens during streaming for real-time cost monitoring")


def example_7_batch_processing_optimization():
    """Optimize token usage in batch processing."""
    print("\n" + "=" * 60)
    print("Example 7: Batch Processing Optimization")
    print("=" * 60)

    class BatchProcessor:
        def __init__(self, max_tokens_per_batch=2000):
            self.max_tokens = max_tokens_per_batch
            self.encoding = tiktoken.encoding_for_model("gpt-4")

        def create_batches(self, items, prompt_template):
            """Group items into token-efficient batches."""
            batches = []
            current_batch = []
            current_tokens = 0

            # Count base template tokens
            base_tokens = len(self.encoding.encode(prompt_template.format(items="")))

            for item in items:
                item_tokens = len(self.encoding.encode(str(item)))

                if current_tokens + item_tokens + base_tokens > self.max_tokens:
                    # Save current batch and start new one
                    if current_batch:
                        batches.append(current_batch)
                    current_batch = [item]
                    current_tokens = item_tokens
                else:
                    current_batch.append(item)
                    current_tokens += item_tokens

            # Add final batch
            if current_batch:
                batches.append(current_batch)

            return batches

    # Example: Processing multiple items
    items_to_process = [
        "Analyze the sentiment of this product review",
        "Extract key entities from this news article",
        "Summarize this research paper",
        "Translate this paragraph to Spanish",
        "Generate test cases for this function",
        "Review this code for security issues",
        "Create documentation for this API",
        "Optimize this SQL query"
    ]

    prompt_template = """
    Process the following tasks:
    {items}

    Provide brief responses for each.
    """

    processor = BatchProcessor(max_tokens_per_batch=500)
    batches = processor.create_batches(items_to_process, prompt_template)

    print("BATCH OPTIMIZATION RESULTS:\n")
    print(f"Total items: {len(items_to_process)}")
    print(f"Batches created: {len(batches)}\n")

    for i, batch in enumerate(batches, 1):
        batch_text = "\n".join([f"- {item}" for item in batch])
        prompt = prompt_template.format(items=batch_text)
        tokens = len(processor.encoding.encode(prompt))

        print(f"Batch {i} ({tokens} tokens):")
        for item in batch:
            print(f"  - {item[:50]}...")
        print()

    print("💡 Batching reduces API calls while respecting token limits")


def run_all_examples():
    """Run all token management examples."""
    examples = [
        example_1_token_counting_basics,
        example_2_model_specific_counting,
        example_3_token_budget_management,
        example_4_cost_calculation,
        example_5_token_optimization,
        example_6_streaming_token_tracking,
        example_7_batch_processing_optimization
    ]

    for example in examples:
        try:
            example()
            print("\n" + "=" * 60 + "\n")
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 07: Token Management")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_token_counting_basics,
            2: example_2_model_specific_counting,
            3: example_3_token_budget_management,
            4: example_4_cost_calculation,
            5: example_5_token_optimization,
            6: example_6_streaming_token_tracking,
            7: example_7_batch_processing_optimization
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 07: Token Management")
        print("\nUsage:")
        print("  python token_management.py --all        # Run all examples")
        print("  python token_management.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: Token Counting Basics")
        print("  2: Model-Specific Counting")
        print("  3: Token Budget Management")
        print("  4: Cost Calculation")
        print("  5: Token Optimization")
        print("  6: Streaming Token Tracking")
        print("  7: Batch Processing Optimization")