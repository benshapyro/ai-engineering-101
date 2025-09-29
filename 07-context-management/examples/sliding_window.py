"""
Module 07: Sliding Window Context Management

Implement sliding window techniques to maintain conversation context within token limits.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import tiktoken
import json
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime


def example_1_basic_sliding_window():
    """Implement a basic sliding window for conversation history."""
    print("=" * 60)
    print("Example 1: Basic Sliding Window")
    print("=" * 60)

    class SlidingWindow:
        def __init__(self, max_messages=5):
            self.max_messages = max_messages
            self.messages = deque(maxlen=max_messages)

        def add_message(self, role, content):
            """Add a message to the window."""
            self.messages.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })

        def get_context(self):
            """Get current window content."""
            return list(self.messages)

        def clear(self):
            """Clear the window."""
            self.messages.clear()

    # Simulate conversation
    window = SlidingWindow(max_messages=4)

    conversation = [
        ("user", "Hello, I need help with Python"),
        ("assistant", "Hello! I'd be happy to help with Python. What do you need?"),
        ("user", "How do I read a file?"),
        ("assistant", "You can use `open()` function: with open('file.txt', 'r') as f: content = f.read()"),
        ("user", "What about writing to a file?"),
        ("assistant", "Use 'w' mode: with open('file.txt', 'w') as f: f.write('content')"),
        ("user", "And appending?"),
        ("assistant", "Use 'a' mode: with open('file.txt', 'a') as f: f.write('new content')")
    ]

    print("SLIDING WINDOW DEMONSTRATION:\n")
    print(f"Window size: {window.max_messages} messages\n")

    for i, (role, content) in enumerate(conversation):
        window.add_message(role, content)
        print(f"After message {i + 1}:")
        print("-" * 40)

        context = window.get_context()
        for msg in context:
            preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
            print(f"  [{msg['role']}]: {preview}")

        print(f"  Window contains {len(context)} messages\n")

    print("💡 Older messages are automatically removed as new ones arrive")


def example_2_token_based_window():
    """Sliding window based on token count rather than message count."""
    print("\n" + "=" * 60)
    print("Example 2: Token-Based Sliding Window")
    print("=" * 60)

    class TokenBasedWindow:
        def __init__(self, max_tokens=500, model="gpt-4"):
            self.max_tokens = max_tokens
            self.messages = []
            self.encoding = tiktoken.encoding_for_model(model)

        def count_tokens(self, messages):
            """Count total tokens in messages."""
            total = 0
            for msg in messages:
                # Include role tokens (approximation)
                text = f"{msg['role']}: {msg['content']}"
                total += len(self.encoding.encode(text))
            return total

        def add_message(self, role, content):
            """Add message and trim to fit token limit."""
            new_message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }

            self.messages.append(new_message)

            # Trim from the beginning until under token limit
            while self.count_tokens(self.messages) > self.max_tokens:
                if len(self.messages) <= 1:
                    # If single message exceeds limit, truncate it
                    self.messages[0]['content'] = self.messages[0]['content'][:100] + "..."
                    break
                self.messages.pop(0)

        def get_context(self):
            """Get current context with token count."""
            tokens = self.count_tokens(self.messages)
            return {
                "messages": self.messages,
                "token_count": tokens,
                "utilization": f"{tokens}/{self.max_tokens} ({tokens/self.max_tokens*100:.1f}%)"
            }

    window = TokenBasedWindow(max_tokens=200)

    messages = [
        ("system", "You are a helpful coding assistant."),
        ("user", "I need help with Python list comprehensions."),
        ("assistant", "List comprehensions provide a concise way to create lists. Basic syntax: [expression for item in iterable if condition]"),
        ("user", "Can you show me an example?"),
        ("assistant", "Sure! Here's an example: squares = [x**2 for x in range(10)] creates [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]"),
        ("user", "What about nested list comprehensions?"),
        ("assistant", "Nested: matrix = [[i*j for j in range(3)] for i in range(3)] creates a 3x3 multiplication table")
    ]

    print(f"TOKEN-BASED WINDOW (Max: {window.max_tokens} tokens)\n")

    for role, content in messages:
        window.add_message(role, content)
        context = window.get_context()

        print(f"Added: [{role}] {content[:40]}...")
        print(f"Window status: {context['utilization']}")
        print(f"Messages in window: {len(context['messages'])}\n")

    print("Final window content:")
    print("-" * 40)
    for msg in context['messages']:
        print(f"[{msg['role']}]: {msg['content'][:60]}...")

    print("\n💡 Token-based windows ensure you stay within API limits")


def example_3_priority_sliding_window():
    """Keep important messages longer using priority."""
    print("\n" + "=" * 60)
    print("Example 3: Priority-Based Sliding Window")
    print("=" * 60)

    class PrioritySlidingWindow:
        def __init__(self, max_tokens=500):
            self.max_tokens = max_tokens
            self.messages = []
            self.encoding = tiktoken.encoding_for_model("gpt-4")

        def count_tokens(self, text):
            """Count tokens in text."""
            return len(self.encoding.encode(text))

        def add_message(self, role, content, priority=1):
            """Add message with priority (higher = more important)."""
            message = {
                "role": role,
                "content": content,
                "priority": priority,
                "timestamp": datetime.now().isoformat(),
                "tokens": self.count_tokens(f"{role}: {content}")
            }
            self.messages.append(message)
            self._trim_to_fit()

        def _trim_to_fit(self):
            """Remove lowest priority messages first."""
            total_tokens = sum(msg['tokens'] for msg in self.messages)

            while total_tokens > self.max_tokens and len(self.messages) > 1:
                # Sort by priority (ascending) and timestamp (ascending)
                # Remove oldest, lowest priority message
                sorted_msgs = sorted(self.messages,
                                   key=lambda x: (x['priority'], x['timestamp']))

                # Don't remove if it's the only high-priority message
                for msg in sorted_msgs:
                    if msg['priority'] < 3:  # Only remove low/medium priority
                        self.messages.remove(msg)
                        total_tokens -= msg['tokens']
                        break
                else:
                    # If all are high priority, remove oldest
                    self.messages.pop(0)
                    total_tokens = sum(msg['tokens'] for msg in self.messages)

        def get_context(self):
            """Get messages ordered by timestamp."""
            return sorted(self.messages, key=lambda x: x['timestamp'])

    window = PrioritySlidingWindow(max_tokens=300)

    # Messages with different priorities
    messages_with_priority = [
        ("system", "You are a helpful assistant.", 3),  # High priority
        ("user", "Tell me about machine learning", 1),  # Low priority
        ("assistant", "Machine learning is a subset of AI...", 1),  # Low
        ("user", "What's the key requirement?", 2),  # Medium
        ("assistant", "The key requirement is quality training data", 3),  # High
        ("user", "How do I get started?", 1),  # Low
        ("assistant", "Start with Python and scikit-learn", 2),  # Medium
        ("user", "IMPORTANT: Remember I'm a beginner", 3),  # High
        ("assistant", "I'll keep explanations simple for beginners", 3),  # High
    ]

    print("PRIORITY-BASED SLIDING WINDOW:\n")
    print("Priority levels: 1=Low, 2=Medium, 3=High\n")

    for role, content, priority in messages_with_priority:
        window.add_message(role, content, priority)
        print(f"Added (P{priority}): [{role}] {content[:30]}...")

    print("\nFinal window (high priority messages preserved):")
    print("-" * 50)

    context = window.get_context()
    total_tokens = sum(msg['tokens'] for msg in context)

    for msg in context:
        print(f"P{msg['priority']} [{msg['role']}]: {msg['content'][:40]}...")

    print(f"\nTotal tokens: {total_tokens}/{window.max_tokens}")
    print("\n💡 Priority ensures important context is preserved")


def example_4_semantic_window():
    """Group related messages to maintain semantic coherence."""
    print("\n" + "=" * 60)
    print("Example 4: Semantic Sliding Window")
    print("=" * 60)

    class SemanticWindow:
        def __init__(self, max_tokens=500):
            self.max_tokens = max_tokens
            self.conversations = []  # List of conversation blocks
            self.encoding = tiktoken.encoding_for_model("gpt-4")

        def add_exchange(self, user_msg, assistant_msg):
            """Add a Q&A exchange as a semantic unit."""
            exchange = {
                "user": user_msg,
                "assistant": assistant_msg,
                "timestamp": datetime.now().isoformat(),
                "tokens": self._count_exchange_tokens(user_msg, assistant_msg)
            }
            self.conversations.append(exchange)
            self._trim_to_fit()

        def _count_exchange_tokens(self, user_msg, assistant_msg):
            """Count tokens in an exchange."""
            text = f"user: {user_msg}\nassistant: {assistant_msg}"
            return len(self.encoding.encode(text))

        def _trim_to_fit(self):
            """Remove complete exchanges to maintain coherence."""
            total_tokens = sum(exc['tokens'] for exc in self.conversations)

            while total_tokens > self.max_tokens and len(self.conversations) > 1:
                # Remove oldest complete exchange
                removed = self.conversations.pop(0)
                total_tokens -= removed['tokens']

        def get_context(self):
            """Get conversations as message list."""
            messages = []
            for exc in self.conversations:
                messages.append({"role": "user", "content": exc['user']})
                messages.append({"role": "assistant", "content": exc['assistant']})
            return messages

    window = SemanticWindow(max_tokens=400)

    # Q&A exchanges
    exchanges = [
        ("What is Python?", "Python is a high-level programming language known for readability."),
        ("What are its main uses?", "Web development, data science, AI, automation, and more."),
        ("How do I install it?", "Download from python.org or use package managers like brew or apt."),
        ("What's pip?", "Pip is Python's package manager for installing libraries."),
        ("Show me a hello world", "print('Hello, World!') - That's all you need!"),
        ("What about variables?", "Variables store data: name = 'Alice', age = 30, pi = 3.14")
    ]

    print("SEMANTIC SLIDING WINDOW:\n")
    print("Maintains complete Q&A pairs for coherence\n")

    for i, (user, assistant) in enumerate(exchanges):
        window.add_exchange(user, assistant)

        print(f"Exchange {i + 1}:")
        print(f"  User: {user}")
        print(f"  Assistant: {assistant[:50]}...")

        context = window.get_context()
        total_tokens = sum(
            len(window.encoding.encode(f"{msg['role']}: {msg['content']}"))
            for msg in context
        )

        print(f"  Window: {len(context)//2} exchanges, {total_tokens} tokens\n")

    print("Final window (complete exchanges only):")
    print("-" * 50)

    for i in range(0, len(context), 2):
        if i + 1 < len(context):
            print(f"Q: {context[i]['content']}")
            print(f"A: {context[i+1]['content'][:50]}...\n")

    print("💡 Semantic windows preserve conversation coherence")


def example_5_adaptive_window():
    """Dynamically adjust window size based on conversation needs."""
    print("\n" + "=" * 60)
    print("Example 5: Adaptive Sliding Window")
    print("=" * 60)

    class AdaptiveWindow:
        def __init__(self, min_tokens=200, max_tokens=1000):
            self.min_tokens = min_tokens
            self.max_tokens = max_tokens
            self.current_limit = min_tokens
            self.messages = []
            self.encoding = tiktoken.encoding_for_model("gpt-4")
            self.complexity_score = 0

        def analyze_complexity(self, content):
            """Analyze message complexity to adjust window size."""
            # Simple heuristics for complexity
            complexity = 0

            # Code blocks
            if "```" in content or "def " in content or "class " in content:
                complexity += 3

            # Technical terms
            tech_terms = ["algorithm", "function", "database", "API", "framework"]
            for term in tech_terms:
                if term.lower() in content.lower():
                    complexity += 1

            # Questions
            if "?" in content:
                complexity += 2

            # Long content
            if len(content) > 200:
                complexity += 2

            return complexity

        def add_message(self, role, content):
            """Add message and adapt window size."""
            complexity = self.analyze_complexity(content)
            self.complexity_score = (self.complexity_score * 0.7) + (complexity * 0.3)

            # Adjust window size based on complexity
            if self.complexity_score > 5:
                self.current_limit = min(self.max_tokens,
                                        self.current_limit + 100)
            elif self.complexity_score < 2:
                self.current_limit = max(self.min_tokens,
                                        self.current_limit - 50)

            message = {
                "role": role,
                "content": content,
                "complexity": complexity,
                "tokens": len(self.encoding.encode(f"{role}: {content}"))
            }

            self.messages.append(message)
            self._trim_to_fit()

        def _trim_to_fit(self):
            """Trim messages to current limit."""
            total_tokens = sum(msg['tokens'] for msg in self.messages)

            while total_tokens > self.current_limit and len(self.messages) > 1:
                removed = self.messages.pop(0)
                total_tokens -= removed['tokens']

        def get_stats(self):
            """Get current window statistics."""
            total_tokens = sum(msg['tokens'] for msg in self.messages)
            return {
                "messages": len(self.messages),
                "tokens": total_tokens,
                "limit": self.current_limit,
                "complexity": round(self.complexity_score, 2),
                "utilization": f"{total_tokens}/{self.current_limit}"
            }

    window = AdaptiveWindow(min_tokens=200, max_tokens=600)

    # Conversation with varying complexity
    conversation = [
        ("user", "Hi there!"),
        ("assistant", "Hello! How can I help you today?"),
        ("user", "Can you explain how to implement a binary search algorithm in Python?"),
        ("assistant", "```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1```"),
        ("user", "What's the time complexity?"),
        ("assistant", "O(log n) - it halves the search space each iteration"),
        ("user", "Thanks! Now, how's the weather?"),
        ("assistant", "I don't have weather data, but I can help with programming!")
    ]

    print("ADAPTIVE SLIDING WINDOW:\n")
    print("Window adjusts size based on conversation complexity\n")

    for role, content in conversation:
        window.add_message(role, content)
        stats = window.get_stats()

        print(f"[{role}]: {content[:50]}...")
        print(f"Stats: {stats['messages']} msgs, {stats['utilization']} tokens")
        print(f"Complexity: {stats['complexity']}, Limit: {stats['limit']}\n")

    print("💡 Adaptive windows allocate more space for complex discussions")


def example_6_checkpoint_window():
    """Save checkpoints for context recovery."""
    print("\n" + "=" * 60)
    print("Example 6: Checkpoint-Based Window")
    print("=" * 60)

    class CheckpointWindow:
        def __init__(self, max_tokens=500):
            self.max_tokens = max_tokens
            self.messages = []
            self.checkpoints = []
            self.encoding = tiktoken.encoding_for_model("gpt-4")

        def add_message(self, role, content, create_checkpoint=False):
            """Add message and optionally create checkpoint."""
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }

            self.messages.append(message)

            if create_checkpoint:
                self.create_checkpoint(f"After: {content[:30]}...")

            self._trim_to_fit()

        def create_checkpoint(self, label=""):
            """Save current state as checkpoint."""
            checkpoint = {
                "label": label,
                "messages": self.messages.copy(),
                "timestamp": datetime.now().isoformat()
            }
            self.checkpoints.append(checkpoint)

            # Keep only last 3 checkpoints
            if len(self.checkpoints) > 3:
                self.checkpoints.pop(0)

        def restore_checkpoint(self, index):
            """Restore from checkpoint."""
            if 0 <= index < len(self.checkpoints):
                self.messages = self.checkpoints[index]["messages"].copy()
                return True
            return False

        def _trim_to_fit(self):
            """Standard trimming."""
            total = sum(
                len(self.encoding.encode(f"{m['role']}: {m['content']}"))
                for m in self.messages
            )

            while total > self.max_tokens and len(self.messages) > 1:
                removed = self.messages.pop(0)
                total -= len(self.encoding.encode(
                    f"{removed['role']}: {removed['content']}"
                ))

        def list_checkpoints(self):
            """List available checkpoints."""
            return [(i, cp["label"]) for i, cp in enumerate(self.checkpoints)]

    window = CheckpointWindow(max_tokens=300)

    # Conversation with checkpoints
    print("CHECKPOINT-BASED WINDOW:\n")

    # Part 1: Initial context
    window.add_message("user", "I'm working on a web app with Flask")
    window.add_message("assistant", "Great! Flask is perfect for web apps. What features do you need?")
    window.add_message("user", "User authentication and a database")
    window.create_checkpoint("Initial requirements")

    # Part 2: Deep dive into auth
    window.add_message("assistant", "For auth, consider Flask-Login and werkzeug for password hashing")
    window.add_message("user", "How do I hash passwords securely?")
    window.add_message("assistant", "Use werkzeug.security: generate_password_hash() and check_password_hash()")
    window.create_checkpoint("Authentication discussion")

    # Part 3: Database discussion
    window.add_message("user", "What about the database?")
    window.add_message("assistant", "SQLAlchemy is the go-to ORM for Flask. Use Flask-SQLAlchemy for integration")
    window.add_message("user", "Can you show me a model example?")

    print("Current checkpoints:")
    for idx, label in window.list_checkpoints():
        print(f"  {idx}: {label}")

    print("\nCurrent window:")
    for msg in window.messages:
        print(f"  [{msg['role']}]: {msg['content'][:40]}...")

    # Restore earlier checkpoint
    print("\nRestoring checkpoint 0...")
    window.restore_checkpoint(0)

    print("Window after restore:")
    for msg in window.messages:
        print(f"  [{msg['role']}]: {msg['content'][:40]}...")

    print("\n💡 Checkpoints allow returning to important conversation states")


def run_all_examples():
    """Run all sliding window examples."""
    examples = [
        example_1_basic_sliding_window,
        example_2_token_based_window,
        example_3_priority_sliding_window,
        example_4_semantic_window,
        example_5_adaptive_window,
        example_6_checkpoint_window
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

    parser = argparse.ArgumentParser(description="Module 07: Sliding Window")
    parser.add_argument("--example", type=int, help="Run specific example (1-6)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_basic_sliding_window,
            2: example_2_token_based_window,
            3: example_3_priority_sliding_window,
            4: example_4_semantic_window,
            5: example_5_adaptive_window,
            6: example_6_checkpoint_window
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 07: Sliding Window Context Management")
        print("\nUsage:")
        print("  python sliding_window.py --all        # Run all examples")
        print("  python sliding_window.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: Basic Sliding Window")
        print("  2: Token-Based Window")
        print("  3: Priority-Based Window")
        print("  4: Semantic Window")
        print("  5: Adaptive Window")
        print("  6: Checkpoint-Based Window")