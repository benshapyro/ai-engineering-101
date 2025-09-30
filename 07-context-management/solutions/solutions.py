"""
Module 07: Context Management - Solutions

Complete solutions for all exercises demonstrating context management mastery.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import tiktoken
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import hashlib


# ===== Exercise 1 Solution: Token Budget Calculator =====

def solution_1_token_calculator():
    """
    Solution: Comprehensive token budget calculator with optimization.
    """
    print("Solution 1: Token Budget Calculator")
    print("=" * 50)

    class TokenCalculator:
        def __init__(self):
            # Real pricing as of 2024 (per 1K tokens)
            self.pricing = {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-32k": {"input": 0.06, "output": 0.12},
                "gpt-5-mini": {"input": 0.0015, "output": 0.002},
                "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004}
            }
            self.usage_history = []
            self.encodings = {}

        def get_encoding(self, model):
            """Get cached encoding for model."""
            if model not in self.encodings:
                # Normalize model name for tiktoken
                base_model = model.replace("-32k", "").replace("-16k", "")
                self.encodings[model] = tiktoken.encoding_for_model(base_model)
            return self.encodings[model]

        def calculate_tokens(self, text, model="gpt-4"):
            """Count tokens for given text and model."""
            encoding = self.get_encoding(model)
            return len(encoding.encode(text))

        def estimate_cost(self, input_text, expected_output_length, model="gpt-4"):
            """Estimate cost for a request."""
            if model not in self.pricing:
                return {"error": f"Unknown model: {model}"}

            input_tokens = self.calculate_tokens(input_text, model)
            output_tokens = expected_output_length

            input_cost = (input_tokens / 1000) * self.pricing[model]["input"]
            output_cost = (output_tokens / 1000) * self.pricing[model]["output"]
            total_cost = input_cost + output_cost

            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "model": model
            }

        def track_usage(self, input_tokens, output_tokens, model, cost=None):
            """Track token usage over time."""
            entry = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost": cost or self._calculate_cost(input_tokens, output_tokens, model)
            }
            self.usage_history.append(entry)

        def _calculate_cost(self, input_tokens, output_tokens, model):
            """Calculate cost from token counts."""
            if model not in self.pricing:
                return 0

            input_cost = (input_tokens / 1000) * self.pricing[model]["input"]
            output_cost = (output_tokens / 1000) * self.pricing[model]["output"]
            return input_cost + output_cost

        def generate_report(self):
            """Generate usage report with recommendations."""
            if not self.usage_history:
                return "No usage history available."

            # Calculate statistics
            total_cost = sum(entry["cost"] for entry in self.usage_history)
            total_tokens = sum(entry["total_tokens"] for entry in self.usage_history)

            model_usage = {}
            for entry in self.usage_history:
                model = entry["model"]
                if model not in model_usage:
                    model_usage[model] = {"count": 0, "tokens": 0, "cost": 0}
                model_usage[model]["count"] += 1
                model_usage[model]["tokens"] += entry["total_tokens"]
                model_usage[model]["cost"] += entry["cost"]

            # Generate report
            report = "TOKEN USAGE REPORT\n" + "=" * 40 + "\n"
            report += f"Total requests: {len(self.usage_history)}\n"
            report += f"Total tokens: {total_tokens:,}\n"
            report += f"Total cost: ${total_cost:.4f}\n\n"

            report += "BY MODEL:\n"
            for model, stats in model_usage.items():
                report += f"  {model}:\n"
                report += f"    Requests: {stats['count']}\n"
                report += f"    Tokens: {stats['tokens']:,}\n"
                report += f"    Cost: ${stats['cost']:.4f}\n"

            # Optimization recommendations
            report += "\nOPTIMIZATION RECOMMENDATIONS:\n"

            # Check if using expensive models for simple tasks
            if "gpt-4" in model_usage and "gpt-5-mini" not in model_usage:
                report += "• Consider using gpt-3.5-turbo for simpler tasks\n"

            # Check average request size
            avg_tokens = total_tokens / len(self.usage_history)
            if avg_tokens > 2000:
                report += "• Large average request size - consider compression\n"

            # Check for potential batching
            if len(self.usage_history) > 10:
                time_diffs = []
                for i in range(1, len(self.usage_history)):
                    t1 = datetime.fromisoformat(self.usage_history[i-1]["timestamp"])
                    t2 = datetime.fromisoformat(self.usage_history[i]["timestamp"])
                    time_diffs.append((t2 - t1).total_seconds())

                if time_diffs and sum(time_diffs) / len(time_diffs) < 5:
                    report += "• Rapid requests detected - consider batching\n"

            return report

    # Test the calculator
    calculator = TokenCalculator()

    test_texts = [
        "Short prompt",
        "A much longer prompt that contains multiple sentences and should use more tokens for demonstration purposes",
        """def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"""
    ]

    print("TOKEN CALCULATION TESTS:\n")

    for i, text in enumerate(test_texts, 1):
        print(f"Text {i}: '{text[:30]}...'")

        for model in ["gpt-5-mini", "gpt-4"]:
            tokens = calculator.calculate_tokens(text, model)
            estimate = calculator.estimate_cost(text, expected_output_length=50, model=model)

            print(f"  {model}:")
            print(f"    Input tokens: {estimate['input_tokens']}")
            print(f"    Est. total cost: ${estimate['total_cost']:.5f}")

            # Track usage
            calculator.track_usage(
                estimate['input_tokens'],
                estimate['output_tokens'],
                model
            )
        print()

    # Generate report
    print(calculator.generate_report())


# ===== Exercise 2 Solution: Intelligent Sliding Window =====

def solution_2_sliding_window():
    """
    Solution: Intelligent sliding window with importance scoring.
    """
    print("\nSolution 2: Intelligent Sliding Window")
    print("=" * 50)

    class IntelligentSlidingWindow:
        def __init__(self, max_tokens=1000, model="gpt-4"):
            self.max_tokens = max_tokens
            self.messages = []
            self.system_message = None
            self.encoding = tiktoken.encoding_for_model(model)

        def count_tokens(self, text):
            """Count tokens in text."""
            return len(self.encoding.encode(text))

        def set_system_message(self, content):
            """Set and protect system message."""
            self.system_message = {
                "role": "system",
                "content": content,
                "tokens": self.count_tokens(f"system: {content}"),
                "importance": float('inf'),  # Never removed
                "timestamp": datetime.now()
            }

        def add_message(self, role, content, importance=1):
            """Add message with importance scoring."""
            # Auto-calculate importance if not provided
            if importance == 1:
                importance = self._calculate_importance(role, content)

            message = {
                "role": role,
                "content": content,
                "importance": importance,
                "timestamp": datetime.now(),
                "tokens": self.count_tokens(f"{role}: {content}")
            }

            self.messages.append(message)
            self._slide_window()

        def _calculate_importance(self, role, content):
            """Calculate message importance automatically."""
            importance = 1.0

            # Role-based scoring
            if role == "system":
                importance = float('inf')
            elif role == "error":
                importance = 10.0

            # Keyword-based scoring
            important_keywords = [
                ("error", 5.0),
                ("important", 3.0),
                ("remember", 3.0),
                ("always", 2.5),
                ("never", 2.5),
                ("critical", 4.0),
                ("warning", 3.5),
                ("TODO", 2.0),
                ("FIXME", 3.0)
            ]

            content_lower = content.lower()
            for keyword, score in important_keywords:
                if keyword in content_lower:
                    importance = max(importance, score)

            # Length penalty (very short messages might be less important)
            if len(content) < 20:
                importance *= 0.8

            # Recency boost (newer messages slightly more important)
            importance *= 1.1

            return importance

        def _slide_window(self):
            """Slide window preserving important messages."""
            total_tokens = self._calculate_total_tokens()

            while total_tokens > self.max_tokens and len(self.messages) > 1:
                # Sort by importance (ascending) and timestamp (ascending)
                removable = [m for m in self.messages if m["importance"] < float('inf')]

                if not removable:
                    break

                # Remove least important, oldest message
                removable.sort(key=lambda x: (x["importance"], x["timestamp"]))

                # Ensure we maintain conversation pairs when possible
                to_remove = removable[0]

                # If removing a user message, try to remove the following assistant message too
                if to_remove["role"] == "user":
                    idx = self.messages.index(to_remove)
                    if idx + 1 < len(self.messages) and self.messages[idx + 1]["role"] == "assistant":
                        if self.messages[idx + 1]["importance"] < 3:
                            self.messages.remove(self.messages[idx + 1])

                self.messages.remove(to_remove)
                total_tokens = self._calculate_total_tokens()

        def _calculate_total_tokens(self):
            """Calculate total tokens in current window."""
            total = 0
            if self.system_message:
                total += self.system_message["tokens"]
            total += sum(msg["tokens"] for msg in self.messages)
            return total

        def get_context(self):
            """Get current window context."""
            context = []

            if self.system_message:
                context.append({
                    "role": self.system_message["role"],
                    "content": self.system_message["content"]
                })

            # Sort messages by timestamp for correct order
            sorted_messages = sorted(self.messages, key=lambda x: x["timestamp"])
            for msg in sorted_messages:
                context.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            return context

        def get_stats(self):
            """Get window statistics."""
            return {
                "total_tokens": self._calculate_total_tokens(),
                "max_tokens": self.max_tokens,
                "message_count": len(self.messages) + (1 if self.system_message else 0),
                "utilization": f"{self._calculate_total_tokens()}/{self.max_tokens}",
                "avg_importance": sum(m["importance"] for m in self.messages) / len(self.messages) if self.messages else 0
            }

    # Test the sliding window
    window = IntelligentSlidingWindow(max_tokens=500)

    window.set_system_message("You are a helpful Python programming assistant.")

    # Add messages with varying importance
    test_messages = [
        ("user", "Hello, I need help with Python", 1),
        ("assistant", "Hello! I'd be happy to help with Python. What do you need?", 1),
        ("user", "IMPORTANT: I'm working on a financial application", 5),
        ("assistant", "I understand this is a financial application. I'll ensure my suggestions follow best practices for financial software.", 4),
        ("user", "How do I read a CSV file?", 2),
        ("assistant", "You can use pandas: pd.read_csv('file.csv')", 2),
        ("user", "ERROR: Getting 'FileNotFoundError'", 8),
        ("assistant", "The FileNotFoundError means the file path is incorrect. Check the file exists and the path is correct.", 7),
        ("user", "Fixed it, thanks! Now how about JSON?", 2),
        ("assistant", "For JSON, use the json module: json.load() for files", 2),
    ]

    print("ADDING MESSAGES TO INTELLIGENT WINDOW:\n")

    for role, content, importance in test_messages:
        window.add_message(role, content, importance)
        stats = window.get_stats()
        print(f"Added: [{role}] {content[:40]}... (importance: {importance})")
        print(f"  Stats: {stats['utilization']} tokens, {stats['message_count']} messages\n")

    print("FINAL CONTEXT:")
    print("-" * 40)
    context = window.get_context()
    for msg in context:
        print(f"[{msg['role']}]: {msg['content'][:60]}...")

    print(f"\n💡 High-importance messages (ERROR, IMPORTANT) were preserved")


# ===== Exercise 3 Solution: Context Prioritization =====

def solution_3_context_prioritization():
    """
    Solution: Advanced context prioritization system.
    """
    print("\nSolution 3: Context Prioritization")
    print("=" * 50)

    class ContextPrioritizer:
        def __init__(self, model="gpt-4"):
            self.contexts = []
            self.prioritization_strategy = "balanced"
            self.encoding = tiktoken.encoding_for_model(model)

            # Weights for different context types
            self.type_weights = {
                "system": 10.0,
                "fact": 7.0,
                "example": 5.0,
                "conversation": 3.0,
                "reference": 4.0
            }

        def add_context(self, content, context_type, metadata=None):
            """Add context with type and metadata."""
            context = {
                "content": content,
                "type": context_type,
                "metadata": metadata or {},
                "timestamp": datetime.now(),
                "tokens": self.count_tokens(content),
                "access_count": 0
            }
            self.contexts.append(context)

        def count_tokens(self, text):
            """Count tokens in text."""
            return len(self.encoding.encode(text))

        def score_relevance(self, context, current_query):
            """Score context relevance to current query."""
            score = 0.0

            # Type weight
            type_weight = self.type_weights.get(context["type"], 1.0)
            score += type_weight

            # Keyword overlap (simplified semantic similarity)
            query_words = set(current_query.lower().split())
            context_words = set(context["content"].lower().split())
            overlap = len(query_words.intersection(context_words))
            score += overlap * 2

            # Age decay
            age = (datetime.now() - context["timestamp"]).total_seconds() / 3600  # hours
            age_factor = 1.0 / (1.0 + age * 0.01)  # Gentle decay
            score *= age_factor

            # Access frequency boost
            score += context["access_count"] * 0.5

            # Metadata boosts
            if context["metadata"].get("pinned"):
                score *= 10
            if context["metadata"].get("user_marked"):
                score *= 2

            return score

        def prioritize(self, current_query, token_budget):
            """Select best contexts within budget."""
            # Score all contexts
            scored_contexts = []
            for ctx in self.contexts:
                score = self.score_relevance(ctx, current_query)
                scored_contexts.append((score, ctx))

            # Apply strategy
            if self.prioritization_strategy == "recency_first":
                scored_contexts.sort(key=lambda x: x[1]["timestamp"], reverse=True)
            elif self.prioritization_strategy == "importance_first":
                scored_contexts.sort(key=lambda x: x[0], reverse=True)
            elif self.prioritization_strategy == "balanced":
                # Balance score and recency
                scored_contexts.sort(
                    key=lambda x: (x[0] * 0.7 +
                                  (1.0 / (1.0 + (datetime.now() - x[1]["timestamp"]).total_seconds())) * 0.3),
                    reverse=True
                )

            # Select within budget
            selected = []
            used_tokens = 0

            for score, ctx in scored_contexts:
                if used_tokens + ctx["tokens"] <= token_budget:
                    selected.append(ctx)
                    used_tokens += ctx["tokens"]
                    ctx["access_count"] += 1

            return selected, used_tokens

        def set_strategy(self, strategy):
            """Set prioritization strategy."""
            valid_strategies = ["recency_first", "importance_first", "balanced"]
            if strategy in valid_strategies:
                self.prioritization_strategy = strategy
            else:
                raise ValueError(f"Invalid strategy. Choose from: {valid_strategies}")

    # Test prioritization
    prioritizer = ContextPrioritizer()

    # Add various context types
    contexts = [
        ("You are an expert Python developer", "system", {"pinned": True}),
        ("User prefers functional programming", "fact", {}),
        ("Previous error: ImportError on pandas", "conversation", {}),
        ("Example: df = pd.DataFrame(data)", "example", {}),
        ("pandas.read_csv() documentation: ...", "reference", {}),
        ("IMPORTANT: Use type hints always", "fact", {"user_marked": True}),
        ("Last query was about data processing", "conversation", {}),
    ]

    for content, ctx_type, metadata in contexts:
        prioritizer.add_context(content, ctx_type, metadata)
        print(f"Added {ctx_type}: '{content[:40]}...'")

    print("\nTESTING PRIORITIZATION STRATEGIES:\n")

    query = "How do I process CSV data with pandas?"
    budget = 300

    for strategy in ["recency_first", "importance_first", "balanced"]:
        prioritizer.set_strategy(strategy)
        selected, tokens = prioritizer.prioritize(query, budget)

        print(f"Strategy: {strategy}")
        print(f"  Selected {len(selected)} contexts ({tokens} tokens)")
        for ctx in selected:
            print(f"    [{ctx['type']}] {ctx['content'][:40]}...")
        print()


# ===== Exercise 4 Solution: Dynamic Summarization =====

def solution_4_dynamic_summarization():
    """
    Solution: Dynamic context summarization with information preservation.
    """
    print("\nSolution 4: Dynamic Summarization")
    print("=" * 50)

    client = LLMClient("openai")

    class DynamicSummarizer:
        def __init__(self, compression_threshold=0.7, max_buffer=1000):
            self.compression_threshold = compression_threshold
            self.max_buffer = max_buffer
            self.conversation_buffer = []
            self.summaries = []
            self.key_facts = set()
            self.encoding = tiktoken.encoding_for_model("gpt-4")

        def count_tokens(self, text):
            """Count tokens in text."""
            return len(self.encoding.encode(text))

        def add_conversation_turn(self, user_msg, assistant_msg):
            """Add conversation turn and check if summarization needed."""
            turn = {
                "user": user_msg,
                "assistant": assistant_msg,
                "timestamp": datetime.now(),
                "tokens": self.count_tokens(f"User: {user_msg}\nAssistant: {assistant_msg}")
            }

            self.conversation_buffer.append(turn)

            # Check if summarization needed
            if self.should_summarize():
                self.perform_summarization()

        def should_summarize(self):
            """Determine if summarization is needed."""
            total_tokens = sum(turn["tokens"] for turn in self.conversation_buffer)
            buffer_usage = total_tokens / self.max_buffer

            # Summarize if buffer exceeds threshold
            return buffer_usage > self.compression_threshold

        def perform_summarization(self):
            """Summarize older conversation parts."""
            if len(self.conversation_buffer) < 3:
                return  # Keep at least 3 turns

            # Take first half of buffer for summarization
            to_summarize = self.conversation_buffer[:len(self.conversation_buffer)//2]
            self.conversation_buffer = self.conversation_buffer[len(self.conversation_buffer)//2:]

            # Create conversation text
            conv_text = "\n".join([
                f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                for turn in to_summarize
            ])

            summary = self.summarize_conversation(conv_text)
            self.summaries.append({
                "summary": summary["summary"],
                "key_points": summary["key_points"],
                "timestamp": datetime.now()
            })

            # Extract and store key facts
            for point in summary["key_points"]:
                self.key_facts.add(point)

        def summarize_conversation(self, conversation_chunk):
            """Summarize a conversation chunk preserving key information."""
            # Create summarization prompt
            prompt = f"""
Summarize this conversation, preserving all important information:

{conversation_chunk}

Provide:
1. A concise summary (2-3 sentences)
2. List of key points that must be remembered

Format:
SUMMARY: [your summary]
KEY POINTS:
- [point 1]
- [point 2]
..."""

            response = client.complete(prompt, temperature=0.3, max_tokens=200)

            # Parse response
            lines = response.strip().split("\n")
            summary = ""
            key_points = []

            parsing_summary = False
            parsing_points = False

            for line in lines:
                if line.startswith("SUMMARY:"):
                    summary = line.replace("SUMMARY:", "").strip()
                    parsing_summary = True
                elif line.startswith("KEY POINTS:"):
                    parsing_summary = False
                    parsing_points = True
                elif parsing_summary and line.strip():
                    summary += " " + line.strip()
                elif parsing_points and line.strip().startswith("-"):
                    key_points.append(line.strip().lstrip("- "))

            return {"summary": summary, "key_points": key_points}

        def get_compressed_context(self):
            """Get full context with summaries and recent messages."""
            context_parts = []

            # Add summaries
            if self.summaries:
                context_parts.append("PREVIOUS CONVERSATION SUMMARIES:")
                for summary in self.summaries:
                    context_parts.append(f"• {summary['summary']}")

            # Add key facts
            if self.key_facts:
                context_parts.append("\nKEY FACTS:")
                for fact in self.key_facts:
                    context_parts.append(f"• {fact}")

            # Add recent conversation
            if self.conversation_buffer:
                context_parts.append("\nRECENT CONVERSATION:")
                for turn in self.conversation_buffer:
                    context_parts.append(f"User: {turn['user']}")
                    context_parts.append(f"Assistant: {turn['assistant']}")

            return "\n".join(context_parts)

        def get_stats(self):
            """Get summarization statistics."""
            buffer_tokens = sum(turn["tokens"] for turn in self.conversation_buffer)
            summary_tokens = self.count_tokens("\n".join([s["summary"] for s in self.summaries]))

            return {
                "buffer_turns": len(self.conversation_buffer),
                "buffer_tokens": buffer_tokens,
                "summaries": len(self.summaries),
                "summary_tokens": summary_tokens,
                "key_facts": len(self.key_facts),
                "compression_ratio": summary_tokens / (buffer_tokens + summary_tokens) if buffer_tokens + summary_tokens > 0 else 0
            }

    # Test dynamic summarization
    summarizer = DynamicSummarizer(compression_threshold=0.7, max_buffer=400)

    conversation = [
        ("What's machine learning?", "Machine learning is a subset of AI that enables computers to learn from data without explicit programming."),
        ("How does it work?", "It works by finding patterns in data using algorithms like neural networks, decision trees, and statistical models."),
        ("What are neural networks?", "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes or 'neurons'."),
        ("Can you give an example?", "Sure! Image recognition is a common example where neural networks learn to identify objects in photos."),
        ("What's deep learning?", "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input."),
        ("How is it different from regular ML?", "Deep learning can automatically learn features from raw data, while traditional ML often requires manual feature engineering."),
    ]

    print("PROCESSING CONVERSATION WITH DYNAMIC SUMMARIZATION:\n")

    for i, (user, assistant) in enumerate(conversation):
        summarizer.add_conversation_turn(user, assistant)
        stats = summarizer.get_stats()

        print(f"Turn {i+1}:")
        print(f"  User: {user}")
        print(f"  Buffer: {stats['buffer_turns']} turns, {stats['buffer_tokens']} tokens")
        print(f"  Summaries: {stats['summaries']}, Compression: {stats['compression_ratio']:.2%}\n")

    print("FINAL COMPRESSED CONTEXT:")
    print("-" * 50)
    print(summarizer.get_compressed_context())


# ===== Exercise 5 Solution: Memory System Design =====

def solution_5_memory_system():
    """
    Solution: Complete memory management system with persistence.
    """
    print("\nSolution 5: Memory System Design")
    print("=" * 50)

    class MemorySystem:
        def __init__(self, short_term_size=10, model="gpt-4"):
            self.short_term = deque(maxlen=short_term_size)
            self.long_term = {}
            self.working_memory = []
            self.memory_index = {}
            self.encoding = tiktoken.encoding_for_model(model)

        def store_short_term(self, content, category=None):
            """Store in short-term memory."""
            memory = {
                "content": content,
                "category": category or "general",
                "timestamp": datetime.now(),
                "access_count": 0,
                "id": self._generate_id(content)
            }

            # Check if short-term is full
            if len(self.short_term) == self.short_term.maxlen:
                # Promote oldest to long-term before it's removed
                self.promote_to_long_term([self.short_term[0]])

            self.short_term.append(memory)
            self._update_index(memory)

        def promote_to_long_term(self, items):
            """Move items to long-term memory with compression."""
            for item in items:
                category = item["category"]

                if category not in self.long_term:
                    self.long_term[category] = []

                # Compress if needed
                compressed = self._compress_memory(item)
                self.long_term[category].append(compressed)

                # Update index
                self._update_index(compressed, long_term=True)

        def _compress_memory(self, memory):
            """Compress memory for long-term storage."""
            # Simple compression: truncate content if too long
            content = memory["content"]
            if len(content) > 200:
                content = content[:200] + "..."

            return {
                "content": content,
                "category": memory["category"],
                "timestamp": memory["timestamp"],
                "original_id": memory["id"],
                "compressed": True
            }

        def retrieve(self, query, memory_type="all"):
            """Retrieve relevant memories."""
            results = []

            if memory_type in ["all", "short"]:
                # Search short-term
                for memory in self.short_term:
                    score = self._calculate_relevance(memory["content"], query)
                    if score > 0:
                        memory["access_count"] += 1
                        results.append((score, "short", memory))

            if memory_type in ["all", "long"]:
                # Search long-term
                for category, memories in self.long_term.items():
                    for memory in memories:
                        score = self._calculate_relevance(memory["content"], query)
                        if score > 0:
                            results.append((score, "long", memory))

            # Sort by relevance
            results.sort(key=lambda x: x[0], reverse=True)
            return results[:5]  # Return top 5

        def _calculate_relevance(self, content, query):
            """Calculate relevance score between content and query."""
            # Simple keyword overlap
            content_words = set(content.lower().split())
            query_words = set(query.lower().split())
            overlap = len(content_words.intersection(query_words))
            return overlap

        def consolidate(self):
            """Consolidate and organize memories."""
            # Merge similar memories in long-term
            for category in self.long_term:
                memories = self.long_term[category]

                # Group similar memories
                consolidated = []
                seen_content = set()

                for memory in memories:
                    # Simple duplicate detection
                    content_hash = hashlib.md5(memory["content"][:50].encode()).hexdigest()
                    if content_hash not in seen_content:
                        consolidated.append(memory)
                        seen_content.add(content_hash)

                self.long_term[category] = consolidated

            print(f"Consolidated memories: {sum(len(m) for m in self.long_term.values())} long-term memories")

        def _generate_id(self, content):
            """Generate unique ID for memory."""
            return hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()[:8]

        def _update_index(self, memory, long_term=False):
            """Update memory index for fast retrieval."""
            memory_id = memory.get("id") or memory.get("original_id")
            self.memory_index[memory_id] = {
                "type": "long" if long_term else "short",
                "category": memory.get("category"),
                "timestamp": memory.get("timestamp")
            }

        def export_memories(self):
            """Export memories for persistence."""
            export_data = {
                "short_term": list(self.short_term),
                "long_term": self.long_term,
                "index": self.memory_index,
                "export_time": datetime.now().isoformat()
            }

            # Convert datetime objects to strings
            for memory in export_data["short_term"]:
                memory["timestamp"] = memory["timestamp"].isoformat()

            for category in export_data["long_term"]:
                for memory in export_data["long_term"][category]:
                    memory["timestamp"] = memory["timestamp"].isoformat()

            return json.dumps(export_data, indent=2)

        def import_memories(self, data):
            """Import previously saved memories."""
            import_data = json.loads(data) if isinstance(data, str) else data

            # Import short-term
            self.short_term.clear()
            for memory in import_data.get("short_term", []):
                memory["timestamp"] = datetime.fromisoformat(memory["timestamp"])
                self.short_term.append(memory)

            # Import long-term
            self.long_term = {}
            for category, memories in import_data.get("long_term", {}).items():
                self.long_term[category] = []
                for memory in memories:
                    memory["timestamp"] = datetime.fromisoformat(memory["timestamp"])
                    self.long_term[category].append(memory)

            # Import index
            self.memory_index = import_data.get("index", {})

            print(f"Imported {len(self.short_term)} short-term and {sum(len(m) for m in self.long_term.values())} long-term memories")

        def get_stats(self):
            """Get memory system statistics."""
            return {
                "short_term_count": len(self.short_term),
                "long_term_count": sum(len(m) for m in self.long_term.values()),
                "categories": list(self.long_term.keys()),
                "index_size": len(self.memory_index),
                "working_memory": len(self.working_memory)
            }

    # Test memory system
    memory = MemorySystem(short_term_size=5)

    # Store various information
    test_data = [
        ("Python is a high-level programming language", "facts"),
        ("User prefers functional programming style", "preferences"),
        ("Last error was ImportError on pandas", "errors"),
        ("Successfully implemented binary search", "achievements"),
        ("TODO: Add error handling to file operations", "tasks"),
        ("API rate limit is 1000 requests per hour", "constraints"),
        ("Database connection string: postgresql://...", "configuration"),
    ]

    print("STORING MEMORIES:\n")
    for content, category in test_data:
        memory.store_short_term(content, category)
        print(f"Stored [{category}]: {content[:40]}...")

    print(f"\nMemory stats: {memory.get_stats()}")

    # Test retrieval
    print("\nRETRIEVAL TESTS:")
    queries = ["Python programming", "error", "database"]

    for query in queries:
        results = memory.retrieve(query)
        print(f"\nQuery: '{query}'")
        for score, mem_type, mem in results:
            print(f"  [{mem_type}] Score {score}: {mem['content'][:40]}...")

    # Test consolidation
    print("\nCONSOLIDATION:")
    memory.consolidate()

    # Test export/import
    print("\nEXPORT/IMPORT:")
    exported = memory.export_memories()
    print(f"Exported data size: {len(exported)} characters")

    # Clear and reimport
    memory = MemorySystem()
    memory.import_memories(exported)
    print(f"After import: {memory.get_stats()}")


# ===== Challenge Solution: Production Context Manager =====

def challenge_solution_production_context_manager():
    """
    Challenge Solution: Production-ready context management system.
    """
    print("\nChallenge: Production Context Manager")
    print("=" * 50)

    client = LLMClient("openai")

    class ProductionContextManager:
        def __init__(self, model="gpt-4", max_tokens=4000):
            self.model = model
            self.max_tokens = max_tokens
            self.conversations = {}
            self.analytics = {
                "total_requests": 0,
                "total_tokens": 0,
                "compression_count": 0,
                "errors": []
            }
            self.compression_strategies = {
                "summarize": self._compress_summarize,
                "extract_facts": self._compress_facts,
                "hierarchical": self._compress_hierarchical,
                "progressive": self._compress_progressive
            }
            self.encoding = tiktoken.encoding_for_model(model)

        def create_conversation(self, conversation_id, system_prompt=None):
            """Initialize a new conversation thread."""
            if conversation_id in self.conversations:
                return {"error": "Conversation already exists"}

            self.conversations[conversation_id] = {
                "messages": [],
                "system_prompt": system_prompt,
                "metadata": {
                    "created": datetime.now(),
                    "token_usage": 0,
                    "compression_count": 0,
                    "strategy": "optimal"
                },
                "facts": set(),
                "summaries": []
            }

            if system_prompt:
                self.add_message(conversation_id, "system", system_prompt)

            return {"success": True, "conversation_id": conversation_id}

        def add_message(self, conversation_id, role, content, metadata=None):
            """Add message to specific conversation."""
            if conversation_id not in self.conversations:
                return {"error": "Conversation not found"}

            try:
                conv = self.conversations[conversation_id]
                tokens = len(self.encoding.encode(content))

                message = {
                    "role": role,
                    "content": content,
                    "tokens": tokens,
                    "timestamp": datetime.now(),
                    "metadata": metadata or {}
                }

                conv["messages"].append(message)
                conv["metadata"]["token_usage"] += tokens
                self.analytics["total_tokens"] += tokens
                self.analytics["total_requests"] += 1

                # Check if compression needed
                total_tokens = sum(m["tokens"] for m in conv["messages"])
                if total_tokens > self.max_tokens * 0.8:
                    self.apply_compression(conversation_id, method="auto")

                return {"success": True, "tokens": tokens}

            except Exception as e:
                self.handle_error("add_message_error", conversation_id, str(e))
                return {"error": str(e)}

        def get_context(self, conversation_id, strategy="optimal"):
            """Get optimized context for conversation."""
            if conversation_id not in self.conversations:
                return {"error": "Conversation not found"}

            conv = self.conversations[conversation_id]

            if strategy == "recent":
                return self._get_recent_context(conv)
            elif strategy == "compressed":
                return self._get_compressed_context(conv)
            elif strategy == "hybrid":
                return self._get_hybrid_context(conv)
            else:  # optimal
                return self._get_optimal_context(conv)

        def _get_optimal_context(self, conv):
            """Get optimally balanced context."""
            context = []
            token_count = 0

            # Always include system prompt
            if conv["system_prompt"]:
                context.append({"role": "system", "content": conv["system_prompt"]})
                token_count += len(self.encoding.encode(conv["system_prompt"]))

            # Include facts if any
            if conv["facts"]:
                facts_text = "Key facts:\n" + "\n".join([f"• {fact}" for fact in conv["facts"]])
                context.append({"role": "system", "content": facts_text})
                token_count += len(self.encoding.encode(facts_text))

            # Include summaries
            if conv["summaries"]:
                summary_text = "Previous conversation:\n" + "\n".join(conv["summaries"][-2:])
                context.append({"role": "system", "content": summary_text})
                token_count += len(self.encoding.encode(summary_text))

            # Include recent messages
            remaining_tokens = self.max_tokens - token_count
            recent_messages = []

            for msg in reversed(conv["messages"]):
                if msg["tokens"] <= remaining_tokens:
                    recent_messages.insert(0, {"role": msg["role"], "content": msg["content"]})
                    remaining_tokens -= msg["tokens"]
                else:
                    break

            context.extend(recent_messages)
            return context

        def apply_compression(self, conversation_id, method="auto"):
            """Apply compression to conversation."""
            if conversation_id not in self.conversations:
                return {"error": "Conversation not found"}

            conv = self.conversations[conversation_id]

            if method == "auto":
                # Choose method based on conversation characteristics
                if len(conv["messages"]) < 10:
                    method = "summarize"
                elif len(conv["facts"]) < 5:
                    method = "extract_facts"
                else:
                    method = "hierarchical"

            if method in self.compression_strategies:
                result = self.compression_strategies[method](conv)
                conv["metadata"]["compression_count"] += 1
                self.analytics["compression_count"] += 1
                return result
            else:
                return {"error": f"Unknown compression method: {method}"}

        def _compress_summarize(self, conv):
            """Compress using summarization."""
            if len(conv["messages"]) < 5:
                return {"success": False, "reason": "Too few messages"}

            # Take first half of messages
            to_summarize = conv["messages"][:len(conv["messages"])//2]
            remaining = conv["messages"][len(conv["messages"])//2:]

            # Create conversation text
            text = "\n".join([f"{m['role']}: {m['content']}" for m in to_summarize])

            # Summarize
            prompt = f"Summarize this conversation concisely:\n\n{text}\n\nSummary:"
            summary = client.complete(prompt, temperature=0.3, max_tokens=150).strip()

            conv["summaries"].append(summary)
            conv["messages"] = remaining

            return {"success": True, "method": "summarize", "compressed_messages": len(to_summarize)}

        def _compress_facts(self, conv):
            """Compress by extracting facts."""
            # Extract facts from messages
            for msg in conv["messages"][:5]:  # Process first 5 messages
                if "important" in msg["content"].lower() or msg["role"] == "system":
                    # Extract key info (simplified)
                    words = msg["content"].split()
                    if len(words) > 10:
                        fact = " ".join(words[:20]) + "..."
                        conv["facts"].add(fact)

            # Remove processed messages
            conv["messages"] = conv["messages"][5:]

            return {"success": True, "method": "extract_facts", "facts_extracted": len(conv["facts"])}

        def _compress_hierarchical(self, conv):
            """Hierarchical compression."""
            # Implement hierarchical compression
            return self._compress_summarize(conv)  # Simplified for demo

        def _compress_progressive(self, conv):
            """Progressive compression."""
            # Implement progressive compression
            return self._compress_summarize(conv)  # Simplified for demo

        def analyze_usage(self, conversation_id=None):
            """Provide usage analytics."""
            if conversation_id:
                if conversation_id not in self.conversations:
                    return {"error": "Conversation not found"}

                conv = self.conversations[conversation_id]
                return {
                    "conversation_id": conversation_id,
                    "messages": len(conv["messages"]),
                    "total_tokens": conv["metadata"]["token_usage"],
                    "compressions": conv["metadata"]["compression_count"],
                    "facts": len(conv["facts"]),
                    "summaries": len(conv["summaries"])
                }
            else:
                return self.analytics

        def optimize(self, conversation_id):
            """Automatically optimize conversation context."""
            if conversation_id not in self.conversations:
                return {"error": "Conversation not found"}

            conv = self.conversations[conversation_id]
            total_tokens = sum(m["tokens"] for m in conv["messages"])

            # Determine optimization strategy
            if total_tokens > self.max_tokens * 0.9:
                # Heavy compression needed
                self.apply_compression(conversation_id, "extract_facts")
                self.apply_compression(conversation_id, "summarize")
            elif total_tokens > self.max_tokens * 0.7:
                # Moderate compression
                self.apply_compression(conversation_id, "summarize")
            else:
                # No optimization needed
                return {"success": True, "action": "none_needed"}

            return {"success": True, "action": "optimized", "new_token_count": sum(m["tokens"] for m in conv["messages"])}

        def export_conversation(self, conversation_id):
            """Export conversation with full context."""
            if conversation_id not in self.conversations:
                return {"error": "Conversation not found"}

            conv = self.conversations[conversation_id]
            export_data = {
                "conversation_id": conversation_id,
                "created": conv["metadata"]["created"].isoformat(),
                "messages": [
                    {
                        "role": m["role"],
                        "content": m["content"],
                        "timestamp": m["timestamp"].isoformat()
                    }
                    for m in conv["messages"]
                ],
                "facts": list(conv["facts"]),
                "summaries": conv["summaries"],
                "metadata": {
                    "total_tokens": conv["metadata"]["token_usage"],
                    "compressions": conv["metadata"]["compression_count"]
                }
            }

            return json.dumps(export_data, indent=2)

        def handle_error(self, error_type, conversation_id, details=""):
            """Handle various error scenarios."""
            error_entry = {
                "type": error_type,
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "details": details
            }
            self.analytics["errors"].append(error_entry)

            if error_type == "token_overflow":
                # Force compression
                self.apply_compression(conversation_id, "extract_facts")
                self.apply_compression(conversation_id, "summarize")
            elif error_type == "api_error":
                # Log and retry logic would go here
                pass

            return {"error_handled": True, "type": error_type}

    # Test the production context manager
    manager = ProductionContextManager(max_tokens=1000)

    # Create conversations
    print("CREATING CONVERSATIONS:")
    manager.create_conversation("conv1", "You are a helpful Python programming assistant.")
    manager.create_conversation("conv2", "You are a data science expert.")
    print("Created 2 conversation threads\n")

    # Add messages to conv1
    print("ADDING MESSAGES TO CONV1:")
    messages = [
        ("user", "I need help with Python async programming"),
        ("assistant", "I'll help you with async programming. It's used for concurrent operations."),
        ("user", "Can you show me an example?"),
        ("assistant", "Here's a simple async example:\n```python\nasync def fetch_data():\n    await asyncio.sleep(1)\n    return 'data'\n```"),
        ("user", "How do I run multiple async functions?"),
        ("assistant", "Use asyncio.gather() to run multiple async functions concurrently."),
    ]

    for role, content in messages:
        result = manager.add_message("conv1", role, content)
        print(f"Added {role} message: {result['tokens']} tokens")

    # Analyze usage
    print("\nANALYTICS:")
    print(json.dumps(manager.analyze_usage("conv1"), indent=2))

    # Get optimized context
    print("\nOPTIMIZED CONTEXT:")
    context = manager.get_context("conv1", strategy="optimal")
    for msg in context:
        print(f"[{msg['role']}]: {msg['content'][:50]}...")

    # Test optimization
    print("\nOPTIMIZATION:")
    result = manager.optimize("conv1")
    print(f"Optimization result: {result}")

    # Export conversation
    print("\nEXPORT:")
    exported = manager.export_conversation("conv1")
    print(f"Exported conversation (first 200 chars):\n{exported[:200]}...")

    print("\n💡 Production context manager handles multiple conversations with optimization")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 07: Solutions")
    parser.add_argument("--solution", type=int, help="Run specific solution (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge solution")
    parser.add_argument("--all", action="store_true", help="Run all solutions")

    args = parser.parse_args()

    solutions = {
        1: solution_1_token_calculator,
        2: solution_2_sliding_window,
        3: solution_3_context_prioritization,
        4: solution_4_dynamic_summarization,
        5: solution_5_memory_system
    }

    if args.all:
        for sol_num, sol_func in solutions.items():
            print(f"\n{'='*60}")
            sol_func()
            print(f"\n{'='*60}\n")
        challenge_solution_production_context_manager()
    elif args.challenge:
        challenge_solution_production_context_manager()
    elif args.solution and args.solution in solutions:
        solutions[args.solution]()
    else:
        print("Module 07: Context Management - Solutions")
        print("\nUsage:")
        print("  python solutions.py --solution N  # Run solution N")
        print("  python solutions.py --challenge    # Run challenge solution")
        print("  python solutions.py --all          # Run all solutions")
        print("\nAvailable solutions:")
        print("  1: Token Budget Calculator")
        print("  2: Intelligent Sliding Window")
        print("  3: Context Prioritization")
        print("  4: Dynamic Summarization")
        print("  5: Memory System Design")
        print("  Challenge: Production Context Manager")