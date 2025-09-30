"""
Module 07: Context Management - Exercises

Practice exercises for mastering context window management and compression.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import tiktoken
import json
from typing import Dict, List, Optional
from datetime import datetime


# ===== Exercise 1: Token Budget Calculator =====

def exercise_1_token_calculator():
    """
    Exercise 1: Build a comprehensive token budget calculator.

    TODO:
    1. Create a calculator for different models
    2. Estimate costs for various scenarios
    3. Provide optimization recommendations
    4. Track token usage over time
    """
    print("Exercise 1: Token Budget Calculator")
    print("=" * 50)

    class TokenCalculator:
        def __init__(self):
            # TODO: Initialize pricing for different models
            self.pricing = {
                "gpt-4": {"input": 0, "output": 0},  # TODO: Add real prices
                "gpt-5-mini": {"input": 0, "output": 0}
            }
            self.usage_history = []

        def calculate_tokens(self, text, model="gpt-4"):
            """TODO: Count tokens for given text and model."""
            # TODO: Use tiktoken to count tokens
            pass

        def estimate_cost(self, input_text, expected_output_length, model="gpt-4"):
            """TODO: Estimate cost for a request."""
            # TODO: Calculate input and output costs
            pass

        def track_usage(self, input_tokens, output_tokens, model):
            """TODO: Track token usage over time."""
            # TODO: Add to usage history with timestamp
            pass

        def generate_report(self):
            """TODO: Generate usage report with recommendations."""
            # TODO: Analyze usage patterns and suggest optimizations
            pass

    # TODO: Test your calculator
    calculator = TokenCalculator()

    test_texts = [
        "Short prompt",
        "A much longer prompt that contains multiple sentences and should use more tokens",
        "def calculate_fibonacci(n):\n    if n <= 1:\n        return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"
    ]

    print("TODO: Implement token calculation for each test text")
    print("TODO: Estimate costs for different models")
    print("TODO: Generate optimization recommendations")


# ===== Exercise 2: Implement Sliding Window =====

def exercise_2_sliding_window():
    """
    Exercise 2: Build an intelligent sliding window system.

    TODO:
    1. Implement token-aware sliding
    2. Add importance scoring
    3. Handle system messages specially
    4. Implement smooth transitions
    """
    print("\nExercise 2: Implement Sliding Window")
    print("=" * 50)

    class IntelligentSlidingWindow:
        def __init__(self, max_tokens=1000):
            self.max_tokens = max_tokens
            self.messages = []
            self.system_message = None

        def set_system_message(self, content):
            """TODO: Set and protect system message."""
            # TODO: System message should always be included
            pass

        def add_message(self, role, content, importance=1):
            """TODO: Add message with importance scoring."""
            # TODO: Store message with metadata
            # TODO: Implement sliding based on importance
            pass

        def _calculate_importance(self, message):
            """TODO: Calculate message importance."""
            # TODO: Score based on:
            # - Keywords (error, important, remember)
            # - Message role
            # - Recency
            # - User markers
            pass

        def _slide_window(self):
            """TODO: Slide window preserving important messages."""
            # TODO: Remove low-importance old messages first
            # TODO: Always keep system message
            # TODO: Maintain conversation coherence
            pass

        def get_context(self):
            """TODO: Get current window context."""
            # TODO: Return messages within token budget
            pass

    # TODO: Test your sliding window
    window = IntelligentSlidingWindow(max_tokens=500)

    print("TODO: Add system message")
    print("TODO: Add messages with varying importance")
    print("TODO: Verify important messages are retained")
    print("TODO: Check smooth sliding behavior")


# ===== Exercise 3: Context Prioritization =====

def exercise_3_context_prioritization():
    """
    Exercise 3: Build a context prioritization system.

    TODO:
    1. Score context elements by relevance
    2. Implement different prioritization strategies
    3. Balance recency vs importance
    4. Handle different context types
    """
    print("\nExercise 3: Context Prioritization")
    print("=" * 50)

    class ContextPrioritizer:
        def __init__(self):
            self.contexts = []
            self.prioritization_strategy = "balanced"

        def add_context(self, content, context_type, metadata=None):
            """TODO: Add context with type and metadata."""
            # Context types: system, example, fact, conversation, reference
            # TODO: Store with timestamp and initial score
            pass

        def score_relevance(self, context, current_query):
            """TODO: Score context relevance to current query."""
            # TODO: Implement scoring based on:
            # - Keyword overlap
            # - Semantic similarity (simplified)
            # - Context type weight
            # - Age decay
            pass

        def prioritize(self, current_query, token_budget):
            """TODO: Select best contexts within budget."""
            # TODO: Score all contexts
            # TODO: Apply strategy (recency, importance, balanced)
            # TODO: Select top contexts within token budget
            pass

        def set_strategy(self, strategy):
            """TODO: Set prioritization strategy."""
            # Strategies: recency_first, importance_first, balanced
            pass

    # TODO: Test prioritization
    prioritizer = ContextPrioritizer()

    contexts = [
        ("System prompt", "system"),
        ("User preferences", "fact"),
        ("Previous error", "conversation"),
        ("Code example", "example"),
        ("API documentation", "reference")
    ]

    print("TODO: Add various context types")
    print("TODO: Test different prioritization strategies")
    print("TODO: Verify token budget compliance")


# ===== Exercise 4: Dynamic Summarization =====

def exercise_4_dynamic_summarization():
    """
    Exercise 4: Implement dynamic context summarization.

    TODO:
    1. Detect when summarization is needed
    2. Summarize different content types appropriately
    3. Preserve key information
    4. Maintain conversation flow
    """
    print("\nExercise 4: Dynamic Summarization")
    print("=" * 50)

    client = LLMClient("openai")

    class DynamicSummarizer:
        def __init__(self, compression_threshold=0.7):
            self.compression_threshold = compression_threshold
            self.conversation_buffer = []
            self.summaries = []

        def add_conversation_turn(self, user_msg, assistant_msg):
            """TODO: Add conversation turn and check if summarization needed."""
            # TODO: Add to buffer
            # TODO: Check total size
            # TODO: Trigger summarization if needed
            pass

        def should_summarize(self):
            """TODO: Determine if summarization is needed."""
            # TODO: Check token count
            # TODO: Check conversation length
            # TODO: Check redundancy
            pass

        def summarize_conversation(self, conversation_chunk):
            """TODO: Summarize a conversation chunk."""
            # TODO: Create appropriate prompt for summarization
            # TODO: Call LLM to summarize
            # TODO: Extract key points
            pass

        def get_compressed_context(self):
            """TODO: Get full context with summaries and recent messages."""
            # TODO: Combine summaries and recent buffer
            # TODO: Format appropriately
            pass

    # TODO: Test dynamic summarization
    summarizer = DynamicSummarizer()

    conversation = [
        ("What's machine learning?", "ML is a type of AI that learns from data..."),
        ("How does it work?", "It uses algorithms to find patterns..."),
        ("What are neural networks?", "Neural networks are inspired by the brain..."),
        # Add more conversation turns
    ]

    print("TODO: Process conversation with dynamic summarization")
    print("TODO: Verify key information preserved")
    print("TODO: Check compression ratio")


# ===== Exercise 5: Memory System Design =====

def exercise_5_memory_system():
    """
    Exercise 5: Design a complete memory management system.

    TODO:
    1. Implement short-term and long-term memory
    2. Create memory retrieval mechanisms
    3. Handle memory overflow gracefully
    4. Implement memory persistence
    """
    print("\nExercise 5: Memory System Design")
    print("=" * 50)

    class MemorySystem:
        def __init__(self):
            self.short_term = []  # Recent, detailed
            self.long_term = {}   # Categorized, compressed
            self.working_memory = []  # Current context
            self.memory_index = {}  # For fast retrieval

        def store_short_term(self, content, category=None):
            """TODO: Store in short-term memory."""
            # TODO: Add with timestamp
            # TODO: Limit size
            # TODO: Move old items to long-term
            pass

        def promote_to_long_term(self, items):
            """TODO: Move items to long-term memory."""
            # TODO: Categorize items
            # TODO: Compress if needed
            # TODO: Update index
            pass

        def retrieve(self, query, memory_type="all"):
            """TODO: Retrieve relevant memories."""
            # TODO: Search short-term
            # TODO: Search long-term by category
            # TODO: Rank by relevance
            pass

        def consolidate(self):
            """TODO: Consolidate and organize memories."""
            # TODO: Merge similar memories
            # TODO: Remove redundancies
            # TODO: Update categories
            pass

        def export_memories(self):
            """TODO: Export memories for persistence."""
            # TODO: Serialize all memory types
            # TODO: Include metadata
            pass

        def import_memories(self, data):
            """TODO: Import previously saved memories."""
            # TODO: Deserialize data
            # TODO: Rebuild indices
            pass

    # TODO: Test memory system
    memory = MemorySystem()

    print("TODO: Store various types of information")
    print("TODO: Test retrieval with different queries")
    print("TODO: Implement and test consolidation")
    print("TODO: Test export/import functionality")


# ===== Challenge: Build a Production Context Manager =====

def challenge_production_context_manager():
    """
    Challenge: Build a production-ready context management system.

    Requirements:
    1. Handle multiple conversation threads
    2. Implement all optimization strategies
    3. Support different LLM models
    4. Provide analytics and monitoring
    5. Handle errors gracefully

    TODO: Complete the implementation
    """
    print("\nChallenge: Production Context Manager")
    print("=" * 50)

    class ProductionContextManager:
        def __init__(self, model="gpt-4", max_tokens=4000):
            self.model = model
            self.max_tokens = max_tokens
            self.conversations = {}  # Multiple conversation threads
            self.analytics = {}
            self.compression_strategies = {}

        def create_conversation(self, conversation_id, system_prompt=None):
            """TODO: Initialize a new conversation thread."""
            pass

        def add_message(self, conversation_id, role, content, metadata=None):
            """TODO: Add message to specific conversation."""
            # TODO: Apply compression if needed
            # TODO: Update analytics
            # TODO: Handle overflow
            pass

        def get_context(self, conversation_id, strategy="optimal"):
            """TODO: Get optimized context for conversation."""
            # Strategies: optimal, recent, compressed, hybrid
            pass

        def apply_compression(self, conversation_id, method="auto"):
            """TODO: Apply compression to conversation."""
            # Methods: summarize, extract_facts, hierarchical, progressive
            pass

        def analyze_usage(self, conversation_id=None):
            """TODO: Provide usage analytics."""
            # TODO: Token usage over time
            # TODO: Compression effectiveness
            # TODO: Cost analysis
            pass

        def optimize(self, conversation_id):
            """TODO: Automatically optimize conversation context."""
            # TODO: Analyze patterns
            # TODO: Select best strategy
            # TODO: Apply optimizations
            pass

        def export_conversation(self, conversation_id):
            """TODO: Export conversation with full context."""
            pass

        def handle_error(self, error_type, conversation_id):
            """TODO: Handle various error scenarios."""
            # TODO: Token overflow
            # TODO: API errors
            # TODO: Compression failures
            pass

    # TODO: Implement complete context manager
    manager = ProductionContextManager()

    print("TODO: Create multiple conversation threads")
    print("TODO: Test different compression strategies")
    print("TODO: Implement analytics and monitoring")
    print("TODO: Handle edge cases and errors")
    print("TODO: Optimize for production use")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 07: Context Management Exercises")
    parser.add_argument("--exercise", type=int, help="Run specific exercise (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge")
    parser.add_argument("--all", action="store_true", help="Run all exercises")

    args = parser.parse_args()

    exercises = {
        1: exercise_1_token_calculator,
        2: exercise_2_sliding_window,
        3: exercise_3_context_prioritization,
        4: exercise_4_dynamic_summarization,
        5: exercise_5_memory_system
    }

    if args.all:
        for ex_num, ex_func in exercises.items():
            print(f"\n{'='*60}")
            ex_func()
            print(f"\n{'='*60}\n")
        challenge_production_context_manager()
    elif args.challenge:
        challenge_production_context_manager()
    elif args.exercise and args.exercise in exercises:
        exercises[args.exercise]()
    else:
        print("Module 07: Context Management - Exercises")
        print("\nUsage:")
        print("  python exercises.py --exercise N  # Run exercise N")
        print("  python exercises.py --challenge    # Run challenge")
        print("  python exercises.py --all          # Run all exercises")
        print("\nAvailable exercises:")
        print("  1: Token Budget Calculator")
        print("  2: Implement Sliding Window")
        print("  3: Context Prioritization")
        print("  4: Dynamic Summarization")
        print("  5: Memory System Design")
        print("  Challenge: Production Context Manager")