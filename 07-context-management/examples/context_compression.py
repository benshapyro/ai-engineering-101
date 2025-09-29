"""
Module 07: Context Compression

Learn techniques to compress context while preserving essential information.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import tiktoken
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib


def example_1_basic_summarization():
    """Use summarization to compress older context."""
    print("=" * 60)
    print("Example 1: Basic Summarization")
    print("=" * 60)

    client = LLMClient("openai")

    # Long conversation to compress
    long_conversation = """
    User: I want to build a recommendation system for my e-commerce site.
    Assistant: Great! There are several approaches: collaborative filtering, content-based, and hybrid systems.
    User: What's collaborative filtering?
    Assistant: It recommends items based on similar users' preferences. There's user-based and item-based CF.
    User: Which is better for a small catalog?
    Assistant: For small catalogs, content-based might work better as it doesn't need much user data.
    User: How do I implement content-based?
    Assistant: Extract item features, compute similarity scores, and recommend similar items to what users liked.
    User: What features should I extract?
    Assistant: For e-commerce: category, brand, price range, descriptions, tags, and user ratings.
    """

    print("ORIGINAL CONVERSATION:")
    print(long_conversation)

    encoding = tiktoken.encoding_for_model("gpt-4")
    original_tokens = len(encoding.encode(long_conversation))
    print(f"\nOriginal tokens: {original_tokens}")

    # Compress using summarization
    summary_prompt = f"""
    Summarize this conversation in 2-3 sentences, preserving key decisions and context:

    {long_conversation}

    Summary:"""

    summary = client.complete(summary_prompt, temperature=0.3, max_tokens=100)

    print("\nCOMPRESSED SUMMARY:")
    print(summary.strip())

    summary_tokens = len(encoding.encode(summary))
    print(f"\nSummary tokens: {summary_tokens}")
    print(f"Compression ratio: {original_tokens/summary_tokens:.1f}x")

    # Continue conversation with compressed context
    new_query = "What similarity metrics should I use?"

    continued_prompt = f"""
    Previous conversation summary: {summary}

    User: {new_query}
    Assistant:"""

    response = client.complete(continued_prompt, temperature=0.3, max_tokens=100)
    print(f"\nContinued conversation:")
    print(f"User: {new_query}")
    print(f"Assistant: {response.strip()}")

    print("\n💡 Summarization maintains context continuity with fewer tokens")


def example_2_key_point_extraction():
    """Extract and preserve key points instead of full context."""
    print("\n" + "=" * 60)
    print("Example 2: Key Point Extraction")
    print("=" * 60)

    client = LLMClient("openai")

    class KeyPointExtractor:
        def __init__(self):
            self.key_points = []
            self.encoding = tiktoken.encoding_for_model("gpt-4")

        def extract_from_conversation(self, messages):
            """Extract key points from messages."""
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}" for msg in messages
            ])

            prompt = f"""
            Extract the key facts, decisions, and requirements from this conversation.
            Return as a bulleted list of concise points:

            {conversation_text}

            Key points:"""

            response = client.complete(prompt, temperature=0.2, max_tokens=150)

            # Parse points
            points = []
            for line in response.strip().split("\n"):
                if line.strip().startswith(("•", "-", "*")):
                    points.append(line.strip().lstrip("•-* "))

            return points

        def update_key_points(self, new_points):
            """Merge new points with existing, removing duplicates."""
            all_points = self.key_points + new_points

            # Simple deduplication (in practice, use semantic similarity)
            unique_points = []
            seen = set()
            for point in all_points:
                # Use first 30 chars as simple duplicate check
                key = point[:30].lower()
                if key not in seen:
                    unique_points.append(point)
                    seen.add(key)

            self.key_points = unique_points[-10:]  # Keep last 10 points

        def get_context_string(self):
            """Format key points as context."""
            if not self.key_points:
                return ""
            return "Key points from conversation:\n" + "\n".join(
                [f"• {point}" for point in self.key_points]
            )

    extractor = KeyPointExtractor()

    # Simulate conversation chunks
    conversation_chunks = [
        [
            {"role": "user", "content": "I need a Python web scraper for product prices"},
            {"role": "assistant", "content": "I'll help you build a scraper using BeautifulSoup and requests"}
        ],
        [
            {"role": "user", "content": "It should handle multiple pages and save to CSV"},
            {"role": "assistant", "content": "We'll add pagination support and use pandas for CSV export"}
        ],
        [
            {"role": "user", "content": "How do I avoid being blocked?"},
            {"role": "assistant", "content": "Use delays, rotate user agents, and respect robots.txt"}
        ]
    ]

    print("EXTRACTING KEY POINTS FROM CONVERSATION:\n")

    for i, chunk in enumerate(conversation_chunks, 1):
        print(f"Chunk {i}:")
        for msg in chunk:
            print(f"  {msg['role']}: {msg['content']}")

        points = extractor.extract_from_conversation(chunk)
        extractor.update_key_points(points)

        print(f"\nExtracted points: {points}")
        print(f"Total key points: {len(extractor.key_points)}\n")
        print("-" * 40 + "\n")

    print("FINAL KEY POINTS CONTEXT:")
    context = extractor.get_context_string()
    print(context)

    tokens = len(extractor.encoding.encode(context))
    print(f"\nContext tokens: {tokens}")

    print("\n💡 Key point extraction preserves essential information efficiently")


def example_3_hierarchical_compression():
    """Implement multi-level compression for different time ranges."""
    print("\n" + "=" * 60)
    print("Example 3: Hierarchical Compression")
    print("=" * 60)

    class HierarchicalCompressor:
        def __init__(self):
            self.recent = []  # Last 5 messages - full detail
            self.medium = ""  # Last 20 messages - summarized
            self.long_term = ""  # Everything else - highly compressed
            self.message_count = 0

        def add_message(self, role, content):
            """Add message and manage compression levels."""
            self.message_count += 1
            message = {
                "role": role,
                "content": content,
                "number": self.message_count
            }

            # Add to recent
            self.recent.append(message)

            # Check if we need to compress
            if len(self.recent) > 5:
                # Move oldest to medium-term
                to_compress = self.recent[:3]
                self.recent = self.recent[3:]

                # Compress to medium-term
                self._compress_to_medium(to_compress)

            # Check medium-term compression
            if self.message_count % 20 == 0:
                self._compress_medium_to_long()

        def _compress_to_medium(self, messages):
            """Compress messages to medium-term summary."""
            content = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            summary = f"[Messages {messages[0]['number']}-{messages[-1]['number']}]: "
            summary += self._create_summary(content, level="medium")

            if self.medium:
                self.medium += "\n" + summary
            else:
                self.medium = summary

        def _compress_medium_to_long(self):
            """Compress medium-term to long-term."""
            if self.medium:
                self.long_term = self._create_summary(
                    self.long_term + "\n" + self.medium,
                    level="high"
                )
                self.medium = ""

        def _create_summary(self, content, level="medium"):
            """Create summary based on compression level."""
            if level == "medium":
                # Moderate compression - keep main points
                words = content.split()
                if len(words) > 50:
                    return " ".join(words[:30]) + "... [compressed]"
                return content
            else:
                # High compression - only essential info
                words = content.split()
                if len(words) > 30:
                    return " ".join(words[:15]) + "... [highly compressed]"
                return content

        def get_full_context(self):
            """Get complete context with all levels."""
            context_parts = []

            if self.long_term:
                context_parts.append(f"[LONG-TERM MEMORY]\n{self.long_term}")

            if self.medium:
                context_parts.append(f"[MEDIUM-TERM MEMORY]\n{self.medium}")

            if self.recent:
                recent_text = "\n".join([
                    f"{m['role']}: {m['content']}" for m in self.recent
                ])
                context_parts.append(f"[RECENT MESSAGES]\n{recent_text}")

            return "\n\n".join(context_parts)

    compressor = HierarchicalCompressor()

    # Simulate long conversation
    conversation = [
        ("user", "I want to learn machine learning"),
        ("assistant", "Start with Python basics and math fundamentals"),
        ("user", "What math do I need?"),
        ("assistant", "Linear algebra, calculus, and statistics"),
        ("user", "Recommend some courses"),
        ("assistant", "Andrew Ng's course on Coursera is excellent"),
        ("user", "What about books?"),
        ("assistant", "Pattern Recognition and Machine Learning by Bishop"),
        ("user", "Should I learn deep learning too?"),
        ("assistant", "Master classical ML first, then move to deep learning"),
        ("user", "What frameworks should I use?"),
        ("assistant", "Start with scikit-learn, then TensorFlow or PyTorch"),
    ]

    print("HIERARCHICAL COMPRESSION DEMO:\n")

    for role, content in conversation:
        compressor.add_message(role, content)
        print(f"Message {compressor.message_count}: [{role}] {content[:30]}...")

    print("\nFINAL HIERARCHICAL CONTEXT:")
    print("-" * 50)
    print(compressor.get_full_context())

    print("\n💡 Hierarchical compression balances detail and context length")


def example_4_semantic_compression():
    """Compress based on semantic importance and relevance."""
    print("\n" + "=" * 60)
    print("Example 4: Semantic Compression")
    print("=" * 60)

    client = LLMClient("openai")

    class SemanticCompressor:
        def __init__(self, focus_topic=""):
            self.focus_topic = focus_topic
            self.messages = []
            self.compressed_context = ""

        def add_message(self, role, content):
            """Add message with relevance scoring."""
            relevance = self._calculate_relevance(content)
            self.messages.append({
                "role": role,
                "content": content,
                "relevance": relevance
            })

        def _calculate_relevance(self, content):
            """Score relevance to focus topic (simplified)."""
            if not self.focus_topic:
                return 0.5

            score = 0.0
            content_lower = content.lower()
            focus_words = self.focus_topic.lower().split()

            for word in focus_words:
                if word in content_lower:
                    score += 0.2

            return min(1.0, score)

        def compress_semantically(self, target_tokens=200):
            """Compress keeping most relevant information."""
            encoding = tiktoken.encoding_for_model("gpt-4")

            # Sort by relevance
            sorted_msgs = sorted(self.messages,
                               key=lambda x: x['relevance'],
                               reverse=True)

            # Build context up to token limit
            context_parts = []
            total_tokens = 0

            for msg in sorted_msgs:
                msg_text = f"{msg['role']}: {msg['content']}"
                msg_tokens = len(encoding.encode(msg_text))

                if total_tokens + msg_tokens <= target_tokens:
                    context_parts.append((msg['relevance'], msg_text))
                    total_tokens += msg_tokens
                else:
                    # Truncate and add
                    remaining = target_tokens - total_tokens
                    if remaining > 20:  # Only add if meaningful
                        truncated = msg_text[:remaining * 3]  # Rough estimate
                        context_parts.append((msg['relevance'], truncated + "..."))
                    break

            # Sort by relevance for presentation
            context_parts.sort(key=lambda x: x[0], reverse=True)

            return "\n".join([text for _, text in context_parts])

    # Example with focus on "API integration"
    compressor = SemanticCompressor(focus_topic="API integration")

    messages = [
        ("user", "I need help with my Python project"),
        ("assistant", "I'd be happy to help. What kind of project?"),
        ("user", "It's a web app that needs to integrate with external APIs"),
        ("assistant", "For API integration, consider using requests library and handling authentication properly"),
        ("user", "What about the frontend?"),
        ("assistant", "You could use React or Vue.js for the frontend"),
        ("user", "How do I handle API rate limits?"),
        ("assistant", "Implement exponential backoff, caching, and request queuing for rate limit handling"),
        ("user", "Should I use a database?"),
        ("assistant", "Yes, store API responses in a database to reduce API calls"),
    ]

    print(f"SEMANTIC COMPRESSION (Focus: '{compressor.focus_topic}')\n")

    for role, content in messages:
        compressor.add_message(role, content)
        relevance = compressor.messages[-1]['relevance']
        print(f"[{role}] (relevance: {relevance:.1f}): {content[:50]}...")

    compressed = compressor.compress_semantically(target_tokens=150)

    print("\nSEMANTICALLY COMPRESSED CONTEXT:")
    print("-" * 50)
    print(compressed)

    print("\n💡 Semantic compression preserves topic-relevant information")


def example_5_fact_extraction():
    """Extract and maintain facts separately from conversation."""
    print("\n" + "=" * 60)
    print("Example 5: Fact Extraction and Management")
    print("=" * 60)

    client = LLMClient("openai")

    class FactManager:
        def __init__(self):
            self.facts = {}  # Dict of facts by category
            self.conversation = []

        def extract_facts(self, role, content):
            """Extract facts from a message."""
            prompt = f"""
            Extract factual information from this message.
            Categorize as: USER_INFO, REQUIREMENTS, CONSTRAINTS, DECISIONS, or TECHNICAL.
            Format: CATEGORY: fact

            Message ({role}): {content}

            Facts:"""

            response = client.complete(prompt, temperature=0.2, max_tokens=100)

            # Parse facts
            facts = []
            for line in response.strip().split("\n"):
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        category = parts[0].strip()
                        fact = parts[1].strip()
                        facts.append((category, fact))

            return facts

        def add_message(self, role, content):
            """Add message and extract facts."""
            self.conversation.append({"role": role, "content": content})

            # Extract facts
            facts = self.extract_facts(role, content)

            # Store facts by category
            for category, fact in facts:
                if category not in self.facts:
                    self.facts[category] = []
                # Avoid duplicates
                if fact not in self.facts[category]:
                    self.facts[category].append(fact)

        def get_fact_context(self):
            """Get facts formatted as context."""
            if not self.facts:
                return "No facts extracted yet."

            context = "EXTRACTED FACTS:\n"
            for category, fact_list in self.facts.items():
                context += f"\n{category}:\n"
                for fact in fact_list:
                    context += f"  • {fact}\n"

            return context

        def get_compressed_conversation(self, last_n=3):
            """Get recent conversation plus facts."""
            recent = self.conversation[-last_n:] if len(self.conversation) > last_n else self.conversation

            recent_text = "RECENT CONVERSATION:\n"
            for msg in recent:
                recent_text += f"{msg['role']}: {msg['content']}\n"

            return self.get_fact_context() + "\n" + recent_text

    manager = FactManager()

    conversation = [
        ("user", "I'm John, a Python developer with 5 years experience"),
        ("assistant", "Nice to meet you John! How can I help you today?"),
        ("user", "I need to build a REST API that handles 1000 requests per second"),
        ("assistant", "For high throughput, consider using FastAPI with async handlers"),
        ("user", "It must integrate with PostgreSQL and Redis"),
        ("assistant", "Good choices. Use asyncpg for PostgreSQL and aioredis for async Redis operations"),
        ("user", "Security is critical - we handle financial data"),
        ("assistant", "Implement OAuth2, use HTTPS only, encrypt sensitive data at rest"),
    ]

    print("FACT EXTRACTION DEMONSTRATION:\n")

    for role, content in conversation:
        manager.add_message(role, content)
        print(f"[{role}]: {content}")

    print("\n" + "=" * 50)
    print(manager.get_fact_context())

    print("\nCOMPRESSED CONTEXT WITH FACTS:")
    print("-" * 50)
    print(manager.get_compressed_conversation(last_n=2))

    print("\n💡 Fact extraction preserves key information efficiently")


def example_6_progressive_compression():
    """Progressively compress context as it ages."""
    print("\n" + "=" * 60)
    print("Example 6: Progressive Compression")
    print("=" * 60)

    client = LLMClient("openai")

    class ProgressiveCompressor:
        def __init__(self):
            self.messages = []
            self.compression_levels = {
                "full": 0,      # No compression (last 3 messages)
                "light": 0.3,   # 30% compression (last 10 messages)
                "medium": 0.6,  # 60% compression (last 20 messages)
                "heavy": 0.8,   # 80% compression (older messages)
            }

        def add_message(self, role, content):
            """Add message with timestamp."""
            self.messages.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now(),
                "compressed": False,
                "compression_level": "full"
            })
            self._apply_progressive_compression()

        def _apply_progressive_compression(self):
            """Apply compression based on message age."""
            message_count = len(self.messages)

            for i, msg in enumerate(self.messages):
                position = message_count - i

                if position <= 3:
                    level = "full"
                elif position <= 10:
                    level = "light"
                elif position <= 20:
                    level = "medium"
                else:
                    level = "heavy"

                if msg["compression_level"] != level:
                    msg["compression_level"] = level
                    msg["compressed_content"] = self._compress_content(
                        msg["content"],
                        self.compression_levels[level]
                    )

        def _compress_content(self, content, compression_ratio):
            """Compress content by specified ratio."""
            if compression_ratio == 0:
                return content

            words = content.split()
            keep_words = int(len(words) * (1 - compression_ratio))
            keep_words = max(5, keep_words)  # Keep at least 5 words

            if len(words) <= keep_words:
                return content

            # Keep beginning and end for context
            if keep_words > 10:
                start_words = keep_words // 2
                end_words = keep_words // 2
                compressed = " ".join(words[:start_words])
                compressed += " [...] "
                compressed += " ".join(words[-end_words:])
            else:
                compressed = " ".join(words[:keep_words]) + "..."

            return compressed

        def get_context(self):
            """Get progressively compressed context."""
            context = []

            for msg in self.messages:
                level = msg["compression_level"]
                if level == "full":
                    content = msg["content"]
                else:
                    content = msg.get("compressed_content", msg["content"])

                context.append({
                    "role": msg["role"],
                    "content": content,
                    "level": level
                })

            return context

    compressor = ProgressiveCompressor()

    # Long conversation
    long_conversation = [
        ("user", "I want to build a machine learning model for sentiment analysis of customer reviews"),
        ("assistant", "Great choice! Sentiment analysis is valuable for understanding customer feedback. You can use approaches like bag-of-words, TF-IDF, or modern transformer models"),
        ("user", "What data do I need to collect for training?"),
        ("assistant", "You'll need labeled reviews with sentiment scores. Aim for at least 10,000 examples balanced across positive, negative, and neutral sentiments"),
        ("user", "Should I use a pre-trained model or train from scratch?"),
        ("assistant", "Start with pre-trained models like BERT or RoBERTa fine-tuned on your data. They offer better performance with less training data"),
        ("user", "How do I handle sarcasm and irony in reviews?"),
        ("assistant", "Sarcasm is challenging. Use context-aware models, add sarcasm-labeled data, and consider features like punctuation patterns and emoji usage"),
        ("user", "What metrics should I use for evaluation?"),
        ("assistant", "Use accuracy, precision, recall, and F1-score. For imbalanced datasets, focus on macro-averaged F1 and confusion matrices"),
        ("user", "How do I deploy the model to production?"),
        ("assistant", "Container" "ize with Docker, use model serving frameworks like TensorFlow Serving or TorchServe, and implement monitoring for model drift"),
    ]

    print("PROGRESSIVE COMPRESSION DEMONSTRATION:\n")

    for role, content in long_conversation:
        compressor.add_message(role, content)

    context = compressor.get_context()

    for i, msg in enumerate(context, 1):
        level_indicator = {
            "full": "✓",
            "light": "░",
            "medium": "▒",
            "heavy": "▓"
        }[msg["level"]]

        print(f"{i}. [{level_indicator}] {msg['role']}: {msg['content'][:60]}...")

    print("\nCompression levels:")
    print("✓ Full | ░ Light (30%) | ▒ Medium (60%) | ▓ Heavy (80%)")

    print("\n💡 Progressive compression maintains recent detail while compacting history")


def run_all_examples():
    """Run all context compression examples."""
    examples = [
        example_1_basic_summarization,
        example_2_key_point_extraction,
        example_3_hierarchical_compression,
        example_4_semantic_compression,
        example_5_fact_extraction,
        example_6_progressive_compression
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

    parser = argparse.ArgumentParser(description="Module 07: Context Compression")
    parser.add_argument("--example", type=int, help="Run specific example (1-6)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_basic_summarization,
            2: example_2_key_point_extraction,
            3: example_3_hierarchical_compression,
            4: example_4_semantic_compression,
            5: example_5_fact_extraction,
            6: example_6_progressive_compression
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 07: Context Compression Techniques")
        print("\nUsage:")
        print("  python context_compression.py --all        # Run all examples")
        print("  python context_compression.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: Basic Summarization")
        print("  2: Key Point Extraction")
        print("  3: Hierarchical Compression")
        print("  4: Semantic Compression")
        print("  5: Fact Extraction")
        print("  6: Progressive Compression")