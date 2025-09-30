"""
Module 07: Context Manager Project

A comprehensive context management system that handles token budgets, sliding windows,
compression strategies, and memory persistence for production LLM applications.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import tiktoken
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import hashlib
import sqlite3


class CompressionStrategy(Enum):
    """Available compression strategies."""
    NONE = "none"
    SUMMARIZE = "summarize"
    EXTRACT_FACTS = "extract_facts"
    HIERARCHICAL = "hierarchical"
    PROGRESSIVE = "progressive"
    HYBRID = "hybrid"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    MEDIUM = 5
    HIGH = 10
    CRITICAL = 100
    SYSTEM = float('inf')


class ContextManager:
    """
    Advanced context management system for LLM applications.

    Features:
    - Token budget management
    - Multiple compression strategies
    - Sliding window with priorities
    - Memory persistence
    - Conversation analytics
    - Cost optimization
    """

    def __init__(self,
                 model: str = "gpt-4",
                 max_tokens: int = 4000,
                 compression_ratio: float = 0.7,
                 db_path: str = "context_memory.db"):
        """
        Initialize the context manager.

        Args:
            model: LLM model name
            max_tokens: Maximum token budget
            compression_ratio: Trigger compression at this ratio of max_tokens
            db_path: Path to SQLite database for persistence
        """
        self.model = model
        self.max_tokens = max_tokens
        self.compression_ratio = compression_ratio
        self.db_path = db_path

        # Initialize components
        self.encoding = tiktoken.encoding_for_model(model)
        self.client = LLMClient("openai")

        # Message storage
        self.messages = deque()
        self.system_message = None
        self.compressed_history = []
        self.facts = set()
        self.key_points = []

        # Analytics
        self.analytics = {
            "total_messages": 0,
            "total_tokens": 0,
            "compressions": 0,
            "cost_estimate": 0.0,
            "session_start": datetime.now()
        }

        # Pricing (per 1K tokens)
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-5-mini": {"input": 0.0015, "output": 0.002}
        }

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for persistence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created TIMESTAMP,
                model TEXT,
                max_tokens INTEGER,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                tokens INTEGER,
                priority INTEGER,
                timestamp TIMESTAMP,
                compressed BOOLEAN,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                fact TEXT,
                category TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                conversation_id TEXT PRIMARY KEY,
                total_messages INTEGER,
                total_tokens INTEGER,
                compressions INTEGER,
                cost_estimate REAL,
                last_updated TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        conn.commit()
        conn.close()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def set_system_message(self, content: str):
        """Set the system message that's always included."""
        self.system_message = {
            "role": "system",
            "content": content,
            "tokens": self.count_tokens(content),
            "priority": MessagePriority.SYSTEM,
            "timestamp": datetime.now()
        }

    def add_message(self,
                   role: str,
                   content: str,
                   priority: Optional[MessagePriority] = None,
                   metadata: Optional[Dict] = None) -> Dict:
        """
        Add a message to the context.

        Args:
            role: Message role (user, assistant, etc.)
            content: Message content
            priority: Message priority
            metadata: Additional metadata

        Returns:
            Status dictionary with token counts and actions taken
        """
        # Auto-detect priority if not provided
        if priority is None:
            priority = self._detect_priority(content)

        tokens = self.count_tokens(content)

        message = {
            "role": role,
            "content": content,
            "tokens": tokens,
            "priority": priority,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }

        self.messages.append(message)
        self.analytics["total_messages"] += 1
        self.analytics["total_tokens"] += tokens

        # Check if compression needed
        current_tokens = self._calculate_total_tokens()
        compression_triggered = False

        if current_tokens > self.max_tokens * self.compression_ratio:
            compression_triggered = True
            strategy = self._select_compression_strategy()
            self.compress(strategy)

        return {
            "tokens_added": tokens,
            "total_tokens": self._calculate_total_tokens(),
            "compression_triggered": compression_triggered,
            "messages_count": len(self.messages)
        }

    def _detect_priority(self, content: str) -> MessagePriority:
        """Automatically detect message priority."""
        content_lower = content.lower()

        if any(word in content_lower for word in ["error", "critical", "urgent", "important"]):
            return MessagePriority.HIGH
        elif any(word in content_lower for word in ["note", "remember", "key", "essential"]):
            return MessagePriority.MEDIUM
        else:
            return MessagePriority.LOW

    def _calculate_total_tokens(self) -> int:
        """Calculate total tokens in current context."""
        total = 0

        if self.system_message:
            total += self.system_message["tokens"]

        total += sum(msg["tokens"] for msg in self.messages)

        # Add compressed history tokens
        for compressed in self.compressed_history:
            total += self.count_tokens(compressed)

        return total

    def _select_compression_strategy(self) -> CompressionStrategy:
        """Select optimal compression strategy based on context."""
        message_count = len(self.messages)
        has_facts = len(self.facts) > 0
        has_compressed = len(self.compressed_history) > 0

        if message_count < 5:
            return CompressionStrategy.NONE
        elif message_count < 10 and not has_facts:
            return CompressionStrategy.EXTRACT_FACTS
        elif has_compressed:
            return CompressionStrategy.HIERARCHICAL
        else:
            return CompressionStrategy.SUMMARIZE

    def compress(self, strategy: CompressionStrategy = CompressionStrategy.HYBRID) -> Dict:
        """
        Apply compression strategy to the context.

        Args:
            strategy: Compression strategy to use

        Returns:
            Compression results
        """
        if strategy == CompressionStrategy.NONE:
            return {"compressed": False, "reason": "No compression needed"}

        elif strategy == CompressionStrategy.SUMMARIZE:
            return self._compress_summarize()

        elif strategy == CompressionStrategy.EXTRACT_FACTS:
            return self._compress_extract_facts()

        elif strategy == CompressionStrategy.HIERARCHICAL:
            return self._compress_hierarchical()

        elif strategy == CompressionStrategy.PROGRESSIVE:
            return self._compress_progressive()

        elif strategy == CompressionStrategy.HYBRID:
            # Combine multiple strategies
            results = []
            if len(self.messages) > 10:
                results.append(self._compress_extract_facts())
            if len(self.messages) > 5:
                results.append(self._compress_summarize())
            return {"compressed": True, "strategies_applied": results}

        else:
            return {"compressed": False, "error": "Unknown strategy"}

    def _compress_summarize(self) -> Dict:
        """Compress using summarization."""
        if len(self.messages) < 5:
            return {"compressed": False, "reason": "Too few messages"}

        # Take oldest 50% of messages
        cutoff = len(self.messages) // 2
        to_compress = list(self.messages)[:cutoff]

        # Create conversation text
        conv_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in to_compress
        ])

        # Summarize
        prompt = f"""
Summarize this conversation concisely, preserving all important information:

{conv_text}

Summary:"""

        summary = self.client.complete(prompt, temperature=0.3, max_tokens=200).strip()

        # Store summary and remove compressed messages
        self.compressed_history.append(f"[Previous conversation]: {summary}")

        # Remove compressed messages
        for _ in range(cutoff):
            self.messages.popleft()

        self.analytics["compressions"] += 1

        return {
            "compressed": True,
            "method": "summarize",
            "messages_compressed": cutoff,
            "summary_length": len(summary)
        }

    def _compress_extract_facts(self) -> Dict:
        """Extract facts from messages."""
        facts_extracted = 0

        for msg in list(self.messages)[:5]:  # Process first 5 messages
            # Extract facts using LLM
            prompt = f"""
Extract key facts from this message:
{msg['role']}: {msg['content']}

Return only the facts, one per line:"""

            response = self.client.complete(prompt, temperature=0.2, max_tokens=100)

            for line in response.strip().split("\n"):
                if line.strip():
                    self.facts.add(line.strip())
                    facts_extracted += 1

        # Remove processed messages
        for _ in range(min(5, len(self.messages))):
            self.messages.popleft()

        return {
            "compressed": True,
            "method": "extract_facts",
            "facts_extracted": facts_extracted
        }

    def _compress_hierarchical(self) -> Dict:
        """Apply hierarchical compression."""
        # Group messages by age
        recent = []  # Last 3 messages
        medium = []  # Last 10 messages
        old = []     # Everything else

        messages_list = list(self.messages)

        if len(messages_list) <= 3:
            recent = messages_list
        elif len(messages_list) <= 10:
            recent = messages_list[-3:]
            medium = messages_list[:-3]
        else:
            recent = messages_list[-3:]
            medium = messages_list[-10:-3]
            old = messages_list[:-10]

        compressed_count = 0

        # Heavy compression for old messages
        if old:
            old_text = " ".join([msg['content'][:50] for msg in old])
            self.compressed_history.append(f"[Old context]: {old_text}")
            compressed_count += len(old)

        # Medium compression for medium-age messages
        if medium:
            medium_text = " ".join([msg['content'][:100] for msg in medium])
            self.compressed_history.append(f"[Recent context]: {medium_text}")
            compressed_count += len(medium)

        # Keep recent messages as-is
        self.messages = deque(recent)

        return {
            "compressed": True,
            "method": "hierarchical",
            "messages_compressed": compressed_count
        }

    def _compress_progressive(self) -> Dict:
        """Progressive compression based on message age."""
        compressed_messages = []

        for i, msg in enumerate(list(self.messages)):
            age_factor = i / len(self.messages)  # 0 = oldest, 1 = newest

            if age_factor < 0.3:  # Old messages - heavy compression
                compressed = msg['content'][:50] + "..."
            elif age_factor < 0.7:  # Medium age - moderate compression
                compressed = msg['content'][:150] + "..."
            else:  # Recent - no compression
                compressed = msg['content']

            compressed_messages.append({
                **msg,
                "content": compressed,
                "compressed": age_factor < 0.7
            })

        self.messages = deque(compressed_messages)

        return {
            "compressed": True,
            "method": "progressive",
            "messages_processed": len(compressed_messages)
        }

    def get_context(self,
                   include_compressed: bool = True,
                   include_facts: bool = True,
                   max_messages: Optional[int] = None) -> List[Dict]:
        """
        Get the current context for LLM input.

        Args:
            include_compressed: Include compressed history
            include_facts: Include extracted facts
            max_messages: Maximum number of recent messages

        Returns:
            List of messages formatted for LLM
        """
        context = []

        # System message
        if self.system_message:
            context.append({
                "role": self.system_message["role"],
                "content": self.system_message["content"]
            })

        # Compressed history
        if include_compressed and self.compressed_history:
            compressed_content = "\n".join(self.compressed_history)
            context.append({
                "role": "system",
                "content": f"Previous context:\n{compressed_content}"
            })

        # Facts
        if include_facts and self.facts:
            facts_content = "\n".join([f"• {fact}" for fact in self.facts])
            context.append({
                "role": "system",
                "content": f"Key facts:\n{facts_content}"
            })

        # Recent messages
        messages_to_include = list(self.messages)
        if max_messages:
            messages_to_include = messages_to_include[-max_messages:]

        for msg in messages_to_include:
            context.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return context

    def sliding_window(self, window_size: int = 10) -> None:
        """
        Apply sliding window to messages.

        Args:
            window_size: Number of messages to keep
        """
        while len(self.messages) > window_size:
            # Remove oldest message with lowest priority
            messages_list = list(self.messages)
            messages_list.sort(key=lambda x: (x["priority"].value, x["timestamp"]))

            # Don't remove high priority messages
            for msg in messages_list:
                if msg["priority"].value < MessagePriority.HIGH.value:
                    self.messages.remove(msg)
                    break

            # If all are high priority, remove oldest
            if len(self.messages) > window_size:
                self.messages.popleft()

    def estimate_cost(self, input_tokens: Optional[int] = None, output_tokens: int = 100) -> Dict:
        """
        Estimate cost for current context.

        Args:
            input_tokens: Override input token count
            output_tokens: Expected output tokens

        Returns:
            Cost estimation
        """
        if input_tokens is None:
            input_tokens = self._calculate_total_tokens()

        if self.model in self.pricing:
            input_cost = (input_tokens / 1000) * self.pricing[self.model]["input"]
            output_cost = (output_tokens / 1000) * self.pricing[self.model]["output"]
            total_cost = input_cost + output_cost

            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost
            }
        else:
            return {"error": f"No pricing for model: {self.model}"}

    def save_conversation(self, conversation_id: str) -> bool:
        """
        Save conversation to database.

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Save conversation metadata
            cursor.execute("""
                INSERT OR REPLACE INTO conversations (id, created, model, max_tokens, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                conversation_id,
                datetime.now(),
                self.model,
                self.max_tokens,
                json.dumps({"compression_ratio": self.compression_ratio})
            ))

            # Save messages
            for msg in self.messages:
                cursor.execute("""
                    INSERT INTO messages (conversation_id, role, content, tokens, priority, timestamp, compressed)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    conversation_id,
                    msg["role"],
                    msg["content"],
                    msg["tokens"],
                    msg["priority"].value if isinstance(msg["priority"], MessagePriority) else msg["priority"],
                    msg["timestamp"],
                    msg.get("compressed", False)
                ))

            # Save facts
            for fact in self.facts:
                cursor.execute("""
                    INSERT INTO facts (conversation_id, fact, category, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (
                    conversation_id,
                    fact,
                    "general",
                    datetime.now()
                ))

            # Save analytics
            cursor.execute("""
                INSERT OR REPLACE INTO analytics
                (conversation_id, total_messages, total_tokens, compressions, cost_estimate, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                conversation_id,
                self.analytics["total_messages"],
                self.analytics["total_tokens"],
                self.analytics["compressions"],
                self.analytics["cost_estimate"],
                datetime.now()
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False

    def load_conversation(self, conversation_id: str) -> bool:
        """
        Load conversation from database.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Load conversation metadata
            cursor.execute("""
                SELECT model, max_tokens, metadata FROM conversations WHERE id = ?
            """, (conversation_id,))

            row = cursor.fetchone()
            if not row:
                return False

            self.model = row[0]
            self.max_tokens = row[1]

            # Load messages
            cursor.execute("""
                SELECT role, content, tokens, priority, timestamp, compressed
                FROM messages WHERE conversation_id = ?
                ORDER BY timestamp
            """, (conversation_id,))

            self.messages.clear()
            for row in cursor.fetchall():
                self.messages.append({
                    "role": row[0],
                    "content": row[1],
                    "tokens": row[2],
                    "priority": MessagePriority(row[3]) if row[3] else MessagePriority.LOW,
                    "timestamp": datetime.fromisoformat(row[4]) if isinstance(row[4], str) else row[4],
                    "compressed": row[5]
                })

            # Load facts
            cursor.execute("""
                SELECT fact FROM facts WHERE conversation_id = ?
            """, (conversation_id,))

            self.facts.clear()
            for row in cursor.fetchall():
                self.facts.add(row[0])

            # Load analytics
            cursor.execute("""
                SELECT total_messages, total_tokens, compressions, cost_estimate
                FROM analytics WHERE conversation_id = ?
            """, (conversation_id,))

            row = cursor.fetchone()
            if row:
                self.analytics.update({
                    "total_messages": row[0],
                    "total_tokens": row[1],
                    "compressions": row[2],
                    "cost_estimate": row[3]
                })

            conn.close()
            return True

        except Exception as e:
            print(f"Error loading conversation: {e}")
            return False

    def get_analytics(self) -> Dict:
        """Get comprehensive analytics."""
        session_duration = (datetime.now() - self.analytics["session_start"]).total_seconds() / 60

        return {
            **self.analytics,
            "current_tokens": self._calculate_total_tokens(),
            "token_utilization": self._calculate_total_tokens() / self.max_tokens,
            "message_count": len(self.messages),
            "facts_count": len(self.facts),
            "compressed_count": len(self.compressed_history),
            "session_duration_minutes": round(session_duration, 2),
            "avg_message_tokens": self.analytics["total_tokens"] / max(self.analytics["total_messages"], 1)
        }

    def clear(self):
        """Clear all context except system message."""
        self.messages.clear()
        self.compressed_history.clear()
        self.facts.clear()
        self.key_points.clear()


def interactive_demo():
    """Interactive demonstration of the context manager."""
    print("=" * 60)
    print("CONTEXT MANAGER DEMONSTRATION")
    print("=" * 60)

    manager = ContextManager(model="gpt-4", max_tokens=500)

    # Set system message
    manager.set_system_message("You are a helpful AI assistant.")

    # Simulate conversation
    conversation = [
        ("user", "I need help building a web scraper", MessagePriority.MEDIUM),
        ("assistant", "I'll help you build a web scraper. What website do you want to scrape?", MessagePriority.LOW),
        ("user", "I want to scrape product prices from e-commerce sites", MessagePriority.MEDIUM),
        ("assistant", "For e-commerce scraping, you'll need BeautifulSoup and requests libraries.", MessagePriority.MEDIUM),
        ("user", "IMPORTANT: It must handle rate limiting", MessagePriority.HIGH),
        ("assistant", "Rate limiting is crucial. Implement delays, rotate user agents, and use proxies.", MessagePriority.HIGH),
        ("user", "How do I handle dynamic content?", MessagePriority.LOW),
        ("assistant", "For dynamic content, use Selenium or Playwright for JavaScript rendering.", MessagePriority.MEDIUM),
        ("user", "ERROR: Getting blocked after 10 requests", MessagePriority.CRITICAL),
        ("assistant", "You're being blocked due to detection. Add random delays, rotate headers, and respect robots.txt.", MessagePriority.HIGH),
    ]

    print("\nADDING MESSAGES:\n")

    for role, content, priority in conversation:
        result = manager.add_message(role, content, priority)
        print(f"[{role}] {content[:40]}... (Priority: {priority.name})")
        print(f"  Tokens: {result['tokens_added']}, Total: {result['total_tokens']}")
        if result["compression_triggered"]:
            print("  🔄 Compression triggered!")
        print()

    # Get context
    print("\nCURRENT CONTEXT:")
    print("-" * 40)
    context = manager.get_context()
    for msg in context:
        print(f"[{msg['role']}]: {msg['content'][:60]}...")

    # Analytics
    print("\nANALYTICS:")
    print("-" * 40)
    analytics = manager.get_analytics()
    for key, value in analytics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # Cost estimate
    print("\nCOST ESTIMATE:")
    print("-" * 40)
    cost = manager.estimate_cost()
    print(f"Input tokens: {cost['input_tokens']}")
    print(f"Estimated total cost: ${cost['total_cost']:.4f}")

    # Save conversation
    print("\nSAVING CONVERSATION:")
    conversation_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if manager.save_conversation(conversation_id):
        print(f"✅ Saved as: {conversation_id}")

    # Test loading
    print("\nTESTING LOAD:")
    new_manager = ContextManager()
    if new_manager.load_conversation(conversation_id):
        print(f"✅ Loaded {len(new_manager.messages)} messages")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Context Manager System")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--load", type=str, help="Load conversation by ID")
    parser.add_argument("--analyze", type=str, help="Analyze conversation by ID")

    args = parser.parse_args()

    if args.demo:
        interactive_demo()
    elif args.load:
        manager = ContextManager()
        if manager.load_conversation(args.load):
            print(f"Loaded conversation: {args.load}")
            context = manager.get_context()
            for msg in context:
                print(f"[{msg['role']}]: {msg['content'][:100]}...")
        else:
            print(f"Failed to load conversation: {args.load}")
    elif args.analyze:
        manager = ContextManager()
        if manager.load_conversation(args.analyze):
            analytics = manager.get_analytics()
            print(f"\nAnalytics for {args.analyze}:")
            for key, value in analytics.items():
                print(f"  {key}: {value}")
        else:
            print(f"Failed to load conversation: {args.analyze}")
    else:
        print("Context Manager System")
        print("\nUsage:")
        print("  python context_manager.py --demo           # Run demo")
        print("  python context_manager.py --load ID        # Load conversation")
        print("  python context_manager.py --analyze ID     # Analyze conversation")
        print("\nFeatures:")
        print("  • Token budget management")
        print("  • Multiple compression strategies")
        print("  • Priority-based message retention")
        print("  • Conversation persistence")
        print("  • Cost optimization")
        print("  • Comprehensive analytics")


if __name__ == "__main__":
    main()