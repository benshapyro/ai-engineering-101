"""
Context management utilities for scalable context windows.

This module provides tools for:
- Scoring and selecting relevant context snippets
- Summarizing long conversation threads
- Managing token budgets efficiently
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from shared.utils import count_tokens


@dataclass
class ContextSnippet:
    """Represents a piece of context with metadata."""
    content: str
    relevance_score: float = 0.0
    priority: int = 0  # Higher = more important
    metadata: Dict[str, Any] = None
    token_count: int = 0

    def __post_init__(self):
        """Calculate token count if not provided."""
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)
        if self.metadata is None:
            self.metadata = {}


class ContextRouter:
    """
    Routes and selects context snippets based on relevance and token budget.

    Helps manage long contexts by:
    - Scoring snippets by relevance to query
    - Selecting top-k within token budget
    - Supporting priority-based selection
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        model: str = "gpt-5"
    ):
        """
        Initialize context router.

        Args:
            max_tokens: Maximum tokens for selected context
            model: Model name for token counting
        """
        self.max_tokens = max_tokens
        self.model = model
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )

    def score_tfidf(
        self,
        query: str,
        snippets: List[str]
    ) -> List[float]:
        """
        Score snippets using TF-IDF similarity to query.

        Args:
            query: Query string
            snippets: List of context snippets

        Returns:
            List of relevance scores (0-1)
        """
        if not snippets:
            return []

        # Combine query and snippets for vectorization
        all_texts = [query] + snippets

        try:
            # Fit and transform
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)

            # Query is first row
            query_vec = tfidf_matrix[0:1]
            snippet_vecs = tfidf_matrix[1:]

            # Compute cosine similarity
            similarities = cosine_similarity(query_vec, snippet_vecs)[0]

            return similarities.tolist()

        except Exception:
            # Fallback: return uniform scores
            return [0.5] * len(snippets)

    def score_keyword(
        self,
        query: str,
        snippets: List[str]
    ) -> List[float]:
        """
        Score snippets by keyword overlap with query.

        Simple and fast, good for quick filtering.

        Args:
            query: Query string
            snippets: List of context snippets

        Returns:
            List of relevance scores (0-1)
        """
        # Extract keywords (simple: lowercase words)
        query_words = set(re.findall(r'\w+', query.lower()))

        scores = []
        for snippet in snippets:
            snippet_words = set(re.findall(r'\w+', snippet.lower()))
            overlap = len(query_words & snippet_words)
            max_possible = len(query_words)

            score = overlap / max_possible if max_possible > 0 else 0.0
            scores.append(score)

        return scores

    def select_context(
        self,
        query: str,
        snippets: List[str],
        method: str = "tfidf",
        priorities: Optional[List[int]] = None
    ) -> List[Tuple[int, str, float]]:
        """
        Select best snippets within token budget.

        Args:
            query: Query to match against
            snippets: Candidate snippets
            method: Scoring method ("tfidf" or "keyword")
            priorities: Optional priority per snippet

        Returns:
            List of (index, snippet, score) tuples for selected snippets
        """
        if not snippets:
            return []

        # Score relevance
        if method == "tfidf":
            scores = self.score_tfidf(query, snippets)
        elif method == "keyword":
            scores = self.score_keyword(query, snippets)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create context snippets with metadata
        context_items = []
        for i, (snippet, score) in enumerate(zip(snippets, scores)):
            priority = priorities[i] if priorities else 0
            context_items.append(ContextSnippet(
                content=snippet,
                relevance_score=score,
                priority=priority,
                metadata={"index": i}
            ))

        # Sort by priority (descending), then relevance (descending)
        context_items.sort(
            key=lambda x: (x.priority, x.relevance_score),
            reverse=True
        )

        # Select within token budget
        selected = []
        total_tokens = 0

        for item in context_items:
            if total_tokens + item.token_count <= self.max_tokens:
                selected.append((
                    item.metadata["index"],
                    item.content,
                    item.relevance_score
                ))
                total_tokens += item.token_count
            else:
                # Budget exhausted
                break

        # Sort by original index to preserve order
        selected.sort(key=lambda x: x[0])

        return selected

    def sliding_window(
        self,
        text: str,
        window_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into overlapping windows.

        Useful for processing long documents.

        Args:
            text: Text to split
            window_size: Tokens per window
            overlap: Overlap tokens between windows

        Returns:
            List of text windows
        """
        words = text.split()
        windows = []

        step = window_size - overlap
        for i in range(0, len(words), step):
            window = ' '.join(words[i:i + window_size])
            windows.append(window)

            # Stop if we've covered the whole text
            if i + window_size >= len(words):
                break

        return windows


class Summarizer:
    """
    Structured memory for long conversation threads.

    Progressively summarizes conversations to fit token budgets
    while preserving critical information.
    """

    def __init__(
        self,
        client,
        max_summary_tokens: int = 500,
        model: str = "gpt-5"
    ):
        """
        Initialize summarizer.

        Args:
            client: LLMClient instance
            max_summary_tokens: Target tokens for summaries
            model: Model name
        """
        self.client = client
        self.max_summary_tokens = max_summary_tokens
        self.model = model

    def summarize_text(
        self,
        text: str,
        target_tokens: Optional[int] = None,
        focus: Optional[str] = None
    ) -> str:
        """
        Summarize text to target length.

        Args:
            text: Text to summarize
            target_tokens: Target summary length (default: max_summary_tokens)
            focus: Optional focus area

        Returns:
            Summary text
        """
        if target_tokens is None:
            target_tokens = self.max_summary_tokens

        # Build prompt
        prompt = f"Summarize the following text in approximately {target_tokens} tokens"

        if focus:
            prompt += f", focusing on {focus}"

        prompt += f":\n\n{text}\n\nSummary:"

        # Generate summary
        response = self.client.generate(
            input_text=prompt,
            temperature=0.3,  # Low temp for consistency
            max_tokens=target_tokens * 2  # Allow some buffer
        )

        summary = self.client.get_output_text(response)
        return summary.strip()

    def summarize_conversation(
        self,
        messages: List[Dict[str, str]],
        preserve_last_n: int = 3
    ) -> List[Dict[str, str]]:
        """
        Summarize conversation history, preserving recent messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            preserve_last_n: Number of recent messages to keep as-is

        Returns:
            New message list with summarized history
        """
        if len(messages) <= preserve_last_n + 1:
            return messages  # No need to summarize

        # Split into old (to summarize) and recent (to preserve)
        old_messages = messages[:-preserve_last_n]
        recent_messages = messages[-preserve_last_n:]

        # Build text to summarize
        history_text = "\n\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in old_messages
        ])

        # Summarize
        summary = self.summarize_text(
            history_text,
            focus="key points and decisions"
        )

        # Create new message list
        summarized = [
            {"role": "system", "content": f"Previous conversation summary:\n{summary}"}
        ]
        summarized.extend(recent_messages)

        return summarized

    def progressive_summarize(
        self,
        text: str,
        chunk_size: int = 2000,
        final_target: int = 500
    ) -> str:
        """
        Progressively summarize long text (summary of summaries).

        For very long texts:
        1. Split into chunks
        2. Summarize each chunk
        3. Summarize the summaries

        Args:
            text: Very long text
            chunk_size: Tokens per chunk
            final_target: Final summary length

        Returns:
            Final summary
        """
        # Check if text is short enough
        text_tokens = count_tokens(text)
        if text_tokens <= final_target * 2:
            return self.summarize_text(text, final_target)

        # Split into chunks
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = self.summarize_text(
                chunk,
                target_tokens=chunk_size // 4  # Compress 4:1
            )
            chunk_summaries.append(summary)

        # Combine summaries
        combined = "\n\n".join(chunk_summaries)

        # Final summarization
        final_summary = self.summarize_text(combined, final_target)

        return final_summary


# Example usage
if __name__ == "__main__":
    from llm.client import LLMClient

    print("Context Management Examples")
    print("=" * 60)

    # Example 1: Context Router
    print("\nExample 1: Context Selection")
    print("-" * 60)

    router = ContextRouter(max_tokens=200)

    query = "machine learning performance optimization"
    snippets = [
        "Machine learning models can be optimized using various techniques.",
        "The weather is nice today with clear skies.",
        "Performance tuning involves profiling and benchmarking.",
        "Deep learning frameworks like PyTorch and TensorFlow.",
        "Database indexing improves query performance."
    ]

    selected = router.select_context(query, snippets, method="tfidf")

    print(f"Query: {query}")
    print(f"\nSelected {len(selected)} snippets:")
    for idx, snippet, score in selected:
        print(f"  [{idx}] Score: {score:.3f}")
        print(f"      {snippet[:60]}...")

    # Example 2: Summarizer
    print("\n\nExample 2: Conversation Summarization")
    print("-" * 60)

    # This example shows the interface (requires actual LLM client)
    example_code = """
    client = LLMClient()
    summarizer = Summarizer(client, max_summary_tokens=100)

    messages = [
        {"role": "user", "content": "Tell me about Python"},
        {"role": "assistant", "content": "Python is a programming language..."},
        {"role": "user", "content": "What about its performance?"},
        {"role": "assistant", "content": "Python is interpreted..."},
        {"role": "user", "content": "How do I optimize it?"},
        {"role": "assistant", "content": "Use Cython, PyPy..."},
        {"role": "user", "content": "Latest question here"}
    ]

    # Summarize old messages, keep last 2
    summarized = summarizer.summarize_conversation(messages, preserve_last_n=2)

    # Result: Summary + last 2 messages
    print(f"Original: {len(messages)} messages")
    print(f"Summarized: {len(summarized)} messages")
    """
    print(example_code)

    # Example 3: Sliding Window
    print("\n\nExample 3: Sliding Window")
    print("-" * 60)

    long_text = "word " * 1000  # Simulate long text

    windows = router.sliding_window(long_text, window_size=100, overlap=20)

    print(f"Split {len(long_text.split())} words into {len(windows)} windows")
    print(f"Window 1: {len(windows[0].split())} words")
    print(f"Window 2: {len(windows[1].split())} words")
