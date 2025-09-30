"""
Module 10: RAG Basics - Solutions

Complete solutions for RAG (Retrieval-Augmented Generation) exercises.

Author: Claude
Date: 2024
"""

import os
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import re
import asyncio
from collections import defaultdict, Counter
import time
import json

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI()

# ================================
# Solution 1: Simple Vector Store
# ================================
print("=" * 50)
print("Solution 1: Simple Vector Store")
print("=" * 50)

class SimpleVectorStore:
    """In-memory vector store implementation."""

    def __init__(self):
        """Initialize the vector store."""
        self.documents = {}
        self.embeddings = {}
        self.metadata = {}
        self.id_counter = 0

    def _generate_id(self) -> str:
        """Generate unique document ID."""
        self.id_counter += 1
        return f"doc_{self.id_counter}"

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)

    def add(self, document: str, metadata: Dict = None) -> str:
        """
        Add a document to the store.

        Args:
            document: Text to store
            metadata: Optional metadata

        Returns:
            Document ID
        """
        # Check for duplicates
        doc_hash = hashlib.md5(document.encode()).hexdigest()
        for doc_id, doc in self.documents.items():
            if hashlib.md5(doc.encode()).hexdigest() == doc_hash:
                print(f"Document already exists with ID: {doc_id}")
                return doc_id

        # Generate ID and embedding
        doc_id = self._generate_id()
        embedding = self._get_embedding(document)

        # Store
        self.documents[doc_id] = document
        self.embeddings[doc_id] = embedding
        self.metadata[doc_id] = metadata or {}

        return doc_id

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of results with documents and scores
        """
        if not self.documents:
            return []

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Calculate similarities
        similarities = []
        for doc_id, doc_embedding in self.embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append({
                "doc_id": doc_id,
                "document": self.documents[doc_id],
                "metadata": self.metadata[doc_id],
                "score": float(similarity)
            })

        # Sort and return top-k
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:k]

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document by ID.

        Args:
            doc_id: Document ID to delete

        Returns:
            Success status
        """
        if doc_id in self.documents:
            del self.documents[doc_id]
            del self.embeddings[doc_id]
            del self.metadata[doc_id]
            return True
        return False

    def update_metadata(self, doc_id: str, metadata: Dict) -> bool:
        """
        Update document metadata.

        Args:
            doc_id: Document ID
            metadata: New metadata

        Returns:
            Success status
        """
        if doc_id in self.documents:
            self.metadata[doc_id].update(metadata)
            return True
        return False

# Test implementation
store = SimpleVectorStore()

# Add documents
doc1_id = store.add("Python is a versatile programming language.", {"type": "programming"})
doc2_id = store.add("Machine learning uses algorithms to learn from data.", {"type": "ml"})
doc3_id = store.add("Natural language processing handles human language.", {"type": "nlp"})

print(f"Added documents: {doc1_id}, {doc2_id}, {doc3_id}")

# Search
results = store.search("What is Python used for?", k=2)
print(f"\nSearch results for 'What is Python used for?':")
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result['score']:.3f} - {result['document'][:50]}...")

# Update metadata
store.update_metadata(doc1_id, {"language": "Python", "updated": True})
print(f"\nUpdated metadata for {doc1_id}")

# Delete document
store.delete(doc3_id)
print(f"Deleted {doc3_id}")

# ================================
# Solution 2: Document Chunking
# ================================
print("\n" + "=" * 50)
print("Solution 2: Document Chunking")
print("=" * 50)

class DocumentChunker:
    """Document chunking utilities."""

    def fixed_size_chunks(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[Dict]:
        """
        Create fixed-size chunks with overlap.

        Args:
            text: Document text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List of chunks with metadata
        """
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            # Skip very small chunks at the end
            if len(chunk_text) < 50 and start > 0:
                # Extend the previous chunk instead
                if chunks:
                    chunks[-1]["content"] += chunk_text
                    chunks[-1]["end_pos"] = end
                break

            chunks.append({
                "content": chunk_text,
                "chunk_index": len(chunks),
                "start_pos": start,
                "end_pos": end,
                "chunk_size": len(chunk_text)
            })

            start += chunk_size - overlap

        return chunks

    def sentence_chunks(
        self,
        text: str,
        sentences_per_chunk: int = 3,
        overlap_sentences: int = 1
    ) -> List[Dict]:
        """
        Create sentence-based chunks.

        Args:
            text: Document text
            sentences_per_chunk: Sentences per chunk
            overlap_sentences: Overlapping sentences

        Returns:
            List of chunks with metadata
        """
        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        chunks = []
        i = 0

        while i < len(sentences):
            # Get chunk sentences
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk_text = ' '.join(chunk_sentences)

            chunks.append({
                "content": chunk_text,
                "chunk_index": len(chunks),
                "start_sentence": i,
                "end_sentence": min(i + sentences_per_chunk, len(sentences)),
                "num_sentences": len(chunk_sentences)
            })

            # Move forward with overlap
            i += sentences_per_chunk - overlap_sentences

        return chunks

    def paragraph_chunks(
        self,
        text: str,
        min_size: int = 100,
        max_size: int = 500
    ) -> List[Dict]:
        """
        Create paragraph-based chunks.

        Args:
            text: Document text
            min_size: Minimum chunk size
            max_size: Maximum chunk size

        Returns:
            List of chunks with metadata
        """
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk = ""
        current_paragraphs = []

        for para in paragraphs:
            # If adding this paragraph exceeds max_size, save current chunk
            if current_chunk and len(current_chunk) + len(para) + 2 > max_size:
                chunks.append({
                    "content": current_chunk,
                    "chunk_index": len(chunks),
                    "num_paragraphs": len(current_paragraphs),
                    "paragraphs": current_paragraphs.copy()
                })
                current_chunk = para
                current_paragraphs = [para]
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_paragraphs.append(para)

                # Check if we should continue merging
                if len(current_chunk) >= min_size:
                    # Look ahead to see if next paragraph would exceed max
                    next_idx = paragraphs.index(para) + 1
                    if next_idx >= len(paragraphs) or \
                       len(current_chunk) + len(paragraphs[next_idx]) + 2 > max_size:
                        chunks.append({
                            "content": current_chunk,
                            "chunk_index": len(chunks),
                            "num_paragraphs": len(current_paragraphs),
                            "paragraphs": current_paragraphs.copy()
                        })
                        current_chunk = ""
                        current_paragraphs = []

        # Add remaining content
        if current_chunk:
            chunks.append({
                "content": current_chunk,
                "chunk_index": len(chunks),
                "num_paragraphs": len(current_paragraphs),
                "paragraphs": current_paragraphs
            })

        return chunks

    def semantic_chunks(
        self,
        text: str,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Create semantic chunks based on meaning.

        Args:
            text: Document text
            similarity_threshold: Threshold for splitting

        Returns:
            List of chunks with metadata
        """
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        # Get embeddings for all sentences
        embeddings = []
        for sentence in sentences:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=sentence
            )
            embeddings.append(np.array(response.data[0].embedding))

        # Group sentences by semantic similarity
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]

        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            similarity = np.dot(current_embedding, embeddings[i]) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(embeddings[i])
            )

            if similarity >= similarity_threshold:
                # Add to current chunk
                current_chunk.append(sentences[i])
                # Update chunk embedding (average)
                chunk_embeddings = embeddings[i - len(current_chunk) + 1:i + 1]
                current_embedding = np.mean(chunk_embeddings, axis=0)
            else:
                # Save current chunk and start new one
                chunks.append({
                    "content": ' '.join(current_chunk),
                    "chunk_index": len(chunks),
                    "num_sentences": len(current_chunk),
                    "avg_similarity": similarity
                })
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]

        # Add last chunk
        if current_chunk:
            chunks.append({
                "content": ' '.join(current_chunk),
                "chunk_index": len(chunks),
                "num_sentences": len(current_chunk),
                "avg_similarity": 1.0
            })

        return chunks

# Test chunking
chunker = DocumentChunker()

sample_text = """
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.
It focuses on developing computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction.
The goal is to allow computers to learn automatically without human intervention or assistance and adjust actions accordingly.

Deep learning is a specialized form of machine learning.
It uses neural networks with multiple layers to progressively extract higher-level features from raw input.
"""

# Test different chunking strategies
print("Fixed-size chunks:")
fixed_chunks = chunker.fixed_size_chunks(sample_text, chunk_size=150, overlap=30)
for chunk in fixed_chunks:
    print(f"  Chunk {chunk['chunk_index']}: {len(chunk['content'])} chars")

print("\nSentence-based chunks:")
sentence_chunks = chunker.sentence_chunks(sample_text, sentences_per_chunk=2)
for chunk in sentence_chunks:
    print(f"  Chunk {chunk['chunk_index']}: {chunk['num_sentences']} sentences")

print("\nParagraph-based chunks:")
para_chunks = chunker.paragraph_chunks(sample_text, min_size=100, max_size=300)
for chunk in para_chunks:
    print(f"  Chunk {chunk['chunk_index']}: {chunk['num_paragraphs']} paragraphs")

# ================================
# Solution 3: Hybrid Search System
# ================================
print("\n" + "=" * 50)
print("Solution 3: Hybrid Search System")
print("=" * 50)

class HybridSearch:
    """Hybrid search combining keyword and semantic search."""

    def __init__(self):
        """Initialize the hybrid search system."""
        self.documents = {}
        self.embeddings = {}
        self.inverted_index = defaultdict(set)
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.total_docs = 0

    def index_document(self, doc_id: str, content: str):
        """
        Index a document for both search methods.

        Args:
            doc_id: Document identifier
            content: Document content
        """
        # Store document
        self.documents[doc_id] = content

        # Generate and store embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=content
        )
        self.embeddings[doc_id] = np.array(response.data[0].embedding)

        # Build inverted index for BM25
        words = content.lower().split()
        self.doc_lengths[doc_id] = len(words)

        for word in words:
            self.inverted_index[word].add(doc_id)

        # Update statistics
        self.total_docs += 1
        self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def bm25_score(
        self,
        query: str,
        doc_id: str,
        k1: float = 1.2,
        b: float = 0.75
    ) -> float:
        """
        Calculate BM25 score.

        Args:
            query: Search query
            doc_id: Document ID
            k1: BM25 parameter
            b: BM25 parameter

        Returns:
            BM25 score
        """
        if doc_id not in self.documents:
            return 0.0

        query_words = query.lower().split()
        doc_words = self.documents[doc_id].lower().split()
        doc_length = self.doc_lengths[doc_id]

        score = 0.0
        for word in query_words:
            # Term frequency in document
            tf = doc_words.count(word)

            # Document frequency
            df = len(self.inverted_index.get(word, set()))

            # IDF calculation
            idf = np.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

            # BM25 formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length))

            score += idf * (numerator / denominator)

        return score

    def semantic_score(self, query: str, doc_id: str) -> float:
        """
        Calculate semantic similarity score.

        Args:
            query: Search query
            doc_id: Document ID

        Returns:
            Similarity score
        """
        if doc_id not in self.embeddings:
            return 0.0

        # Get query embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = np.array(response.data[0].embedding)

        # Calculate cosine similarity
        doc_embedding = self.embeddings[doc_id]
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )

        return float(similarity)

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> List[Dict]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            k: Number of results
            keyword_weight: Weight for keyword search
            semantic_weight: Weight for semantic search

        Returns:
            Ranked results
        """
        scores = {}

        # Calculate scores for all documents
        for doc_id in self.documents:
            bm25 = self.bm25_score(query, doc_id)
            semantic = self.semantic_score(query, doc_id)

            # Normalize scores (min-max normalization would be better with more docs)
            scores[doc_id] = {
                "bm25_score": bm25,
                "semantic_score": semantic,
                "hybrid_score": keyword_weight * bm25 + semantic_weight * semantic,
                "document": self.documents[doc_id]
            }

        # Sort by hybrid score
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1]["hybrid_score"],
            reverse=True
        )

        # Return top-k
        return [
            {
                "doc_id": doc_id,
                **result
            }
            for doc_id, result in sorted_results[:k]
        ]

    def reciprocal_rank_fusion(
        self,
        keyword_results: List[str],
        semantic_results: List[str],
        k: int = 60
    ) -> List[str]:
        """
        Combine rankings using Reciprocal Rank Fusion.

        Args:
            keyword_results: Ranked list from keyword search
            semantic_results: Ranked list from semantic search
            k: RRF parameter

        Returns:
            Fused ranking
        """
        rrf_scores = defaultdict(float)

        # Add scores from keyword results
        for rank, doc_id in enumerate(keyword_results, 1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

        # Add scores from semantic results
        for rank, doc_id in enumerate(semantic_results, 1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc_id for doc_id, _ in sorted_docs]

# Test hybrid search
hybrid = HybridSearch()

# Index documents
documents = {
    "doc1": "Python programming language is known for its simplicity and readability.",
    "doc2": "Machine learning models require large amounts of training data.",
    "doc3": "Python is widely used in data science and machine learning projects."
}

for doc_id, content in documents.items():
    hybrid.index_document(doc_id, content)

# Search with different weight combinations
print("Searching for: 'Python for data science'")
for kw_weight in [0.0, 0.3, 0.5, 0.7, 1.0]:
    sem_weight = 1.0 - kw_weight
    results = hybrid.hybrid_search(
        "Python for data science",
        k=3,
        keyword_weight=kw_weight,
        semantic_weight=sem_weight
    )
    print(f"\nWeights - Keyword: {kw_weight}, Semantic: {sem_weight}")
    for result in results[:2]:
        print(f"  {result['doc_id']}: Score={result['hybrid_score']:.3f}")

# ================================
# Solution 4: Query Expansion
# ================================
print("\n" + "=" * 50)
print("Solution 4: Query Expansion")
print("=" * 50)

class QueryExpander:
    """Query expansion for improved retrieval."""

    def expand_with_synonyms(self, query: str, n_synonyms: int = 3) -> List[str]:
        """
        Expand query with synonyms.

        Args:
            query: Original query
            n_synonyms: Number of synonyms per term

        Returns:
            Expanded queries
        """
        # Use LLM to generate synonyms
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Generate {n_synonyms} synonyms for each important word. Return as comma-separated list."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.5
        )

        synonyms = response.choices[0].message.content.strip().split(',')
        synonyms = [s.strip() for s in synonyms]

        # Create expanded queries
        expanded = [query]  # Original query
        words = query.split()

        # Add queries with synonyms
        for synonym in synonyms[:n_synonyms]:
            if synonym and synonym not in query:
                expanded.append(f"{query} {synonym}")

        return expanded

    def rewrite_query(self, query: str, context: str = None) -> str:
        """
        Rewrite query using LLM.

        Args:
            query: Original query
            context: Optional context

        Returns:
            Rewritten query
        """
        prompt = "Rewrite this search query to be more effective. Keep it concise."
        if context:
            prompt += f"\nContext: {context}"

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    def pseudo_relevance_feedback(
        self,
        query: str,
        initial_results: List[str],
        n_terms: int = 5
    ) -> str:
        """
        Expand query using initial results.

        Args:
            query: Original query
            initial_results: Top initial results
            n_terms: Number of terms to add

        Returns:
            Expanded query
        """
        if not initial_results:
            return query

        # Extract important terms from initial results
        all_words = []
        for result in initial_results[:3]:  # Use top 3 results
            words = result.lower().split()
            all_words.extend(words)

        # Count word frequencies
        word_freq = Counter(all_words)

        # Remove common words and query terms
        common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'])
        query_words = set(query.lower().split())

        # Get top terms
        top_terms = []
        for word, freq in word_freq.most_common():
            if word not in common_words and word not in query_words and len(word) > 2:
                top_terms.append(word)
                if len(top_terms) >= n_terms:
                    break

        # Expand query
        expanded_query = query
        if top_terms:
            expanded_query += " " + " ".join(top_terms)

        return expanded_query

    def generate_multi_queries(
        self,
        query: str,
        n_variations: int = 3
    ) -> List[str]:
        """
        Generate multiple query variations.

        Args:
            query: Original query
            n_variations: Number of variations

        Returns:
            Query variations
        """
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Generate {n_variations} different ways to search for this information. Each on a new line."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.7
        )

        variations = response.choices[0].message.content.strip().split('\n')
        variations = [v.strip('- ').strip('1234567890. ').strip() for v in variations]

        # Include original query
        all_queries = [query] + variations[:n_variations-1]

        return all_queries

    def combine_expansions(
        self,
        original: str,
        expansions: List[str]
    ) -> str:
        """
        Combine original query with expansions.

        Args:
            original: Original query
            expansions: Expansion terms/queries

        Returns:
            Combined query
        """
        # Remove duplicates while preserving order
        seen = set()
        combined_parts = [original]

        for expansion in expansions:
            # Extract unique terms from expansion
            expansion_words = expansion.split()
            unique_words = []

            for word in expansion_words:
                word_lower = word.lower()
                if word_lower not in seen and word_lower not in original.lower():
                    unique_words.append(word)
                    seen.add(word_lower)

            if unique_words:
                combined_parts.append(' '.join(unique_words))

        # Combine all parts
        combined = ' '.join(combined_parts)

        # Limit length
        max_length = 100
        if len(combined) > max_length:
            combined = combined[:max_length].rsplit(' ', 1)[0]

        return combined

# Test query expansion
expander = QueryExpander()

query = "machine learning algorithms"

print(f"Original query: {query}")

# Test synonym expansion
synonyms = expander.expand_with_synonyms(query, n_synonyms=2)
print(f"\nSynonym expansion:")
for syn in synonyms:
    print(f"  - {syn}")

# Test query rewriting
rewritten = expander.rewrite_query(query, context="Focus on neural networks")
print(f"\nRewritten query: {rewritten}")

# Test pseudo-relevance feedback
initial_results = [
    "Deep learning uses neural network architectures",
    "Supervised learning requires labeled training data",
    "Reinforcement learning optimizes through trial and error"
]
expanded = expander.pseudo_relevance_feedback(query, initial_results, n_terms=3)
print(f"\nPseudo-relevance feedback: {expanded}")

# Test multi-query generation
variations = expander.generate_multi_queries(query, n_variations=3)
print(f"\nQuery variations:")
for var in variations:
    print(f"  - {var}")

# ================================
# Solution 5: RAG Evaluation System
# ================================
print("\n" + "=" * 50)
print("Solution 5: RAG Evaluation System")
print("=" * 50)

@dataclass
class RAGMetrics:
    """Metrics for RAG evaluation."""
    relevance_score: float
    answer_quality: float
    faithfulness: float
    latency_ms: float
    tokens_used: int
    sources_used: int

class RAGEvaluator:
    """Evaluation framework for RAG systems."""

    def evaluate_relevance(
        self,
        query: str,
        documents: List[str]
    ) -> float:
        """
        Evaluate relevance of retrieved documents.

        Args:
            query: Search query
            documents: Retrieved documents

        Returns:
            Relevance score (0-1)
        """
        if not documents:
            return 0.0

        # Use LLM to score relevance
        scores = []
        for doc in documents:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Rate the relevance of the document to the query from 0 to 10. Respond with just a number."
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nDocument: {doc[:500]}"
                    }
                ],
                temperature=0
            )

            try:
                score = float(response.choices[0].message.content.strip())
                scores.append(score / 10.0)
            except:
                scores.append(0.5)

        # Return average relevance
        return sum(scores) / len(scores)

    def evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        ground_truth: str = None
    ) -> float:
        """
        Evaluate quality of generated answer.

        Args:
            question: Original question
            answer: Generated answer
            ground_truth: Optional correct answer

        Returns:
            Quality score (0-1)
        """
        # Create evaluation prompt
        if ground_truth:
            prompt = f"""
            Evaluate the answer quality on a scale of 0-10.
            Question: {question}
            Generated Answer: {answer}
            Reference Answer: {ground_truth}
            Consider accuracy, completeness, and clarity.
            Respond with just a number.
            """
        else:
            prompt = f"""
            Evaluate the answer quality on a scale of 0-10.
            Question: {question}
            Answer: {answer}
            Consider completeness, clarity, and apparent accuracy.
            Respond with just a number.
            """

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        try:
            score = float(response.choices[0].message.content.strip())
            return score / 10.0
        except:
            return 0.5

    def evaluate_faithfulness(
        self,
        answer: str,
        source_documents: List[str]
    ) -> float:
        """
        Check if answer is faithful to sources.

        Args:
            answer: Generated answer
            source_documents: Source documents

        Returns:
            Faithfulness score (0-1)
        """
        if not source_documents:
            return 0.0

        # Combine sources
        sources = "\n\n".join(source_documents)

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Rate from 0-10 how well the answer is supported by the sources. Consider if claims are grounded in the provided documents. Respond with just a number."
                },
                {
                    "role": "user",
                    "content": f"Sources:\n{sources[:1500]}\n\nAnswer:\n{answer}"
                }
            ],
            temperature=0
        )

        try:
            score = float(response.choices[0].message.content.strip())
            return score / 10.0
        except:
            return 0.5

    def evaluate_rag_pipeline(
        self,
        rag_function,
        test_queries: List[Dict]
    ) -> Dict:
        """
        Evaluate complete RAG pipeline.

        Args:
            rag_function: RAG query function
            test_queries: Test queries with ground truth

        Returns:
            Evaluation report
        """
        all_metrics = []

        for test in test_queries:
            start_time = time.time()

            # Run RAG query
            result = rag_function(test["question"])

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Evaluate components
            relevance = self.evaluate_relevance(
                test["question"],
                result.get("sources", [])
            )

            quality = self.evaluate_answer_quality(
                test["question"],
                result["answer"],
                test.get("ground_truth")
            )

            faithfulness = self.evaluate_faithfulness(
                result["answer"],
                result.get("sources", [])
            )

            # Count tokens (approximate)
            tokens = len(result["answer"].split()) * 1.3

            metrics = RAGMetrics(
                relevance_score=relevance,
                answer_quality=quality,
                faithfulness=faithfulness,
                latency_ms=latency_ms,
                tokens_used=int(tokens),
                sources_used=len(result.get("sources", []))
            )

            all_metrics.append(metrics)

        # Generate summary statistics
        report = {
            "num_queries": len(test_queries),
            "avg_relevance": sum(m.relevance_score for m in all_metrics) / len(all_metrics),
            "avg_quality": sum(m.answer_quality for m in all_metrics) / len(all_metrics),
            "avg_faithfulness": sum(m.faithfulness for m in all_metrics) / len(all_metrics),
            "avg_latency_ms": sum(m.latency_ms for m in all_metrics) / len(all_metrics),
            "total_tokens": sum(m.tokens_used for m in all_metrics),
            "avg_sources": sum(m.sources_used for m in all_metrics) / len(all_metrics),
            "detailed_metrics": all_metrics
        }

        return report

    def generate_report(
        self,
        metrics: List[RAGMetrics]
    ) -> str:
        """
        Generate evaluation report.

        Args:
            metrics: List of metrics

        Returns:
            Formatted report
        """
        if not metrics:
            return "No metrics to report"

        # Calculate aggregates
        num_queries = len(metrics)
        avg_relevance = sum(m.relevance_score for m in metrics) / num_queries
        avg_quality = sum(m.answer_quality for m in metrics) / num_queries
        avg_faithfulness = sum(m.faithfulness for m in metrics) / num_queries
        avg_latency = sum(m.latency_ms for m in metrics) / num_queries
        total_tokens = sum(m.tokens_used for m in metrics)

        report = f"""
RAG Evaluation Report
====================

Summary Statistics:
------------------
Total Queries: {num_queries}
Average Relevance Score: {avg_relevance:.3f}
Average Answer Quality: {avg_quality:.3f}
Average Faithfulness: {avg_faithfulness:.3f}
Average Latency: {avg_latency:.1f}ms
Total Tokens Used: {total_tokens:,}

Performance Distribution:
------------------------
Excellent (>0.8): {sum(1 for m in metrics if m.answer_quality > 0.8)} queries
Good (0.6-0.8): {sum(1 for m in metrics if 0.6 <= m.answer_quality <= 0.8)} queries
Poor (<0.6): {sum(1 for m in metrics if m.answer_quality < 0.6)} queries

Recommendations:
---------------
"""

        # Add recommendations based on metrics
        if avg_relevance < 0.7:
            report += "- Improve retrieval system (low relevance scores)\n"
        if avg_faithfulness < 0.7:
            report += "- Enhance answer grounding in source documents\n"
        if avg_latency > 2000:
            report += "- Optimize for latency (consider caching)\n"
        if avg_quality < 0.7:
            report += "- Improve answer generation quality\n"

        return report

# Test evaluation
evaluator = RAGEvaluator()

# Mock RAG function for testing
def mock_rag_function(question: str) -> Dict:
    """Mock RAG function for testing."""
    return {
        "answer": f"This is a mock answer to: {question}",
        "sources": [
            "Source document 1 with relevant information.",
            "Source document 2 with supporting details."
        ]
    }

# Test individual evaluations
test_query = "What is machine learning?"
test_docs = [
    "Machine learning is a type of AI that learns from data.",
    "ML algorithms improve through experience."
]
test_answer = "Machine learning is a branch of AI that enables systems to learn from data."

relevance = evaluator.evaluate_relevance(test_query, test_docs)
print(f"Relevance score: {relevance:.3f}")

quality = evaluator.evaluate_answer_quality(test_query, test_answer)
print(f"Answer quality: {quality:.3f}")

faithfulness = evaluator.evaluate_faithfulness(test_answer, test_docs)
print(f"Faithfulness score: {faithfulness:.3f}")

# Generate sample metrics
sample_metrics = [
    RAGMetrics(0.8, 0.85, 0.9, 1500, 150, 3),
    RAGMetrics(0.7, 0.75, 0.8, 1800, 180, 4),
    RAGMetrics(0.9, 0.88, 0.95, 1200, 120, 2)
]

report = evaluator.generate_report(sample_metrics)
print(report)

# ================================
# Challenge Solution: Production RAG Service
# ================================
print("\n" + "=" * 50)
print("Challenge Solution: Production RAG Service")
print("=" * 50)

class ProductionRAGService:
    """Production-ready RAG service."""

    def __init__(self, config: Dict):
        """
        Initialize the service.

        Args:
            config: Service configuration
        """
        self.config = config
        self.documents = {}
        self.embeddings = {}
        self.metadata = {}
        self.cache = {}
        self.metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "total_documents": 0,
            "errors": 0,
            "avg_latency": 0
        }
        self.rate_limiter = defaultdict(list)
        self.health_status = {"status": "healthy", "checks": {}}

    def _validate_input(self, content: str) -> bool:
        """Validate input for security."""
        # Check length
        if len(content) > 10000:
            raise ValueError("Content too long")

        # Check for potential injection attempts
        dangerous_patterns = ['<script', 'javascript:', 'onclick=']
        content_lower = content.lower()
        for pattern in dangerous_patterns:
            if pattern in content_lower:
                raise ValueError(f"Potentially dangerous content detected: {pattern}")

        return True

    def _check_rate_limit(self, client_id: str = "default") -> bool:
        """Check rate limiting."""
        current_time = time.time()
        window = 60  # 1 minute window

        # Clean old entries
        self.rate_limiter[client_id] = [
            t for t in self.rate_limiter[client_id]
            if current_time - t < window
        ]

        # Check limit
        if len(self.rate_limiter[client_id]) >= self.config.get("rate_limit", 100):
            return False

        # Add current request
        self.rate_limiter[client_id].append(current_time)
        return True

    async def add_document(
        self,
        content: str,
        metadata: Dict,
        doc_id: str = None
    ) -> str:
        """
        Add document to the service.

        Args:
            content: Document content
            metadata: Document metadata
            doc_id: Optional document ID

        Returns:
            Document ID
        """
        # Validate input
        self._validate_input(content)

        # Generate ID if not provided
        if not doc_id:
            doc_id = f"doc_{hashlib.md5(content.encode()).hexdigest()[:8]}"

        # Check if document already exists
        if doc_id in self.documents:
            return doc_id

        # Generate embedding asynchronously
        await asyncio.sleep(0)  # Simulate async operation

        response = client.embeddings.create(
            model=self.config.get("embedding_model", "text-embedding-3-small"),
            input=content
        )
        embedding = response.data[0].embedding

        # Store document
        self.documents[doc_id] = content
        self.embeddings[doc_id] = np.array(embedding)
        self.metadata[doc_id] = metadata

        # Update metrics
        self.metrics["total_documents"] += 1

        return doc_id

    async def query(
        self,
        question: str,
        filters: Dict = None,
        options: Dict = None
    ) -> Dict:
        """
        Process a RAG query.

        Args:
            question: User question
            filters: Metadata filters
            options: Query options

        Returns:
            Query result with answer and metadata
        """
        start_time = time.time()

        try:
            # Rate limiting
            if not self._check_rate_limit():
                raise ValueError("Rate limit exceeded")

            # Validate input
            self._validate_input(question)

            # Update metrics
            self.metrics["total_queries"] += 1

            # Check cache
            cache_key = f"{question}_{str(filters)}_{str(options)}"
            if cache_key in self.cache:
                self.metrics["cache_hits"] += 1
                cached_result = self.cache[cache_key].copy()
                cached_result["from_cache"] = True
                return cached_result

            # Retrieve relevant documents
            await asyncio.sleep(0)  # Simulate async operation

            response = client.embeddings.create(
                model=self.config.get("embedding_model", "text-embedding-3-small"),
                input=question
            )
            query_embedding = np.array(response.data[0].embedding)

            # Find similar documents
            similarities = []
            for doc_id, doc_embedding in self.embeddings.items():
                # Apply filters
                if filters:
                    doc_meta = self.metadata.get(doc_id, {})
                    skip = False
                    for key, value in filters.items():
                        if doc_meta.get(key) != value:
                            skip = True
                            break
                    if skip:
                        continue

                # Calculate similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((doc_id, similarity))

            # Sort and get top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            k = options.get("top_k", 3) if options else 3
            top_docs = similarities[:k]

            # Generate answer
            context = "\n\n".join([
                self.documents[doc_id]
                for doc_id, _ in top_docs
            ])

            response = client.chat.completions.create(
                model=self.config.get("llm_model", "gpt-5"),
                messages=[
                    {
                        "role": "system",
                        "content": "Answer the question based on the provided context."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ],
                max_tokens=options.get("max_tokens", 500) if options else 500
            )

            answer = response.choices[0].message.content

            # Create result
            result = {
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "doc_id": doc_id,
                        "score": score,
                        "content": self.documents[doc_id][:200] + "..."
                    }
                    for doc_id, score in top_docs
                ],
                "metadata": {
                    "latency_ms": (time.time() - start_time) * 1000,
                    "model": self.config.get("llm_model"),
                    "timestamp": datetime.now().isoformat()
                },
                "from_cache": False
            }

            # Cache result
            self.cache[cache_key] = result

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.metrics["avg_latency"] = (
                (self.metrics["avg_latency"] * (self.metrics["total_queries"] - 1) + latency) /
                self.metrics["total_queries"]
            )

            return result

        except Exception as e:
            self.metrics["errors"] += 1
            self.health_status["checks"]["last_error"] = str(e)
            raise e

    def get_metrics(self) -> Dict:
        """
        Get service metrics.

        Returns:
            Service metrics
        """
        cache_size = len(self.cache)
        cache_hit_rate = (
            self.metrics["cache_hits"] / self.metrics["total_queries"]
            if self.metrics["total_queries"] > 0 else 0
        )

        return {
            **self.metrics,
            "cache_size": cache_size,
            "cache_hit_rate": cache_hit_rate,
            "document_count": len(self.documents)
        }

    def health_check(self) -> Dict:
        """
        Perform health check.

        Returns:
            Health status
        """
        checks = {}

        # Check document store
        checks["document_store"] = len(self.documents) < self.config.get("max_documents", 10000)

        # Check cache size
        checks["cache"] = len(self.cache) < 1000

        # Check error rate
        error_rate = (
            self.metrics["errors"] / self.metrics["total_queries"]
            if self.metrics["total_queries"] > 0 else 0
        )
        checks["error_rate"] = error_rate < 0.1

        # Overall status
        all_healthy = all(checks.values())

        return {
            "status": "healthy" if all_healthy else "degraded",
            "checks": checks,
            "metrics": self.get_metrics(),
            "timestamp": datetime.now().isoformat()
        }

# Test production service
async def test_production_service():
    """Test the production RAG service."""
    config = {
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-5",
        "vector_store": "in_memory",
        "cache_ttl": 3600,
        "max_documents": 10000,
        "rate_limit": 100,
    }

    service = ProductionRAGService(config)

    # Add documents
    doc1_id = await service.add_document(
        "RAG combines retrieval and generation for better answers.",
        {"category": "rag", "importance": "high"}
    )
    print(f"Added document: {doc1_id}")

    doc2_id = await service.add_document(
        "Production systems need monitoring, caching, and error handling.",
        {"category": "production", "importance": "critical"}
    )
    print(f"Added document: {doc2_id}")

    # Query the service
    result = await service.query(
        "What makes a RAG system production-ready?",
        filters=None,
        options={"top_k": 2, "max_tokens": 200}
    )

    print(f"\nQuery: {result['question']}")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Latency: {result['metadata']['latency_ms']:.2f}ms")
    print(f"Sources: {len(result['sources'])}")

    # Query again (should hit cache)
    result2 = await service.query(
        "What makes a RAG system production-ready?",
        filters=None,
        options={"top_k": 2, "max_tokens": 200}
    )
    print(f"\nCached: {result2['from_cache']}")

    # Check health
    health = service.health_check()
    print(f"\nHealth Status: {health['status']}")
    print(f"Metrics: {json.dumps(health['metrics'], indent=2)}")

# Run async test
asyncio.run(test_production_service())

print("\n" + "=" * 50)
print("All Solutions Complete!")
print("=" * 50)