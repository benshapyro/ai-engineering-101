"""
Module 10: RAG Basics - Vector Embeddings

Learn the fundamentals of text embeddings for RAG systems.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass
import json


# ===== Example 1: Basic Text Embeddings =====

def example_1_basic_embeddings():
    """Create and compare text embeddings."""
    print("Example 1: Basic Text Embeddings")
    print("=" * 50)

    # Simulated embedding model (in practice, use sentence-transformers or OpenAI)
    class SimpleEmbedder:
        """Simple text embedder for demonstration."""

        def __init__(self, dimension: int = 384):
            self.dimension = dimension
            # Simulate word vectors
            self.word_vectors = {}

        def encode(self, text: str) -> np.ndarray:
            """Encode text to vector (simplified)."""
            # In practice, use real embedding models
            # This is a simplified demonstration
            words = text.lower().split()

            # Create pseudo-embedding based on word features
            embedding = np.zeros(self.dimension)

            for i, word in enumerate(words):
                # Generate consistent pseudo-random values for each word
                np.random.seed(hash(word) % 2**32)
                word_vec = np.random.randn(self.dimension) * 0.1
                embedding += word_vec

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        def encode_batch(self, texts: List[str]) -> np.ndarray:
            """Encode multiple texts."""
            return np.array([self.encode(text) for text in texts])

    # Create embedder
    embedder = SimpleEmbedder(dimension=128)

    # Test texts
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "AI and machine learning are closely related fields",
        "The weather today is sunny and warm",
        "Python is a popular programming language",
        "Deep learning uses neural networks"
    ]

    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = embedder.encode_batch(texts)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Calculate similarity matrix
    print("\n" + "-" * 30)
    print("Similarity Matrix:")
    print("-" * 30)

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarity_matrix = np.zeros((len(texts), len(texts)))
    for i in range(len(texts)):
        for j in range(len(texts)):
            similarity_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])

    # Display similarity matrix
    print("\n     ", end="")
    for i in range(len(texts)):
        print(f"  T{i+1}  ", end="")
    print()

    for i, text in enumerate(texts):
        print(f"T{i+1}: ", end="")
        for j in range(len(texts)):
            sim = similarity_matrix[i, j]
            print(f" {sim:.2f} ", end="")
        print(f"  # {text[:30]}...")

    # Find most similar pairs
    print("\n" + "-" * 30)
    print("Most Similar Text Pairs:")
    print("-" * 30)

    similarities = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarities.append((i, j, similarity_matrix[i, j]))

    similarities.sort(key=lambda x: x[2], reverse=True)

    for i, j, sim in similarities[:3]:
        print(f"\nSimilarity: {sim:.3f}")
        print(f"  Text 1: {texts[i][:50]}")
        print(f"  Text 2: {texts[j][:50]}")


# ===== Example 2: Document Chunking Strategies =====

def example_2_document_chunking():
    """Implement different document chunking strategies."""
    print("\nExample 2: Document Chunking Strategies")
    print("=" * 50)

    @dataclass
    class Chunk:
        """Document chunk with metadata."""
        text: str
        start_pos: int
        end_pos: int
        chunk_id: int
        metadata: Dict = None

    class DocumentChunker:
        """Various document chunking strategies."""

        @staticmethod
        def fixed_size_chunking(text: str, chunk_size: int = 500,
                               overlap: int = 50) -> List[Chunk]:
            """Chunk by fixed character size with overlap."""
            chunks = []
            chunk_id = 0

            for i in range(0, len(text), chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]

                # Don't create very small final chunk
                if len(chunk_text) < chunk_size // 4 and chunks:
                    # Merge with previous chunk
                    chunks[-1].text += chunk_text
                    chunks[-1].end_pos = i + len(chunk_text)
                else:
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_pos=i,
                        end_pos=min(i + chunk_size, len(text)),
                        chunk_id=chunk_id,
                        metadata={"method": "fixed_size"}
                    ))
                    chunk_id += 1

            return chunks

        @staticmethod
        def sentence_chunking(text: str, sentences_per_chunk: int = 3,
                            overlap_sentences: int = 1) -> List[Chunk]:
            """Chunk by sentences."""
            # Simple sentence splitting (use NLTK in production)
            sentences = text.replace('!', '.').replace('?', '.').split('.')
            sentences = [s.strip() for s in sentences if s.strip()]

            chunks = []
            chunk_id = 0

            for i in range(0, len(sentences),
                          sentences_per_chunk - overlap_sentences):
                chunk_sentences = sentences[i:i + sentences_per_chunk]
                chunk_text = '. '.join(chunk_sentences) + '.'

                chunks.append(Chunk(
                    text=chunk_text,
                    start_pos=i,
                    end_pos=min(i + sentences_per_chunk, len(sentences)),
                    chunk_id=chunk_id,
                    metadata={"method": "sentence", "sentence_count": len(chunk_sentences)}
                ))
                chunk_id += 1

            return chunks

        @staticmethod
        def semantic_chunking(text: str, embedder,
                            similarity_threshold: float = 0.7) -> List[Chunk]:
            """Chunk based on semantic similarity."""
            # Split into sentences first
            sentences = text.replace('!', '.').replace('?', '.').split('.')
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                return []

            # Embed sentences
            embeddings = embedder.encode_batch(sentences)

            chunks = []
            current_chunk = [sentences[0]]
            current_start = 0
            chunk_id = 0

            for i in range(1, len(sentences)):
                # Calculate similarity with current chunk
                current_embedding = embedder.encode(' '.join(current_chunk))
                similarity = np.dot(current_embedding, embeddings[i]) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(embeddings[i])
                )

                if similarity >= similarity_threshold:
                    # Add to current chunk
                    current_chunk.append(sentences[i])
                else:
                    # Start new chunk
                    chunk_text = '. '.join(current_chunk) + '.'
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_pos=current_start,
                        end_pos=i,
                        chunk_id=chunk_id,
                        metadata={
                            "method": "semantic",
                            "sentence_count": len(current_chunk)
                        }
                    ))

                    current_chunk = [sentences[i]]
                    current_start = i
                    chunk_id += 1

            # Add final chunk
            if current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append(Chunk(
                    text=chunk_text,
                    start_pos=current_start,
                    end_pos=len(sentences),
                    chunk_id=chunk_id,
                    metadata={
                        "method": "semantic",
                        "sentence_count": len(current_chunk)
                    }
                ))

            return chunks

        @staticmethod
        def sliding_window_chunking(text: str, window_size: int = 400,
                                   step_size: int = 200) -> List[Chunk]:
            """Sliding window chunking with configurable overlap."""
            chunks = []
            chunk_id = 0

            for i in range(0, len(text) - window_size + 1, step_size):
                chunk_text = text[i:i + window_size]

                chunks.append(Chunk(
                    text=chunk_text,
                    start_pos=i,
                    end_pos=i + window_size,
                    chunk_id=chunk_id,
                    metadata={
                        "method": "sliding_window",
                        "window_size": window_size,
                        "step_size": step_size
                    }
                ))
                chunk_id += 1

            # Handle remainder
            if len(text) > chunks[-1].end_pos if chunks else 0:
                remaining = text[chunks[-1].end_pos:] if chunks else text
                if len(remaining) > window_size // 4:  # Only if substantial
                    chunks.append(Chunk(
                        text=remaining,
                        start_pos=chunks[-1].end_pos if chunks else 0,
                        end_pos=len(text),
                        chunk_id=chunk_id,
                        metadata={"method": "sliding_window", "is_remainder": True}
                    ))

            return chunks

    # Sample document
    document = """
    Artificial intelligence is transforming the world. Machine learning algorithms
    can now process vast amounts of data. Deep learning has enabled breakthroughs
    in computer vision and natural language processing. Neural networks are becoming
    more sophisticated every year.

    The applications of AI are diverse. In healthcare, AI helps diagnose diseases.
    In finance, it detects fraud. In transportation, it powers autonomous vehicles.
    The impact is felt across all industries.

    However, challenges remain. Ethical considerations are important. Privacy concerns
    must be addressed. The need for explainable AI is growing. Bias in algorithms is
    a serious issue that requires attention.
    """

    # Clean document
    document = ' '.join(document.split())

    # Create chunker
    chunker = DocumentChunker()
    embedder = SimpleEmbedder(dimension=64)

    # Test different chunking strategies
    strategies = [
        ("Fixed Size", lambda: chunker.fixed_size_chunking(document, 150, 30)),
        ("Sentence-based", lambda: chunker.sentence_chunking(document, 2, 0)),
        ("Sliding Window", lambda: chunker.sliding_window_chunking(document, 200, 100)),
        ("Semantic", lambda: chunker.semantic_chunking(document, embedder, 0.6))
    ]

    for strategy_name, chunk_func in strategies:
        print(f"\n{strategy_name} Chunking:")
        print("-" * 30)

        chunks = chunk_func()
        print(f"Number of chunks: {len(chunks)}")

        # Display first few chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"  Text: {chunk.text[:100]}...")
            print(f"  Length: {len(chunk.text)} chars")
            if chunk.metadata:
                print(f"  Metadata: {chunk.metadata}")


# ===== Example 3: Embedding Models Comparison =====

def example_3_embedding_models():
    """Compare different embedding models and techniques."""
    print("\nExample 3: Embedding Models Comparison")
    print("=" * 50)

    class EmbeddingModel:
        """Base class for embedding models."""

        def __init__(self, name: str, dimension: int):
            self.name = name
            self.dimension = dimension

        def encode(self, text: str) -> np.ndarray:
            """Encode text to vector."""
            raise NotImplementedError

    class BagOfWordsEmbedder(EmbeddingModel):
        """Simple bag-of-words embedding."""

        def __init__(self, vocab_size: int = 1000):
            super().__init__("BagOfWords", vocab_size)
            self.vocab = {}

        def encode(self, text: str) -> np.ndarray:
            """Create bag-of-words embedding."""
            words = text.lower().split()
            embedding = np.zeros(self.dimension)

            for word in words:
                if word not in self.vocab:
                    if len(self.vocab) < self.dimension:
                        self.vocab[word] = len(self.vocab)

                if word in self.vocab:
                    embedding[self.vocab[word]] += 1

            # Normalize
            if embedding.sum() > 0:
                embedding = embedding / embedding.sum()

            return embedding

    class TFIDFEmbedder(EmbeddingModel):
        """TF-IDF based embedding."""

        def __init__(self, dimension: int = 100):
            super().__init__("TF-IDF", dimension)
            self.idf_weights = np.ones(dimension)
            self.vocab = {}

        def encode(self, text: str) -> np.ndarray:
            """Create TF-IDF embedding."""
            words = text.lower().split()
            tf = {}

            # Calculate term frequency
            for word in words:
                tf[word] = tf.get(word, 0) + 1

            # Create embedding
            embedding = np.zeros(self.dimension)
            for word, freq in tf.items():
                if word not in self.vocab and len(self.vocab) < self.dimension:
                    self.vocab[word] = len(self.vocab)

                if word in self.vocab:
                    idx = self.vocab[word]
                    # TF * IDF (simplified)
                    embedding[idx] = (freq / len(words)) * self.idf_weights[idx]

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

    class SemanticEmbedder(EmbeddingModel):
        """Semantic embedding using word relationships."""

        def __init__(self, dimension: int = 128):
            super().__init__("Semantic", dimension)
            # Simulate semantic word groups
            self.semantic_groups = {
                "ai": ["artificial", "intelligence", "machine", "learning", "neural", "deep"],
                "data": ["data", "information", "dataset", "database", "storage"],
                "programming": ["code", "programming", "software", "algorithm", "function"],
                "business": ["business", "company", "market", "finance", "economy"]
            }

        def encode(self, text: str) -> np.ndarray:
            """Create semantic embedding."""
            words = set(text.lower().split())
            embedding = np.zeros(self.dimension)

            # Activate dimensions based on semantic groups
            for i, (group_name, group_words) in enumerate(self.semantic_groups.items()):
                overlap = len(words.intersection(group_words))
                if overlap > 0:
                    # Distribute across dimensions for this semantic group
                    group_dims = slice(i * (self.dimension // len(self.semantic_groups)),
                                      (i + 1) * (self.dimension // len(self.semantic_groups)))
                    embedding[group_dims] = overlap / len(group_words)

            # Add some randomness for words not in groups
            np.random.seed(hash(' '.join(sorted(words))) % 2**32)
            noise = np.random.randn(self.dimension) * 0.05
            embedding += noise

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

    # Test texts
    test_texts = [
        "Machine learning and artificial intelligence are transforming technology",
        "Python is a versatile programming language for data science",
        "Business analytics requires understanding of market dynamics",
        "Neural networks enable deep learning applications"
    ]

    # Create models
    models = [
        BagOfWordsEmbedder(vocab_size=50),
        TFIDFEmbedder(dimension=64),
        SemanticEmbedder(dimension=128)
    ]

    # Compare embeddings
    for model in models:
        print(f"\n{model.name} Embeddings:")
        print("-" * 30)

        embeddings = [model.encode(text) for text in test_texts]

        # Calculate inter-text similarities
        print(f"Dimension: {model.dimension}")
        print("\nPairwise Similarities:")

        for i in range(len(test_texts)):
            for j in range(i + 1, len(test_texts)):
                similarity = np.dot(embeddings[i], embeddings[j])
                print(f"  Text {i+1} <-> Text {j+1}: {similarity:.3f}")

        # Measure embedding sparsity
        sparsity = np.mean([np.mean(emb == 0) for emb in embeddings])
        print(f"\nAverage Sparsity: {sparsity:.1%}")


# ===== Example 4: Query vs Document Embeddings =====

def example_4_query_document_embeddings():
    """Optimize embeddings for queries vs documents."""
    print("\nExample 4: Query vs Document Embeddings")
    print("=" * 50)

    class AsymmetricEmbedder:
        """Different encoding for queries and documents."""

        def __init__(self, dimension: int = 128):
            self.dimension = dimension
            self.query_prefix = "[QUERY]"
            self.doc_prefix = "[DOC]"

        def encode_query(self, query: str) -> np.ndarray:
            """Encode a search query."""
            # Add query prefix for different encoding
            prefixed = f"{self.query_prefix} {query}"

            # Simulate query-specific encoding
            # - Emphasize keywords
            # - Ignore stop words
            # - Focus on intent

            keywords = self._extract_keywords(query)
            embedding = np.zeros(self.dimension)

            # Activate dimensions based on keywords
            for i, keyword in enumerate(keywords[:self.dimension // 4]):
                np.random.seed(hash(keyword) % 2**32)
                # Strong activation for query keywords
                embedding[i * 4:(i + 1) * 4] = np.random.randn(4) * 0.5 + 0.5

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        def encode_document(self, document: str) -> np.ndarray:
            """Encode a document."""
            # Add document prefix
            prefixed = f"{self.doc_prefix} {document}"

            # Simulate document-specific encoding
            # - Comprehensive representation
            # - Include context
            # - Preserve nuance

            words = document.lower().split()
            embedding = np.zeros(self.dimension)

            # Distributed representation across all dimensions
            for i, word in enumerate(words):
                np.random.seed(hash(word) % 2**32)
                # Softer activation for documents
                word_vec = np.random.randn(self.dimension) * 0.1
                embedding += word_vec / len(words)

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        def _extract_keywords(self, text: str) -> List[str]:
            """Extract keywords from text."""
            # Simple keyword extraction
            stop_words = {"the", "is", "at", "in", "on", "and", "a", "to", "of", "for"}
            words = text.lower().split()
            return [w for w in words if w not in stop_words and len(w) > 2]

    # Create embedder
    embedder = AsymmetricEmbedder(dimension=128)

    # Documents
    documents = [
        "Machine learning is a method of data analysis that automates analytical model building",
        "Python is an interpreted high-level programming language for general-purpose programming",
        "Deep learning is part of a broader family of machine learning methods based on neural networks",
        "Natural language processing is a subfield of linguistics and artificial intelligence",
        "Computer vision is an interdisciplinary field that deals with how computers gain understanding from digital images"
    ]

    # Queries
    queries = [
        "what is machine learning",
        "python programming",
        "neural networks",
        "image recognition"
    ]

    # Encode documents
    print("Encoding documents...")
    doc_embeddings = [embedder.encode_document(doc) for doc in documents]

    # Encode queries and find matches
    print("\nQuery-Document Matching:")
    print("-" * 30)

    for query in queries:
        query_embedding = embedder.encode_query(query)

        # Calculate similarities
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            similarity = np.dot(query_embedding, doc_emb)
            similarities.append((i, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        print(f"\nQuery: '{query}'")
        print("Top matches:")
        for idx, sim in similarities[:3]:
            print(f"  {sim:.3f}: {documents[idx][:60]}...")


# ===== Example 5: Embedding Visualization =====

def example_5_embedding_visualization():
    """Visualize and analyze embeddings."""
    print("\nExample 5: Embedding Visualization and Analysis")
    print("=" * 50)

    class EmbeddingAnalyzer:
        """Analyze embedding properties."""

        @staticmethod
        def calculate_metrics(embeddings: np.ndarray) -> Dict:
            """Calculate various metrics for embeddings."""
            metrics = {}

            # Basic statistics
            metrics["mean"] = float(np.mean(embeddings))
            metrics["std"] = float(np.std(embeddings))
            metrics["min"] = float(np.min(embeddings))
            metrics["max"] = float(np.max(embeddings))

            # Sparsity
            metrics["sparsity"] = float(np.mean(embeddings == 0))

            # Average pairwise similarity
            n = len(embeddings)
            if n > 1:
                similarities = []
                for i in range(n):
                    for j in range(i + 1, n):
                        sim = np.dot(embeddings[i], embeddings[j])
                        similarities.append(sim)
                metrics["avg_similarity"] = float(np.mean(similarities))
                metrics["similarity_std"] = float(np.std(similarities))

            # Dimensionality metrics
            # Estimate intrinsic dimension using PCA-like analysis
            if n > 1:
                centered = embeddings - np.mean(embeddings, axis=0)
                cov = np.dot(centered.T, centered) / n
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = sorted(eigenvalues, reverse=True)

                # Effective dimension (90% variance)
                total_var = np.sum(eigenvalues)
                cumsum = 0
                for i, ev in enumerate(eigenvalues):
                    cumsum += ev
                    if cumsum / total_var >= 0.9:
                        metrics["effective_dimension"] = i + 1
                        break

            return metrics

        @staticmethod
        def create_ascii_heatmap(embedding: np.ndarray, width: int = 60) -> str:
            """Create ASCII visualization of embedding."""
            # Normalize to 0-1
            min_val, max_val = embedding.min(), embedding.max()
            if max_val - min_val > 0:
                normalized = (embedding - min_val) / (max_val - min_val)
            else:
                normalized = embedding

            # Resize to fit width
            n_dims = len(embedding)
            if n_dims > width:
                # Downsample
                indices = np.linspace(0, n_dims - 1, width, dtype=int)
                display_values = normalized[indices]
            else:
                display_values = normalized

            # Convert to ASCII
            chars = " ░▒▓█"
            ascii_art = ""
            for val in display_values:
                char_idx = int(val * (len(chars) - 1))
                ascii_art += chars[char_idx]

            return ascii_art

        @staticmethod
        def find_outliers(embeddings: np.ndarray, threshold: float = 2.0) -> List[int]:
            """Find outlier embeddings."""
            # Calculate mean embedding
            mean_embedding = np.mean(embeddings, axis=0)

            # Calculate distances from mean
            distances = []
            for i, emb in enumerate(embeddings):
                dist = np.linalg.norm(emb - mean_embedding)
                distances.append((i, dist))

            # Find outliers (> threshold * std from mean)
            distances = np.array([d[1] for d in distances])
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)

            outliers = []
            for i, dist in enumerate(distances):
                if dist > mean_dist + threshold * std_dist:
                    outliers.append(i)

            return outliers

    # Create sample embeddings
    embedder = SimpleEmbedder(dimension=64)

    texts = [
        "Artificial intelligence and machine learning",
        "Deep learning with neural networks",
        "Natural language processing applications",
        "Computer vision and image recognition",
        "Reinforcement learning algorithms",
        "This is a completely different topic about cooking",  # Outlier
        "Data science and analytics",
        "Predictive modeling techniques"
    ]

    embeddings = embedder.encode_batch(texts)
    analyzer = EmbeddingAnalyzer()

    # Analyze embeddings
    print("Embedding Analysis:")
    print("-" * 30)

    metrics = analyzer.calculate_metrics(embeddings)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # Visualize individual embeddings
    print("\n" + "-" * 30)
    print("Embedding Visualizations:")
    print("-" * 30)

    for i in range(min(3, len(embeddings))):
        print(f"\nText {i+1}: {texts[i][:40]}...")
        print(f"Embedding: {analyzer.create_ascii_heatmap(embeddings[i])}")

    # Find outliers
    print("\n" + "-" * 30)
    print("Outlier Detection:")
    print("-" * 30)

    outliers = analyzer.find_outliers(embeddings, threshold=1.5)
    if outliers:
        print(f"Found {len(outliers)} outlier(s):")
        for idx in outliers:
            print(f"  - Text {idx+1}: {texts[idx][:50]}...")
    else:
        print("No significant outliers detected")

    # Clustering analysis
    print("\n" + "-" * 30)
    print("Similarity Clustering:")
    print("-" * 30)

    # Simple clustering based on similarity
    threshold = 0.7
    clusters = []
    assigned = set()

    for i in range(len(embeddings)):
        if i in assigned:
            continue

        cluster = [i]
        assigned.add(i)

        for j in range(i + 1, len(embeddings)):
            if j in assigned:
                continue

            similarity = np.dot(embeddings[i], embeddings[j])
            if similarity > threshold:
                cluster.append(j)
                assigned.add(j)

        clusters.append(cluster)

    print(f"Found {len(clusters)} cluster(s):")
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i+1}:")
        for idx in cluster:
            print(f"  - {texts[idx][:50]}...")


# ===== Example 6: Multi-lingual Embeddings =====

def example_6_multilingual_embeddings():
    """Handle multi-lingual text embeddings."""
    print("\nExample 6: Multi-lingual Embeddings")
    print("=" * 50)

    class MultilingualEmbedder:
        """Simple multi-lingual embedding handler."""

        def __init__(self, dimension: int = 128):
            self.dimension = dimension
            # Language-specific adjustments
            self.language_markers = {
                "en": 0.0,
                "es": 0.1,
                "fr": 0.2,
                "de": 0.3,
                "zh": 0.4
            }

        def detect_language(self, text: str) -> str:
            """Detect language (simplified)."""
            # Simple heuristic-based detection
            if any(word in text.lower() for word in ["the", "is", "and", "to"]):
                return "en"
            elif any(word in text.lower() for word in ["el", "la", "es", "y"]):
                return "es"
            elif any(word in text.lower() for word in ["le", "la", "est", "et"]):
                return "fr"
            elif any(word in text.lower() for word in ["der", "die", "ist", "und"]):
                return "de"
            else:
                return "en"  # Default

        def encode(self, text: str, language: Optional[str] = None) -> np.ndarray:
            """Encode text with language awareness."""
            if language is None:
                language = self.detect_language(text)

            # Base embedding
            words = text.lower().split()
            embedding = np.zeros(self.dimension)

            for word in words:
                np.random.seed(hash(word) % 2**32)
                word_vec = np.random.randn(self.dimension) * 0.1
                embedding += word_vec

            # Add language-specific marker
            if language in self.language_markers:
                # Reserve first few dimensions for language
                embedding[0] = self.language_markers[language]

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        def encode_cross_lingual(self, text: str, source_lang: str,
                               target_lang: str) -> np.ndarray:
            """Encode for cross-lingual retrieval."""
            # Simulate translation-invariant encoding
            base_embedding = self.encode(text, source_lang)

            # Apply transformation to align languages
            # In practice, use aligned multi-lingual models
            alignment_factor = abs(
                self.language_markers.get(source_lang, 0) -
                self.language_markers.get(target_lang, 0)
            )

            # Reduce language-specific components
            base_embedding[0] *= (1 - alignment_factor)

            return base_embedding

    # Create embedder
    embedder = MultilingualEmbedder(dimension=128)

    # Multi-lingual texts (simulated translations)
    texts = [
        ("en", "Machine learning is transforming technology"),
        ("es", "El aprendizaje automático está transformando la tecnología"),
        ("fr", "L'apprentissage automatique transforme la technologie"),
        ("en", "Data science requires statistical knowledge"),
        ("de", "Datenwissenschaft erfordert statistisches Wissen"),
        ("en", "Neural networks enable deep learning")
    ]

    print("Multi-lingual Text Embeddings:")
    print("-" * 30)

    # Encode texts
    embeddings = []
    for lang, text in texts:
        emb = embedder.encode(text, lang)
        embeddings.append(emb)
        print(f"{lang}: {text[:40]}...")

    # Test cross-lingual similarity
    print("\n" + "-" * 30)
    print("Cross-lingual Similarity:")
    print("-" * 30)

    # Compare similar content in different languages
    # Texts 0, 1, 2 are translations of each other
    for i in range(3):
        for j in range(i + 1, 3):
            similarity = np.dot(embeddings[i], embeddings[j])
            lang_i = texts[i][0]
            lang_j = texts[j][0]
            print(f"{lang_i} <-> {lang_j}: {similarity:.3f}")

    # Test cross-lingual encoding
    print("\n" + "-" * 30)
    print("Cross-lingual Encoding:")
    print("-" * 30)

    query = "machine learning technology"
    print(f"Query: {query}")

    # Encode for different target languages
    for target_lang in ["en", "es", "fr"]:
        query_emb = embedder.encode_cross_lingual(query, "en", target_lang)

        print(f"\nTarget language: {target_lang}")

        # Find best matches
        similarities = []
        for i, (lang, text) in enumerate(texts):
            if lang == target_lang or target_lang == "en":  # Include English
                sim = np.dot(query_emb, embeddings[i])
                similarities.append((i, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        if similarities:
            best_idx, best_sim = similarities[0]
            print(f"  Best match: {texts[best_idx][1][:40]}... (sim: {best_sim:.3f})")


# ===== Example 7: Performance Benchmarking =====

def example_7_performance_benchmarking():
    """Benchmark embedding performance."""
    print("\nExample 7: Performance Benchmarking")
    print("=" * 50)

    class EmbeddingBenchmark:
        """Benchmark embedding operations."""

        def __init__(self):
            self.results = {}

        def benchmark_encoding_speed(self, embedder, texts: List[str],
                                    batch_sizes: List[int]) -> Dict:
            """Benchmark encoding speed with different batch sizes."""
            results = {}

            for batch_size in batch_sizes:
                start_time = time.time()

                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    if hasattr(embedder, 'encode_batch'):
                        embedder.encode_batch(batch)
                    else:
                        for text in batch:
                            embedder.encode(text)

                elapsed = time.time() - start_time

                results[batch_size] = {
                    "total_time": elapsed,
                    "texts_per_second": len(texts) / elapsed,
                    "time_per_text": elapsed / len(texts)
                }

            return results

        def benchmark_memory_usage(self, embedder, num_texts: List[int]) -> Dict:
            """Benchmark memory usage for different corpus sizes."""
            results = {}

            for n in num_texts:
                # Generate dummy texts
                texts = [f"Sample text number {i} with some content" for i in range(n)]

                # Encode all texts
                embeddings = embedder.encode_batch(texts) if hasattr(embedder, 'encode_batch') else [embedder.encode(t) for t in texts]

                # Calculate memory usage
                if isinstance(embeddings, np.ndarray):
                    memory_bytes = embeddings.nbytes
                else:
                    memory_bytes = sum(e.nbytes for e in embeddings)

                results[n] = {
                    "memory_bytes": memory_bytes,
                    "memory_mb": memory_bytes / (1024 * 1024),
                    "bytes_per_text": memory_bytes / n
                }

            return results

        def benchmark_similarity_computation(self, embeddings: np.ndarray) -> Dict:
            """Benchmark similarity computation speed."""
            n = len(embeddings)

            # Full pairwise similarity
            start_time = time.time()
            similarity_matrix = np.dot(embeddings, embeddings.T)
            full_time = time.time() - start_time

            # Single query similarity
            query = embeddings[0]
            start_time = time.time()
            for _ in range(100):
                similarities = np.dot(embeddings, query)
            single_time = (time.time() - start_time) / 100

            return {
                "full_matrix_time": full_time,
                "full_matrix_comparisons": n * n,
                "comparisons_per_second": (n * n) / full_time,
                "single_query_time": single_time,
                "single_query_comparisons": n
            }

    # Create benchmark
    benchmark = EmbeddingBenchmark()
    embedder = SimpleEmbedder(dimension=384)

    # Generate test texts
    test_texts = [
        f"This is test document number {i} containing various words and information"
        for i in range(100)
    ]

    # Benchmark encoding speed
    print("Encoding Speed Benchmark:")
    print("-" * 30)

    batch_sizes = [1, 10, 50, 100]
    speed_results = benchmark.benchmark_encoding_speed(embedder, test_texts, batch_sizes)

    for batch_size, metrics in speed_results.items():
        print(f"\nBatch size: {batch_size}")
        print(f"  Total time: {metrics['total_time']:.3f}s")
        print(f"  Texts/second: {metrics['texts_per_second']:.1f}")
        print(f"  Time/text: {metrics['time_per_text']*1000:.2f}ms")

    # Benchmark memory usage
    print("\n" + "-" * 30)
    print("Memory Usage Benchmark:")
    print("-" * 30)

    corpus_sizes = [100, 1000, 5000]
    memory_results = benchmark.benchmark_memory_usage(embedder, corpus_sizes)

    for size, metrics in memory_results.items():
        print(f"\nCorpus size: {size}")
        print(f"  Total memory: {metrics['memory_mb']:.2f} MB")
        print(f"  Per text: {metrics['bytes_per_text']:.0f} bytes")

    # Benchmark similarity computation
    print("\n" + "-" * 30)
    print("Similarity Computation Benchmark:")
    print("-" * 30)

    embeddings = embedder.encode_batch(test_texts[:50])
    sim_results = benchmark.benchmark_similarity_computation(embeddings)

    print(f"Full similarity matrix:")
    print(f"  Time: {sim_results['full_matrix_time']:.4f}s")
    print(f"  Comparisons: {sim_results['full_matrix_comparisons']:,}")
    print(f"  Speed: {sim_results['comparisons_per_second']:,.0f} comparisons/s")

    print(f"\nSingle query search:")
    print(f"  Time: {sim_results['single_query_time']*1000:.3f}ms")
    print(f"  Comparisons: {sim_results['single_query_comparisons']}")

    # Performance recommendations
    print("\n" + "-" * 30)
    print("Performance Recommendations:")
    print("-" * 30)

    # Analyze results and make recommendations
    optimal_batch = max(speed_results.keys(),
                       key=lambda k: speed_results[k]['texts_per_second'])
    print(f"• Optimal batch size: {optimal_batch}")

    if embeddings.shape[1] > 512:
        print("• Consider dimensionality reduction for large embeddings")

    if sim_results['full_matrix_time'] > 1.0:
        print("• Use approximate nearest neighbor search for large corpus")

    print("• Cache frequently accessed embeddings")
    print("• Consider GPU acceleration for large-scale processing")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 10: Vector Embeddings Examples")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    examples = {
        1: example_1_basic_embeddings,
        2: example_2_document_chunking,
        3: example_3_embedding_models,
        4: example_4_query_document_embeddings,
        5: example_5_embedding_visualization,
        6: example_6_multilingual_embeddings,
        7: example_7_performance_benchmarking
    }

    if args.all:
        for example in examples.values():
            example()
            print("\n" + "=" * 70 + "\n")
    elif args.example and args.example in examples:
        examples[args.example]()
    else:
        print("Module 10: Vector Embeddings - Examples")
        print("\nUsage:")
        print("  python vector_embeddings.py --example N  # Run example N")
        print("  python vector_embeddings.py --all         # Run all examples")
        print("\nAvailable examples:")
        print("  1: Basic Text Embeddings")
        print("  2: Document Chunking Strategies")
        print("  3: Embedding Models Comparison")
        print("  4: Query vs Document Embeddings")
        print("  5: Embedding Visualization and Analysis")
        print("  6: Multi-lingual Embeddings")
        print("  7: Performance Benchmarking")