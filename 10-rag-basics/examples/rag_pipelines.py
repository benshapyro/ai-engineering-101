"""
Module 10: RAG Basics - Complete RAG Pipeline Examples

This file demonstrates complete end-to-end RAG (Retrieval-Augmented Generation)
pipelines that combine all components into production-ready systems.

Author: Claude
Date: 2024
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import faiss
import pickle
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize clients
client = OpenAI()
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

# ================================
# Example 1: Basic RAG Pipeline
# ================================
print("=" * 50)
print("Example 1: Basic RAG Pipeline")
print("=" * 50)

class BasicRAGPipeline:
    """Simple RAG pipeline with core components."""

    def __init__(self, collection_name: str = "basic_rag"):
        self.collection_name = collection_name
        self.collection = self._init_collection()

    def _init_collection(self):
        """Initialize ChromaDB collection."""
        try:
            return chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except:
            chroma_client.delete_collection(self.collection_name)
            return chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    def index_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Index documents with embeddings."""
        # Generate embeddings
        embeddings = []
        for doc in documents:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=doc
            )
            embeddings.append(response.data[0].embedding)

        # Store in ChromaDB
        ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata or [{}] * len(documents)
        )
        print(f"Indexed {len(documents)} documents")

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant documents."""
        # Generate query embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return [
            {
                "document": doc,
                "metadata": meta,
                "distance": dist
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate response using retrieved context."""
        # Format context
        context_text = "\n\n".join([
            f"Document {i+1}: {doc['document']}"
            for i, doc in enumerate(context)
        ])

        # Generate response
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "Answer questions based on the provided context."
                },
                {
                    "role": "user",
                    "content": f"""Context:
{context_text}

Question: {query}

Answer based on the context provided:"""
                }
            ]
        )

        return response.choices[0].message.content

    def query(self, question: str, k: int = 3) -> Dict:
        """Complete RAG query pipeline."""
        # Retrieve relevant documents
        context = self.retrieve(question, k)

        # Generate response
        answer = self.generate_response(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": context
        }

# Test basic pipeline
pipeline = BasicRAGPipeline()

# Index sample documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand human language.",
    "Computer vision allows machines to interpret visual information.",
    "Reinforcement learning learns through trial and error."
]

pipeline.index_documents(documents)

# Query the pipeline
result = pipeline.query("What is deep learning?")
print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Sources used: {len(result['sources'])}")

# ================================
# Example 2: Advanced RAG with Preprocessing
# ================================
print("\n" + "=" * 50)
print("Example 2: Advanced RAG with Preprocessing")
print("=" * 50)

@dataclass
class Document:
    """Document with preprocessing metadata."""
    content: str
    title: str = ""
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    chunk_size: int = 500
    chunk_overlap: int = 50

class AdvancedRAGPipeline:
    """RAG pipeline with advanced preprocessing."""

    def __init__(self, collection_name: str = "advanced_rag"):
        self.collection_name = collection_name
        self.collection = self._init_collection()
        self.chunk_cache = {}

    def _init_collection(self):
        """Initialize collection with metadata."""
        try:
            return chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except:
            chroma_client.delete_collection(self.collection_name)
            return chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    def preprocess_document(self, doc: Document) -> List[Dict]:
        """Preprocess document into chunks."""
        # Simple chunking with overlap
        text = doc.content
        chunks = []

        for i in range(0, len(text), doc.chunk_size - doc.chunk_overlap):
            chunk = text[i:i + doc.chunk_size]
            if len(chunk) < 50:  # Skip very small chunks
                continue

            chunks.append({
                "content": chunk,
                "metadata": {
                    "title": doc.title,
                    "source": doc.source,
                    "timestamp": doc.timestamp.isoformat(),
                    "chunk_index": len(chunks),
                    "char_start": i,
                    "char_end": min(i + doc.chunk_size, len(text))
                }
            })

        return chunks

    def index_document(self, doc: Document):
        """Index a single document with preprocessing."""
        # Preprocess into chunks
        chunks = self.preprocess_document(doc)

        # Generate embeddings for chunks
        embeddings = []
        for chunk in chunks:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk["content"]
            )
            embeddings.append(response.data[0].embedding)

        # Store in collection
        ids = [
            f"{doc.title}_{i}_{hashlib.md5(chunk['content'].encode()).hexdigest()[:8]}"
            for i, chunk in enumerate(chunks)
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[c["content"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks]
        )

        print(f"Indexed document '{doc.title}' as {len(chunks)} chunks")

    def query_with_reranking(self, query: str, k: int = 5, rerank_k: int = 3) -> Dict:
        """Query with reranking of results."""
        # Initial retrieval
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # Rerank using cross-encoder simulation (using GPT for scoring)
        reranked = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # Score relevance
            score_response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{
                    "role": "user",
                    "content": f"Rate the relevance of this text to the query on a scale of 0-10:\nQuery: {query}\nText: {doc[:200]}\nRespond with just a number."
                }],
                temperature=0
            )

            try:
                score = float(score_response.choices[0].message.content.strip())
            except:
                score = 5.0

            reranked.append({
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "relevance_score": score
            })

        # Sort by relevance score
        reranked.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Generate response with top reranked results
        context_text = "\n\n".join([
            f"Source: {doc['metadata'].get('title', 'Unknown')}\n{doc['document']}"
            for doc in reranked[:rerank_k]
        ])

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "Answer based on the provided context. Cite sources when possible."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {query}"
                }
            ]
        )

        return {
            "question": query,
            "answer": response.choices[0].message.content,
            "sources": reranked[:rerank_k]
        }

# Test advanced pipeline
advanced_pipeline = AdvancedRAGPipeline()

# Index documents with metadata
doc1 = Document(
    content="Artificial Intelligence (AI) has revolutionized many industries. Machine learning, a subset of AI, enables computers to learn from data without explicit programming. Deep learning, using neural networks, has achieved breakthrough results in computer vision and natural language processing.",
    title="Introduction to AI",
    source="AI Textbook Chapter 1"
)

doc2 = Document(
    content="Natural Language Processing (NLP) is the field of AI concerned with enabling computers to understand, interpret, and generate human language. Key tasks include sentiment analysis, named entity recognition, machine translation, and question answering. Modern NLP relies heavily on transformer models.",
    title="NLP Fundamentals",
    source="NLP Guide"
)

advanced_pipeline.index_document(doc1)
advanced_pipeline.index_document(doc2)

# Query with reranking
result = advanced_pipeline.query_with_reranking("What is NLP used for?")
print(f"Question: {result['question']}")
print(f"Answer: {result['answer'][:200]}...")

# ================================
# Example 3: Hybrid Search RAG
# ================================
print("\n" + "=" * 50)
print("Example 3: Hybrid Search RAG")
print("=" * 50)

class HybridSearchRAG:
    """RAG with both semantic and keyword search."""

    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.keyword_index = {}

    def _build_keyword_index(self, doc: str, doc_id: int):
        """Build inverted index for keyword search."""
        words = doc.lower().split()
        for word in set(words):
            if word not in self.keyword_index:
                self.keyword_index[word] = []
            self.keyword_index[word].append(doc_id)

    def index(self, documents: List[str]):
        """Index documents for hybrid search."""
        self.documents = documents

        # Build semantic embeddings
        for doc in documents:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=doc
            )
            self.embeddings.append(response.data[0].embedding)

        # Build keyword index
        for i, doc in enumerate(documents):
            self._build_keyword_index(doc, i)

        print(f"Indexed {len(documents)} documents for hybrid search")

    def hybrid_search(self, query: str, k: int = 3, alpha: float = 0.5) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword search.
        alpha: weight for semantic search (1-alpha for keyword)
        """
        # Semantic search
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = np.array(response.data[0].embedding)

        # Calculate cosine similarities
        semantic_scores = []
        for emb in self.embeddings:
            similarity = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            semantic_scores.append(similarity)

        # Keyword search (BM25-like scoring)
        keyword_scores = [0.0] * len(self.documents)
        query_words = query.lower().split()

        for word in query_words:
            if word in self.keyword_index:
                doc_ids = self.keyword_index[word]
                idf = np.log(len(self.documents) / len(doc_ids))
                for doc_id in doc_ids:
                    tf = self.documents[doc_id].lower().count(word)
                    keyword_scores[doc_id] += tf * idf

        # Normalize scores
        max_semantic = max(semantic_scores) if semantic_scores else 1
        max_keyword = max(keyword_scores) if max(keyword_scores) > 0 else 1

        semantic_scores = [s / max_semantic for s in semantic_scores]
        keyword_scores = [s / max_keyword for s in keyword_scores]

        # Combine scores
        hybrid_scores = [
            alpha * sem + (1 - alpha) * key
            for sem, key in zip(semantic_scores, keyword_scores)
        ]

        # Get top-k results
        top_indices = np.argsort(hybrid_scores)[-k:][::-1]

        return [
            {
                "document": self.documents[i],
                "semantic_score": semantic_scores[i],
                "keyword_score": keyword_scores[i],
                "hybrid_score": hybrid_scores[i]
            }
            for i in top_indices
        ]

    def generate_answer(self, query: str, k: int = 3, alpha: float = 0.5) -> str:
        """Generate answer using hybrid search."""
        # Retrieve using hybrid search
        results = self.hybrid_search(query, k, alpha)

        # Format context
        context = "\n\n".join([
            f"[Score: {r['hybrid_score']:.3f}] {r['document']}"
            for r in results
        ])

        # Generate response
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "Answer based on the context. Mention if information comes from high-scoring sources."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ]
        )

        return response.choices[0].message.content

# Test hybrid search
hybrid_rag = HybridSearchRAG()

documents = [
    "Python is a high-level programming language known for its simplicity.",
    "JavaScript is essential for web development and runs in browsers.",
    "Machine learning models can be deployed using Python frameworks.",
    "React is a JavaScript library for building user interfaces.",
    "Data science often uses Python libraries like pandas and numpy."
]

hybrid_rag.index(documents)

# Test with different alpha values
for alpha in [0.0, 0.5, 1.0]:
    results = hybrid_rag.hybrid_search("Python programming", k=2, alpha=alpha)
    print(f"\nAlpha={alpha} (semantic weight):")
    for r in results:
        print(f"  Score={r['hybrid_score']:.3f}: {r['document'][:50]}...")

# ================================
# Example 4: Streaming RAG Pipeline
# ================================
print("\n" + "=" * 50)
print("Example 4: Streaming RAG Pipeline")
print("=" * 50)

class StreamingRAG:
    """RAG pipeline with streaming responses."""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def index(self, documents: List[str]):
        """Index documents."""
        self.documents = documents
        for doc in documents:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=doc
            )
            self.embeddings.append(response.data[0].embedding)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documents."""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_emb = np.array(response.data[0].embedding)

        # Calculate similarities
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb)
            )
            similarities.append(sim)

        # Get top-k
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_indices]

    def stream_response(self, query: str):
        """Generate streaming response."""
        # Retrieve context
        context_docs = self.retrieve(query)
        context = "\n\n".join(context_docs)

        # Stream response
        stream = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "Answer based on the context provided."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ],
            stream=True
        )

        print(f"Streaming answer for: {query}")
        print("Response: ", end="")
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print()  # New line after streaming

        return full_response

# Test streaming
streaming_rag = StreamingRAG()
streaming_rag.index([
    "Streaming allows real-time data processing.",
    "RAG pipelines can provide streaming responses for better UX.",
    "Chunked responses reduce latency in applications."
])

response = streaming_rag.stream_response("How does streaming improve user experience?")

# ================================
# Example 5: Multi-Query RAG
# ================================
print("\n" + "=" * 50)
print("Example 5: Multi-Query RAG")
print("=" * 50)

class MultiQueryRAG:
    """RAG that generates multiple queries for better retrieval."""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def index(self, documents: List[str]):
        """Index documents."""
        self.documents = documents
        for doc in documents:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=doc
            )
            self.embeddings.append(response.data[0].embedding)

    def generate_queries(self, original_query: str, n: int = 3) -> List[str]:
        """Generate multiple query variations."""
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Generate {n} different ways to ask this question. Return each on a new line."
                },
                {
                    "role": "user",
                    "content": original_query
                }
            ],
            temperature=0.7
        )

        queries = response.choices[0].message.content.strip().split('\n')
        queries = [q.strip('- ').strip('1234567890. ').strip() for q in queries]
        return [original_query] + queries[:n-1]

    def multi_query_retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve using multiple query variations."""
        # Generate query variations
        queries = self.generate_queries(query)
        print(f"Generated queries: {queries}")

        # Retrieve for each query
        all_results = {}
        for q in queries:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=q
            )
            query_emb = np.array(response.data[0].embedding)

            # Calculate similarities
            for i, emb in enumerate(self.embeddings):
                sim = np.dot(query_emb, emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(emb)
                )

                if i not in all_results:
                    all_results[i] = []
                all_results[i].append(sim)

        # Aggregate scores (max pooling)
        final_scores = {}
        for doc_id, scores in all_results.items():
            final_scores[doc_id] = max(scores)

        # Get top-k
        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "document": self.documents[doc_id],
                "score": score,
                "doc_id": doc_id
            }
            for doc_id, score in sorted_docs[:k]
        ]

    def answer(self, query: str) -> str:
        """Generate answer using multi-query retrieval."""
        # Retrieve with multiple queries
        results = self.multi_query_retrieve(query)

        # Format context
        context = "\n\n".join([
            f"[Relevance: {r['score']:.3f}] {r['document']}"
            for r in results
        ])

        # Generate response
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question based on the context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ]
        )

        return response.choices[0].message.content

# Test multi-query RAG
multi_query_rag = MultiQueryRAG()

documents = [
    "Database indexing improves query performance by creating data structures for fast lookups.",
    "SQL optimization involves query planning, index usage, and execution strategies.",
    "NoSQL databases offer flexibility but require different optimization approaches.",
    "Caching can significantly reduce database load and improve response times.",
    "Database sharding distributes data across multiple servers for scalability."
]

multi_query_rag.index(documents)

answer = multi_query_rag.answer("How can I make my database faster?")
print(f"\nAnswer: {answer[:200]}...")

# ================================
# Example 6: RAG with Caching
# ================================
print("\n" + "=" * 50)
print("Example 6: RAG with Caching")
print("=" * 50)

class CachedRAG:
    """RAG pipeline with multi-level caching."""

    def __init__(self, cache_dir: str = ".rag_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.documents = []
        self.embeddings = []
        self.query_cache = {}
        self.embedding_cache = {}

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache an embedding."""
        key = self._get_cache_key(text)
        self.embedding_cache[key] = embedding

        # Persist to disk
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)

    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if available."""
        key = self._get_cache_key(text)

        # Check memory cache
        if key in self.embedding_cache:
            return self.embedding_cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                embedding = pickle.load(f)
                self.embedding_cache[key] = embedding
                return embedding

        return None

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching."""
        # Check cache
        cached = self._get_cached_embedding(text)
        if cached is not None:
            print(f"Cache hit for: {text[:30]}...")
            return cached

        # Generate new embedding
        print(f"Cache miss, generating embedding for: {text[:30]}...")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding

        # Cache it
        self._cache_embedding(text, embedding)

        return embedding

    def index(self, documents: List[str]):
        """Index documents with caching."""
        self.documents = documents
        self.embeddings = []

        for doc in documents:
            embedding = self.get_embedding(doc)
            self.embeddings.append(embedding)

        print(f"Indexed {len(documents)} documents with caching")

    def query_with_cache(self, query: str, k: int = 3) -> Dict:
        """Query with response caching."""
        # Check query cache
        cache_key = f"{query}_{k}"
        if cache_key in self.query_cache:
            print("Returning cached response")
            return self.query_cache[cache_key]

        # Get query embedding
        query_emb = np.array(self.get_embedding(query))

        # Find similar documents
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb)
            )
            similarities.append(sim)

        # Get top-k
        top_indices = np.argsort(similarities)[-k:][::-1]
        context = [self.documents[i] for i in top_indices]

        # Generate response
        context_text = "\n\n".join(context)
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "Answer based on the context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {query}"
                }
            ]
        )

        result = {
            "question": query,
            "answer": response.choices[0].message.content,
            "sources": context
        }

        # Cache the result
        self.query_cache[cache_key] = result

        return result

    def clear_cache(self):
        """Clear all caches."""
        self.query_cache.clear()
        self.embedding_cache.clear()

        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

        print("Cache cleared")

# Test caching
cached_rag = CachedRAG()

documents = [
    "Caching improves performance by storing frequently accessed data.",
    "LRU (Least Recently Used) is a common cache eviction policy.",
    "Cache invalidation is one of the hardest problems in computer science."
]

# Index documents (will cache embeddings)
cached_rag.index(documents)

# Query twice (second should be cached)
print("\nFirst query:")
result1 = cached_rag.query_with_cache("What is caching?")
print(f"Answer: {result1['answer'][:100]}...")

print("\nSecond query (same):")
result2 = cached_rag.query_with_cache("What is caching?")
print(f"Answer: {result2['answer'][:100]}...")

# ================================
# Example 7: Production RAG System
# ================================
print("\n" + "=" * 50)
print("Example 7: Production RAG System")
print("=" * 50)

@dataclass
class RAGConfig:
    """Configuration for production RAG system."""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-5"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 5
    rerank_k: int = 3
    cache_ttl: int = 3600
    max_retries: int = 3
    timeout: int = 30

class ProductionRAG:
    """Production-ready RAG system with all features."""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.cache = {}
        self.metrics = {
            "queries": 0,
            "cache_hits": 0,
            "errors": 0,
            "avg_latency": 0
        }

    async def aindex_documents(self, documents: List[Dict]):
        """Async document indexing."""
        tasks = []
        for doc in documents:
            tasks.append(self._aindex_single(doc))

        results = await asyncio.gather(*tasks)

        for doc_data in results:
            self.documents.extend(doc_data["chunks"])
            self.embeddings.extend(doc_data["embeddings"])
            self.metadata.extend(doc_data["metadata"])

        print(f"Indexed {len(documents)} documents asynchronously")

    async def _aindex_single(self, document: Dict) -> Dict:
        """Index single document asynchronously."""
        # Chunk document
        chunks = self._chunk_text(
            document["content"],
            self.config.chunk_size,
            self.config.chunk_overlap
        )

        # Generate embeddings
        embeddings = []
        for chunk in chunks:
            # In production, use async client
            response = client.embeddings.create(
                model=self.config.embedding_model,
                input=chunk
            )
            embeddings.append(response.data[0].embedding)

        # Create metadata
        metadata = [
            {
                **document.get("metadata", {}),
                "chunk_index": i,
                "source": document.get("source", "unknown")
            }
            for i in range(len(chunks))
        ]

        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "metadata": metadata
        }

    def _chunk_text(self, text: str, size: int, overlap: int) -> List[str]:
        """Chunk text with overlap."""
        chunks = []
        for i in range(0, len(text), size - overlap):
            chunk = text[i:i + size]
            if len(chunk) > 50:
                chunks.append(chunk)
        return chunks

    def query(self, query: str, filters: Dict = None) -> Dict:
        """Production query with monitoring."""
        import time
        start_time = time.time()

        try:
            self.metrics["queries"] += 1

            # Check cache
            cache_key = f"{query}_{str(filters)}"
            if cache_key in self.cache:
                self.metrics["cache_hits"] += 1
                return self.cache[cache_key]

            # Retrieve
            results = self._retrieve(query, filters)

            # Rerank
            reranked = self._rerank(query, results)

            # Generate response
            response = self._generate(query, reranked)

            # Create result
            result = {
                "question": query,
                "answer": response,
                "sources": reranked,
                "metadata": {
                    "latency": time.time() - start_time,
                    "num_sources": len(reranked)
                }
            }

            # Cache result
            self.cache[cache_key] = result

            # Update metrics
            self.metrics["avg_latency"] = (
                (self.metrics["avg_latency"] * (self.metrics["queries"] - 1) +
                 result["metadata"]["latency"]) / self.metrics["queries"]
            )

            return result

        except Exception as e:
            self.metrics["errors"] += 1
            raise e

    def _retrieve(self, query: str, filters: Dict = None) -> List[Dict]:
        """Retrieve with filtering."""
        # Generate query embedding
        response = client.embeddings.create(
            model=self.config.embedding_model,
            input=query
        )
        query_emb = np.array(response.data[0].embedding)

        # Calculate similarities
        results = []
        for i, (doc, emb, meta) in enumerate(zip(
            self.documents, self.embeddings, self.metadata
        )):
            # Apply filters
            if filters:
                skip = False
                for key, value in filters.items():
                    if meta.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            # Calculate similarity
            sim = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb)
            )

            results.append({
                "document": doc,
                "metadata": meta,
                "score": sim
            })

        # Sort and return top-k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:self.config.retrieval_k]

    def _rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results."""
        # Simple reranking based on relevance scoring
        for result in results:
            # In production, use a cross-encoder model
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{
                    "role": "user",
                    "content": f"Rate 0-10 relevance:\nQuery: {query}\nText: {result['document'][:200]}\nJust the number:"
                }],
                temperature=0
            )

            try:
                result["rerank_score"] = float(
                    response.choices[0].message.content.strip()
                )
            except:
                result["rerank_score"] = 5.0

        # Sort by rerank score
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:self.config.rerank_k]

    def _generate(self, query: str, context: List[Dict]) -> str:
        """Generate response with context."""
        context_text = "\n\n".join([
            f"[Source: {c['metadata'].get('source', 'Unknown')}]\n{c['document']}"
            for c in context
        ])

        response = client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer based on context. Cite sources."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {query}"
                }
            ]
        )

        return response.choices[0].message.content

    def get_metrics(self) -> Dict:
        """Get system metrics."""
        return {
            **self.metrics,
            "cache_size": len(self.cache),
            "document_count": len(self.documents),
            "cache_hit_rate": (
                self.metrics["cache_hits"] / self.metrics["queries"]
                if self.metrics["queries"] > 0 else 0
            )
        }

# Test production system
async def test_production_rag():
    """Test the production RAG system."""
    config = RAGConfig(
        chunk_size=200,
        chunk_overlap=20,
        retrieval_k=3,
        rerank_k=2
    )

    rag = ProductionRAG(config)

    # Index documents
    documents = [
        {
            "content": "Production RAG systems require careful consideration of scalability, latency, and accuracy. Key components include efficient indexing, smart caching, and robust error handling.",
            "metadata": {"category": "systems", "priority": "high"},
            "source": "RAG Best Practices"
        },
        {
            "content": "Monitoring and observability are crucial for production deployments. Track metrics like query latency, cache hit rates, and error frequencies to ensure system health.",
            "metadata": {"category": "operations", "priority": "high"},
            "source": "Production Guide"
        }
    ]

    await rag.aindex_documents(documents)

    # Test queries
    result = rag.query("What are important considerations for production RAG?")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Latency: {result['metadata']['latency']:.2f}s")

    # Test with filters
    result = rag.query(
        "How to monitor systems?",
        filters={"category": "operations"}
    )
    print(f"\nFiltered answer: {result['answer'][:150]}...")

    # Show metrics
    print(f"\nSystem metrics: {rag.get_metrics()}")

# Run async test
import asyncio
asyncio.run(test_production_rag())

print("\n" + "=" * 50)
print("RAG Pipeline Examples Complete!")
print("=" * 50)