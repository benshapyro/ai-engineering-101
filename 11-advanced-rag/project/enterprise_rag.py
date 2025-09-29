"""
Module 11: Advanced RAG - Enterprise RAG System Project

A production-grade enterprise RAG system with advanced features including
multi-tenant support, horizontal scaling, real-time indexing, A/B testing,
comprehensive monitoring, and security controls.

This system demonstrates:
- Hybrid retrieval strategies
- Advanced query processing
- Multi-stage reranking
- Adaptive routing
- Multi-level caching
- Real-time performance monitoring
- A/B testing framework
- Security and access control
- Horizontal scaling with load balancing
- WebSocket support for streaming

Author: Claude
Date: 2024
"""

import os
import json
import hashlib
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis
import logging
from collections import defaultdict
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import jwt
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI()
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================
# Data Models
# ================================

class AccessLevel(str, Enum):
    """User access levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"

class QueryType(str, Enum):
    """Query type classification."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    COMPARISON = "comparison"
    EXPLORATORY = "exploratory"

@dataclass
class Tenant:
    """Tenant configuration."""
    id: str
    name: str
    settings: Dict[str, Any] = field(default_factory=dict)
    access_levels: Set[AccessLevel] = field(default_factory=set)
    rate_limit: int = 100  # Queries per minute
    max_documents: int = 10000
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class User:
    """User profile."""
    id: str
    tenant_id: str
    email: str
    access_level: AccessLevel = AccessLevel.PUBLIC
    preferences: Dict[str, Any] = field(default_factory=dict)
    query_history: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Document:
    """Document with enterprise metadata."""
    id: str
    tenant_id: str
    content: str
    access_level: AccessLevel
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: List[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1

@dataclass
class Query:
    """Query with tracking."""
    id: str
    tenant_id: str
    user_id: str
    text: str
    type: QueryType = QueryType.FACTUAL
    filters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    experiment_id: Optional[str] = None

# ================================
# Core Components
# ================================

class TenantManager:
    """Manage multi-tenant configuration."""

    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.user_cache: Dict[str, User] = {}

    def create_tenant(self, name: str, settings: Dict[str, Any] = None) -> Tenant:
        """Create a new tenant."""
        tenant_id = str(uuid.uuid4())
        tenant = Tenant(
            id=tenant_id,
            name=name,
            settings=settings or {},
            access_levels={AccessLevel.PUBLIC, AccessLevel.INTERNAL}
        )
        self.tenants[tenant_id] = tenant
        logger.info(f"Created tenant: {tenant_id}")
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)

    def create_user(
        self,
        tenant_id: str,
        email: str,
        access_level: AccessLevel = AccessLevel.PUBLIC
    ) -> User:
        """Create a user for a tenant."""
        user_id = str(uuid.uuid4())
        user = User(
            id=user_id,
            tenant_id=tenant_id,
            email=email,
            access_level=access_level
        )
        self.user_cache[user_id] = user
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.user_cache.get(user_id)

    def check_access(self, user: User, document: Document) -> bool:
        """Check if user has access to document."""
        # Check tenant match
        if user.tenant_id != document.tenant_id:
            return False

        # Check access level hierarchy
        access_hierarchy = {
            AccessLevel.PUBLIC: 0,
            AccessLevel.INTERNAL: 1,
            AccessLevel.CONFIDENTIAL: 2,
            AccessLevel.SECRET: 3
        }

        user_level = access_hierarchy.get(user.access_level, 0)
        doc_level = access_hierarchy.get(document.access_level, 0)

        return user_level >= doc_level

class HybridRetriever:
    """Advanced hybrid retrieval with caching."""

    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.tfidf_models: Dict[str, TfidfVectorizer] = {}
        self.tfidf_matrices: Dict[str, Any] = {}

    def index_document(self, document: Document):
        """Index a document for retrieval."""
        self.documents[document.id] = document

        # Generate embedding if not present
        if not document.embeddings:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=document.content
            )
            document.embeddings = response.data[0].embedding

        # Cache embedding
        self.embeddings_cache[document.id] = np.array(document.embeddings)

        # Update TF-IDF model for tenant
        self._update_tfidf(document.tenant_id)

    def _update_tfidf(self, tenant_id: str):
        """Update TF-IDF model for tenant."""
        tenant_docs = [
            doc.content for doc in self.documents.values()
            if doc.tenant_id == tenant_id
        ]

        if tenant_docs:
            vectorizer = TfidfVectorizer(max_features=1000)
            matrix = vectorizer.fit_transform(tenant_docs)
            self.tfidf_models[tenant_id] = vectorizer
            self.tfidf_matrices[tenant_id] = matrix

    def search(
        self,
        query: Query,
        user: User,
        k: int = 10,
        method: str = "hybrid"
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents with access control.

        Args:
            query: Search query
            user: User performing search
            k: Number of results
            method: Search method (dense, sparse, hybrid)

        Returns:
            List of (document, score) tuples
        """
        # Filter documents by access
        accessible_docs = [
            doc for doc in self.documents.values()
            if doc.tenant_id == query.tenant_id and
            self._check_access_level(user, doc)
        ]

        if not accessible_docs:
            return []

        if method == "dense":
            results = self._dense_search(query.text, accessible_docs, k)
        elif method == "sparse":
            results = self._sparse_search(query.text, accessible_docs, query.tenant_id, k)
        else:  # hybrid
            dense_results = self._dense_search(query.text, accessible_docs, k*2)
            sparse_results = self._sparse_search(query.text, accessible_docs, query.tenant_id, k*2)
            results = self._fusion(dense_results, sparse_results, k)

        return results

    def _check_access_level(self, user: User, document: Document) -> bool:
        """Check access level compatibility."""
        hierarchy = {
            AccessLevel.PUBLIC: 0,
            AccessLevel.INTERNAL: 1,
            AccessLevel.CONFIDENTIAL: 2,
            AccessLevel.SECRET: 3
        }
        return hierarchy.get(user.access_level, 0) >= hierarchy.get(document.access_level, 0)

    def _dense_search(
        self,
        query_text: str,
        documents: List[Document],
        k: int
    ) -> List[Tuple[Document, float]]:
        """Dense retrieval using embeddings."""
        # Generate query embedding
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text
        )
        query_emb = np.array(response.data[0].embedding)

        # Calculate similarities
        scores = []
        for doc in documents:
            doc_emb = self.embeddings_cache.get(doc.id)
            if doc_emb is not None:
                sim = np.dot(query_emb, doc_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
                )
                scores.append((doc, sim))

        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _sparse_search(
        self,
        query_text: str,
        documents: List[Document],
        tenant_id: str,
        k: int
    ) -> List[Tuple[Document, float]]:
        """Sparse retrieval using TF-IDF."""
        if tenant_id not in self.tfidf_models:
            return []

        vectorizer = self.tfidf_models[tenant_id]
        query_vec = vectorizer.transform([query_text])

        # Calculate scores
        scores = []
        doc_list = [d for d in self.documents.values() if d.tenant_id == tenant_id]

        for i, doc in enumerate(doc_list):
            if doc in documents:
                score = self.tfidf_matrices[tenant_id][i].dot(query_vec.T).toarray()[0, 0]
                scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]],
        k: int
    ) -> List[Tuple[Document, float]]:
        """Fuse dense and sparse results."""
        scores = defaultdict(float)

        # Reciprocal Rank Fusion
        for rank, (doc, _) in enumerate(dense_results, 1):
            scores[doc.id] += 1.0 / (60 + rank)

        for rank, (doc, _) in enumerate(sparse_results, 1):
            scores[doc.id] += 1.0 / (60 + rank)

        # Get documents and sort
        doc_scores = []
        for doc_id, score in scores.items():
            doc = self.documents.get(doc_id)
            if doc:
                doc_scores.append((doc, score))

        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:k]

class QueryProcessor:
    """Advanced query processing with caching."""

    def __init__(self):
        self.cache = {}
        self.query_history = defaultdict(list)

    def process(self, query: Query) -> Dict[str, Any]:
        """Process query with understanding and expansion."""
        # Check cache
        cache_key = f"{query.tenant_id}:{query.text}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Classify query type
        query.type = self._classify_query(query.text)

        # Extract entities
        entities = self._extract_entities(query.text)

        # Expand query
        expansions = self._expand_query(query.text)

        # Decompose if complex
        sub_queries = self._decompose_query(query.text)

        result = {
            "original": query.text,
            "type": query.type,
            "entities": entities,
            "expansions": expansions,
            "sub_queries": sub_queries
        }

        # Cache result
        self.cache[cache_key] = result

        # Update history
        self.query_history[query.user_id].append(query.text)

        return result

    def _classify_query(self, text: str) -> QueryType:
        """Classify query type."""
        text_lower = text.lower()

        if "compare" in text_lower or "versus" in text_lower:
            return QueryType.COMPARISON
        elif "how" in text_lower or "steps" in text_lower:
            return QueryType.PROCEDURAL
        elif "why" in text_lower or "analyze" in text_lower:
            return QueryType.ANALYTICAL
        elif "explore" in text_lower or "discover" in text_lower:
            return QueryType.EXPLORATORY
        else:
            return QueryType.FACTUAL

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from query."""
        # Simplified entity extraction
        entities = []

        # Look for capitalized words (potential entities)
        words = text.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append(word)

        return entities

    def _expand_query(self, text: str) -> List[str]:
        """Expand query with variations."""
        # Use LLM for expansion
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate 2 query variations. Return each on a new line."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.5,
                max_tokens=100
            )
            variations = response.choices[0].message.content.strip().split('\n')
            return [v.strip() for v in variations if v.strip()][:2]
        except:
            return []

    def _decompose_query(self, text: str) -> List[str]:
        """Decompose complex queries."""
        if " and " in text.lower() or len(text.split()) > 15:
            parts = text.split(" and ")
            return [p.strip() for p in parts]
        return []

class Reranker:
    """Multi-strategy reranking with personalization."""

    def __init__(self):
        self.strategies = {
            "relevance": self._relevance_rerank,
            "diversity": self._diversity_rerank,
            "personalized": self._personalized_rerank,
            "temporal": self._temporal_rerank
        }

    def rerank(
        self,
        query: Query,
        documents: List[Tuple[Document, float]],
        user: User,
        strategy: str = "relevance",
        k: int = 5
    ) -> List[Document]:
        """
        Rerank documents using specified strategy.

        Args:
            query: Search query
            documents: Documents with initial scores
            user: User profile
            strategy: Reranking strategy
            k: Number of documents to return

        Returns:
            Reranked documents
        """
        if strategy not in self.strategies:
            strategy = "relevance"

        rerank_func = self.strategies[strategy]
        reranked = rerank_func(query, documents, user, k)

        return reranked

    def _relevance_rerank(
        self,
        query: Query,
        documents: List[Tuple[Document, float]],
        user: User,
        k: int
    ) -> List[Document]:
        """Rerank by relevance using cross-encoder."""
        scored_docs = []

        for doc, initial_score in documents[:k*2]:
            # Use LLM for relevance scoring
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "Rate relevance from 0-10. Just the number."
                        },
                        {
                            "role": "user",
                            "content": f"Query: {query.text}\nDocument: {doc.content[:300]}"
                        }
                    ],
                    temperature=0,
                    max_tokens=10
                )
                relevance = float(response.choices[0].message.content.strip()) / 10.0
            except:
                relevance = initial_score

            scored_docs.append((doc, relevance))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]

    def _diversity_rerank(
        self,
        query: Query,
        documents: List[Tuple[Document, float]],
        user: User,
        k: int
    ) -> List[Document]:
        """Rerank for diversity using MMR."""
        if not documents:
            return []

        selected = []
        remaining = list(documents)
        lambda_param = 0.5

        # Select first document
        selected.append(remaining.pop(0)[0])

        # Iteratively select diverse documents
        while len(selected) < k and remaining:
            best_score = -1
            best_idx = -1

            for i, (doc, rel_score) in enumerate(remaining):
                # Calculate diversity from selected
                max_sim = 0
                for sel_doc in selected:
                    # Simple text similarity (could use embeddings)
                    sim = len(set(doc.content.split()) & set(sel_doc.content.split())) / \
                          max(len(doc.content.split()), len(sel_doc.content.split()))
                    max_sim = max(max_sim, sim)

                # MMR score
                mmr = lambda_param * rel_score - (1 - lambda_param) * max_sim

                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            if best_idx >= 0:
                selected.append(remaining.pop(best_idx)[0])

        return selected

    def _personalized_rerank(
        self,
        query: Query,
        documents: List[Tuple[Document, float]],
        user: User,
        k: int
    ) -> List[Document]:
        """Rerank based on user preferences."""
        scored_docs = []

        for doc, initial_score in documents:
            # Check user preferences
            pref_score = 0
            if user.preferences:
                # Simple keyword matching
                for pref_key, pref_value in user.preferences.items():
                    if str(pref_value).lower() in doc.content.lower():
                        pref_score += 0.1

            # Check query history relevance
            history_score = 0
            for hist_query in user.query_history[-5:]:
                if any(word in doc.content.lower() for word in hist_query.lower().split()):
                    history_score += 0.05

            final_score = initial_score + pref_score + history_score
            scored_docs.append((doc, final_score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]

    def _temporal_rerank(
        self,
        query: Query,
        documents: List[Tuple[Document, float]],
        user: User,
        k: int
    ) -> List[Document]:
        """Rerank considering recency."""
        current_time = datetime.now()
        scored_docs = []

        for doc, initial_score in documents:
            # Calculate age penalty
            age = (current_time - doc.created_at).days
            freshness = np.exp(-age / 30)  # 30-day half-life

            final_score = initial_score * 0.7 + freshness * 0.3
            scored_docs.append((doc, final_score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]

class ResponseGenerator:
    """Generate responses with citations and streaming."""

    def __init__(self):
        self.template_cache = {}

    async def generate(
        self,
        query: Query,
        documents: List[Document],
        streaming: bool = False
    ) -> Dict[str, Any]:
        """Generate response from documents."""
        # Format context
        context = self._format_context(documents)

        # Select template based on query type
        system_prompt = self._get_system_prompt(query.type)

        if streaming:
            return await self._generate_streaming(query.text, context, system_prompt)
        else:
            return self._generate_standard(query.text, context, system_prompt)

    def _format_context(self, documents: List[Document]) -> str:
        """Format documents as context."""
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(f"[{i+1}] {doc.content[:500]}")
        return "\n\n".join(context_parts)

    def _get_system_prompt(self, query_type: QueryType) -> str:
        """Get system prompt based on query type."""
        prompts = {
            QueryType.FACTUAL: "Answer the question directly based on the context. Cite sources using [number].",
            QueryType.ANALYTICAL: "Analyze the information and provide insights. Cite sources using [number].",
            QueryType.PROCEDURAL: "Provide step-by-step instructions based on the context. Cite sources using [number].",
            QueryType.COMPARISON: "Compare and contrast the information. Cite sources using [number].",
            QueryType.EXPLORATORY: "Explore the topic comprehensively. Cite sources using [number]."
        }
        return prompts.get(query_type, prompts[QueryType.FACTUAL])

    def _generate_standard(
        self,
        query_text: str,
        context: str,
        system_prompt: str
    ) -> Dict[str, Any]:
        """Generate standard response."""
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
            ],
            temperature=0.3,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        # Extract citations
        import re
        citations = re.findall(r'\[(\d+)\]', answer)

        return {
            "answer": answer,
            "citations": list(set(citations)),
            "model": "gpt-4-turbo-preview",
            "tokens": response.usage.total_tokens if hasattr(response, 'usage') else 0
        }

    async def _generate_streaming(
        self,
        query_text: str,
        context: str,
        system_prompt: str
    ) -> Dict[str, Any]:
        """Generate streaming response."""
        stream = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
            ],
            temperature=0.3,
            max_tokens=500,
            stream=True
        )

        # Return generator
        return {
            "stream": stream,
            "model": "gpt-4-turbo-preview"
        }

# ================================
# Monitoring and Analytics
# ================================

class MetricsCollector:
    """Collect and aggregate system metrics."""

    def __init__(self):
        self.metrics = defaultdict(lambda: defaultdict(float))
        self.events = []

    def record_query(self, query: Query, latency: float, success: bool):
        """Record query metrics."""
        tenant_metrics = self.metrics[query.tenant_id]
        tenant_metrics["total_queries"] += 1
        tenant_metrics["total_latency"] += latency

        if success:
            tenant_metrics["successful_queries"] += 1
        else:
            tenant_metrics["failed_queries"] += 1

        # Record event
        self.events.append({
            "type": "query",
            "tenant_id": query.tenant_id,
            "user_id": query.user_id,
            "latency": latency,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })

    def record_indexing(self, tenant_id: str, doc_count: int, duration: float):
        """Record indexing metrics."""
        tenant_metrics = self.metrics[tenant_id]
        tenant_metrics["documents_indexed"] += doc_count
        tenant_metrics["indexing_time"] += duration

    def get_metrics(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for tenant or all tenants."""
        if tenant_id:
            metrics = self.metrics.get(tenant_id, {})
            if metrics.get("total_queries", 0) > 0:
                metrics["avg_latency"] = metrics["total_latency"] / metrics["total_queries"]
                metrics["success_rate"] = metrics["successful_queries"] / metrics["total_queries"]
            return dict(metrics)
        else:
            # Aggregate all metrics
            total_metrics = defaultdict(float)
            for tenant_metrics in self.metrics.values():
                for key, value in tenant_metrics.items():
                    total_metrics[key] += value

            if total_metrics.get("total_queries", 0) > 0:
                total_metrics["avg_latency"] = total_metrics["total_latency"] / total_metrics["total_queries"]
                total_metrics["success_rate"] = total_metrics["successful_queries"] / total_metrics["total_queries"]

            return dict(total_metrics)

    def get_recent_events(self, limit: int = 100) -> List[Dict]:
        """Get recent events."""
        return self.events[-limit:]

# ================================
# A/B Testing Framework
# ================================

class ABTestManager:
    """Manage A/B testing experiments."""

    def __init__(self):
        self.experiments = {}
        self.results = defaultdict(lambda: defaultdict(list))

    def create_experiment(
        self,
        name: str,
        variants: List[str],
        allocation: Dict[str, float]
    ) -> str:
        """Create a new experiment."""
        experiment_id = str(uuid.uuid4())
        self.experiments[experiment_id] = {
            "id": experiment_id,
            "name": name,
            "variants": variants,
            "allocation": allocation,
            "created_at": datetime.now(),
            "active": True
        }
        return experiment_id

    def assign_variant(self, experiment_id: str, user_id: str) -> str:
        """Assign user to experiment variant."""
        experiment = self.experiments.get(experiment_id)
        if not experiment or not experiment["active"]:
            return "control"

        # Simple hash-based assignment
        hash_value = int(hashlib.md5(f"{experiment_id}:{user_id}".encode()).hexdigest(), 16)
        rand_value = (hash_value % 100) / 100.0

        cumulative = 0
        for variant, allocation in experiment["allocation"].items():
            cumulative += allocation
            if rand_value < cumulative:
                return variant

        return "control"

    def record_result(
        self,
        experiment_id: str,
        variant: str,
        metric: str,
        value: float
    ):
        """Record experiment result."""
        self.results[experiment_id][f"{variant}:{metric}"].append(value)

    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment results with statistical significance."""
        if experiment_id not in self.results:
            return {}

        results = {}
        experiment_results = self.results[experiment_id]

        for key, values in experiment_results.items():
            variant, metric = key.split(":")
            if variant not in results:
                results[variant] = {}

            results[variant][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "count": len(values)
            }

        return results

# ================================
# Cache Manager
# ================================

class CacheManager:
    """Multi-level caching system."""

    def __init__(self):
        self.memory_cache = {}
        self.cache_stats = defaultdict(int)

    def get(self, key: str, level: str = "memory") -> Optional[Any]:
        """Get from cache."""
        self.cache_stats["requests"] += 1

        if level == "memory":
            if key in self.memory_cache:
                self.cache_stats["hits"] += 1
                return self.memory_cache[key]["value"]
        elif level == "redis":
            try:
                value = redis_client.get(key)
                if value:
                    self.cache_stats["hits"] += 1
                    return json.loads(value)
            except:
                pass

        self.cache_stats["misses"] += 1
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
        level: str = "memory"
    ):
        """Set cache value."""
        if level == "memory":
            self.memory_cache[key] = {
                "value": value,
                "expires": time.time() + ttl
            }
        elif level == "redis":
            try:
                redis_client.setex(key, ttl, json.dumps(value))
            except:
                pass

    def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        # Memory cache
        keys_to_delete = [
            k for k in self.memory_cache.keys()
            if pattern in k
        ]
        for key in keys_to_delete:
            del self.memory_cache[key]

        # Redis cache
        try:
            for key in redis_client.scan_iter(f"*{pattern}*"):
                redis_client.delete(key)
        except:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.cache_stats["hits"] / self.cache_stats["requests"] if self.cache_stats["requests"] > 0 else 0

        return {
            "requests": self.cache_stats["requests"],
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "memory_size": len(self.memory_cache)
        }

# ================================
# Main Enterprise RAG System
# ================================

class EnterpriseRAGSystem:
    """Complete enterprise RAG system."""

    def __init__(self):
        self.tenant_manager = TenantManager()
        self.retriever = HybridRetriever()
        self.query_processor = QueryProcessor()
        self.reranker = Reranker()
        self.generator = ResponseGenerator()
        self.metrics = MetricsCollector()
        self.ab_test_manager = ABTestManager()
        self.cache = CacheManager()

    async def process_query(
        self,
        query_text: str,
        tenant_id: str,
        user_id: str,
        session_id: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query end-to-end.

        Args:
            query_text: Query text
            tenant_id: Tenant ID
            user_id: User ID
            session_id: Optional session ID
            experiment_id: Optional experiment ID

        Returns:
            Query response
        """
        start_time = time.time()

        try:
            # Get user and tenant
            user = self.tenant_manager.get_user(user_id)
            tenant = self.tenant_manager.get_tenant(tenant_id)

            if not user or not tenant:
                raise ValueError("Invalid user or tenant")

            # Check rate limit
            if not self._check_rate_limit(tenant_id):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            # Create query object
            query = Query(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                user_id=user_id,
                text=query_text,
                session_id=session_id,
                experiment_id=experiment_id
            )

            # Check cache
            cache_key = f"query:{tenant_id}:{hashlib.md5(query_text.encode()).hexdigest()}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result

            # Process query
            processed_query = self.query_processor.process(query)

            # Determine variant for A/B testing
            variant = "control"
            if experiment_id:
                variant = self.ab_test_manager.assign_variant(experiment_id, user_id)

            # Retrieve documents
            search_method = "hybrid" if variant == "treatment" else "dense"
            search_results = self.retriever.search(query, user, k=10, method=search_method)

            # Rerank
            rerank_strategy = "diversity" if variant == "treatment" else "relevance"
            reranked_docs = self.reranker.rerank(
                query,
                search_results,
                user,
                strategy=rerank_strategy,
                k=5
            )

            # Generate response
            response = await self.generator.generate(query, reranked_docs)

            # Prepare result
            result = {
                "query_id": query.id,
                "query": query_text,
                "response": response["answer"],
                "citations": response.get("citations", []),
                "sources": [
                    {
                        "id": doc.id,
                        "content": doc.content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in reranked_docs
                ],
                "metadata": {
                    "query_type": query.type.value,
                    "variant": variant,
                    "latency": time.time() - start_time,
                    "processed_query": processed_query
                }
            }

            # Cache result
            self.cache.set(cache_key, result, ttl=3600)

            # Record metrics
            latency = time.time() - start_time
            self.metrics.record_query(query, latency, success=True)

            # Record A/B test results
            if experiment_id:
                self.ab_test_manager.record_result(
                    experiment_id,
                    variant,
                    "latency",
                    latency
                )

            return result

        except Exception as e:
            logger.error(f"Query processing error: {e}")
            latency = time.time() - start_time
            self.metrics.record_query(query, latency, success=False)
            raise e

    def _check_rate_limit(self, tenant_id: str) -> bool:
        """Check if tenant is within rate limit."""
        # Simple rate limiting using Redis
        try:
            key = f"rate_limit:{tenant_id}:{int(time.time() / 60)}"
            current = redis_client.incr(key)
            redis_client.expire(key, 60)

            tenant = self.tenant_manager.get_tenant(tenant_id)
            return current <= tenant.rate_limit
        except:
            return True  # Allow if Redis is unavailable

    async def index_document(
        self,
        content: str,
        tenant_id: str,
        access_level: AccessLevel,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Index a new document."""
        start_time = time.time()

        # Create document
        doc = Document(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            content=content,
            access_level=access_level,
            metadata=metadata or {}
        )

        # Index in retriever
        self.retriever.index_document(doc)

        # Invalidate relevant caches
        self.cache.invalidate(f"query:{tenant_id}")

        # Record metrics
        duration = time.time() - start_time
        self.metrics.record_indexing(tenant_id, 1, duration)

        logger.info(f"Indexed document {doc.id} for tenant {tenant_id}")
        return doc.id

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        return {
            "status": "healthy",
            "components": {
                "retriever": len(self.retriever.documents) > 0,
                "cache": self.cache.get_stats()["requests"] > 0,
                "metrics": self.metrics.get_metrics().get("total_queries", 0) > 0
            },
            "metrics": self.metrics.get_metrics(),
            "cache_stats": self.cache.get_stats(),
            "timestamp": datetime.now().isoformat()
        }

# ================================
# FastAPI Application
# ================================

app = FastAPI(title="Enterprise RAG System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize system
rag_system = EnterpriseRAGSystem()

# WebSocket connections
active_websockets: List[WebSocket] = []

# ================================
# API Models
# ================================

class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    session_id: Optional[str] = None
    experiment_id: Optional[str] = None

class DocumentRequest(BaseModel):
    """Document indexing request."""
    content: str
    access_level: AccessLevel = AccessLevel.PUBLIC
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TenantRequest(BaseModel):
    """Tenant creation request."""
    name: str
    settings: Dict[str, Any] = Field(default_factory=dict)

class UserRequest(BaseModel):
    """User creation request."""
    email: str
    access_level: AccessLevel = AccessLevel.PUBLIC

# ================================
# Authentication
# ================================

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Verify JWT token."""
    token = credentials.credentials
    try:
        # Simplified token verification
        payload = jwt.decode(token, "secret_key", algorithms=["HS256"])
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# ================================
# API Endpoints
# ================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Enterprise RAG System",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/api/v1/query")
async def query(
    request: QueryRequest,
    token_data: Dict = Depends(verify_token)
):
    """Process a RAG query."""
    try:
        result = await rag_system.process_query(
            query_text=request.query,
            tenant_id=token_data["tenant_id"],
            user_id=token_data["user_id"],
            session_id=request.session_id,
            experiment_id=request.experiment_id
        )
        return result
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/documents")
async def index_document(
    request: DocumentRequest,
    token_data: Dict = Depends(verify_token)
):
    """Index a new document."""
    try:
        doc_id = await rag_system.index_document(
            content=request.content,
            tenant_id=token_data["tenant_id"],
            access_level=request.access_level,
            metadata=request.metadata
        )
        return {"document_id": doc_id}
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/tenants")
async def create_tenant(request: TenantRequest):
    """Create a new tenant."""
    tenant = rag_system.tenant_manager.create_tenant(
        name=request.name,
        settings=request.settings
    )
    return {"tenant_id": tenant.id}

@app.post("/api/v1/users")
async def create_user(
    request: UserRequest,
    token_data: Dict = Depends(verify_token)
):
    """Create a new user."""
    user = rag_system.tenant_manager.create_user(
        tenant_id=token_data["tenant_id"],
        email=request.email,
        access_level=request.access_level
    )
    return {"user_id": user.id}

@app.get("/api/v1/health")
async def health():
    """System health check."""
    return rag_system.get_system_health()

@app.get("/api/v1/metrics")
async def metrics(tenant_id: Optional[str] = None):
    """Get system metrics."""
    return rag_system.metrics.get_metrics(tenant_id)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for streaming queries."""
    await websocket.accept()
    active_websockets.append(websocket)

    try:
        while True:
            # Receive query
            data = await websocket.receive_json()

            # Process query with streaming
            query = Query(
                id=str(uuid.uuid4()),
                tenant_id=data["tenant_id"],
                user_id=data["user_id"],
                text=data["query"]
            )

            # Get user
            user = rag_system.tenant_manager.get_user(data["user_id"])

            # Search and rerank
            results = rag_system.retriever.search(query, user)
            reranked = rag_system.reranker.rerank(query, results, user)

            # Generate streaming response
            response = await rag_system.generator.generate(query, reranked, streaming=True)

            # Stream response
            if "stream" in response:
                for chunk in response["stream"]:
                    if chunk.choices[0].delta.content:
                        await websocket.send_json({
                            "type": "chunk",
                            "content": chunk.choices[0].delta.content
                        })

                await websocket.send_json({"type": "complete"})

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_websockets.remove(websocket)

# ================================
# Main Entry Point
# ================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Enterprise RAG System...")
    uvicorn.run(app, host="0.0.0.0", port=8000)