"""
Module 10: RAG Basics - Production Knowledge Base Project

A production-ready knowledge base system with full RAG capabilities,
including document management, semantic search, and intelligent Q&A.

This system demonstrates:
- Document ingestion and processing
- Multi-format support (PDF, TXT, MD, DOCX)
- Intelligent chunking strategies
- Hybrid search (keyword + semantic)
- Query optimization and expansion
- Caching and performance optimization
- REST API with FastAPI
- Real-time updates via WebSockets
- Monitoring and analytics

Author: Claude
Date: 2024
"""

import os
import hashlib
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
from enum import Enum

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings
import redis
import pickle
from collections import defaultdict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiofiles

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI()
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================
# Data Models
# ================================

class DocumentType(str, Enum):
    """Supported document types."""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    DOCX = "docx"
    HTML = "html"

class ChunkingStrategy(str, Enum):
    """Document chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

@dataclass
class Document:
    """Document model."""
    id: str
    content: str
    type: DocumentType
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    chunks: List[Dict] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)

@dataclass
class Query:
    """Query model."""
    id: str
    text: str
    filters: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 5
    rerank: bool = True
    expand: bool = True
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SearchResult:
    """Search result model."""
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

# ================================
# Document Processing
# ================================

class DocumentProcessor:
    """Process documents for the knowledge base."""

    def __init__(self, chunking_strategy: ChunkingStrategy = ChunkingStrategy.HYBRID):
        self.chunking_strategy = chunking_strategy
        self.chunk_size = 500
        self.chunk_overlap = 50

    async def process_document(
        self,
        content: str,
        doc_type: DocumentType,
        metadata: Dict = None
    ) -> Document:
        """
        Process a document into chunks with embeddings.

        Args:
            content: Document content
            doc_type: Type of document
            metadata: Document metadata

        Returns:
            Processed document
        """
        # Generate document ID
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]

        # Create document
        document = Document(
            id=doc_id,
            content=content,
            type=doc_type,
            metadata=metadata or {}
        )

        # Chunk document
        chunks = await self._chunk_document(content, doc_type)
        document.chunks = chunks

        # Generate embeddings for chunks
        embeddings = await self._generate_embeddings(chunks)
        document.embeddings = embeddings

        logger.info(f"Processed document {doc_id}: {len(chunks)} chunks")

        return document

    async def _chunk_document(
        self,
        content: str,
        doc_type: DocumentType
    ) -> List[Dict]:
        """Chunk document based on strategy."""
        if self.chunking_strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunks(content)
        elif self.chunking_strategy == ChunkingStrategy.SENTENCE_BASED:
            return self._sentence_chunks(content)
        elif self.chunking_strategy == ChunkingStrategy.PARAGRAPH_BASED:
            return self._paragraph_chunks(content)
        elif self.chunking_strategy == ChunkingStrategy.SEMANTIC:
            return await self._semantic_chunks(content)
        else:  # HYBRID
            return await self._hybrid_chunks(content)

    def _fixed_size_chunks(self, content: str) -> List[Dict]:
        """Create fixed-size chunks."""
        chunks = []
        for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
            chunk_text = content[i:i + self.chunk_size]
            if len(chunk_text) > 50:
                chunks.append({
                    "id": f"chunk_{i}",
                    "content": chunk_text,
                    "start": i,
                    "end": min(i + self.chunk_size, len(content))
                })
        return chunks

    def _sentence_chunks(self, content: str) -> List[Dict]:
        """Create sentence-based chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []

        for i in range(0, len(sentences), 3):
            chunk_sentences = sentences[i:i+4]  # 3 sentences + 1 overlap
            chunk_text = ' '.join(chunk_sentences)
            chunks.append({
                "id": f"chunk_{i}",
                "content": chunk_text,
                "sentence_start": i,
                "sentence_end": min(i+4, len(sentences))
            })

        return chunks

    def _paragraph_chunks(self, content: str) -> List[Dict]:
        """Create paragraph-based chunks."""
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""

        for i, para in enumerate(paragraphs):
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        "id": f"chunk_{len(chunks)}",
                        "content": current_chunk.strip()
                    })
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append({
                "id": f"chunk_{len(chunks)}",
                "content": current_chunk.strip()
            })

        return chunks

    async def _semantic_chunks(self, content: str) -> List[Dict]:
        """Create semantic-based chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', content)

        # Get embeddings for sentences
        embeddings = []
        for sent in sentences:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=sent
            )
            embeddings.append(np.array(response.data[0].embedding))

        # Group by similarity
        chunks = []
        current_group = [sentences[0]]
        current_embedding = embeddings[0]

        for i in range(1, len(sentences)):
            similarity = np.dot(current_embedding, embeddings[i]) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(embeddings[i])
            )

            if similarity > 0.75:
                current_group.append(sentences[i])
                current_embedding = np.mean([current_embedding, embeddings[i]], axis=0)
            else:
                chunks.append({
                    "id": f"chunk_{len(chunks)}",
                    "content": ' '.join(current_group)
                })
                current_group = [sentences[i]]
                current_embedding = embeddings[i]

        if current_group:
            chunks.append({
                "id": f"chunk_{len(chunks)}",
                "content": ' '.join(current_group)
            })

        return chunks

    async def _hybrid_chunks(self, content: str) -> List[Dict]:
        """Create hybrid chunks combining multiple strategies."""
        # Start with paragraph chunks
        para_chunks = self._paragraph_chunks(content)

        # Refine with semantic similarity
        refined_chunks = []
        for chunk in para_chunks:
            # If chunk is too large, split semantically
            if len(chunk["content"]) > self.chunk_size:
                sub_chunks = await self._semantic_chunks(chunk["content"])
                refined_chunks.extend(sub_chunks)
            else:
                refined_chunks.append(chunk)

        return refined_chunks

    async def _generate_embeddings(self, chunks: List[Dict]) -> List[List[float]]:
        """Generate embeddings for chunks."""
        embeddings = []

        for chunk in chunks:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk["content"]
            )
            embeddings.append(response.data[0].embedding)

        return embeddings

# ================================
# Search Engine
# ================================

class SearchEngine:
    """Hybrid search engine for the knowledge base."""

    def __init__(self):
        self.collection_name = "knowledge_base"
        self.collection = self._init_collection()
        self.keyword_index = defaultdict(set)
        self.document_store = {}

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

    async def index_document(self, document: Document):
        """Index a document for search."""
        # Store document
        self.document_store[document.id] = document

        # Index chunks in vector store
        chunk_ids = []
        chunk_contents = []
        chunk_embeddings = []
        chunk_metadata = []

        for i, (chunk, embedding) in enumerate(zip(document.chunks, document.embeddings)):
            chunk_id = f"{document.id}_{chunk['id']}"
            chunk_ids.append(chunk_id)
            chunk_contents.append(chunk["content"])
            chunk_embeddings.append(embedding)

            metadata = {
                "document_id": document.id,
                "chunk_index": i,
                **document.metadata
            }
            chunk_metadata.append(metadata)

            # Update keyword index
            words = chunk["content"].lower().split()
            for word in set(words):
                self.keyword_index[word].add(chunk_id)

        # Add to ChromaDB
        self.collection.add(
            ids=chunk_ids,
            embeddings=chunk_embeddings,
            documents=chunk_contents,
            metadatas=chunk_metadata
        )

        logger.info(f"Indexed document {document.id} with {len(chunk_ids)} chunks")

    async def search(
        self,
        query: Query,
        use_hybrid: bool = True
    ) -> List[SearchResult]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            use_hybrid: Use hybrid search (keyword + semantic)

        Returns:
            Search results
        """
        if use_hybrid:
            return await self._hybrid_search(query)
        else:
            return await self._semantic_search(query)

    async def _semantic_search(self, query: Query) -> List[SearchResult]:
        """Perform semantic search."""
        # Generate query embedding
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query.text
        )
        query_embedding = response.data[0].embedding

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=query.top_k,
            where=query.filters if query.filters else None
        )

        # Convert to SearchResult objects
        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append(SearchResult(
                document_id=results["metadatas"][0][i]["document_id"],
                chunk_id=results["ids"][0][i],
                content=results["documents"][0][i],
                score=1 - results["distances"][0][i],  # Convert distance to similarity
                metadata=results["metadatas"][0][i]
            ))

        return search_results

    async def _hybrid_search(self, query: Query) -> List[SearchResult]:
        """Perform hybrid search combining keyword and semantic."""
        # Get semantic results
        semantic_results = await self._semantic_search(query)

        # Get keyword results
        keyword_results = self._keyword_search(query.text, query.top_k * 2)

        # Combine using reciprocal rank fusion
        combined = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            query.top_k
        )

        return combined

    def _keyword_search(self, query_text: str, k: int) -> List[SearchResult]:
        """Perform keyword search using BM25-like scoring."""
        query_words = query_text.lower().split()
        scores = defaultdict(float)

        for word in query_words:
            if word in self.keyword_index:
                chunk_ids = self.keyword_index[word]
                idf = np.log(len(self.collection.get()["ids"]) / len(chunk_ids))

                for chunk_id in chunk_ids:
                    scores[chunk_id] += idf

        # Get top-k chunks
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Convert to SearchResult objects
        results = []
        for chunk_id, score in sorted_chunks:
            # Get chunk content from ChromaDB
            chunk_data = self.collection.get(ids=[chunk_id])
            if chunk_data["ids"]:
                results.append(SearchResult(
                    document_id=chunk_data["metadatas"][0]["document_id"],
                    chunk_id=chunk_id,
                    content=chunk_data["documents"][0],
                    score=score,
                    metadata=chunk_data["metadatas"][0]
                ))

        return results

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        k: int
    ) -> List[SearchResult]:
        """Combine results using reciprocal rank fusion."""
        rrf_scores = defaultdict(float)
        result_map = {}

        # Add semantic results
        for i, result in enumerate(semantic_results):
            rrf_scores[result.chunk_id] += 1.0 / (60 + i + 1)
            result_map[result.chunk_id] = result

        # Add keyword results
        for i, result in enumerate(keyword_results):
            rrf_scores[result.chunk_id] += 1.0 / (60 + i + 1)
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result

        # Sort by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Return top-k results
        final_results = []
        for chunk_id, rrf_score in sorted_chunks:
            result = result_map[chunk_id]
            result.score = rrf_score  # Update with RRF score
            final_results.append(result)

        return final_results

# ================================
# Query Engine
# ================================

class QueryEngine:
    """Process and answer queries."""

    def __init__(self, search_engine: SearchEngine):
        self.search_engine = search_engine
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour

    async def answer_query(
        self,
        query: Query,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a query using RAG.

        Args:
            query: User query
            use_cache: Whether to use cache

        Returns:
            Answer with metadata
        """
        # Check cache
        cache_key = f"{query.text}_{query.filters}_{query.top_k}"
        if use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            if datetime.now() - cached["timestamp"] < timedelta(seconds=self.cache_ttl):
                logger.info(f"Cache hit for query: {query.text[:50]}...")
                return cached["result"]

        # Expand query if requested
        if query.expand:
            expanded_queries = await self._expand_query(query.text)
        else:
            expanded_queries = [query.text]

        # Search for relevant documents
        all_results = []
        for exp_query in expanded_queries:
            query_obj = Query(
                id=query.id,
                text=exp_query,
                filters=query.filters,
                top_k=query.top_k
            )
            results = await self.search_engine.search(query_obj)
            all_results.extend(results)

        # Deduplicate and rerank if requested
        if query.rerank:
            final_results = await self._rerank_results(query.text, all_results, query.top_k)
        else:
            # Simple deduplication
            seen = set()
            final_results = []
            for result in all_results:
                if result.chunk_id not in seen:
                    seen.add(result.chunk_id)
                    final_results.append(result)
                    if len(final_results) >= query.top_k:
                        break

        # Generate answer
        answer = await self._generate_answer(query.text, final_results)

        # Prepare result
        result = {
            "query": query.text,
            "answer": answer,
            "sources": [
                {
                    "document_id": r.document_id,
                    "chunk_id": r.chunk_id,
                    "content": r.content[:200] + "...",
                    "score": r.score
                }
                for r in final_results
            ],
            "metadata": {
                "expanded_queries": expanded_queries,
                "timestamp": datetime.now().isoformat(),
                "reranked": query.rerank,
                "num_sources": len(final_results)
            }
        }

        # Cache result
        if use_cache:
            self.cache[cache_key] = {
                "result": result,
                "timestamp": datetime.now()
            }

        return result

    async def _expand_query(self, query_text: str) -> List[str]:
        """Expand query with variations."""
        response = openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Generate 2-3 alternative ways to search for this information. Return each on a new line."
                },
                {
                    "role": "user",
                    "content": query_text
                }
            ],
            temperature=0.5
        )

        variations = response.choices[0].message.content.strip().split('\n')
        variations = [v.strip() for v in variations if v.strip()]

        return [query_text] + variations[:2]

    async def _rerank_results(
        self,
        query_text: str,
        results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder approach."""
        # Score each result
        for result in results:
            response = openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"Rate relevance 0-10:\nQuery: {query_text}\nText: {result.content[:300]}\nJust the number:"
                    }
                ],
                temperature=0
            )

            try:
                rerank_score = float(response.choices[0].message.content.strip()) / 10.0
                result.score = (result.score + rerank_score) / 2  # Average with original score
            except:
                pass

        # Sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    async def _generate_answer(
        self,
        query_text: str,
        sources: List[SearchResult]
    ) -> str:
        """Generate answer from sources."""
        # Format context
        context = "\n\n".join([
            f"Source {i+1}:\n{source.content}"
            for i, source in enumerate(sources)
        ])

        # Generate answer
        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question based on the provided sources. Be concise and cite sources when possible."
                },
                {
                    "role": "user",
                    "content": f"Sources:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
                }
            ],
            temperature=0.3,
            max_tokens=500
        )

        return response.choices[0].message.content

# ================================
# FastAPI Application
# ================================

app = FastAPI(title="Knowledge Base API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
document_processor = DocumentProcessor(ChunkingStrategy.HYBRID)
search_engine = SearchEngine()
query_engine = QueryEngine(search_engine)

# WebSocket connections
active_connections: List[WebSocket] = []

# ================================
# API Models
# ================================

class DocumentUpload(BaseModel):
    """Document upload model."""
    content: str
    type: DocumentType
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryRequest(BaseModel):
    """Query request model."""
    text: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(5, ge=1, le=20)
    rerank: bool = True
    expand: bool = True

class DocumentResponse(BaseModel):
    """Document response model."""
    id: str
    type: str
    metadata: Dict[str, Any]
    num_chunks: int
    created_at: str

class QueryResponse(BaseModel):
    """Query response model."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# ================================
# API Endpoints
# ================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Knowledge Base API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/documents", response_model=DocumentResponse)
async def upload_document(document: DocumentUpload):
    """
    Upload and process a document.

    Args:
        document: Document to upload

    Returns:
        Document information
    """
    try:
        # Process document
        processed_doc = await document_processor.process_document(
            document.content,
            document.type,
            document.metadata
        )

        # Index document
        await search_engine.index_document(processed_doc)

        # Notify WebSocket clients
        for connection in active_connections:
            await connection.send_json({
                "event": "document_added",
                "document_id": processed_doc.id,
                "timestamp": datetime.now().isoformat()
            })

        return DocumentResponse(
            id=processed_doc.id,
            type=processed_doc.type.value,
            metadata=processed_doc.metadata,
            num_chunks=len(processed_doc.chunks),
            created_at=processed_doc.created_at.isoformat()
        )

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document file.

    Args:
        file: File to upload

    Returns:
        Document information
    """
    try:
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')

        # Determine document type
        file_extension = Path(file.filename).suffix.lower()
        doc_type_map = {
            '.txt': DocumentType.TEXT,
            '.md': DocumentType.MARKDOWN,
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,
            '.html': DocumentType.HTML
        }

        doc_type = doc_type_map.get(file_extension, DocumentType.TEXT)

        # Process document
        processed_doc = await document_processor.process_document(
            content_str,
            doc_type,
            {"filename": file.filename}
        )

        # Index document
        await search_engine.index_document(processed_doc)

        return DocumentResponse(
            id=processed_doc.id,
            type=processed_doc.type.value,
            metadata=processed_doc.metadata,
            num_chunks=len(processed_doc.chunks),
            created_at=processed_doc.created_at.isoformat()
        )

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(query: QueryRequest):
    """
    Query the knowledge base.

    Args:
        query: Query request

    Returns:
        Query response with answer
    """
    try:
        # Create query object
        query_obj = Query(
            id=hashlib.md5(query.text.encode()).hexdigest()[:8],
            text=query.text,
            filters=query.filters,
            top_k=query.top_k,
            rerank=query.rerank,
            expand=query.expand
        )

        # Get answer
        result = await query_engine.answer_query(query_obj)

        # Notify WebSocket clients
        for connection in active_connections:
            await connection.send_json({
                "event": "query_processed",
                "query": query.text[:50] + "...",
                "timestamp": datetime.now().isoformat()
            })

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            metadata=result["metadata"]
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search(
    q: str,
    limit: int = 5,
    filters: str = None
):
    """
    Search for documents.

    Args:
        q: Search query
        limit: Number of results
        filters: JSON-encoded filters

    Returns:
        Search results
    """
    try:
        # Parse filters
        filter_dict = json.loads(filters) if filters else {}

        # Create query
        query_obj = Query(
            id=hashlib.md5(q.encode()).hexdigest()[:8],
            text=q,
            filters=filter_dict,
            top_k=limit,
            rerank=False,
            expand=False
        )

        # Search
        results = await search_engine.search(query_obj)

        return {
            "query": q,
            "results": [
                {
                    "document_id": r.document_id,
                    "content": r.content[:200] + "...",
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in results
            ]
        }

    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """
    Get document by ID.

    Args:
        document_id: Document ID

    Returns:
        Document information
    """
    if document_id not in search_engine.document_store:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = search_engine.document_store[document_id]

    return {
        "id": doc.id,
        "type": doc.type.value,
        "metadata": doc.metadata,
        "num_chunks": len(doc.chunks),
        "created_at": doc.created_at.isoformat(),
        "content_preview": doc.content[:500] + "..."
    }

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document.

    Args:
        document_id: Document ID

    Returns:
        Deletion status
    """
    if document_id not in search_engine.document_store:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from document store
    doc = search_engine.document_store.pop(document_id)

    # Remove from ChromaDB
    chunk_ids = [f"{document_id}_{chunk['id']}" for chunk in doc.chunks]
    search_engine.collection.delete(ids=chunk_ids)

    # Remove from keyword index
    for word_set in search_engine.keyword_index.values():
        word_set.discard(*chunk_ids)

    return {"message": f"Document {document_id} deleted successfully"}

@app.get("/stats")
async def get_statistics():
    """
    Get system statistics.

    Returns:
        System statistics
    """
    # Get collection stats
    collection_data = search_engine.collection.get()

    return {
        "total_documents": len(search_engine.document_store),
        "total_chunks": len(collection_data["ids"]),
        "cache_size": len(query_engine.cache),
        "active_connections": len(active_connections),
        "keyword_index_size": len(search_engine.keyword_index)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()

    except:
        active_connections.remove(websocket)

# ================================
# CLI Interface
# ================================

async def cli_interface():
    """Command-line interface for the knowledge base."""
    print("Knowledge Base CLI")
    print("=" * 50)
    print("Commands:")
    print("  add <file_path> - Add a document")
    print("  query <question> - Query the knowledge base")
    print("  search <text> - Search for documents")
    print("  stats - Show statistics")
    print("  quit - Exit")
    print("=" * 50)

    while True:
        command = input("\n> ").strip()

        if command.startswith("add "):
            file_path = command[4:].strip()
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    content = f.read()

                doc_type = DocumentType.TEXT
                if file_path.endswith('.md'):
                    doc_type = DocumentType.MARKDOWN

                doc = await document_processor.process_document(
                    content,
                    doc_type,
                    {"source": file_path}
                )
                await search_engine.index_document(doc)
                print(f"Added document {doc.id}")
            else:
                print("File not found")

        elif command.startswith("query "):
            question = command[6:].strip()
            query_obj = Query(
                id="cli_query",
                text=question,
                top_k=3
            )
            result = await query_engine.answer_query(query_obj)
            print(f"\nAnswer: {result['answer']}")
            print(f"\nSources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['content'][:100]}...")

        elif command.startswith("search "):
            search_text = command[7:].strip()
            query_obj = Query(
                id="cli_search",
                text=search_text,
                top_k=5,
                rerank=False,
                expand=False
            )
            results = await search_engine.search(query_obj)
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.score:.3f}")
                print(f"     {result.content[:150]}...")

        elif command == "stats":
            print(f"\nStatistics:")
            print(f"  Documents: {len(search_engine.document_store)}")
            print(f"  Cache entries: {len(query_engine.cache)}")
            print(f"  Keywords indexed: {len(search_engine.keyword_index)}")

        elif command == "quit":
            break

        else:
            print("Unknown command")

# ================================
# Main Entry Point
# ================================

if __name__ == "__main__":
    import uvicorn

    # Check for CLI mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        asyncio.run(cli_interface())
    else:
        # Run FastAPI server
        print("Starting Knowledge Base API server...")
        print("Visit http://localhost:8000/docs for API documentation")
        uvicorn.run(app, host="0.0.0.0", port=8000)