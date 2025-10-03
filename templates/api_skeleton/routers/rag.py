"""
RAG (Retrieval-Augmented Generation) endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from llm.client import LLMClient
from rag.retrievers import BM25Retriever
from rag.rerankers import CrossEncoderReranker

router = APIRouter(prefix="/rag")

# Initialize components (in production, use dependency injection)
client = LLMClient()

# Example documents (in production, load from database)
SAMPLE_DOCS = [
    {
        "id": "1",
        "content": "Prompt engineering is the practice of designing effective prompts for LLMs.",
        "metadata": {"category": "prompting"}
    },
    {
        "id": "2",
        "content": "RAG combines retrieval with generation for better accuracy.",
        "metadata": {"category": "rag"}
    },
    {
        "id": "3",
        "content": "Few-shot learning uses examples to guide model behavior.",
        "metadata": {"category": "prompting"}
    },
]

# Initialize retriever
retriever = BM25Retriever(SAMPLE_DOCS)

# Initialize reranker (lazy loading)
reranker = None


class RAGQueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    top_k: int = Field(3, ge=1, le=20, description="Number of documents to retrieve")
    use_rerank: bool = Field(False, description="Whether to use reranking")
    generate_answer: bool = Field(True, description="Whether to generate answer")


class Source(BaseModel):
    """Source document."""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = {}


class RAGQueryResponse(BaseModel):
    """Response model for RAG query."""
    query: str
    answer: Optional[str] = None
    sources: List[Source]
    metadata: Dict[str, Any] = {}


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest) -> RAGQueryResponse:
    """
    RAG query endpoint.

    Retrieves relevant documents and optionally generates answer.

    Args:
        request: RAG query request

    Returns:
        Answer and source documents

    Example:
        ```python
        import requests

        response = requests.post(
            "http://localhost:8000/api/v1/rag/query",
            json={
                "query": "What is prompt engineering?",
                "top_k": 3,
                "use_rerank": True,
                "generate_answer": True
            }
        )

        result = response.json()
        print(result["answer"])
        for source in result["sources"]:
            print(f"- {source['content']}")
        ```
    """
    try:
        # 1. Retrieve documents
        results = retriever.retrieve(request.query, top_k=request.top_k * 2)

        # 2. Optional reranking
        if request.use_rerank:
            global reranker
            if reranker is None:
                reranker = CrossEncoderReranker()

            docs = [r.content for r in results]
            reranked_indices = reranker.rerank(
                request.query,
                docs,
                top_k=request.top_k,
                return_scores=True
            )

            # Rebuild results in reranked order
            reranked_results = []
            for idx, score in reranked_indices:
                result = results[idx]
                result.score = score  # Update with rerank score
                reranked_results.append(result)

            results = reranked_results
        else:
            results = results[:request.top_k]

        # 3. Build sources
        sources = [
            Source(
                doc_id=r.doc_id,
                content=r.content,
                score=r.score,
                metadata=r.metadata
            )
            for r in results
        ]

        # 4. Optional answer generation
        answer = None
        if request.generate_answer:
            # Build context from sources
            context = "\n\n".join([
                f"[{i+1}] {s.content}"
                for i, s in enumerate(sources)
            ])

            # Generate answer
            prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {request.query}

Answer:"""

            response = client.generate(
                input_text=prompt,
                temperature=0.3,
                max_tokens=500
            )

            answer = client.get_output_text(response)

        return RAGQueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            metadata={
                "retrieved_count": len(results),
                "reranked": request.use_rerank
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )


@router.get("/documents")
async def list_documents() -> Dict[str, Any]:
    """
    List available documents.

    Returns:
        Document list and stats
    """
    return {
        "total_documents": len(SAMPLE_DOCS),
        "documents": SAMPLE_DOCS
    }
