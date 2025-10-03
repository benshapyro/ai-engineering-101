"""
Text generation endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from llm.client import LLMClient
from metrics.tracing import trace_call

router = APIRouter()

# Initialize client (in production, use dependency injection)
client = LLMClient()


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt")
    instructions: Optional[str] = Field(None, description="System instructions")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(1000, ge=1, le=4000, description="Maximum tokens to generate")
    model: Optional[str] = Field(None, description="Model name")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    text: str = Field(..., description="Generated text")
    tokens: Optional[int] = Field(None, description="Total tokens used")
    cost: Optional[float] = Field(None, description="Estimated cost in USD")
    model: str = Field(..., description="Model used")


@trace_call(model="gpt-5")
@router.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest) -> GenerateResponse:
    """
    Generate text from prompt.

    Args:
        request: Generation request

    Returns:
        Generated text and metadata

    Example:
        ```python
        import requests

        response = requests.post(
            "http://localhost:8000/api/v1/generate",
            json={
                "prompt": "Explain quantum computing",
                "temperature": 0.7,
                "max_tokens": 500
            }
        )

        print(response.json()["text"])
        ```
    """
    try:
        # Generate
        response = client.generate(
            input_text=request.prompt,
            instructions=request.instructions,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            model=request.model
        )

        # Get output
        text = client.get_output_text(response)

        # Extract metadata
        tokens = None
        cost = None

        if hasattr(response, 'usage'):
            tokens = getattr(response.usage, 'total_tokens', None)
            # Simple cost estimate
            if tokens:
                cost = tokens * 10 / 1_000_000  # Rough estimate

        return GenerateResponse(
            text=text,
            tokens=tokens,
            cost=cost,
            model=request.model or client.model
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


@router.post("/complete")
async def complete_text(request: GenerateRequest) -> dict:
    """
    Alternative completion endpoint with simpler response.

    Args:
        request: Generation request

    Returns:
        Dictionary with completion
    """
    try:
        response = client.generate(
            input_text=request.prompt,
            instructions=request.instructions,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return {
            "completion": client.get_output_text(response),
            "model": client.model
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
