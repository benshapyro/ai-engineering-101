"""
LLM Client wrapper using OpenAI Responses API.

This module provides a standardized interface for LLM interactions
using the modern Responses API (primary) with Chat Completions fallback.

The Responses API is OpenAI's primary API for building agentic applications,
released in March 2025. It provides a cleaner interface with built-in support
for tools, web search, file search, and structured outputs.

Example usage:
    from llm.client import LLMClient

    # Basic generation
    client = LLMClient()
    response = client.generate(
        input_text="Explain quantum computing in simple terms",
        instructions="You are a helpful science educator"
    )
    print(client.get_output_text(response))

    # Structured output
    schema = {...}  # JSON schema
    response = client.structured(
        input_text="Analyze this sentiment",
        schema=schema
    )
"""

from typing import Dict, List, Any, Optional, Union
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Unified client for LLM interactions using OpenAI Responses API.

    This client supports both the modern Responses API (default) and the
    legacy Chat Completions API for backwards compatibility.
    """

    def __init__(
        self,
        model: str = None,
        use_responses_api: bool = True,
        api_key: str = None
    ):
        """
        Initialize LLM client.

        Args:
            model: Model name (defaults to OPENAI_MODEL env var or gpt-5)
            use_responses_api: Use Responses API (True) or Chat Completions (False)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5")
        self.use_responses_api = use_responses_api

    def generate(
        self,
        input_text: Union[str, List[Dict[str, str]]],
        instructions: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Any:
        """
        Generate completion using Responses API or Chat Completions.

        The Responses API is simpler: just provide input text and optional
        instructions (similar to system message). For Chat Completions API,
        you can provide a messages list directly.

        Args:
            input_text: User input (str) or messages list (for Chat Completions fallback)
            instructions: System instructions (Responses API only)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            **kwargs: Additional API parameters

        Returns:
            OpenAI response object

        Example:
            # Responses API (recommended)
            response = client.generate(
                input_text="What is Python?",
                instructions="Be concise"
            )

            # Chat Completions API (legacy)
            client = LLMClient(use_responses_api=False)
            response = client.generate(
                input_text=[
                    {"role": "user", "content": "What is Python?"}
                ]
            )
        """
        if self.use_responses_api:
            # Use modern Responses API
            return self.client.responses.create(
                model=self.model,
                input=input_text,
                instructions=instructions,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        else:
            # Fallback to Chat Completions API
            if isinstance(input_text, str):
                messages = []
                if instructions:
                    messages.append({"role": "developer", "content": instructions})
                messages.append({"role": "user", "content": input_text})
            else:
                messages = input_text

            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

    def structured(
        self,
        input_text: Union[str, List[Dict[str, str]]],
        schema: dict,
        instructions: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Any:
        """
        Generate structured output with strict JSON schema validation.

        Both Responses API and Chat Completions support structured outputs.
        The Responses API is preferred for its cleaner interface.

        Args:
            input_text: User input (str) or messages list
            schema: JSON schema dict with 'name', 'strict', and 'schema' keys
            instructions: System instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional API parameters

        Returns:
            OpenAI response object with validated JSON

        Example:
            schema = {
                "name": "sentiment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["sentiment", "confidence"],
                    "additionalProperties": False
                }
            }

            response = client.structured(
                input_text="I love this product!",
                schema=schema,
                instructions="Analyze sentiment"
            )
        """
        if self.use_responses_api:
            # Use Responses API with structured output
            return self.client.responses.create(
                model=self.model,
                input=input_text,
                instructions=instructions,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": schema,
                    "strict": True
                },
                **kwargs
            )
        else:
            # Use Chat Completions with structured output
            if isinstance(input_text, str):
                messages = []
                if instructions:
                    messages.append({"role": "developer", "content": instructions})
                messages.append({"role": "user", "content": input_text})
            else:
                messages = input_text

            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": schema,
                    "strict": True
                },
                **kwargs
            )

    def stream(
        self,
        input_text: Union[str, List[Dict[str, str]]],
        instructions: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Stream completion in real-time using Responses API or Chat Completions.

        Args:
            input_text: User input (str) or messages list
            instructions: System instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional API parameters

        Returns:
            Stream iterator

        Example:
            for event in client.stream("Tell me a story"):
                if hasattr(event, 'delta'):
                    print(event.delta, end='', flush=True)
        """
        if self.use_responses_api:
            return self.client.responses.create(
                model=self.model,
                input=input_text,
                instructions=instructions,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
        else:
            if isinstance(input_text, str):
                messages = []
                if instructions:
                    messages.append({"role": "developer", "content": instructions})
                messages.append({"role": "user", "content": input_text})
            else:
                messages = input_text

            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

    def get_output_text(self, response: Any) -> str:
        """
        Extract output text from response (handles both APIs).

        Args:
            response: Response object from generate() or structured()

        Returns:
            Output text string

        Example:
            response = client.generate("Hello")
            text = client.get_output_text(response)
        """
        if hasattr(response, 'output_text'):
            # Responses API
            return response.output_text
        elif hasattr(response, 'choices'):
            # Chat Completions API
            return response.choices[0].message.content
        else:
            raise ValueError("Unknown response format")


# Example usage
if __name__ == "__main__":
    # Test basic generation with Responses API
    print("Testing LLM Client with Responses API\n")
    print("=" * 60)

    client = LLMClient()

    # Example 1: Basic generation
    print("\nExample 1: Basic Generation")
    print("-" * 60)
    response = client.generate(
        input_text="What is the capital of France?",
        instructions="Be concise and direct.",
        temperature=0
    )
    print(f"Response: {client.get_output_text(response)}")

    # Example 2: Structured output
    print("\nExample 2: Structured Output")
    print("-" * 60)

    sentiment_schema = {
        "name": "sentiment_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "reasoning": {"type": "string"}
            },
            "required": ["sentiment", "confidence"],
            "additionalProperties": False
        }
    }

    response = client.structured(
        input_text="I absolutely love this product! It's amazing!",
        schema=sentiment_schema,
        instructions="Analyze the sentiment of the given text.",
        temperature=0
    )
    print(f"Response: {client.get_output_text(response)}")

    # Example 3: Streaming
    print("\nExample 3: Streaming")
    print("-" * 60)
    print("Stream output: ", end="", flush=True)
    stream = client.stream(
        input_text="Count to 5 slowly with words",
        temperature=0
    )
    for event in stream:
        if hasattr(event, 'delta'):
            print(event.delta, end="", flush=True)
    print()

    print("\n" + "=" * 60)
    print("✓ All examples completed successfully")
