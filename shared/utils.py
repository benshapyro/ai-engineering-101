"""
Shared utility functions for the Prompt Engineering curriculum.
"""

import os
import json
import time
import tiktoken
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import openai
from anthropic import Anthropic

# Load environment variables
load_dotenv()


class LLMClient:
    """Unified client for OpenAI and Anthropic models."""

    def __init__(self, provider: str = "openai"):
        """
        Initialize LLM client.

        Args:
            provider: "openai" or "anthropic"
        """
        self.provider = provider.lower()

        if self.provider == "openai":
            self.client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.default_model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        elif self.provider == "anthropic":
            self.client = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            self.default_model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Get completion from the LLM.

        Args:
            prompt: User prompt
            system_message: System message (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            model: Model to use (defaults to provider default)
            **kwargs: Additional provider-specific parameters

        Returns:
            The model's response text
        """
        model = model or self.default_model

        if self.provider == "openai":
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            kwargs_anthropic = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
            if system_message:
                kwargs_anthropic["system"] = system_message
            kwargs_anthropic.update(kwargs)

            response = self.client.messages.create(**kwargs_anthropic)
            return response.content[0].text

    def stream_complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        model: Optional[str] = None,
        **kwargs
    ):
        """
        Stream completion from the LLM.

        Yields:
            Chunks of the response text
        """
        model = model or self.default_model

        if self.provider == "openai":
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        elif self.provider == "anthropic":
            kwargs_anthropic = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }
            if system_message:
                kwargs_anthropic["system"] = system_message
            kwargs_anthropic.update(kwargs)

            with self.client.messages.stream(**kwargs_anthropic) as stream:
                for text in stream.text_stream:
                    yield text


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in a text string.

    Args:
        text: Text to count tokens for
        model: Model to use for encoding

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4-turbo-preview"
) -> Dict[str, float]:
    """
    Estimate API cost for a completion.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name

    Returns:
        Dictionary with cost breakdown
    """
    # Pricing as of 2024 (per 1M tokens)
    pricing = {
        "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    if model not in pricing:
        return {"error": f"Unknown model: {model}"}

    input_cost = (input_tokens / 1_000_000) * pricing[model]["input"]
    output_cost = (output_tokens / 1_000_000) * pricing[model]["output"]

    return {
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(input_cost + output_cost, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }


def format_messages_for_display(messages: List[Dict[str, str]]) -> str:
    """
    Format messages for display.

    Args:
        messages: List of message dictionaries

    Returns:
        Formatted string
    """
    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        formatted.append(f"[{role}]\n{content}\n")

    return "\n".join(formatted)


def save_conversation(
    messages: List[Dict[str, str]],
    filename: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save a conversation to a JSON file.

    Args:
        messages: List of message dictionaries
        filename: Output filename
        metadata: Optional metadata to include
    """
    data = {
        "messages": messages,
        "metadata": metadata or {},
        "timestamp": time.time()
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def load_conversation(filename: str) -> Dict[str, Any]:
    """
    Load a conversation from a JSON file.

    Args:
        filename: Input filename

    Returns:
        Dictionary with messages and metadata
    """
    with open(filename, "r") as f:
        return json.load(f)


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    max_retries: int = 5,
    errors: tuple = (Exception,)
):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        initial_delay: Initial delay in seconds
        exponential_base: Base for exponential backoff
        max_retries: Maximum number of retries
        errors: Tuple of errors to catch

    Returns:
        Function result
    """
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except errors as e:
                if i == max_retries - 1:
                    raise
                print(f"Error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= exponential_base

    return wrapper


def validate_json_response(response: str, schema: Optional[Dict] = None) -> Dict:
    """
    Validate and parse JSON response from LLM.

    Args:
        response: Response string from LLM
        schema: Optional JSON schema for validation

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If JSON is invalid
    """
    # Try to extract JSON from response
    response = response.strip()

    # Handle markdown code blocks
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]

    response = response.strip()

    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    # Validate against schema if provided
    if schema:
        import jsonschema
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Schema validation failed: {e}")

    return data


def truncate_text(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """
    Truncate text to fit within token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model to use for encoding

    Returns:
        Truncated text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    model: str = "gpt-4"
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to split
        chunk_size: Size of each chunk in tokens
        overlap: Number of overlapping tokens
        model: Model to use for encoding

    Returns:
        List of text chunks
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    chunks = []

    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(encoding.decode(chunk_tokens))
        i += chunk_size - overlap

    return chunks


# Example usage
if __name__ == "__main__":
    # Test OpenAI client
    openai_client = LLMClient("openai")
    response = openai_client.complete(
        "What is prompt engineering?",
        temperature=0.7,
        max_tokens=100
    )
    print("OpenAI Response:", response)

    # Test token counting
    text = "This is a test sentence for token counting."
    tokens = count_tokens(text)
    print(f"Token count: {tokens}")

    # Test cost estimation
    cost = estimate_cost(100, 50, "gpt-4-turbo-preview")
    print(f"Estimated cost: ${cost['total_cost']}")