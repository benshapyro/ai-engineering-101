"""
Reproducibility utilities for deterministic LLM behavior.

This module provides tools for making LLM outputs reproducible across runs,
essential for testing, debugging, and production consistency.
"""

import os
import random
import hashlib
from typing import Optional, Dict, Any


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    While LLM API calls aren't fully deterministic (temperature=0 is close
    but not guaranteed), setting seeds helps with:
    - Consistent test data generation
    - Reproducible data splits
    - Deterministic random sampling

    Args:
        seed: Random seed value (default: 42)

    Example:
        set_seed(42)
        # Now random operations will be reproducible
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_deterministic_params(
    temperature: float = 0.0,
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get parameters for deterministic LLM behavior.

    Args:
        temperature: Temperature (0.0 for most deterministic)
        seed: Random seed (if supported by API)
        **kwargs: Additional parameters

    Returns:
        Dictionary of parameters optimized for determinism

    Example:
        params = get_deterministic_params()
        response = client.generate("test", **params)
        # Should get same response on repeated calls
    """
    params = {
        "temperature": temperature,
        "top_p": 1.0,  # Don't use nucleus sampling
        **kwargs
    }

    # Add seed if provided and supported
    if seed is not None:
        params["seed"] = seed

    return params


def hash_prompt(prompt: str) -> str:
    """
    Create deterministic hash of a prompt for caching.

    Args:
        prompt: Prompt text

    Returns:
        Hexadecimal hash string

    Example:
        cache_key = hash_prompt("Translate to French: Hello")
        # Use as cache key for storing/retrieving responses
    """
    return hashlib.sha256(prompt.encode()).hexdigest()


def cache_key_from_params(
    prompt: str,
    model: str,
    temperature: float,
    **kwargs
) -> str:
    """
    Create cache key from prompt and parameters.

    Useful for caching LLM responses to avoid repeated API calls
    with identical parameters.

    Args:
        prompt: Input prompt
        model: Model name
        temperature: Temperature parameter
        **kwargs: Additional parameters to include in cache key

    Returns:
        Unique cache key string

    Example:
        key = cache_key_from_params(
            "Summarize this text",
            model="gpt-5",
            temperature=0.0
        )
        # Use key to store/retrieve cached response
    """
    # Sort kwargs for consistent ordering
    sorted_kwargs = sorted(kwargs.items())

    # Create deterministic string representation
    param_string = f"{prompt}|{model}|{temperature}|{sorted_kwargs}"

    # Hash it
    return hashlib.sha256(param_string.encode()).hexdigest()


class ResponseCache:
    """
    Simple in-memory cache for LLM responses.

    Useful for:
    - Testing (avoid repeated API calls)
    - Development (faster iteration)
    - Cost reduction (cache common queries)

    Note: For production, use Redis or similar.
    """

    def __init__(self):
        """Initialize empty cache."""
        self._cache: Dict[str, str] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[str]:
        """
        Get cached response.

        Args:
            key: Cache key

        Returns:
            Cached response or None
        """
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, key: str, value: str):
        """
        Store response in cache.

        Args:
            key: Cache key
            value: Response to cache
        """
        self._cache[key] = value

    def clear(self):
        """Clear all cached responses."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0

        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate
        }


class DeterministicClient:
    """
    Wrapper for LLMClient with deterministic defaults and caching.

    Example:
        from llm.client import LLMClient

        base_client = LLMClient()
        det_client = DeterministicClient(base_client)

        # First call hits API
        response1 = det_client.generate("Test prompt")

        # Second call returns cached result
        response2 = det_client.generate("Test prompt")
        assert response1 == response2
    """

    def __init__(
        self,
        client,
        use_cache: bool = True,
        default_temperature: float = 0.0,
        seed: Optional[int] = 42
    ):
        """
        Initialize deterministic client.

        Args:
            client: Base LLMClient instance
            use_cache: Enable response caching
            default_temperature: Default temperature (0.0 for deterministic)
            seed: Random seed for reproducibility
        """
        self.client = client
        self.use_cache = use_cache
        self.default_temperature = default_temperature
        self.seed = seed
        self.cache = ResponseCache()

        if seed is not None:
            set_seed(seed)

    def generate(
        self,
        input_text: str,
        instructions: str = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """
        Generate with deterministic defaults and caching.

        Args:
            input_text: Input text
            instructions: System instructions
            temperature: Temperature (uses default if None)
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        # Use default temperature if not specified
        if temperature is None:
            temperature = self.default_temperature

        # Add seed if not already present
        if self.seed is not None and "seed" not in kwargs:
            kwargs["seed"] = self.seed

        # Check cache
        if self.use_cache:
            cache_key = cache_key_from_params(
                f"{input_text}|{instructions or ''}",
                model=self.client.model,
                temperature=temperature,
                **kwargs
            )

            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Generate
        response = self.client.generate(
            input_text=input_text,
            instructions=instructions,
            temperature=temperature,
            **kwargs
        )

        # Get output text
        output = self.client.get_output_text(response)

        # Cache result
        if self.use_cache:
            self.cache.set(cache_key, output)

        return output

    def clear_cache(self):
        """Clear response cache."""
        self.cache.clear()

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()


# Example usage
if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    print("Random seed set to 42")

    # Get deterministic parameters
    params = get_deterministic_params(temperature=0.0, seed=42)
    print(f"\nDeterministic parameters: {params}")

    # Create cache key
    cache_key = cache_key_from_params(
        prompt="Translate to French: Hello",
        model="gpt-5",
        temperature=0.0
    )
    print(f"\nCache key: {cache_key}")

    # Test response cache
    cache = ResponseCache()

    # Simulate caching
    key1 = "prompt1_hash"
    cache.set(key1, "Response 1")

    # Hit
    result = cache.get(key1)
    print(f"\nCache hit: {result}")

    # Miss
    result = cache.get("prompt2_hash")
    print(f"Cache miss: {result}")

    # Stats
    stats = cache.stats()
    print(f"\nCache stats: {stats}")

    # Example with DeterministicClient
    print("\n" + "="*60)
    print("DeterministicClient Example")
    print("="*60)

    # This would work with actual LLMClient:
    # from llm.client import LLMClient
    # client = LLMClient()
    # det_client = DeterministicClient(client)
    # response = det_client.generate("Test prompt")
    # print(f"Response: {response}")

    print("""
    Usage with actual client:

    from llm.client import LLMClient
    from shared.repro import DeterministicClient

    client = LLMClient()
    det_client = DeterministicClient(client, use_cache=True, seed=42)

    # First call - hits API
    response1 = det_client.generate("Translate: Hello")

    # Second call - returns cached result (no API call)
    response2 = det_client.generate("Translate: Hello")

    assert response1 == response2  # Should be identical

    # Check cache stats
    print(det_client.cache_stats())
    # {"size": 1, "hits": 1, "misses": 1, "hit_rate": 0.5}
    """)
