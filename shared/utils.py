"""
Shared utility functions for the Prompt Engineering curriculum.
"""

import os
import json
import time
import tiktoken
import yaml
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import openai
from anthropic import Anthropic

# Load environment variables
load_dotenv()


def load_model_config(config_path: str = "config/models.yml") -> dict:
    """
    Load model configuration from YAML file.

    Args:
        config_path: Path to config file (defaults to config/models.yml)

    Returns:
        Configuration dictionary, or empty dict if file not found
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


class LLMClient:
    """
    Unified client for OpenAI and Anthropic models.

    Sampling Parameter Conventions:
    --------------------------------
    This client follows standardized parameter naming and defaults:

    temperature (float, default=0.7):
        Controls randomness. Range: 0.0 to 2.0
        - 0.0-0.3: Deterministic, factual tasks, code generation
        - 0.7-1.0: Balanced creativity and consistency (recommended default)
        - 1.0-2.0: Creative writing, brainstorming, diverse outputs

    max_tokens (int, default=1000):
        Maximum tokens in response. Set based on expected output length.
        - Short answers: 100-500
        - Paragraphs: 500-2000
        - Long form: 2000-4000

    top_p (float, default=1.0):
        Nucleus sampling. Use temperature OR top_p, not both.
        - 1.0: No filtering (default)
        - 0.9: Filter low-probability tokens
        - Lower values (0.1-0.5): More focused outputs

    Best Practices:
    - Use temperature=0.0 for reproducible testing
    - Use temperature=0.7 as default for balanced outputs
    - Don't adjust both temperature and top_p simultaneously
    - Set max_tokens based on actual needs to control costs
    """

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
            self.default_model = os.getenv("OPENAI_MODEL", "gpt-5")
        elif self.provider == "anthropic":
            self.client = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            self.default_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
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

            # GPT-5 and o1 models have different parameter requirements
            # - Use max_completion_tokens instead of max_tokens
            # - Only support temperature=1 (default)
            uses_new_api = any(m in model for m in ["gpt-5", "o1", "o3"])

            api_params = {
                "model": model,
                "messages": messages,
            }

            # Handle temperature - GPT-5/o1 only support temperature=1
            if not uses_new_api or temperature == 1.0:
                api_params["temperature"] = temperature
            # Otherwise, skip temperature parameter (uses default of 1)

            # Handle token limits
            if uses_new_api:
                api_params["max_completion_tokens"] = max_tokens
            else:
                api_params["max_tokens"] = max_tokens

            api_params.update(kwargs)

            response = self.client.chat.completions.create(**api_params)
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

            # GPT-5 and o1 models have different parameter requirements
            uses_new_api = any(m in model for m in ["gpt-5", "o1", "o3"])

            api_params = {
                "model": model,
                "messages": messages,
                "stream": True,
            }

            # Handle temperature - GPT-5/o1 only support temperature=1
            if not uses_new_api or temperature == 1.0:
                api_params["temperature"] = temperature

            # Handle token limits
            if uses_new_api:
                api_params["max_completion_tokens"] = max_tokens
            else:
                api_params["max_tokens"] = max_tokens

            api_params.update(kwargs)

            stream = self.client.chat.completions.create(**api_params)

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


def get_encoding(model: str):
    """
    Get tiktoken encoding with robust fallback for unknown models.

    Falls back through:
    1. tiktoken.encoding_for_model(model) - standard lookup
    2. o200k_base - for newer models (GPT-5, o1, etc.)
    3. cl100k_base - universal fallback (GPT-4, GPT-3.5)

    Args:
        model: Model name

    Returns:
        tiktoken Encoding object
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Try o200k_base for newer models
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            # Ultimate fallback to cl100k_base
            return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-5") -> int:
    """
    Count tokens in a text string with robust encoding fallback.

    Args:
        text: Text to count tokens for
        model: Model to use for encoding (defaults to gpt-5)

    Returns:
        Number of tokens

    Example:
        tokens = count_tokens("Hello world", "gpt-5")
        # Works even with unknown models
        tokens = count_tokens("Hello world", "future-model-xyz")
    """
    encoding = get_encoding(model)
    return len(encoding.encode(text))


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-5",
    config: dict = None
) -> Dict[str, float]:
    """
    Estimate API cost for a completion.

    Pricing is loaded from (in priority order):
    1. Provided config parameter
    2. config/models.yml file
    3. Environment variables (COST_PER_1M_INPUT, COST_PER_1M_OUTPUT)
    4. Default illustrative pricing (with warning)

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name
        config: Optional config dict (loaded from file if None)

    Returns:
        Dictionary with cost breakdown

    Example:
        # Using config file
        cost = estimate_cost(1000, 500, "gpt-5")

        # Using environment variables
        os.environ["COST_PER_1M_INPUT"] = "5.00"
        os.environ["COST_PER_1M_OUTPUT"] = "15.00"
        cost = estimate_cost(1000, 500)
    """
    # Try to load config if not provided
    if config is None:
        config = load_model_config()

    # Check environment variables
    env_input = os.getenv("COST_PER_1M_INPUT")
    env_output = os.getenv("COST_PER_1M_OUTPUT")

    # Determine pricing source
    if env_input and env_output:
        # Use environment variables (highest priority)
        input_cost = (input_tokens / 1_000_000) * float(env_input)
        output_cost = (output_tokens / 1_000_000) * float(env_output)
    elif config:
        # Try to find model in config
        # Handle model names like "gpt-5" or "openai/gpt-5" or "claude-sonnet-4-5-20250929"
        provider = None
        model_name = model

        # Check if model has provider prefix
        if "/" in model:
            provider, model_name = model.split("/", 1)
        else:
            # Try to guess provider
            if model.startswith("gpt") or model.startswith("o1"):
                provider = "openai"
            elif model.startswith("claude"):
                provider = "anthropic"

        # Search for model in config
        found = False
        if provider and provider in config:
            if model_name in config[provider]:
                model_config = config[provider][model_name]
                input_cost = (input_tokens / 1_000_000) * model_config["input_per_1m"]
                output_cost = (output_tokens / 1_000_000) * model_config["output_per_1m"]
                found = True

        if not found:
            # Model not in config, fall back to defaults
            print(f"⚠️  Model '{model}' not found in config/models.yml")
            print(f"⚠️  Using illustrative pricing. Set COST_PER_1M_INPUT/OUTPUT env vars or update config.")
            input_cost, output_cost = _get_default_pricing(input_tokens, output_tokens, model)
    else:
        # No config or env vars, use defaults with warning
        print(f"⚠️  No pricing config found (config/models.yml missing)")
        print(f"⚠️  Costs are illustrative only. Set COST_PER_1M_INPUT/OUTPUT env vars or create config/models.yml")
        input_cost, output_cost = _get_default_pricing(input_tokens, output_tokens, model)

    return {
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(input_cost + output_cost, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model
    }


def _get_default_pricing(input_tokens: int, output_tokens: int, model: str) -> tuple:
    """
    Get default illustrative pricing for a model.

    These are example prices from September 2025 and should not be used
    for actual cost calculations.

    Returns:
        Tuple of (input_cost, output_cost)
    """
    # Default pricing (illustrative only, as of September 2025)
    default_pricing = {
        "gpt-5": {"input": 5.00, "output": 15.00},
        "gpt-5-mini": {"input": 0.30, "output": 1.20},
        "gpt-5-nano": {"input": 0.10, "output": 0.40},
        "gpt-5-codex": {"input": 6.00, "output": 18.00},
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    }

    if model in default_pricing:
        pricing = default_pricing[model]
    else:
        # Unknown model, use GPT-5 pricing as default
        pricing = default_pricing["gpt-5"]

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return (input_cost, output_cost)


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
    cost = estimate_cost(100, 50, "gpt-5")
    print(f"Estimated cost: ${cost['total_cost']}")