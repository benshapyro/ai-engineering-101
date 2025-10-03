"""
Prompt chaining utilities for multi-step workflows.

This module provides tools for building and executing prompt chains,
where the output of one LLM call becomes the input to the next.
"""

from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import json
import time


@dataclass
class ChainStep:
    """
    Represents a single step in a prompt chain.

    Attributes:
        name: Step identifier
        prompt_template: Template with {placeholders}
        instructions: Optional system instructions
        temperature: Temperature for this step
        max_tokens: Max tokens for this step
        output_key: Key to store output (default: step name)
        extract_fn: Optional function to extract/transform output
    """
    name: str
    prompt_template: str
    instructions: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    output_key: Optional[str] = None
    extract_fn: Optional[Callable[[str], Any]] = None

    def __post_init__(self):
        """Set output_key to name if not provided."""
        if self.output_key is None:
            self.output_key = self.name


class PromptChain:
    """
    Execute multi-step prompt chains with state management.

    Example:
        from llm.client import LLMClient

        client = LLMClient()
        chain = PromptChain(client)

        # Define steps
        chain.add_step(
            "summarize",
            "Summarize this text in one sentence: {text}"
        )
        chain.add_step(
            "sentiment",
            "What is the sentiment of this: {summarize}",
            temperature=0.0
        )

        # Execute
        result = chain.run({"text": "Long article text..."})
        print(result["sentiment"])
    """

    def __init__(self, client, verbose: bool = False):
        """
        Initialize prompt chain.

        Args:
            client: LLMClient instance
            verbose: Print step execution details
        """
        self.client = client
        self.verbose = verbose
        self.steps: List[ChainStep] = []
        self.state: Dict[str, Any] = {}
        self.execution_log: List[Dict[str, Any]] = []

    def add_step(
        self,
        name: str,
        prompt_template: str,
        instructions: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        output_key: str = None,
        extract_fn: Callable = None
    ):
        """
        Add a step to the chain.

        Args:
            name: Step identifier
            prompt_template: Template with {placeholders}
            instructions: Optional system instructions
            temperature: Temperature for this step
            max_tokens: Max tokens for this step
            output_key: Key to store output
            extract_fn: Optional extraction function

        Example:
            chain.add_step(
                "extract_topic",
                "What is the main topic of: {text}",
                temperature=0.0,
                max_tokens=50
            )
        """
        step = ChainStep(
            name=name,
            prompt_template=prompt_template,
            instructions=instructions,
            temperature=temperature,
            max_tokens=max_tokens,
            output_key=output_key,
            extract_fn=extract_fn
        )
        self.steps.append(step)

    def run(self, initial_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the prompt chain.

        Args:
            initial_state: Initial variables for templates

        Returns:
            Final state dictionary with all outputs

        Example:
            result = chain.run({"text": "Input text"})
            # result contains outputs from all steps
        """
        # Initialize state
        self.state = initial_state or {}
        self.execution_log = []

        # Execute each step
        for i, step in enumerate(self.steps):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Step {i+1}/{len(self.steps)}: {step.name}")
                print(f"{'='*60}")

            # Format prompt with current state
            try:
                prompt = step.prompt_template.format(**self.state)
            except KeyError as e:
                raise ValueError(
                    f"Step '{step.name}': Missing variable {e} in state. "
                    f"Available: {list(self.state.keys())}"
                )

            if self.verbose:
                print(f"Prompt: {prompt[:100]}...")

            # Execute step
            start_time = time.time()

            response = self.client.generate(
                input_text=prompt,
                instructions=step.instructions,
                temperature=step.temperature,
                max_tokens=step.max_tokens
            )

            # Get output
            output = self.client.get_output_text(response)

            # Apply extraction function if provided
            if step.extract_fn:
                output = step.extract_fn(output)

            execution_time = time.time() - start_time

            # Store in state
            self.state[step.output_key] = output

            # Log execution
            log_entry = {
                "step": step.name,
                "prompt": prompt,
                "output": output,
                "temperature": step.temperature,
                "execution_time": execution_time
            }
            self.execution_log.append(log_entry)

            if self.verbose:
                print(f"Output: {str(output)[:200]}...")
                print(f"Time: {execution_time:.2f}s")

        return self.state

    def get_log(self) -> List[Dict[str, Any]]:
        """
        Get execution log.

        Returns:
            List of log entries with prompts, outputs, and timings
        """
        return self.execution_log

    def clear(self):
        """Clear steps and state."""
        self.steps = []
        self.state = {}
        self.execution_log = []


def sequential_chain(
    client,
    steps: List[tuple],
    initial_input: str,
    verbose: bool = False
) -> List[str]:
    """
    Simple sequential chain helper.

    Each step receives the output of the previous step.

    Args:
        client: LLMClient instance
        steps: List of (prompt_template, temperature) tuples
        initial_input: First input
        verbose: Print execution details

    Returns:
        List of outputs (one per step)

    Example:
        outputs = sequential_chain(
            client,
            [
                ("Summarize: {input}", 0.7),
                ("Translate to French: {input}", 0.3),
                ("Make it more formal: {input}", 0.5)
            ],
            initial_input="Long text to process..."
        )
    """
    outputs = []
    current_input = initial_input

    for i, (template, temp) in enumerate(steps):
        if verbose:
            print(f"\nStep {i+1}/{len(steps)}")

        prompt = template.format(input=current_input)

        output = client.generate(
            input_text=prompt,
            temperature=temp
        )
        output_text = client.get_output_text(output)

        outputs.append(output_text)
        current_input = output_text

        if verbose:
            print(f"Output: {output_text[:100]}...")

    return outputs


def parallel_chain(
    client,
    prompts: List[str],
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> List[str]:
    """
    Execute multiple prompts in parallel (conceptually - still sequential in implementation).

    Useful when multiple independent analyses are needed.

    Args:
        client: LLMClient instance
        prompts: List of prompts to execute
        temperature: Temperature for all prompts
        max_tokens: Max tokens for all prompts

    Returns:
        List of outputs (one per prompt)

    Example:
        results = parallel_chain(
            client,
            [
                "Summarize this: " + text,
                "Extract key facts from: " + text,
                "Identify sentiment of: " + text
            ]
        )
    """
    outputs = []

    for prompt in prompts:
        output = client.generate(
            input_text=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        outputs.append(client.get_output_text(output))

    return outputs


def map_reduce_chain(
    client,
    items: List[str],
    map_template: str,
    reduce_template: str,
    map_temperature: float = 0.7,
    reduce_temperature: float = 0.7
) -> str:
    """
    Map-reduce pattern for processing lists.

    Maps operation over items, then reduces results to single output.

    Args:
        client: LLMClient instance
        items: List of items to process
        map_template: Template for map step (use {item})
        reduce_template: Template for reduce step (use {results})
        map_temperature: Temperature for map steps
        reduce_temperature: Temperature for reduce step

    Returns:
        Final reduced output

    Example:
        # Summarize multiple articles
        summary = map_reduce_chain(
            client,
            articles,
            map_template="Summarize this article: {item}",
            reduce_template="Synthesize these summaries into one: {results}"
        )
    """
    # Map: Process each item
    map_outputs = []
    for item in items:
        prompt = map_template.format(item=item)
        output = client.generate(
            input_text=prompt,
            temperature=map_temperature
        )
        map_outputs.append(client.get_output_text(output))

    # Reduce: Combine results
    combined = "\n\n".join(
        [f"Item {i+1}: {out}" for i, out in enumerate(map_outputs)]
    )

    reduce_prompt = reduce_template.format(results=combined)
    final_output = client.generate(
        input_text=reduce_prompt,
        temperature=reduce_temperature
    )

    return client.get_output_text(final_output)


# Example usage
if __name__ == "__main__":
    print("Prompt Chain Examples")
    print("=" * 60)

    # Example 1: Basic chain
    print("\nExample 1: Document analysis chain")
    print("-" * 60)

    example_chain = """
    from llm.client import LLMClient
    from shared.chains import PromptChain

    client = LLMClient()
    chain = PromptChain(client, verbose=True)

    # Step 1: Summarize
    chain.add_step(
        "summary",
        "Summarize this article in 2 sentences: {article}",
        temperature=0.7,
        max_tokens=100
    )

    # Step 2: Extract key points
    chain.add_step(
        "key_points",
        "List 3 key points from: {summary}",
        temperature=0.5,
        max_tokens=200
    )

    # Step 3: Generate questions
    chain.add_step(
        "questions",
        "Generate 3 questions about: {key_points}",
        temperature=0.8,
        max_tokens=200
    )

    # Run chain
    result = chain.run({"article": "Long article text..."})

    print(result["summary"])
    print(result["key_points"])
    print(result["questions"])
    """
    print(example_chain)

    # Example 2: Sequential chain
    print("\nExample 2: Sequential processing")
    print("-" * 60)

    sequential_example = """
    from shared.chains import sequential_chain

    outputs = sequential_chain(
        client,
        [
            ("Summarize this: {input}", 0.7),
            ("Make it more concise: {input}", 0.5),
            ("Add a title: {input}", 0.8)
        ],
        initial_input="Long text...",
        verbose=True
    )

    final_output = outputs[-1]
    """
    print(sequential_example)

    # Example 3: Map-reduce
    print("\nExample 3: Map-reduce pattern")
    print("-" * 60)

    mapreduce_example = """
    from shared.chains import map_reduce_chain

    articles = [
        "Article 1 text...",
        "Article 2 text...",
        "Article 3 text..."
    ]

    overall_summary = map_reduce_chain(
        client,
        articles,
        map_template="Summarize: {item}",
        reduce_template="Combine these summaries: {results}"
    )

    print(overall_summary)
    """
    print(mapreduce_example)

    # Example 4: Conditional chaining
    print("\nExample 4: Conditional logic in chains")
    print("-" * 60)

    conditional_example = """
    def extract_sentiment(output: str) -> str:
        # Extract sentiment from output
        if "positive" in output.lower():
            return "positive"
        elif "negative" in output.lower():
            return "negative"
        return "neutral"

    chain = PromptChain(client)

    chain.add_step(
        "sentiment",
        "Classify sentiment: {text}",
        temperature=0.0,
        extract_fn=extract_sentiment
    )

    chain.add_step(
        "response",
        "Generate a {sentiment} response to: {text}",
        temperature=0.8
    )

    result = chain.run({"text": "Customer feedback..."})
    """
    print(conditional_example)
