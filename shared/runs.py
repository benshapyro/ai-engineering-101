"""
Run Tracking System

This module provides utilities for logging and tracking LLM API calls for
reproducibility, debugging, and cost analysis.

Features:
- Log prompts, responses, and metadata to JSONL files
- Hash prompts for deduplication and caching
- Track token usage and costs
- Enable reproducibility with deterministic settings

Usage:
    from shared.runs import log_run, hash_prompt, load_runs

    # Log an API call
    response = client.chat.completions.create(...)
    log_run(
        prompt="What is the capital of France?",
        response=response.choices[0].message.content,
        model="gpt-4",
        temperature=0.7,
        metadata={"exercise": "geography_quiz"}
    )

    # Hash a prompt for caching
    prompt_hash = hash_prompt("What is the capital of France?")

    # Load previous runs
    runs = load_runs(limit=10)
"""

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


# Default runs directory
RUNS_DIR = Path(__file__).parent.parent / "runs"


@dataclass
class RunRecord:
    """
    Record of a single LLM API call.

    Attributes:
        timestamp: ISO 8601 timestamp of when the run occurred
        prompt: The prompt sent to the LLM
        response: The response received from the LLM
        model: Model identifier (e.g., "gpt-4", "claude-sonnet-4-5")
        temperature: Temperature parameter used
        max_tokens: Maximum tokens parameter (if specified)
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used (prompt + completion)
        cost_usd: Estimated cost in USD (if calculable)
        prompt_hash: SHA-256 hash of the prompt for deduplication
        metadata: Additional metadata (exercise name, tags, etc.)
    """
    timestamp: str
    prompt: str
    response: str
    model: str
    temperature: float
    max_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    prompt_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def hash_prompt(prompt: str) -> str:
    """
    Generate deterministic hash of a prompt for deduplication and caching.

    Args:
        prompt: The prompt text to hash

    Returns:
        str: Hexadecimal SHA-256 hash of the prompt

    Examples:
        >>> hash_prompt("What is 2+2?")
        'a3c65c2974270fd093ee8a9bf8ae7d0b5f28d4d1d9de1c6f3c6c6c0d1c6f3c6c'
        >>> hash_prompt("What is 2+2?") == hash_prompt("What is 2+2?")
        True
    """
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()


def log_run(
    prompt: str,
    response: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    runs_dir: Optional[Path] = None
) -> RunRecord:
    """
    Log an LLM API call to a JSONL file for reproducibility tracking.

    Args:
        prompt: The prompt sent to the LLM
        response: The response received from the LLM
        model: Model identifier
        temperature: Temperature parameter used
        max_tokens: Maximum tokens parameter (if specified)
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used
        cost_usd: Estimated cost in USD
        metadata: Additional metadata (exercise name, tags, etc.)
        runs_dir: Directory to save runs (defaults to ./runs/)

    Returns:
        RunRecord: The logged run record

    Examples:
        >>> from openai import OpenAI
        >>> client = OpenAI()
        >>> response = client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "What is 2+2?"}],
        ...     temperature=0
        ... )
        >>> log_run(
        ...     prompt="What is 2+2?",
        ...     response=response.choices[0].message.content,
        ...     model="gpt-4",
        ...     temperature=0,
        ...     prompt_tokens=response.usage.prompt_tokens,
        ...     completion_tokens=response.usage.completion_tokens,
        ...     total_tokens=response.usage.total_tokens,
        ...     metadata={"exercise": "math_quiz"}
        ... )
    """
    # Use default runs directory if not specified
    if runs_dir is None:
        runs_dir = RUNS_DIR

    # Create runs directory if it doesn't exist
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create run record
    record = RunRecord(
        timestamp=datetime.utcnow().isoformat() + "Z",
        prompt=prompt,
        response=response,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        prompt_hash=hash_prompt(prompt),
        metadata=metadata or {}
    )

    # Determine log file name (one file per day for organization)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    log_file = runs_dir / f"runs_{date_str}.jsonl"

    # Append to JSONL file
    with open(log_file, 'a') as f:
        f.write(json.dumps(record.to_dict()) + '\n')

    return record


def load_runs(
    runs_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    date: Optional[str] = None
) -> List[RunRecord]:
    """
    Load previous runs from JSONL files.

    Args:
        runs_dir: Directory containing run logs (defaults to ./runs/)
        limit: Maximum number of runs to load (most recent first)
        date: Specific date to load (YYYY-MM-DD format). If None, loads all dates.

    Returns:
        List[RunRecord]: List of run records, most recent first

    Examples:
        >>> # Load last 10 runs
        >>> recent_runs = load_runs(limit=10)
        >>> # Load all runs from a specific date
        >>> runs = load_runs(date="2025-09-15")
    """
    # Use default runs directory if not specified
    if runs_dir is None:
        runs_dir = RUNS_DIR

    if not runs_dir.exists():
        return []

    runs = []

    # Determine which files to read
    if date:
        log_files = [runs_dir / f"runs_{date}.jsonl"]
    else:
        log_files = sorted(runs_dir.glob("runs_*.jsonl"), reverse=True)

    # Read JSONL files
    for log_file in log_files:
        if not log_file.exists():
            continue

        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    runs.append(RunRecord(**data))

        # Stop if we've reached the limit
        if limit and len(runs) >= limit:
            break

    # Return most recent first
    runs.sort(key=lambda r: r.timestamp, reverse=True)

    if limit:
        return runs[:limit]
    return runs


def get_run_stats(
    runs_dir: Optional[Path] = None,
    date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get statistics about logged runs.

    Args:
        runs_dir: Directory containing run logs (defaults to ./runs/)
        date: Specific date to analyze (YYYY-MM-DD). If None, analyzes all runs.

    Returns:
        Dict containing statistics:
        - total_runs: Total number of runs
        - total_tokens: Total tokens used
        - total_cost_usd: Total estimated cost
        - models: Count of runs per model
        - avg_temperature: Average temperature used

    Examples:
        >>> stats = get_run_stats()
        >>> print(f"Total runs: {stats['total_runs']}")
        >>> print(f"Total cost: ${stats['total_cost_usd']:.2f}")
    """
    runs = load_runs(runs_dir=runs_dir, date=date)

    if not runs:
        return {
            "total_runs": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "models": {},
            "avg_temperature": 0.0
        }

    total_tokens = sum(r.total_tokens or 0 for r in runs)
    total_cost = sum(r.cost_usd or 0.0 for r in runs)

    models = {}
    for run in runs:
        models[run.model] = models.get(run.model, 0) + 1

    temperatures = [r.temperature for r in runs if r.temperature is not None]
    avg_temp = sum(temperatures) / len(temperatures) if temperatures else 0.0

    return {
        "total_runs": len(runs),
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "models": models,
        "avg_temperature": avg_temp
    }


def find_duplicate_prompts(
    runs_dir: Optional[Path] = None
) -> Dict[str, List[RunRecord]]:
    """
    Find duplicate prompts across all runs for cache analysis.

    Args:
        runs_dir: Directory containing run logs (defaults to ./runs/)

    Returns:
        Dict mapping prompt hashes to list of runs with that prompt

    Examples:
        >>> duplicates = find_duplicate_prompts()
        >>> for prompt_hash, runs in duplicates.items():
        ...     if len(runs) > 1:
        ...         print(f"Prompt hash {prompt_hash[:8]}... used {len(runs)} times")
    """
    runs = load_runs(runs_dir=runs_dir)

    prompt_map: Dict[str, List[RunRecord]] = {}
    for run in runs:
        if run.prompt_hash:
            if run.prompt_hash not in prompt_map:
                prompt_map[run.prompt_hash] = []
            prompt_map[run.prompt_hash].append(run)

    # Return only duplicates
    return {h: runs for h, runs in prompt_map.items() if len(runs) > 1}
