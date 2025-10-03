"""
Solutions Access Control

This module provides utilities for controlling access to exercise solutions.
Solutions are only accessible when ALLOW_SOLUTIONS=1 is set in the environment.

Usage:
    from shared.solutions import solutions_enabled, require_solutions

    if solutions_enabled():
        # Show solution
        pass
    else:
        # Hide solution
        pass

    # Or use as decorator
    @require_solutions
    def show_solution():
        return "The answer is 42"
"""

import os
import functools
from typing import Callable, Any


def solutions_enabled() -> bool:
    """
    Check if solutions are accessible.

    Returns:
        bool: True if ALLOW_SOLUTIONS=1 is set in environment

    Examples:
        >>> os.environ["ALLOW_SOLUTIONS"] = "1"
        >>> solutions_enabled()
        True
        >>> os.environ["ALLOW_SOLUTIONS"] = "0"
        >>> solutions_enabled()
        False
    """
    return os.getenv("ALLOW_SOLUTIONS", "0") == "1"


def require_solutions(func: Callable) -> Callable:
    """
    Decorator to require ALLOW_SOLUTIONS=1 for function execution.

    Args:
        func: Function to protect with access control

    Returns:
        Wrapped function that checks solutions_enabled()

    Raises:
        PermissionError: If solutions are not enabled

    Examples:
        >>> @require_solutions
        ... def get_answer():
        ...     return 42
        >>> get_answer()  # Raises PermissionError if ALLOW_SOLUTIONS != "1"
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not solutions_enabled():
            raise PermissionError(
                f"Access to {func.__name__} requires ALLOW_SOLUTIONS=1 environment variable.\n"
                "Solutions are hidden by default to encourage independent learning.\n"
                "Set ALLOW_SOLUTIONS=1 only after attempting exercises yourself."
            )
        return func(*args, **kwargs)
    return wrapper


def load_solution_file(filepath: str) -> str:
    """
    Load a solution file with access control.

    Args:
        filepath: Path to solution file

    Returns:
        str: Contents of solution file

    Raises:
        PermissionError: If solutions are not enabled
        FileNotFoundError: If solution file doesn't exist

    Examples:
        >>> content = load_solution_file("01-fundamentals/solutions/exercise1.py")
    """
    if not solutions_enabled():
        raise PermissionError(
            f"Access to solution file '{filepath}' requires ALLOW_SOLUTIONS=1.\n"
            "Solutions are hidden by default to encourage independent learning.\n"
            "Set ALLOW_SOLUTIONS=1 only after attempting exercises yourself."
        )

    with open(filepath, 'r') as f:
        return f.read()


def get_solution_message() -> str:
    """
    Get message to display when solutions are not accessible.

    Returns:
        str: Helpful message explaining how to enable solutions
    """
    return (
        "Solutions are currently hidden to encourage independent learning.\n"
        "To access solutions after attempting the exercises:\n"
        "1. Set the environment variable: export ALLOW_SOLUTIONS=1\n"
        "2. Or add ALLOW_SOLUTIONS=1 to your .env file\n"
        "3. Re-run your code\n"
        "\n"
        "Remember: You'll learn more by attempting exercises first!"
    )
