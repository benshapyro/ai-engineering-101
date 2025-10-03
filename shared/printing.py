"""
UX Printing Utilities

This module provides rich CLI formatting utilities for improved user experience
when running examples and displaying RAG results.

Uses the `rich` library for:
- Colored output
- Formatted tables
- Syntax highlighting
- Progress indicators
- Panels and boxes
"""

from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn


# Global console instance
console = Console()


def print_answer(
    answer: str,
    title: str = "Answer",
    style: str = "bold cyan"
) -> None:
    """
    Print an LLM answer in a formatted panel.

    Args:
        answer: The answer text to display
        title: Panel title (default: "Answer")
        style: Panel border style (default: "bold cyan")

    Example:
        >>> print_answer("List comprehensions are...", "Python Expert Says")
    """
    panel = Panel(
        answer,
        title=f"[{style}]{title}[/{style}]",
        border_style=style,
        box=box.ROUNDED
    )
    console.print(panel)


def print_sources(
    sources: List[Dict[str, Any]],
    title: str = "Sources",
    show_scores: bool = True
) -> None:
    """
    Print source citations in a formatted table.

    Args:
        sources: List of source dictionaries with keys:
                 - doc_id: Document identifier
                 - title: Document title
                 - score: Relevance score (optional)
        title: Table title (default: "Sources")
        show_scores: Whether to display relevance scores

    Example:
        >>> sources = [
        ...     {"doc_id": "doc_001", "title": "List Comprehensions", "score": 0.95},
        ...     {"doc_id": "doc_002", "title": "Generators", "score": 0.87}
        ... ]
        >>> print_sources(sources)
    """
    if not sources:
        console.print("[dim]No sources available[/dim]")
        return

    table = Table(title=title, box=box.SIMPLE, show_header=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("Document ID", style="magenta")
    table.add_column("Title", style="green")

    if show_scores:
        table.add_column("Score", justify="right", style="yellow")

    for i, source in enumerate(sources, 1):
        row = [
            str(i),
            source.get("doc_id", "N/A"),
            source.get("title", "Untitled")
        ]
        if show_scores and "score" in source:
            row.append(f"{source['score']:.2f}")
        table.add_row(*row)

    console.print(table)


def print_metrics(
    metrics: Dict[str, Any],
    title: str = "Metrics",
    precision: int = 4
) -> None:
    """
    Print metrics in a formatted table.

    Args:
        metrics: Dictionary of metric name -> value
        title: Table title (default: "Metrics")
        precision: Decimal places for float values

    Example:
        >>> metrics = {
        ...     "accuracy": 0.87,
        ...     "latency_ms": 245.3,
        ...     "cost_usd": 0.0023
        ... }
        >>> print_metrics(metrics)
    """
    table = Table(title=title, box=box.SIMPLE, show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    for key, value in metrics.items():
        # Format value based on type
        if isinstance(value, float):
            formatted_value = f"{value:.{precision}f}"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)

        # Add unit hints for common metrics
        if "cost" in key.lower() and isinstance(value, (int, float)):
            formatted_value = f"${formatted_value}"
        elif "ms" in key.lower() or "latency" in key.lower():
            formatted_value = f"{formatted_value}ms"
        elif "percent" in key.lower() or "accuracy" in key.lower():
            if isinstance(value, float) and value <= 1:
                formatted_value = f"{value * 100:.2f}%"

        table.add_row(key.replace("_", " ").title(), formatted_value)

    console.print(table)


def print_code(
    code: str,
    language: str = "python",
    theme: str = "monokai",
    line_numbers: bool = True
) -> None:
    """
    Print code with syntax highlighting.

    Args:
        code: Code string to display
        language: Programming language for syntax highlighting
        theme: Color theme (monokai, github, etc.)
        line_numbers: Whether to show line numbers

    Example:
        >>> code = '''
        ... def fibonacci(n):
        ...     if n <= 1:
        ...         return n
        ...     return fibonacci(n-1) + fibonacci(n-2)
        ... '''
        >>> print_code(code, language="python")
    """
    syntax = Syntax(
        code,
        language,
        theme=theme,
        line_numbers=line_numbers,
        word_wrap=True
    )
    console.print(syntax)


def print_markdown(
    markdown_text: str,
    style: Optional[str] = None
) -> None:
    """
    Print markdown-formatted text with rich rendering.

    Args:
        markdown_text: Markdown string
        style: Optional style override

    Example:
        >>> md = '''
        ... # Hello
        ... This is **bold** and this is *italic*.
        ... - Item 1
        ... - Item 2
        ... '''
        >>> print_markdown(md)
    """
    md = Markdown(markdown_text, style=style)
    console.print(md)


def print_cost_warning(
    estimated_cost: float,
    threshold: float = 0.10
) -> None:
    """
    Print a cost warning if estimated cost exceeds threshold.

    Args:
        estimated_cost: Estimated cost in USD
        threshold: Warning threshold (default: $0.10)

    Example:
        >>> print_cost_warning(0.15)  # Shows warning
        >>> print_cost_warning(0.05)  # No warning
    """
    if estimated_cost > threshold:
        warning_panel = Panel(
            f"[bold yellow]⚠️  Estimated cost: ${estimated_cost:.4f}[/bold yellow]\n"
            f"This exceeds the threshold of ${threshold:.2f}.\n"
            f"Consider using a smaller model or reducing max_tokens.",
            title="[bold red]Cost Warning[/bold red]",
            border_style="yellow",
            box=box.HEAVY
        )
        console.print(warning_panel)


def print_rag_result(
    query: str,
    answer: str,
    sources: List[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]] = None,
    show_query: bool = True
) -> None:
    """
    Print a complete RAG result with query, answer, sources, and metrics.

    Args:
        query: The original query
        answer: The generated answer
        sources: List of source documents
        metrics: Optional metrics dictionary
        show_query: Whether to display the query

    Example:
        >>> print_rag_result(
        ...     query="How do list comprehensions work?",
        ...     answer="List comprehensions provide...",
        ...     sources=[{"doc_id": "001", "title": "Lists", "score": 0.95}],
        ...     metrics={"cost_usd": 0.0023, "latency_ms": 245}
        ... )
    """
    if show_query:
        console.print("\n[bold blue]Query:[/bold blue]", query, "\n")

    print_answer(answer)

    console.print()  # Blank line

    if sources:
        print_sources(sources)
    else:
        console.print("[dim]No sources cited[/dim]")

    if metrics:
        console.print()  # Blank line
        print_metrics(metrics, title="Performance Metrics")


def print_progress(
    iterable,
    description: str = "Processing...",
    total: Optional[int] = None
) -> Any:
    """
    Display a progress bar for iterables.

    Args:
        iterable: Items to iterate over
        description: Task description
        total: Total items (auto-detected if None)

    Yields:
        Items from iterable

    Example:
        >>> for item in print_progress(range(100), "Indexing documents"):
        ...     process(item)
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(description, total=total)

        for item in iterable:
            yield item
            progress.update(task, advance=1)


def print_comparison_table(
    comparisons: List[Dict[str, Any]],
    title: str = "Comparison",
    metric_columns: Optional[List[str]] = None
) -> None:
    """
    Print a comparison table for multiple experiments or models.

    Args:
        comparisons: List of dictionaries with comparison data
        title: Table title
        metric_columns: Specific columns to display (None = all)

    Example:
        >>> comparisons = [
        ...     {"name": "GPT-5", "accuracy": 0.87, "cost": 0.005},
        ...     {"name": "GPT-5-mini", "accuracy": 0.82, "cost": 0.001}
        ... ]
        >>> print_comparison_table(comparisons, title="Model Comparison")
    """
    if not comparisons:
        console.print("[dim]No comparisons available[/dim]")
        return

    table = Table(title=title, box=box.ROUNDED, show_header=True)

    # Get all keys from first comparison
    all_keys = list(comparisons[0].keys())

    # Filter columns if specified
    if metric_columns:
        columns = [k for k in all_keys if k in metric_columns]
    else:
        columns = all_keys

    # Add columns
    for col in columns:
        table.add_column(col.replace("_", " ").title(), style="cyan")

    # Add rows
    for comparison in comparisons:
        row = []
        for col in columns:
            value = comparison.get(col, "N/A")
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        table.add_row(*row)

    console.print(table)


def print_header(
    text: str,
    style: str = "bold magenta",
    char: str = "="
) -> None:
    """
    Print a formatted section header.

    Args:
        text: Header text
        style: Text style
        char: Character for line (default: "=")

    Example:
        >>> print_header("Module 01: Fundamentals")
    """
    console.print()
    console.print(f"[{style}]{char * 60}[/{style}]")
    console.print(f"[{style}]{text.center(60)}[/{style}]")
    console.print(f"[{style}]{char * 60}[/{style}]")
    console.print()


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


# Example usage
if __name__ == "__main__":
    print_header("UX Printing Utilities Demo")

    # Example 1: Print answer
    print_answer(
        "List comprehensions provide a concise way to create lists in Python. "
        "The syntax is [expression for item in iterable if condition].",
        title="Python Expert Says"
    )

    # Example 2: Print sources
    sources = [
        {"doc_id": "python_001", "title": "List Comprehensions", "score": 0.95},
        {"doc_id": "python_002", "title": "Python Basics", "score": 0.78}
    ]
    print_sources(sources)

    # Example 3: Print metrics
    metrics = {
        "accuracy": 0.87,
        "latency_ms": 245.3,
        "cost_usd": 0.0023,
        "tokens": 156
    }
    print_metrics(metrics)

    # Example 4: Print comparison
    comparisons = [
        {"model": "GPT-5", "accuracy": 0.87, "cost": 0.005, "latency_ms": 250},
        {"model": "GPT-5-mini", "accuracy": 0.82, "cost": 0.001, "latency_ms": 180}
    ]
    print_comparison_table(comparisons, title="Model Comparison")

    # Example 5: Status messages
    print_success("All tests passed!")
    print_warning("Cost approaching limit")
    print_error("API key not found")
    print_info("Loading documents...")
