"""
Module 03: Project - Few-Shot Learning Manager

A production-ready system for managing and optimizing few-shot prompts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient, count_tokens
import json
import random
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import statistics


@dataclass
class Example:
    """Represents a single few-shot example."""
    input_text: str
    output_text: str
    metadata: Dict[str, Any] = None
    performance_score: float = 0.0
    usage_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "input": self.input_text,
            "output": self.output_text,
            "metadata": self.metadata or {},
            "performance_score": self.performance_score,
            "usage_count": self.usage_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Example':
        """Create from dictionary."""
        return cls(
            input_text=data["input"],
            output_text=data["output"],
            metadata=data.get("metadata", {}),
            performance_score=data.get("performance_score", 0.0),
            usage_count=data.get("usage_count", 0)
        )


@dataclass
class ExampleSet:
    """Collection of examples for a specific task."""
    name: str
    description: str
    examples: List[Example]
    format_template: str
    optimal_shot_count: int = 3
    performance_metrics: Dict[str, float] = None

    def select_examples(self, n: int, strategy: str = "best") -> List[Example]:
        """
        Select n examples using specified strategy.

        Strategies:
        - best: Select highest performing examples
        - random: Random selection
        - diverse: Maximum diversity (placeholder for embedding-based)
        - stratified: Balanced selection across categories
        """
        if strategy == "best":
            sorted_examples = sorted(self.examples,
                                    key=lambda x: x.performance_score,
                                    reverse=True)
            return sorted_examples[:n]

        elif strategy == "random":
            return random.sample(self.examples, min(n, len(self.examples)))

        elif strategy == "diverse":
            # Simplified diversity: select examples with different characteristics
            if not self.examples:
                return []

            selected = [self.examples[0]]
            for ex in self.examples[1:]:
                if len(selected) >= n:
                    break
                # Simple diversity check: different lengths or starting words
                if all(len(ex.input_text) != len(s.input_text) or
                      ex.input_text.split()[0] != s.input_text.split()[0]
                      for s in selected):
                    selected.append(ex)

            # Fill remaining with random if needed
            while len(selected) < n and len(selected) < len(self.examples):
                ex = random.choice(self.examples)
                if ex not in selected:
                    selected.append(ex)

            return selected

        elif strategy == "stratified":
            # Group by metadata categories if available
            categories = {}
            for ex in self.examples:
                cat = ex.metadata.get("category", "default") if ex.metadata else "default"
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(ex)

            # Select proportionally from each category
            selected = []
            per_category = max(1, n // len(categories))

            for cat_examples in categories.values():
                selected.extend(random.sample(cat_examples,
                                             min(per_category, len(cat_examples))))

            return selected[:n]

        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")


class FewShotManager:
    """Main manager for few-shot learning operations."""

    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """Initialize the manager."""
        self.client = LLMClient(provider)
        self.model = model
        self.example_sets: Dict[str, ExampleSet] = {}
        self.performance_history = []
        self.library_path = os.path.join(os.path.dirname(__file__), "example_library.json")
        self.load_library()

    def load_library(self):
        """Load example library from JSON."""
        if not os.path.exists(self.library_path):
            return

        with open(self.library_path, 'r') as f:
            library_data = json.load(f)

        # Convert JSON to ExampleSet objects
        for category, tasks in library_data.items():
            if category == "meta":
                continue  # Skip metadata section

            for task_name, task_data in tasks.items():
                if isinstance(task_data, dict) and "examples" in task_data:
                    examples = []
                    for ex_data in task_data["examples"]:
                        example = Example(
                            input_text=ex_data["input"],
                            output_text=ex_data["output"],
                            metadata={k: v for k, v in ex_data.items()
                                    if k not in ["input", "output"]}
                        )
                        examples.append(example)

                    example_set = ExampleSet(
                        name=f"{category}.{task_name}",
                        description=task_data.get("description", ""),
                        examples=examples,
                        format_template=task_data.get("format_template",
                                                     "Input: {input}\nOutput: {output}")
                    )
                    self.example_sets[example_set.name] = example_set

    def add_example_set(self, example_set: ExampleSet):
        """Add a new example set."""
        self.example_sets[example_set.name] = example_set

    def create_prompt(
        self,
        task_name: str,
        input_text: str,
        shot_count: Optional[int] = None,
        selection_strategy: str = "best"
    ) -> str:
        """
        Create a few-shot prompt for the given task and input.

        Args:
            task_name: Name of the task/example set
            input_text: The input to process
            shot_count: Number of examples (None = use optimal)
            selection_strategy: How to select examples

        Returns:
            Formatted few-shot prompt
        """
        if task_name not in self.example_sets:
            raise ValueError(f"Unknown task: {task_name}")

        example_set = self.example_sets[task_name]
        n_shots = shot_count or example_set.optimal_shot_count

        # Select examples
        selected_examples = example_set.select_examples(n_shots, selection_strategy)

        # Build prompt
        prompt_parts = []

        # Add task description if available
        if example_set.description:
            prompt_parts.append(example_set.description)
            prompt_parts.append("")

        # Add examples
        for example in selected_examples:
            formatted = example_set.format_template.format(
                input=example.input_text,
                output=example.output_text
            )
            prompt_parts.append(formatted)
            prompt_parts.append("")

        # Add the actual input
        final_input = example_set.format_template.split("{output}")[0].format(
            input=input_text
        )
        prompt_parts.append(final_input.strip())

        return "\n".join(prompt_parts)

    def test_configurations(
        self,
        task_name: str,
        test_inputs: List[Tuple[str, str]],
        configurations: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Test different few-shot configurations.

        Args:
            task_name: Name of the task
            test_inputs: List of (input, expected_output) tuples
            configurations: List of configuration dicts with shot_count and strategy

        Returns:
            Test results with performance metrics
        """
        if configurations is None:
            configurations = [
                {"shot_count": 1, "strategy": "best"},
                {"shot_count": 3, "strategy": "best"},
                {"shot_count": 3, "strategy": "diverse"},
                {"shot_count": 5, "strategy": "best"},
                {"shot_count": 5, "strategy": "stratified"},
            ]

        results = []

        for config in configurations:
            config_results = {
                "configuration": config,
                "accuracy": 0.0,
                "consistency": 0.0,
                "avg_tokens": 0,
                "avg_time": 0.0,
                "responses": []
            }

            correct = 0
            tokens_used = []
            times = []

            for test_input, expected_output in test_inputs:
                start_time = time.time()

                # Create prompt
                prompt = self.create_prompt(
                    task_name,
                    test_input,
                    shot_count=config["shot_count"],
                    selection_strategy=config["strategy"]
                )

                # Get response
                response = self.client.complete(
                    prompt,
                    temperature=0.2,
                    max_tokens=100,
                    model=self.model
                )

                elapsed = time.time() - start_time
                times.append(elapsed)
                tokens_used.append(count_tokens(prompt))

                # Check accuracy
                response_clean = response.strip().lower()
                expected_clean = expected_output.strip().lower()
                if expected_clean in response_clean or response_clean in expected_clean:
                    correct += 1

                config_results["responses"].append({
                    "input": test_input,
                    "expected": expected_output,
                    "actual": response.strip()
                })

            # Calculate metrics
            config_results["accuracy"] = correct / len(test_inputs) if test_inputs else 0
            config_results["avg_tokens"] = statistics.mean(tokens_used) if tokens_used else 0
            config_results["avg_time"] = statistics.mean(times) if times else 0

            # Calculate consistency (variance in similar responses)
            response_lengths = [len(r["actual"]) for r in config_results["responses"]]
            if len(response_lengths) > 1:
                config_results["consistency"] = 1.0 - (statistics.stdev(response_lengths) /
                                                       statistics.mean(response_lengths))
            else:
                config_results["consistency"] = 1.0

            results.append(config_results)

        # Find best configuration
        best_config = max(results, key=lambda x: x["accuracy"])

        return {
            "task_name": task_name,
            "test_size": len(test_inputs),
            "configurations_tested": len(configurations),
            "results": results,
            "best_configuration": best_config["configuration"],
            "best_accuracy": best_config["accuracy"]
        }

    def optimize_examples(
        self,
        task_name: str,
        validation_data: List[Tuple[str, str]],
        max_iterations: int = 3
    ) -> ExampleSet:
        """
        Optimize example set based on validation data.

        Args:
            task_name: Name of the task
            validation_data: List of (input, expected_output) for validation
            max_iterations: Maximum optimization iterations

        Returns:
            Optimized ExampleSet
        """
        if task_name not in self.example_sets:
            raise ValueError(f"Unknown task: {task_name}")

        example_set = self.example_sets[task_name]
        best_accuracy = 0.0

        for iteration in range(max_iterations):
            # Test each example's contribution
            example_scores = []

            for i, example in enumerate(example_set.examples):
                # Test without this example
                test_examples = example_set.examples[:i] + example_set.examples[i+1:]

                if len(test_examples) < 1:
                    continue

                # Create temporary set
                temp_set = ExampleSet(
                    name=example_set.name,
                    description=example_set.description,
                    examples=test_examples[:3],  # Use top 3
                    format_template=example_set.format_template
                )

                # Test accuracy
                correct = 0
                for val_input, val_expected in validation_data[:5]:  # Quick test
                    prompt = self._create_prompt_from_set(temp_set, val_input)
                    response = self.client.complete(prompt, temperature=0.2, max_tokens=50)

                    if val_expected.lower() in response.lower():
                        correct += 1

                accuracy = correct / min(5, len(validation_data))
                example_scores.append((example, accuracy))

            # Update example performance scores
            for example, score in example_scores:
                example.performance_score = score

            # Sort examples by performance
            example_set.examples.sort(key=lambda x: x.performance_score, reverse=True)

            # Test current configuration
            test_results = self.test_configurations(
                task_name,
                validation_data[:10],
                [{"shot_count": 3, "strategy": "best"}]
            )

            current_accuracy = test_results["results"][0]["accuracy"]

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
            else:
                break  # No improvement, stop optimizing

        # Determine optimal shot count
        shot_test_results = self.test_configurations(
            task_name,
            validation_data[:10],
            [
                {"shot_count": 1, "strategy": "best"},
                {"shot_count": 3, "strategy": "best"},
                {"shot_count": 5, "strategy": "best"},
            ]
        )

        best_shot_config = max(shot_test_results["results"], key=lambda x: x["accuracy"])
        example_set.optimal_shot_count = best_shot_config["configuration"]["shot_count"]

        return example_set

    def _create_prompt_from_set(self, example_set: ExampleSet, input_text: str) -> str:
        """Helper to create prompt from example set."""
        prompt_parts = []

        for example in example_set.examples:
            formatted = example_set.format_template.format(
                input=example.input_text,
                output=example.output_text
            )
            prompt_parts.append(formatted)
            prompt_parts.append("")

        final_input = example_set.format_template.split("{output}")[0].format(
            input=input_text
        )
        prompt_parts.append(final_input.strip())

        return "\n".join(prompt_parts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the manager's performance."""
        stats = {
            "total_example_sets": len(self.example_sets),
            "total_examples": sum(len(es.examples) for es in self.example_sets.values()),
            "example_sets": {}
        }

        for name, example_set in self.example_sets.items():
            stats["example_sets"][name] = {
                "num_examples": len(example_set.examples),
                "optimal_shots": example_set.optimal_shot_count,
                "avg_performance": statistics.mean([e.performance_score
                                                   for e in example_set.examples])
                                  if example_set.examples else 0
            }

        return stats


# ===== Interactive Demo =====

def interactive_demo():
    """Run an interactive demonstration of the Few-Shot Manager."""
    print("=" * 60)
    print("Few-Shot Learning Manager - Interactive Demo")
    print("=" * 60)

    manager = FewShotManager()

    # Display available tasks
    print("\nAvailable tasks:")
    for i, task_name in enumerate(manager.example_sets.keys(), 1):
        print(f"{i}. {task_name}")

    while True:
        print("\nOptions:")
        print("1. Create few-shot prompt")
        print("2. Test configurations")
        print("3. Optimize examples")
        print("4. View statistics")
        print("5. Exit")

        choice = input("\nSelect option (1-5): ")

        if choice == "1":
            task_name = input("Enter task name (e.g., classification.sentiment): ")
            if task_name in manager.example_sets:
                input_text = input("Enter your input text: ")

                shot_count = input("Number of examples (press Enter for optimal): ")
                shot_count = int(shot_count) if shot_count else None

                strategy = input("Selection strategy (best/random/diverse/stratified): ") or "best"

                try:
                    prompt = manager.create_prompt(task_name, input_text, shot_count, strategy)
                    print("\nGenerated Prompt:")
                    print("-" * 40)
                    print(prompt)

                    # Get completion
                    response = manager.client.complete(prompt, temperature=0.3, max_tokens=100)
                    print("\nModel Response:")
                    print(response.strip())
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Unknown task name")

        elif choice == "2":
            task_name = input("Enter task name: ")
            if task_name in manager.example_sets:
                print("\nEnter test data (input:expected_output), empty line to finish:")
                test_data = []
                while True:
                    line = input()
                    if not line:
                        break
                    if ":" in line:
                        input_part, output_part = line.split(":", 1)
                        test_data.append((input_part.strip(), output_part.strip()))

                if test_data:
                    results = manager.test_configurations(task_name, test_data)

                    print("\nTest Results:")
                    print("-" * 40)
                    for result in results["results"]:
                        config = result["configuration"]
                        print(f"\nShot count: {config['shot_count']}, Strategy: {config['strategy']}")
                        print(f"  Accuracy: {result['accuracy']:.1%}")
                        print(f"  Avg tokens: {result['avg_tokens']:.0f}")
                        print(f"  Consistency: {result['consistency']:.1%}")

                    print(f"\nBest configuration: {results['best_configuration']}")
                    print(f"Best accuracy: {results['best_accuracy']:.1%}")
            else:
                print("Unknown task name")

        elif choice == "3":
            task_name = input("Enter task name: ")
            if task_name in manager.example_sets:
                print("\nEnter validation data (input:expected_output), empty line to finish:")
                validation_data = []
                while True:
                    line = input()
                    if not line:
                        break
                    if ":" in line:
                        input_part, output_part = line.split(":", 1)
                        validation_data.append((input_part.strip(), output_part.strip()))

                if validation_data:
                    print("\nOptimizing examples...")
                    optimized_set = manager.optimize_examples(task_name, validation_data)
                    print(f"Optimization complete!")
                    print(f"Optimal shot count: {optimized_set.optimal_shot_count}")
                    print(f"Top performing examples:")
                    for i, ex in enumerate(optimized_set.examples[:3], 1):
                        print(f"  {i}. {ex.input_text[:50]}... (score: {ex.performance_score:.2f})")
            else:
                print("Unknown task name")

        elif choice == "4":
            stats = manager.get_statistics()
            print("\nManager Statistics:")
            print("-" * 40)
            print(f"Total example sets: {stats['total_example_sets']}")
            print(f"Total examples: {stats['total_examples']}")

            print("\nPer-task statistics:")
            for task_name, task_stats in stats["example_sets"].items():
                print(f"\n{task_name}:")
                print(f"  Examples: {task_stats['num_examples']}")
                print(f"  Optimal shots: {task_stats['optimal_shots']}")
                print(f"  Avg performance: {task_stats['avg_performance']:.2f}")

        elif choice == "5":
            print("Exiting...")
            break

        else:
            print("Invalid option")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Few-Shot Learning Manager")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--task", type=str, help="Task name")
    parser.add_argument("--input", type=str, help="Input text")
    parser.add_argument("--shots", type=int, help="Number of examples")
    parser.add_argument("--strategy", type=str, default="best",
                       help="Selection strategy (best/random/diverse/stratified)")

    args = parser.parse_args()

    if args.demo:
        interactive_demo()
    elif args.task and args.input:
        manager = FewShotManager()
        try:
            prompt = manager.create_prompt(args.task, args.input, args.shots, args.strategy)
            print("Generated Prompt:")
            print("-" * 40)
            print(prompt)

            response = manager.client.complete(prompt, temperature=0.3, max_tokens=100)
            print("\nModel Response:")
            print(response.strip())
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Few-Shot Learning Manager")
        print("\nUsage:")
        print("  python few_shot_manager.py --demo")
        print("  python few_shot_manager.py --task TASK --input TEXT [--shots N] [--strategy S]")
        print("\nExample:")
        print("  python few_shot_manager.py --task classification.sentiment --input 'Great product!' --shots 3")