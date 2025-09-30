"""
Module 02: Project - Zero-Shot Task Processor

A comprehensive system for automatically generating and optimizing zero-shot prompts
for various task types, with reliability testing and multi-model support.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient, count_tokens, estimate_cost
import json
import time
import statistics
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum


class TaskType(Enum):
    """Supported task types for zero-shot processing."""
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    GENERATION = "generation"
    ANALYSIS = "analysis"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    SUMMARIZATION = "summarization"


class ZeroShotProcessor:
    """
    Main processor for handling zero-shot tasks with automatic prompt optimization.
    """

    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize the processor.

        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Specific model to use (optional)
        """
        self.client = LLMClient(provider)
        self.model = model
        self.task_history = []
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict:
        """Load task templates from JSON file."""
        template_file = os.path.join(os.path.dirname(__file__), "task_templates.json")
        if os.path.exists(template_file):
            with open(template_file, 'r') as f:
                return json.load(f)
        return {}

    def detect_task_type(self, description: str) -> TaskType:
        """
        Automatically detect the type of task from user description.

        Args:
            description: User's task description

        Returns:
            Detected TaskType
        """
        detection_prompt = f"""Classify this task description into one category:

        Task: {description}

        Categories:
        - CLASSIFICATION: Categorizing or labeling data
        - EXTRACTION: Pulling out specific information
        - GENERATION: Creating new content
        - ANALYSIS: Examining and evaluating
        - TRANSFORMATION: Converting format or style
        - VALIDATION: Checking correctness or compliance
        - SUMMARIZATION: Condensing information

        Output only the category name:"""

        response = self.client.complete(
            detection_prompt,
            temperature=0.0,
            max_tokens=20,
            model=self.model
        )

        # Map response to TaskType
        response_clean = response.strip().upper()
        for task_type in TaskType:
            if task_type.name in response_clean:
                return task_type

        # Default to ANALYSIS if unclear
        return TaskType.ANALYSIS

    def generate_prompt(
        self,
        task_description: str,
        task_type: Optional[TaskType] = None,
        constraints: Optional[List[str]] = None,
        output_format: Optional[str] = None
    ) -> str:
        """
        Generate an optimized zero-shot prompt for the task.

        Args:
            task_description: Description of what to do
            task_type: Type of task (auto-detected if not provided)
            constraints: Additional constraints or requirements
            output_format: Desired output format

        Returns:
            Optimized zero-shot prompt
        """
        if task_type is None:
            task_type = self.detect_task_type(task_description)

        # Get base template for task type
        template = self._get_template(task_type)

        # Build the prompt
        prompt_parts = []

        # Add task description
        prompt_parts.append(f"Task: {task_description}\n")

        # Add specific instructions based on task type
        if task_type == TaskType.CLASSIFICATION:
            prompt_parts.append(template.get("instructions", ""))
            prompt_parts.append("\nClassification criteria:")
            prompt_parts.append("- Be consistent across similar inputs")
            prompt_parts.append("- Consider all aspects mentioned")
            prompt_parts.append("- Choose the most appropriate category")

        elif task_type == TaskType.EXTRACTION:
            prompt_parts.append(template.get("instructions", ""))
            prompt_parts.append("\nExtraction rules:")
            prompt_parts.append("- Extract only explicitly mentioned information")
            prompt_parts.append("- Use 'N/A' for missing information")
            prompt_parts.append("- Preserve original formatting when relevant")

        elif task_type == TaskType.GENERATION:
            prompt_parts.append(template.get("instructions", ""))
            prompt_parts.append("\nGeneration guidelines:")
            prompt_parts.append("- Be creative but relevant")
            prompt_parts.append("- Maintain consistent tone and style")
            prompt_parts.append("- Meet all specified requirements")

        elif task_type == TaskType.ANALYSIS:
            prompt_parts.append(template.get("instructions", ""))
            prompt_parts.append("\nAnalysis framework:")
            prompt_parts.append("- Examine all relevant aspects")
            prompt_parts.append("- Provide specific evidence")
            prompt_parts.append("- Draw clear conclusions")

        elif task_type == TaskType.TRANSFORMATION:
            prompt_parts.append(template.get("instructions", ""))
            prompt_parts.append("\nTransformation rules:")
            prompt_parts.append("- Preserve essential meaning")
            prompt_parts.append("- Apply format consistently")
            prompt_parts.append("- Handle edge cases gracefully")

        elif task_type == TaskType.VALIDATION:
            prompt_parts.append(template.get("instructions", ""))
            prompt_parts.append("\nValidation checklist:")
            prompt_parts.append("- Check against all criteria")
            prompt_parts.append("- Provide specific feedback")
            prompt_parts.append("- Suggest corrections when possible")

        elif task_type == TaskType.SUMMARIZATION:
            prompt_parts.append(template.get("instructions", ""))
            prompt_parts.append("\nSummarization principles:")
            prompt_parts.append("- Capture key points")
            prompt_parts.append("- Maintain accuracy")
            prompt_parts.append("- Be concise but complete")

        # Add constraints if provided
        if constraints:
            prompt_parts.append("\n\nConstraints:")
            for constraint in constraints:
                prompt_parts.append(f"- {constraint}")

        # Add output format specification
        if output_format:
            prompt_parts.append(f"\n\nOutput format:\n{output_format}")
        else:
            prompt_parts.append("\n\nProvide clear, structured output.")

        # Add input placeholder
        prompt_parts.append("\n\nInput: {input_text}")
        prompt_parts.append("\n\nOutput:")

        return "\n".join(prompt_parts)

    def _get_template(self, task_type: TaskType) -> Dict:
        """Get template for specific task type."""
        return self.templates.get(task_type.value, {
            "instructions": f"Perform the {task_type.value} task as described."
        })

    def test_reliability(
        self,
        prompt: str,
        test_input: str,
        num_runs: int = 5,
        temperature: float = 0.3
    ) -> Dict:
        """
        Test the reliability of a prompt across multiple runs.

        Args:
            prompt: The prompt to test
            test_input: Input to use for testing
            num_runs: Number of test runs
            temperature: Temperature setting for testing

        Returns:
            Dictionary with reliability metrics
        """
        responses = []
        response_times = []

        formatted_prompt = prompt.replace("{input_text}", test_input)

        for _ in range(num_runs):
            start_time = time.time()

            response = self.client.complete(
                formatted_prompt,
                temperature=temperature,
                max_tokens=200,
                model=self.model
            )

            response_times.append(time.time() - start_time)
            responses.append(response.strip())

        # Calculate metrics
        unique_responses = len(set(responses))
        avg_length = statistics.mean([len(r) for r in responses])
        avg_time = statistics.mean(response_times)

        # Calculate consistency score
        if unique_responses == 1:
            consistency_score = 100.0
        else:
            # Use similarity of responses
            consistency_score = ((num_runs - unique_responses) / num_runs) * 100

        return {
            "num_runs": num_runs,
            "unique_responses": unique_responses,
            "consistency_score": consistency_score,
            "average_length": avg_length,
            "average_time": avg_time,
            "responses": responses
        }

    def optimize_prompt(
        self,
        initial_prompt: str,
        test_input: str,
        target_consistency: float = 80.0
    ) -> Tuple[str, Dict]:
        """
        Automatically optimize a prompt for better consistency.

        Args:
            initial_prompt: Starting prompt
            test_input: Input for testing
            target_consistency: Desired consistency score

        Returns:
            Tuple of (optimized prompt, metrics)
        """
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_score = 0.0
        iterations = []

        for iteration in range(3):  # Maximum 3 optimization iterations
            # Test current prompt
            metrics = self.test_reliability(current_prompt, test_input)
            consistency = metrics["consistency_score"]

            iterations.append({
                "iteration": iteration + 1,
                "consistency": consistency,
                "unique_responses": metrics["unique_responses"]
            })

            if consistency >= target_consistency:
                return current_prompt, {
                    "optimized": True,
                    "iterations": iterations,
                    "final_consistency": consistency
                }

            if consistency > best_score:
                best_score = consistency
                best_prompt = current_prompt

            # Generate improvement suggestions
            improvement_prompt = f"""This zero-shot prompt has {consistency:.1f}% consistency (target: {target_consistency}%).

            Current prompt:
            {current_prompt}

            Suggest specific improvements to increase consistency:
            1. Add more precise instructions
            2. Specify exact output format
            3. Add constraints to reduce ambiguity
            4. Include step-by-step structure if needed

            Provide an improved version:"""

            improved = self.client.complete(
                improvement_prompt,
                temperature=0.4,
                max_tokens=500,
                model=self.model
            )

            # Extract the improved prompt (simple heuristic)
            if ":" in improved:
                current_prompt = improved.split(":", 1)[1].strip()
            else:
                current_prompt = improved.strip()

        return best_prompt, {
            "optimized": best_score > metrics["consistency_score"],
            "iterations": iterations,
            "final_consistency": best_score
        }

    def process_task(
        self,
        task_description: str,
        input_text: str,
        auto_optimize: bool = True,
        test_reliability: bool = True
    ) -> Dict:
        """
        Complete end-to-end processing of a zero-shot task.

        Args:
            task_description: What task to perform
            input_text: Input to process
            auto_optimize: Whether to automatically optimize the prompt
            test_reliability: Whether to test reliability

        Returns:
            Dictionary with results and metrics
        """
        start_time = time.time()

        # Detect task type
        task_type = self.detect_task_type(task_description)

        # Generate initial prompt
        prompt = self.generate_prompt(task_description, task_type)

        # Optimize if requested
        optimization_metrics = None
        if auto_optimize:
            prompt, optimization_metrics = self.optimize_prompt(prompt, input_text[:100])

        # Test reliability if requested
        reliability_metrics = None
        if test_reliability:
            reliability_metrics = self.test_reliability(prompt, input_text[:100], num_runs=3)

        # Execute the task
        final_prompt = prompt.replace("{input_text}", input_text)
        result = self.client.complete(
            final_prompt,
            temperature=0.2,
            max_tokens=500,
            model=self.model
        )

        # Calculate costs
        input_tokens = count_tokens(final_prompt)
        output_tokens = count_tokens(result)
        cost_info = estimate_cost(
            input_tokens,
            output_tokens,
            self.model or "gpt-5"
        )

        # Record in history
        task_record = {
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type.value,
            "description": task_description,
            "processing_time": time.time() - start_time,
            "consistency_score": reliability_metrics["consistency_score"] if reliability_metrics else None
        }
        self.task_history.append(task_record)

        return {
            "task_type": task_type.value,
            "prompt": prompt,
            "result": result,
            "metrics": {
                "processing_time": time.time() - start_time,
                "optimization": optimization_metrics,
                "reliability": reliability_metrics,
                "cost": cost_info
            },
            "task_record": task_record
        }

    def get_statistics(self) -> Dict:
        """Get statistics about processed tasks."""
        if not self.task_history:
            return {"message": "No tasks processed yet"}

        task_types = [t["task_type"] for t in self.task_history]
        consistency_scores = [t["consistency_score"] for t in self.task_history if t["consistency_score"]]

        return {
            "total_tasks": len(self.task_history),
            "task_distribution": {
                task_type: task_types.count(task_type)
                for task_type in set(task_types)
            },
            "average_consistency": statistics.mean(consistency_scores) if consistency_scores else None,
            "average_processing_time": statistics.mean([t["processing_time"] for t in self.task_history])
        }


# ===== Interactive Demo =====

def interactive_demo():
    """Run an interactive demonstration of the Zero-Shot Task Processor."""
    print("=" * 60)
    print("Zero-Shot Task Processor - Interactive Demo")
    print("=" * 60)

    processor = ZeroShotProcessor()

    while True:
        print("\nOptions:")
        print("1. Process a task")
        print("2. Test reliability of a prompt")
        print("3. View statistics")
        print("4. Run example tasks")
        print("5. Exit")

        choice = input("\nSelect option (1-5): ")

        if choice == "1":
            print("\nDescribe your task:")
            task_desc = input("> ")

            print("\nProvide input text:")
            input_text = input("> ")

            print("\nProcessing...")
            result = processor.process_task(task_desc, input_text)

            print(f"\nDetected Task Type: {result['task_type']}")
            print(f"\nGenerated Prompt:\n{result['prompt'][:300]}...")
            print(f"\nResult:\n{result['result']}")

            if result["metrics"]["reliability"]:
                print(f"\nConsistency Score: {result['metrics']['reliability']['consistency_score']:.1f}%")

            print(f"\nEstimated Cost: ${result['metrics']['cost'].get('total_cost', 0):.4f}")

        elif choice == "2":
            print("\nEnter your zero-shot prompt (use {input_text} as placeholder):")
            prompt = input("> ")

            print("\nProvide test input:")
            test_input = input("> ")

            print("\nTesting reliability...")
            metrics = processor.test_reliability(prompt, test_input)

            print(f"\nConsistency Score: {metrics['consistency_score']:.1f}%")
            print(f"Unique Responses: {metrics['unique_responses']}")
            print(f"Average Response Time: {metrics['average_time']:.2f}s")

        elif choice == "3":
            stats = processor.get_statistics()
            print("\nProcessor Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        elif choice == "4":
            run_example_tasks(processor)

        elif choice == "5":
            print("\nExiting...")
            break

        else:
            print("\nInvalid option, please try again.")


def run_example_tasks(processor: ZeroShotProcessor):
    """Run predefined example tasks."""
    examples = [
        {
            "description": "Classify the sentiment of customer feedback",
            "input": "The product quality is excellent but shipping was terribly slow."
        },
        {
            "description": "Extract key information from this business description",
            "input": "Acme Corp, founded in 2020, has 50 employees and $10M in revenue."
        },
        {
            "description": "Generate a professional email subject line",
            "input": "Meeting rescheduled from Tuesday to Thursday, same time"
        },
        {
            "description": "Analyze this code for potential issues",
            "input": "def divide(a, b): return a / b"
        }
    ]

    print("\n" + "=" * 60)
    print("Running Example Tasks")
    print("=" * 60)

    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example['description']}")
        print("-" * 40)

        result = processor.process_task(
            example["description"],
            example["input"],
            auto_optimize=False,
            test_reliability=False
        )

        print(f"Task Type: {result['task_type']}")
        print(f"Result: {result['result'][:200]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Zero-Shot Task Processor")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--task", type=str, help="Task description")
    parser.add_argument("--input", type=str, help="Input text")
    parser.add_argument("--optimize", action="store_true", help="Auto-optimize prompt")
    parser.add_argument("--test", action="store_true", help="Test reliability")

    args = parser.parse_args()

    if args.demo:
        interactive_demo()
    elif args.task and args.input:
        processor = ZeroShotProcessor()
        result = processor.process_task(
            args.task,
            args.input,
            auto_optimize=args.optimize,
            test_reliability=args.test
        )

        print(f"Task Type: {result['task_type']}")
        print(f"\nPrompt:\n{result['prompt']}")
        print(f"\nResult:\n{result['result']}")

        if result["metrics"]["reliability"]:
            print(f"\nConsistency: {result['metrics']['reliability']['consistency_score']:.1f}%")
    else:
        print("Zero-Shot Task Processor")
        print("\nUsage:")
        print("  python zero_shot_processor.py --demo")
        print("  python zero_shot_processor.py --task 'description' --input 'text' [--optimize] [--test]")
        print("\nRun with --demo for interactive mode")