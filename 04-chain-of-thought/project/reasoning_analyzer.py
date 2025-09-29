"""
Module 04: Reasoning Chain Analyzer

A production-ready system for analyzing, validating, and optimizing
Chain-of-Thought reasoning in LLM outputs.
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict


class StepType(Enum):
    """Types of reasoning steps."""
    SETUP = "setup"
    CALCULATION = "calculation"
    LOGIC = "logic"
    VERIFICATION = "verification"
    CONCLUSION = "conclusion"
    ERROR_CORRECTION = "error_correction"


class ReasoningQuality(Enum):
    """Quality levels for reasoning chains."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class ReasoningStep:
    """Individual step in a reasoning chain."""
    number: int
    content: str
    step_type: StepType
    dependencies: List[int] = field(default_factory=list)
    confidence: float = 1.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ReasoningChain:
    """Complete reasoning chain analysis."""
    raw_text: str
    problem: str
    steps: List[ReasoningStep]
    conclusion: Optional[str] = None
    quality: ReasoningQuality = ReasoningQuality.FAIR
    metrics: Dict = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    validation_results: Dict = field(default_factory=dict)


class ReasoningAnalyzer:
    """Analyzes and validates Chain-of-Thought reasoning."""

    def __init__(self):
        self.step_patterns = {
            'numbered': r'Step\s*(\d+)[:\.]?\s*(.*?)(?=Step\s*\d+|$)',
            'bulleted': r'[-•]\s*(.*?)(?=[-•]|$)',
            'labeled': r'([A-Z][^:]+):\s*(.*?)(?=[A-Z][^:]+:|$)',
            'conclusion': r'(?:Answer|Conclusion|Result|Therefore)[:\s]+(.*?)(?:\n|$)'
        }

        self.quality_criteria = {
            'has_steps': 0.2,
            'clear_conclusion': 0.2,
            'logical_flow': 0.2,
            'verification': 0.15,
            'no_errors': 0.15,
            'appropriate_depth': 0.1
        }

    def analyze(self, reasoning_text: str, problem: str = "") -> ReasoningChain:
        """
        Analyze a complete reasoning chain.

        Args:
            reasoning_text: The raw reasoning output
            problem: The original problem (optional)

        Returns:
            ReasoningChain with full analysis
        """
        chain = ReasoningChain(
            raw_text=reasoning_text,
            problem=problem,
            steps=[]
        )

        # Parse steps
        chain.steps = self._parse_steps(reasoning_text)

        # Extract conclusion
        chain.conclusion = self._extract_conclusion(reasoning_text)

        # Validate reasoning
        chain.validation_results = self._validate_chain(chain)

        # Calculate metrics
        chain.metrics = self._calculate_metrics(chain)

        # Assess quality
        chain.quality = self._assess_quality(chain)

        # Generate suggestions
        chain.suggestions = self._generate_suggestions(chain)

        return chain

    def _parse_steps(self, text: str) -> List[ReasoningStep]:
        """Parse reasoning steps from text."""
        steps = []

        # Try numbered steps first
        matches = re.findall(self.step_patterns['numbered'], text, re.DOTALL | re.IGNORECASE)

        if matches:
            for i, (num, content) in enumerate(matches):
                step = ReasoningStep(
                    number=int(num),
                    content=content.strip(),
                    step_type=self._classify_step(content)
                )
                steps.append(step)
        else:
            # Try other patterns
            sections = text.split('\n\n')
            for i, section in enumerate(sections):
                if section.strip():
                    step = ReasoningStep(
                        number=i + 1,
                        content=section.strip(),
                        step_type=self._classify_step(section)
                    )
                    steps.append(step)

        # Identify dependencies
        for i, step in enumerate(steps):
            step.dependencies = self._find_dependencies(step, steps[:i])

        return steps

    def _classify_step(self, content: str) -> StepType:
        """Classify the type of reasoning step."""
        content_lower = content.lower()

        if any(word in content_lower for word in ['given', 'know', 'have', 'start']):
            return StepType.SETUP
        elif any(word in content_lower for word in ['calculate', 'compute', 'multiply', 'divide']):
            return StepType.CALCULATION
        elif any(word in content_lower for word in ['therefore', 'thus', 'hence', 'conclude']):
            return StepType.CONCLUSION
        elif any(word in content_lower for word in ['check', 'verify', 'confirm', 'validate']):
            return StepType.VERIFICATION
        elif any(word in content_lower for word in ['error', 'mistake', 'wrong', 'correct']):
            return StepType.ERROR_CORRECTION
        else:
            return StepType.LOGIC

    def _find_dependencies(self, current_step: ReasoningStep,
                           previous_steps: List[ReasoningStep]) -> List[int]:
        """Identify which previous steps the current step depends on."""
        dependencies = []

        for prev_step in previous_steps:
            # Check for explicit references
            if f"Step {prev_step.number}" in current_step.content:
                dependencies.append(prev_step.number)
            # Check for implicit references (using results)
            elif self._uses_result_from(current_step.content, prev_step.content):
                dependencies.append(prev_step.number)

        return dependencies

    def _uses_result_from(self, current_content: str, previous_content: str) -> bool:
        """Check if current step uses results from previous step."""
        # Extract numbers from previous step
        prev_numbers = re.findall(r'\b\d+\.?\d*\b', previous_content)

        # Check if those numbers appear in current step
        for num in prev_numbers:
            if num in current_content and len(num) > 1:  # Avoid single digits
                return True

        return False

    def _extract_conclusion(self, text: str) -> Optional[str]:
        """Extract the final conclusion or answer."""
        conclusion_match = re.search(self.step_patterns['conclusion'], text, re.IGNORECASE)

        if conclusion_match:
            return conclusion_match.group(1).strip()

        # Look for last substantive line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            return lines[-1]

        return None

    def _validate_chain(self, chain: ReasoningChain) -> Dict:
        """Validate the reasoning chain for correctness."""
        validation = {
            'has_problem': bool(chain.problem),
            'has_steps': len(chain.steps) > 0,
            'has_conclusion': bool(chain.conclusion),
            'steps_connected': self._check_step_connectivity(chain.steps),
            'no_circular_deps': self._check_no_circular_dependencies(chain.steps),
            'math_correct': self._validate_math(chain),
            'logic_sound': self._validate_logic(chain)
        }

        return validation

    def _check_step_connectivity(self, steps: List[ReasoningStep]) -> bool:
        """Check if steps form a connected chain."""
        if len(steps) <= 1:
            return True

        # Check if each step (except first) has dependencies or follows naturally
        for i, step in enumerate(steps[1:], 1):
            if not step.dependencies:
                # Check if it naturally follows
                if not self._follows_naturally(steps[i-1], step):
                    return False

        return True

    def _follows_naturally(self, prev_step: ReasoningStep, current_step: ReasoningStep) -> bool:
        """Check if current step naturally follows previous."""
        # Simple heuristic: check for continuation words
        continuation_words = ['then', 'next', 'now', 'therefore', 'thus']
        return any(word in current_step.content.lower()[:20] for word in continuation_words)

    def _check_no_circular_dependencies(self, steps: List[ReasoningStep]) -> bool:
        """Ensure no circular dependencies in reasoning."""
        for step in steps:
            if step.number in step.dependencies:
                return False
            # Check for indirect circular deps (would need more complex graph analysis)

        return True

    def _validate_math(self, chain: ReasoningChain) -> bool:
        """Validate mathematical calculations in the chain."""
        for step in chain.steps:
            if step.step_type == StepType.CALCULATION:
                # Extract and verify calculations
                calculations = re.findall(r'(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',
                                        step.content)
                for calc in calculations:
                    try:
                        left = float(calc[0])
                        right = float(calc[2])
                        result = float(calc[3])
                        operator = calc[1]

                        if operator == '+':
                            expected = left + right
                        elif operator == '-':
                            expected = left - right
                        elif operator == '*':
                            expected = left * right
                        elif operator == '/':
                            expected = left / right if right != 0 else None

                        if expected and abs(expected - result) > 0.01:
                            step.errors.append(f"Math error: {left} {operator} {right} ≠ {result}")
                            return False
                    except:
                        continue

        return True

    def _validate_logic(self, chain: ReasoningChain) -> bool:
        """Validate logical consistency."""
        # Check for contradictions
        for i, step1 in enumerate(chain.steps):
            for step2 in chain.steps[i+1:]:
                if self._contradicts(step1, step2):
                    return False

        return True

    def _contradicts(self, step1: ReasoningStep, step2: ReasoningStep) -> bool:
        """Check if two steps contradict each other."""
        # Simple contradiction detection
        negation_pairs = [
            ('increase', 'decrease'),
            ('greater', 'less'),
            ('positive', 'negative'),
            ('true', 'false')
        ]

        for pos, neg in negation_pairs:
            if pos in step1.content.lower() and neg in step2.content.lower():
                # Check if referring to same entity
                return True  # Simplified - would need entity resolution

        return False

    def _calculate_metrics(self, chain: ReasoningChain) -> Dict:
        """Calculate quality metrics for the reasoning chain."""
        metrics = {
            'step_count': len(chain.steps),
            'avg_step_length': sum(len(s.content) for s in chain.steps) / max(len(chain.steps), 1),
            'verification_steps': sum(1 for s in chain.steps if s.step_type == StepType.VERIFICATION),
            'error_corrections': sum(1 for s in chain.steps if s.step_type == StepType.ERROR_CORRECTION),
            'total_dependencies': sum(len(s.dependencies) for s in chain.steps),
            'max_dependency_depth': self._calculate_max_depth(chain.steps),
            'total_errors': sum(len(s.errors) for s in chain.steps)
        }

        return metrics

    def _calculate_max_depth(self, steps: List[ReasoningStep]) -> int:
        """Calculate maximum dependency depth."""
        if not steps:
            return 0

        depths = {}
        for step in steps:
            if not step.dependencies:
                depths[step.number] = 1
            else:
                depths[step.number] = max(depths.get(dep, 0) for dep in step.dependencies) + 1

        return max(depths.values()) if depths else 0

    def _assess_quality(self, chain: ReasoningChain) -> ReasoningQuality:
        """Assess overall quality of reasoning."""
        score = 0.0

        # Check each quality criterion
        if chain.validation_results.get('has_steps'):
            score += self.quality_criteria['has_steps']

        if chain.validation_results.get('has_conclusion'):
            score += self.quality_criteria['clear_conclusion']

        if chain.validation_results.get('logic_sound'):
            score += self.quality_criteria['logical_flow']

        if chain.metrics.get('verification_steps', 0) > 0:
            score += self.quality_criteria['verification']

        if chain.metrics.get('total_errors', 0) == 0:
            score += self.quality_criteria['no_errors']

        # Check appropriate depth
        step_count = chain.metrics.get('step_count', 0)
        if 3 <= step_count <= 7:
            score += self.quality_criteria['appropriate_depth']

        # Map score to quality level
        if score >= 0.8:
            return ReasoningQuality.EXCELLENT
        elif score >= 0.6:
            return ReasoningQuality.GOOD
        elif score >= 0.4:
            return ReasoningQuality.FAIR
        elif score >= 0.2:
            return ReasoningQuality.POOR
        else:
            return ReasoningQuality.INVALID

    def _generate_suggestions(self, chain: ReasoningChain) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []

        if not chain.validation_results.get('has_conclusion'):
            suggestions.append("Add a clear conclusion or final answer")

        if chain.metrics.get('step_count', 0) < 2:
            suggestions.append("Break down reasoning into more explicit steps")

        if chain.metrics.get('step_count', 0) > 10:
            suggestions.append("Consider consolidating steps for clarity")

        if chain.metrics.get('verification_steps', 0) == 0:
            suggestions.append("Add verification steps to check your work")

        if chain.metrics.get('total_errors', 0) > 0:
            suggestions.append("Review and correct calculation errors")

        if not chain.validation_results.get('steps_connected'):
            suggestions.append("Ensure steps flow logically from one to the next")

        return suggestions

    def optimize_reasoning(self, chain: ReasoningChain) -> str:
        """Generate optimized version of reasoning."""
        optimized = []

        # Add problem statement if missing
        if chain.problem:
            optimized.append(f"Problem: {chain.problem}\n")

        # Reorganize steps
        optimized.append("Solution:\n")
        for i, step in enumerate(chain.steps, 1):
            # Skip error correction steps in optimized version
            if step.step_type != StepType.ERROR_CORRECTION:
                optimized.append(f"Step {i}: {step.content}\n")

        # Add verification if missing
        if chain.metrics.get('verification_steps', 0) == 0:
            optimized.append("\nVerification: [Check answer against original problem]\n")

        # Add clear conclusion
        if chain.conclusion:
            optimized.append(f"\nAnswer: {chain.conclusion}")
        else:
            optimized.append("\nAnswer: [State final answer clearly]")

        return "\n".join(optimized)


class ReasoningOptimizer:
    """Optimizes reasoning chains for different use cases."""

    def __init__(self):
        self.analyzer = ReasoningAnalyzer()
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict:
        """Load reasoning templates for different domains."""
        return {
            'math': {
                'structure': ['Setup', 'Formula', 'Calculation', 'Verification', 'Answer'],
                'keywords': ['calculate', 'solve', 'find', 'compute']
            },
            'logic': {
                'structure': ['Premises', 'Rules', 'Application', 'Conclusion'],
                'keywords': ['if', 'then', 'therefore', 'implies']
            },
            'analysis': {
                'structure': ['Context', 'Factors', 'Evaluation', 'Recommendation'],
                'keywords': ['analyze', 'evaluate', 'consider', 'assess']
            }
        }

    def optimize_for_domain(self, reasoning_text: str, domain: str) -> str:
        """Optimize reasoning for specific domain."""
        if domain not in self.templates:
            domain = 'math'  # Default

        template = self.templates[domain]
        chain = self.analyzer.analyze(reasoning_text)

        optimized = []
        for section in template['structure']:
            matching_steps = [s for s in chain.steps
                            if any(kw in s.content.lower()
                                  for kw in template['keywords'])]
            if matching_steps:
                optimized.append(f"{section}:\n{matching_steps[0].content}\n")

        return "\n".join(optimized)

    def compress_reasoning(self, reasoning_text: str, target_length: int = 200) -> str:
        """Compress reasoning while maintaining key points."""
        chain = self.analyzer.analyze(reasoning_text)

        # Prioritize steps
        priority_steps = []
        for step in chain.steps:
            if step.step_type in [StepType.SETUP, StepType.CONCLUSION]:
                priority_steps.append(step)

        # Add calculation steps if room
        calc_steps = [s for s in chain.steps if s.step_type == StepType.CALCULATION]
        priority_steps.extend(calc_steps[:2])  # Max 2 calculation steps

        # Build compressed version
        compressed = []
        current_length = 0

        for step in priority_steps:
            step_text = f"• {step.content[:100]}"
            if current_length + len(step_text) < target_length:
                compressed.append(step_text)
                current_length += len(step_text)

        if chain.conclusion:
            compressed.append(f"\nAnswer: {chain.conclusion}")

        return "\n".join(compressed)


# Usage Example
if __name__ == "__main__":
    analyzer = ReasoningAnalyzer()
    optimizer = ReasoningOptimizer()

    # Example reasoning to analyze
    sample_reasoning = """
    Step 1: We have a rectangle with length 10m and width 5m.
    Step 2: Calculate the area: 10 × 5 = 50 square meters.
    Step 3: Calculate the perimeter: 2 × (10 + 5) = 30 meters.
    Step 4: Verify: Area should be length × width = 10 × 5 = 50 ✓
    Answer: Area is 50 square meters, perimeter is 30 meters.
    """

    # Analyze the reasoning
    analysis = analyzer.analyze(sample_reasoning, "Find area and perimeter of a rectangle")

    print(f"Quality: {analysis.quality.value}")
    print(f"Metrics: {json.dumps(analysis.metrics, indent=2)}")
    print(f"Suggestions: {analysis.suggestions}")
    print("\nOptimized version:")
    print(analyzer.optimize_reasoning(analysis))

    print("\n" + "="*50)
    print("Compressed version:")
    print(optimizer.compress_reasoning(sample_reasoning))