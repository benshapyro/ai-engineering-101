"""
Module 03: Example Selection Strategies

Advanced techniques for selecting and ordering examples in few-shot prompts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import random
import json
from typing import List, Dict, Tuple


def example_1_random_vs_curated():
    """Compare random vs carefully curated example selection."""
    print("=" * 60)
    print("Example 1: Random vs Curated Selection")
    print("=" * 60)

    client = LLMClient("openai")

    # Pool of examples for sentiment classification
    example_pool = [
        ("I love this!", "Positive"),
        ("Terrible experience", "Negative"),
        ("It's okay", "Neutral"),
        ("Amazing product, highly recommend!", "Positive"),
        ("Waste of money", "Negative"),
        ("Not bad, not great", "Neutral"),
        ("Best purchase ever!", "Positive"),
        ("Completely disappointed", "Negative"),
        ("It works as expected", "Neutral"),
        ("Exceptional quality!", "Positive"),
    ]

    test_input = "The product is fine but overpriced"

    # Random selection
    random_examples = random.sample(example_pool, 3)
    random_prompt = "Classify sentiment:\n\n"
    for text, label in random_examples:
        random_prompt += f'Text: "{text}"\nSentiment: {label}\n\n'
    random_prompt += f'Text: "{test_input}"\nSentiment:'

    print("RANDOM SELECTION:")
    print(f"Selected: {[ex[1] for ex in random_examples]}")
    random_response = client.complete(random_prompt, temperature=0.2, max_tokens=20)
    print(f"Response: {random_response.strip()}")

    # Curated selection (one of each type)
    curated_examples = [
        ("Amazing product, highly recommend!", "Positive"),
        ("Completely disappointed", "Negative"),
        ("It works as expected", "Neutral"),
    ]
    curated_prompt = "Classify sentiment:\n\n"
    for text, label in curated_examples:
        curated_prompt += f'Text: "{text}"\nSentiment: {label}\n\n'
    curated_prompt += f'Text: "{test_input}"\nSentiment:'

    print("\n" + "-" * 40)
    print("\nCURATED SELECTION:")
    print("Selected: One clear example of each category")
    curated_response = client.complete(curated_prompt, temperature=0.2, max_tokens=20)
    print(f"Response: {curated_response.strip()}")

    print("\nAnalysis: Curated selection ensures balanced representation")


def example_2_similarity_based_selection():
    """Select examples based on similarity to input."""
    print("\n" + "=" * 60)
    print("Example 2: Similarity-Based Selection")
    print("=" * 60)

    client = LLMClient("openai")

    # Example bank with different types of questions
    qa_examples = [
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "type": "geography"
        },
        {
            "question": "How do I reset my password?",
            "answer": "To reset your password, click 'Forgot Password' on the login page.",
            "type": "technical"
        },
        {
            "question": "What are the symptoms of flu?",
            "answer": "Common flu symptoms include fever, cough, and body aches.",
            "type": "medical"
        },
        {
            "question": "What is the capital of Japan?",
            "answer": "The capital of Japan is Tokyo.",
            "type": "geography"
        },
        {
            "question": "How do I update my software?",
            "answer": "To update software, go to Settings > Updates and click 'Check for Updates'.",
            "type": "technical"
        },
        {
            "question": "What causes headaches?",
            "answer": "Headaches can be caused by stress, dehydration, or lack of sleep.",
            "type": "medical"
        }
    ]

    def calculate_similarity(q1: str, q2: str) -> float:
        """Simple word overlap similarity."""
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

    def select_similar_examples(query: str, examples: List[Dict], n: int = 3) -> List[Dict]:
        """Select n most similar examples."""
        similarities = [
            (ex, calculate_similarity(query, ex["question"]))
            for ex in examples
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in similarities[:n]]

    # Test queries
    test_queries = [
        "What is the capital of Germany?",
        "How do I change my email address?",
        "What are signs of dehydration?"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # Select similar examples
        selected = select_similar_examples(query, qa_examples, n=2)
        print(f"Selected examples: {[ex['type'] for ex in selected]}")

        # Build prompt
        prompt = "Answer questions based on these examples:\n\n"
        for ex in selected:
            prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
        prompt += f"Q: {query}\nA:"

        response = client.complete(prompt, temperature=0.3, max_tokens=50)
        print(f"Answer: {response.strip()}")


def example_3_diversity_selection():
    """Select diverse examples to cover edge cases."""
    print("\n" + "=" * 60)
    print("Example 3: Diversity in Example Selection")
    print("=" * 60)

    client = LLMClient("openai")

    # Examples with different characteristics
    text_examples = [
        {"text": "Hi", "length": "very_short", "formality": "informal"},
        {"text": "Good morning, how are you?", "length": "short", "formality": "formal"},
        {"text": "The quarterly financial report shows a 15% increase in revenue compared to the previous year, driven primarily by strong performance in the Asian markets.", "length": "long", "formality": "formal"},
        {"text": "hey whats up", "length": "very_short", "formality": "very_informal"},
        {"text": "I would like to schedule a meeting to discuss the project timeline.", "length": "medium", "formality": "formal"},
        {"text": "Can't make it tonight, sorry!", "length": "short", "formality": "informal"},
    ]

    def select_diverse_examples(examples: List[Dict], n: int = 3) -> List[Dict]:
        """Select diverse examples covering different characteristics."""
        selected = []
        characteristics_covered = set()

        for ex in examples:
            # Check if this example adds new diversity
            chars = (ex["length"], ex["formality"])
            if chars not in characteristics_covered:
                selected.append(ex)
                characteristics_covered.add(chars)
                if len(selected) >= n:
                    break

        return selected

    # Task: Make text more formal
    diverse_examples = select_diverse_examples(text_examples)

    prompt = "Make these texts more formal:\n\n"
    for ex in diverse_examples:
        formal_version = ex["text"]
        if ex["formality"] == "informal":
            formal_version = "Good day" if ex["text"] == "Hi" else ex["text"].capitalize()
        elif ex["formality"] == "very_informal":
            formal_version = "Hello, how are you doing?"

        prompt += f'Informal: "{ex["text"]}"\n'
        prompt += f'Formal: "{formal_version}"\n\n'

    test_text = "gonna grab lunch, wanna come?"
    prompt += f'Informal: "{test_text}"\nFormal:'

    print("DIVERSE EXAMPLE SELECTION:")
    print(f"Selected {len(diverse_examples)} diverse examples")
    print(f"Covering: {[(ex['length'], ex['formality']) for ex in diverse_examples]}")

    response = client.complete(prompt, temperature=0.3, max_tokens=50)
    print(f"\nInput: '{test_text}'")
    print(f"Formalized: {response.strip()}")


def example_4_balanced_selection():
    """Ensure balanced representation of all categories."""
    print("\n" + "=" * 60)
    print("Example 4: Balanced Category Selection")
    print("=" * 60)

    client = LLMClient("openai")

    # Imbalanced example pool
    category_examples = {
        "Technical": [
            "API rate limiting prevents server overload",
            "Database indexing improves query speed",
            "Cache invalidation is a complex problem",
            "Microservices enable independent scaling",
            "Load balancing distributes traffic evenly",
        ],
        "Business": [
            "Q3 revenue exceeded projections",
            "Customer retention improved by 20%",
        ],
        "Scientific": [
            "Photosynthesis converts light to energy",
        ]
    }

    def balanced_selection(examples_dict: Dict[str, List], examples_per_category: int = 1) -> List[Tuple[str, str]]:
        """Select balanced examples from each category."""
        selected = []
        for category, examples in examples_dict.items():
            # Take up to examples_per_category from each
            n_select = min(examples_per_category, len(examples))
            for ex in random.sample(examples, n_select):
                selected.append((ex, category))
        random.shuffle(selected)  # Mix categories
        return selected

    # Without balancing (might be biased toward Technical)
    all_examples = [(ex, cat) for cat, exs in category_examples.items() for ex in exs]
    unbalanced = random.sample(all_examples, min(3, len(all_examples)))

    print("UNBALANCED SELECTION:")
    for text, category in unbalanced:
        print(f"  [{category}] {text[:50]}...")

    # With balancing
    balanced = balanced_selection(category_examples, examples_per_category=1)

    print("\n" + "-" * 40)
    print("\nBALANCED SELECTION:")
    for text, category in balanced:
        print(f"  [{category}] {text[:50]}...")

    # Create classification prompt
    prompt = "Classify these texts by domain:\n\n"
    for text, category in balanced:
        prompt += f'Text: "{text}"\nDomain: {category}\n\n'

    test_text = "Machine learning models require training data"
    prompt += f'Text: "{test_text}"\nDomain:'

    response = client.complete(prompt, temperature=0.2, max_tokens=20)
    print(f"\nTest: '{test_text}'")
    print(f"Classification: {response.strip()}")


def example_5_example_ordering():
    """Show how example ordering affects performance."""
    print("\n" + "=" * 60)
    print("Example 5: Impact of Example Ordering")
    print("=" * 60)

    client = LLMClient("openai")

    # Examples of increasing complexity
    math_examples = [
        ("2 + 3", "5"),
        ("10 - 4", "6"),
        ("(5 + 3) * 2", "16"),
        ("15 / 3 + 2", "7"),
        ("(10 - 3) * (4 + 1)", "35"),
    ]

    test_problem = "((8 + 2) * 3) - 5"

    # Random order
    random_order = math_examples.copy()
    random.shuffle(random_order)

    random_prompt = "Solve these math problems:\n\n"
    for problem, answer in random_order[:3]:
        random_prompt += f"Problem: {problem}\nAnswer: {answer}\n\n"
    random_prompt += f"Problem: {test_problem}\nAnswer:"

    print("RANDOM ORDER:")
    print([p for p, _ in random_order[:3]])
    random_response = client.complete(random_prompt, temperature=0.1, max_tokens=20)
    print(f"Response: {random_response.strip()}")

    # Progressive order (simple to complex)
    progressive_prompt = "Solve these math problems:\n\n"
    for problem, answer in math_examples[:3]:
        progressive_prompt += f"Problem: {problem}\nAnswer: {answer}\n\n"
    progressive_prompt += f"Problem: {test_problem}\nAnswer:"

    print("\n" + "-" * 40)
    print("\nPROGRESSIVE ORDER (Simple → Complex):")
    print([p for p, _ in math_examples[:3]])
    progressive_response = client.complete(progressive_prompt, temperature=0.1, max_tokens=20)
    print(f"Response: {progressive_response.strip()}")

    # Most similar first
    def complexity_score(expr):
        return expr.count('(') * 2 + expr.count('+') + expr.count('-') + expr.count('*') + expr.count('/')

    test_complexity = complexity_score(test_problem)
    similarities = [(p, a, abs(complexity_score(p) - test_complexity)) for p, a in math_examples]
    similarities.sort(key=lambda x: x[2])

    similar_prompt = "Solve these math problems:\n\n"
    for problem, answer, _ in similarities[:3]:
        similar_prompt += f"Problem: {problem}\nAnswer: {answer}\n\n"
    similar_prompt += f"Problem: {test_problem}\nAnswer:"

    print("\n" + "-" * 40)
    print("\nSIMILAR COMPLEXITY FIRST:")
    print([p for p, _, _ in similarities[:3]])
    similar_response = client.complete(similar_prompt, temperature=0.1, max_tokens=20)
    print(f"Response: {similar_response.strip()}")

    print("\nAnalysis: Similar complexity examples often yield better results")


def example_6_adaptive_selection():
    """Dynamically adapt example selection based on context."""
    print("\n" + "=" * 60)
    print("Example 6: Adaptive Example Selection")
    print("=" * 60)

    client = LLMClient("openai")

    # Comprehensive example bank
    example_bank = {
        "code_review": [
            {
                "input": "def add(a,b): return a+b",
                "output": "Consider adding type hints and docstring for clarity",
                "language": "python",
                "issue_type": "documentation"
            },
            {
                "input": "while True: process()",
                "output": "Add break condition to prevent infinite loop",
                "language": "python",
                "issue_type": "logic"
            },
            {
                "input": "password = '12345'",
                "output": "Never hardcode passwords. Use environment variables",
                "language": "python",
                "issue_type": "security"
            },
            {
                "input": "SELECT * FROM users",
                "output": "Specify needed columns instead of using SELECT *",
                "language": "sql",
                "issue_type": "performance"
            },
            {
                "input": "var x = document.getElementById('id').value",
                "output": "Check if element exists before accessing value",
                "language": "javascript",
                "issue_type": "error_handling"
            }
        ]
    }

    def detect_context(code: str) -> Dict:
        """Detect code context for example selection."""
        context = {
            "language": "unknown",
            "has_loop": False,
            "has_hardcoded": False,
            "has_sql": False
        }

        code_lower = code.lower()

        # Detect language
        if "def " in code or "import " in code:
            context["language"] = "python"
        elif "function" in code or "var " in code or "const " in code:
            context["language"] = "javascript"
        elif "select " in code_lower or "from " in code_lower:
            context["language"] = "sql"

        # Detect patterns
        context["has_loop"] = "while" in code_lower or "for" in code_lower
        context["has_hardcoded"] = "password" in code_lower or "token" in code_lower
        context["has_sql"] = "select" in code_lower

        return context

    def select_adaptive_examples(code: str, bank: List[Dict], n: int = 2) -> List[Dict]:
        """Select examples based on code context."""
        context = detect_context(code)
        scored_examples = []

        for ex in bank:
            score = 0
            # Language match
            if ex["language"] == context["language"]:
                score += 3
            # Pattern match
            if context["has_loop"] and ex["issue_type"] == "logic":
                score += 2
            if context["has_hardcoded"] and ex["issue_type"] == "security":
                score += 2
            if context["has_sql"] and ex["language"] == "sql":
                score += 2

            scored_examples.append((ex, score))

        # Sort by relevance score
        scored_examples.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in scored_examples[:n]]

    # Test with different code snippets
    test_codes = [
        "def process_user(id): data = get_user(id); return data['name']",
        "SELECT * FROM orders WHERE user_id = '$user_input'",
        "api_key = 'sk-1234567890abcdef'"
    ]

    for test_code in test_codes:
        print(f"\nCode: {test_code[:60]}...")
        context = detect_context(test_code)
        print(f"Detected: Language={context['language']}")

        selected = select_adaptive_examples(test_code, example_bank["code_review"])

        # Build prompt
        prompt = "Review this code:\n\n"
        for ex in selected:
            prompt += f"Code: {ex['input']}\nReview: {ex['output']}\n\n"
        prompt += f"Code: {test_code}\nReview:"

        response = client.complete(prompt, temperature=0.3, max_tokens=100)
        print(f"Review: {response.strip()}")


def run_all_examples():
    """Run all example selection examples."""
    examples = [
        example_1_random_vs_curated,
        example_2_similarity_based_selection,
        example_3_diversity_selection,
        example_4_balanced_selection,
        example_5_example_ordering,
        example_6_adaptive_selection
    ]

    for example in examples:
        try:
            example()
            print("\n" + "=" * 60 + "\n")
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 03: Example Selection")
    parser.add_argument("--example", type=int, help="Run specific example (1-6)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_random_vs_curated,
            2: example_2_similarity_based_selection,
            3: example_3_diversity_selection,
            4: example_4_balanced_selection,
            5: example_5_example_ordering,
            6: example_6_adaptive_selection
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Run with --all for all examples or --example N for a specific example")
        print("\nAvailable examples:")
        print("1. Random vs Curated Selection")
        print("2. Similarity-Based Selection")
        print("3. Diversity Selection")
        print("4. Balanced Selection")
        print("5. Example Ordering Impact")
        print("6. Adaptive Selection")