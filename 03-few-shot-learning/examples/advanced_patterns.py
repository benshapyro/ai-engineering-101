"""
Module 03: Advanced Few-Shot Patterns

Complex few-shot techniques including chain-of-thought, multi-task, and meta-learning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json


def example_1_chain_of_thought_few_shot():
    """Combine few-shot learning with chain-of-thought reasoning."""
    print("=" * 60)
    print("Example 1: Chain-of-Thought Few-Shot")
    print("=" * 60)

    client = LLMClient("openai")

    # Few-shot with reasoning steps
    cot_prompt = """Solve these word problems step by step:

Problem: A bakery sells cupcakes for $3 each. If Sarah buys 4 cupcakes and pays with a $20 bill, how much change does she get?
Reasoning:
1. Calculate cost of cupcakes: 4 cupcakes × $3 = $12
2. Calculate change: $20 - $12 = $8
Answer: Sarah gets $8 in change.

Problem: A train travels 240 miles in 3 hours. If it maintains the same speed, how far will it travel in 5 hours?
Reasoning:
1. Calculate speed: 240 miles ÷ 3 hours = 80 miles per hour
2. Calculate distance for 5 hours: 80 mph × 5 hours = 400 miles
Answer: The train will travel 400 miles in 5 hours.

Problem: A restaurant bill is $84. If you want to leave a 20% tip, what is the total amount you should pay?
Reasoning:
1. Calculate tip amount: $84 × 0.20 = $16.80
2. Calculate total: $84 + $16.80 = $100.80
Answer: The total amount to pay is $100.80.

Problem: A swimming pool holds 15,000 gallons. If it's being filled at 25 gallons per minute, how long will it take to fill completely?
Reasoning:"""

    print("CHAIN-OF-THOUGHT FEW-SHOT:")
    print("Teaching both the answer pattern AND reasoning process")
    print("\nTest problem: Pool filling calculation")

    response = client.complete(cot_prompt, temperature=0.2, max_tokens=200)
    print(f"\nResponse:\n{response}")

    print("\nBenefit: Model learns to show work, making answers verifiable")


def example_2_multi_task_few_shot():
    """Single prompt handling multiple task types."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Task Few-Shot")
    print("=" * 60)

    client = LLMClient("openai")

    # Examples covering different tasks
    multi_task_prompt = """Process these requests according to their type:

Request: "Translate to Spanish: Good morning"
Type: Translation
Output: "Buenos días"

Request: "Sentiment of: This product exceeded my expectations!"
Type: Sentiment Analysis
Output: Positive

Request: "Summarize: The meeting discussed Q3 results showing 15% growth, new product launches planned for Q4, and expansion into Asian markets."
Type: Summarization
Output: Q3 showed 15% growth; Q4 product launches and Asian expansion planned.

Request: "Extract email: Contact John at john.doe@example.com or call 555-0123"
Type: Information Extraction
Output: Email: john.doe@example.com

Request: "Translate to French: Thank you"
Type: Translation
Output: "Merci"

Request: "Sentiment of: The service was terrible and the food was cold"
Type: Sentiment Analysis
Output: Negative

Request: "Extract phone: Reach us at support@company.com or 1-800-555-5555"
Type:"""

    print("MULTI-TASK FEW-SHOT:")
    print("Single prompt handling translation, sentiment, summarization, extraction")

    response = client.complete(multi_task_prompt, temperature=0.2, max_tokens=100)
    print(f"\nResponse: {response}")

    # Test with different task
    test_prompt = multi_task_prompt + " Information Extraction\nOutput:"
    response = client.complete(test_prompt, temperature=0.2, max_tokens=50)
    print(f"\nCompleted response: {response.strip()}")

    print("\nBenefit: One versatile prompt handles multiple task types")


def example_3_cross_lingual_few_shot():
    """Few-shot examples work across languages."""
    print("\n" + "=" * 60)
    print("Example 3: Cross-Lingual Few-Shot")
    print("=" * 60)

    client = LLMClient("openai")

    # Examples in different languages teaching same pattern
    cross_lingual_prompt = """Extract the main subject from these sentences:

English: "The cat sleeps on the warm windowsill."
Subject: cat

Spanish: "El perro corre en el parque."
Subject: perro (dog)

French: "Les enfants jouent dans le jardin."
Subject: enfants (children)

German: "Das Auto fährt schnell auf der Autobahn."
Subject: Auto (car)

Japanese: "学生は図書館で勉強しています。"
Subject: 学生 (student)

Italian: "La pizza è deliziosa e calda."
Subject:"""

    print("CROSS-LINGUAL FEW-SHOT:")
    print("Examples in multiple languages teaching subject extraction")

    response = client.complete(cross_lingual_prompt, temperature=0.2, max_tokens=30)
    print(f"\nResponse: {response.strip()}")

    # Test with new language
    test_additions = [
        '\n\nPortuguese: "O livro está sobre a mesa."\nSubject:',
        '\n\nDutch: "De bloemen bloeien in de tuin."\nSubject:',
        '\n\nRussian: "Кошка спит на диване."\nSubject:'
    ]

    for addition in test_additions[:1]:  # Test one
        extended_prompt = cross_lingual_prompt + response.strip() + addition
        new_response = client.complete(extended_prompt, temperature=0.2, max_tokens=30)
        print(f"Additional test: {new_response.strip()}")

    print("\nBenefit: Pattern learning transfers across languages")


def example_4_few_shot_with_constraints():
    """Examples demonstrating specific constraints and rules."""
    print("\n" + "=" * 60)
    print("Example 4: Few-Shot with Constraints")
    print("=" * 60)

    client = LLMClient("openai")

    # Examples showing constraint application
    constraint_prompt = """Rewrite these sentences following ALL constraints:
Constraints:
- Maximum 10 words
- Must include a number
- Use present tense only
- End with exclamation mark

Original: "Yesterday, we completed the project successfully after many attempts."
Rewritten: "We complete 5 projects successfully every week now!"

Original: "The ancient castle was built hundreds of years ago by skilled craftsmen."
Rewritten: "This 500-year-old castle stands magnificently today!"

Original: "She will be traveling to Paris next month for a conference."
Rewritten: "She visits Paris for 3 conferences annually!"

Original: "The committee had been reviewing applications throughout the entire week."
Rewritten: "The committee reviews 50 applications this week!"

Original: "Scientists discovered a new species of butterfly in the Amazon rainforest."
Rewritten:"""

    print("FEW-SHOT WITH CONSTRAINTS:")
    print("Teaching complex multi-constraint application")
    print("\nConstraints: ≤10 words, include number, present tense, end with !")

    response = client.complete(constraint_prompt, temperature=0.3, max_tokens=50)
    print(f"\nResponse: {response.strip()}")

    # Verify constraints
    response_clean = response.strip()
    word_count = len(response_clean.rstrip('!').split())
    has_number = any(char.isdigit() for char in response_clean)
    ends_exclamation = response_clean.endswith('!')

    print(f"\nConstraint Check:")
    print(f"  ≤10 words: {'✓' if word_count <= 10 else '✗'} ({word_count} words)")
    print(f"  Has number: {'✓' if has_number else '✗'}")
    print(f"  Ends with !: {'✓' if ends_exclamation else '✗'}")


def example_5_meta_learning_patterns():
    """Learning to learn - extracting patterns from examples."""
    print("\n" + "=" * 60)
    print("Example 5: Meta-Learning Patterns")
    print("=" * 60)

    client = LLMClient("openai")

    # Teach the model to identify and apply patterns
    meta_learning_prompt = """Identify the transformation pattern and apply it:

Set 1:
Input: "hello" → Output: "HELLO_5"
Input: "world" → Output: "WORLD_5"
Input: "python" → Output: "PYTHON_6"
Pattern: Uppercase the word and append underscore with length

Set 2:
Input: "cat" → Output: "tac"
Input: "dog" → Output: "god"
Input: "bird" → Output: "drib"
Pattern: Reverse the word

Set 3:
Input: "apple" → Output: "a2p2l1e1"
Input: "book" → Output: "b1o2k1"
Input: "good" → Output: "g1o2d1"
Pattern: Show each letter with its count

Now identify and apply this pattern:
Input: "red" → Output: "r#e#d"
Input: "blue" → Output: "b#l#u#e"
Input: "green" → Output: "g#r#e#e#n"
Pattern:"""

    print("META-LEARNING PATTERNS:")
    print("Teaching the model to identify patterns from examples")

    response = client.complete(meta_learning_prompt, temperature=0.3, max_tokens=100)
    print(f"\nIdentified pattern: {response.strip()}")

    # Test pattern application
    test_prompt = meta_learning_prompt + response.strip() + "\n\nApply the pattern:\nInput: \"yellow\" → Output:"

    test_response = client.complete(test_prompt, temperature=0.2, max_tokens=30)
    print(f"Pattern application test: {test_response.strip()}")

    print("\nBenefit: Model learns to extract and generalize patterns")


def example_6_recursive_few_shot():
    """Using outputs as new examples in recursive patterns."""
    print("\n" + "=" * 60)
    print("Example 6: Recursive Few-Shot")
    print("=" * 60)

    client = LLMClient("openai")

    # Start with base examples
    base_prompt = """Progressively elaborate on concepts:

Level 1: "Animal"
Level 2: "Animal → Living creature that can move"
Level 3: "Animal → Living creature that can move → Multicellular organism capable of voluntary movement and response to stimuli"

Level 1: "Computer"
Level 2: "Computer → Electronic device that processes data"
Level 3: "Computer → Electronic device that processes data → Programmable machine that manipulates information according to instructions"

Level 1: "Democracy"
Level 2:"""

    print("RECURSIVE FEW-SHOT:")
    print("Building on previous outputs recursively")

    # Get Level 2
    response_2 = client.complete(base_prompt, temperature=0.3, max_tokens=50)
    print(f"\nLevel 2: {response_2.strip()}")

    # Use Level 2 to get Level 3
    extended_prompt = base_prompt + response_2.strip() + "\nLevel 3:"
    response_3 = client.complete(extended_prompt, temperature=0.3, max_tokens=100)
    print(f"Level 3: {response_3.strip()}")

    # Try with new concept
    new_concept_prompt = extended_prompt + response_3.strip() + "\n\nLevel 1: \"Algorithm\"\nLevel 2:"
    new_response = client.complete(new_concept_prompt, temperature=0.3, max_tokens=100)
    print(f"\nNew concept elaboration: {new_response.strip()}")

    print("\nBenefit: Builds increasingly complex outputs using previous results")


def example_7_contrastive_few_shot():
    """Using contrasting examples to clarify boundaries."""
    print("\n" + "=" * 60)
    print("Example 7: Contrastive Few-Shot")
    print("=" * 60)

    client = LLMClient("openai")

    # Show what IS and ISN'T correct
    contrastive_prompt = """Identify valid email addresses using these examples:

✓ VALID: john.doe@example.com
  Reason: Standard format with name, @, domain

✗ INVALID: john.doe@
  Reason: Missing domain after @

✓ VALID: user+tag@company.co.uk
  Reason: Plus addressing and subdomain are allowed

✗ INVALID: user@company@example.com
  Reason: Multiple @ symbols not allowed

✓ VALID: first_last@my-company.org
  Reason: Underscores and hyphens are valid

✗ INVALID: user name@example.com
  Reason: Spaces not allowed in email addresses

✓ VALID: info@website.io
  Reason: Modern TLDs like .io are valid

✗ INVALID: @example.com
  Reason: Missing local part before @

Check: admin.user@internal-server.local
Result:"""

    print("CONTRASTIVE FEW-SHOT:")
    print("Using positive and negative examples to define boundaries")

    response = client.complete(contrastive_prompt, temperature=0.2, max_tokens=100)
    print(f"\nResponse: {response.strip()}")

    # Test edge cases
    edge_cases = [
        "user.name+filter@email.example.com",
        "user..name@example.com",
        ".username@example.com"
    ]

    for email in edge_cases[:1]:
        test_prompt = contrastive_prompt + response.strip() + f"\n\nCheck: {email}\nResult:"
        test_response = client.complete(test_prompt, temperature=0.2, max_tokens=100)
        print(f"\nEdge case '{email}': {test_response.strip()}")

    print("\nBenefit: Clear boundary definition through contrasting examples")


def run_all_examples():
    """Run all advanced pattern examples."""
    examples = [
        example_1_chain_of_thought_few_shot,
        example_2_multi_task_few_shot,
        example_3_cross_lingual_few_shot,
        example_4_few_shot_with_constraints,
        example_5_meta_learning_patterns,
        example_6_recursive_few_shot,
        example_7_contrastive_few_shot
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

    parser = argparse.ArgumentParser(description="Module 03: Advanced Patterns")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_chain_of_thought_few_shot,
            2: example_2_multi_task_few_shot,
            3: example_3_cross_lingual_few_shot,
            4: example_4_few_shot_with_constraints,
            5: example_5_meta_learning_patterns,
            6: example_6_recursive_few_shot,
            7: example_7_contrastive_few_shot
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Run with --all for all examples or --example N for a specific example")
        print("\nAvailable examples:")
        print("1. Chain-of-Thought Few-Shot")
        print("2. Multi-Task Few-Shot")
        print("3. Cross-Lingual Few-Shot")
        print("4. Few-Shot with Constraints")
        print("5. Meta-Learning Patterns")
        print("6. Recursive Few-Shot")
        print("7. Contrastive Few-Shot")