"""
Module 06: Role-Based Prompting - Exercises

Practice exercises for mastering role-based prompting techniques.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from typing import Dict, List, Optional


# ===== Exercise 1: Design Effective Personas =====

def exercise_1_design_personas():
    """
    Exercise 1: Design effective personas for different use cases.

    TODO:
    1. Create detailed personas for 3 different domains
    2. Include background, expertise, personality, and constraints
    3. Test each persona with the same question
    4. Compare the responses
    """
    client = LLMClient("openai")

    print("Exercise 1: Design Effective Personas")
    print("=" * 50)

    # Universal question to test all personas
    test_question = "How should we handle data validation in our application?"

    print(f"Test Question: {test_question}\n")

    # TODO: Design three detailed personas
    personas = [
        {
            "domain": "Security",
            "role": "TODO: Define security expert role",
            "background": "TODO: Add background",
            "expertise": "TODO: List expertise areas",
            "personality": "TODO: Define personality traits",
            "constraints": "TODO: Add constraints"
        },
        {
            "domain": "User Experience",
            "role": "TODO: Define UX expert role",
            "background": "TODO",
            "expertise": "TODO",
            "personality": "TODO",
            "constraints": "TODO"
        },
        {
            "domain": "Performance",
            "role": "TODO: Define performance expert role",
            "background": "TODO",
            "expertise": "TODO",
            "personality": "TODO",
            "constraints": "TODO"
        }
    ]

    # TODO: Create prompts for each persona
    for persona in personas:
        print(f"\n{persona['domain']} Expert:")
        print("-" * 40)

        # TODO: Build complete persona prompt
        prompt = f"""TODO: Combine all persona attributes into a prompt

Question: {test_question}"""

        print("TODO: Test persona and analyze response")


# ===== Exercise 2: Role Consistency =====

def exercise_2_role_consistency():
    """
    Exercise 2: Maintain role consistency across multiple interactions.

    TODO:
    1. Define a specific role with clear characteristics
    2. Ask 5 related questions
    3. Ensure responses maintain consistent expertise and personality
    4. Implement a consistency checker
    """
    client = LLMClient("openai")

    print("Exercise 2: Role Consistency")
    print("=" * 50)

    # TODO: Define a consistent role
    role_definition = """TODO: Create a detailed, specific role
    Include:
    - Name and title
    - Specific expertise
    - Communication style
    - Common phrases or patterns
    - Knowledge boundaries"""

    # Series of related questions
    questions = [
        "What's your approach to problem-solving?",
        "How do you stay current in your field?",
        "What's the biggest challenge you face?",
        "What advice would you give to beginners?",
        "What tools do you recommend?"
    ]

    print("Testing Role Consistency:")
    print("-" * 40)

    responses = []
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")

        # TODO: Maintain role across questions
        prompt = f"""{role_definition}

Question: {question}"""

        # TODO: Generate response and check consistency
        print("TODO: Generate and analyze response")
        responses.append("TODO: Store response")

    # TODO: Implement consistency checker
    def check_consistency(responses):
        """TODO: Check if responses maintain consistent role characteristics."""
        pass

    print("\nTODO: Analyze consistency across all responses")


# ===== Exercise 3: Multi-Perspective Analysis =====

def exercise_3_multi_perspective():
    """
    Exercise 3: Analyze the same problem from multiple role perspectives.

    TODO:
    1. Define a complex problem
    2. Create 4 different expert roles
    3. Get each role's perspective
    4. Synthesize insights into a comprehensive solution
    """
    client = LLMClient("openai")

    print("Exercise 3: Multi-Perspective Analysis")
    print("=" * 50)

    # Complex problem requiring multiple perspectives
    problem = """Our e-commerce platform is experiencing:
    - 20% cart abandonment increase
    - Customer complaints about checkout process
    - Payment failures during high traffic
    - Mobile users reporting UI issues"""

    print(f"Problem:\n{problem}\n")

    # TODO: Define expert roles
    experts = [
        {
            "role": "TODO: UX Researcher",
            "focus": "TODO: User behavior and experience"
        },
        {
            "role": "TODO: Backend Engineer",
            "focus": "TODO: System performance"
        },
        {
            "role": "TODO: Data Analyst",
            "focus": "TODO: Metrics and patterns"
        },
        {
            "role": "TODO: Business Strategist",
            "focus": "TODO: Business impact"
        }
    ]

    analyses = []

    # TODO: Collect each expert's analysis
    for expert in experts:
        print(f"\n{expert['role']}:")
        print("-" * 30)

        # TODO: Create expert prompt
        prompt = f"""TODO: Define expert role and focus

Problem: {problem}

Your analysis:"""

        print("TODO: Get expert analysis")
        analyses.append("TODO: Store analysis")

    # TODO: Synthesize all perspectives
    print("\nSynthesis:")
    print("-" * 30)
    print("TODO: Combine all expert insights into comprehensive solution")


# ===== Exercise 4: Dynamic Role Selection =====

def exercise_4_dynamic_role_selection():
    """
    Exercise 4: Implement dynamic role selection based on query type.

    TODO:
    1. Create a role registry with different experts
    2. Implement query classification
    3. Automatically select appropriate role
    4. Handle queries requiring multiple roles
    """
    client = LLMClient("openai")

    print("Exercise 4: Dynamic Role Selection")
    print("=" * 50)

    # TODO: Create role registry
    role_registry = {
        "technical": {
            "triggers": ["bug", "error", "code", "performance"],
            "role": "TODO: Senior Software Engineer role"
        },
        "business": {
            "triggers": ["roi", "cost", "budget", "revenue"],
            "role": "TODO: Business Analyst role"
        },
        "security": {
            "triggers": ["vulnerability", "breach", "authentication", "encryption"],
            "role": "TODO: Security Expert role"
        },
        "data": {
            "triggers": ["analytics", "metrics", "dashboard", "reporting"],
            "role": "TODO: Data Scientist role"
        }
    }

    # Test queries
    queries = [
        "We found a SQL injection vulnerability",
        "What's the ROI of this new feature?",
        "The dashboard is loading slowly",
        "How do we implement OAuth2?",
        "Analyze user engagement metrics"
    ]

    # TODO: Implement role selection logic
    def select_role(query, registry):
        """TODO: Select appropriate role based on query content."""
        selected_role = "TODO: Implement selection logic"
        return selected_role

    # TODO: Process each query
    for query in queries:
        print(f"\nQuery: {query}")

        # TODO: Select and apply role
        selected_role = select_role(query, role_registry)
        print(f"Selected Role: TODO")

        # TODO: Generate response with selected role
        print("Response: TODO")


# ===== Exercise 5: Role Evolution =====

def exercise_5_role_evolution():
    """
    Exercise 5: Evolve a role based on interaction history.

    TODO:
    1. Start with a basic role
    2. Track user interactions and preferences
    3. Adapt the role's behavior over time
    4. Maintain core expertise while personalizing style
    """
    client = LLMClient("openai")

    print("Exercise 5: Role Evolution")
    print("=" * 50)

    # TODO: Define base role
    class EvolvingRole:
        def __init__(self):
            self.base_role = "TODO: Define base expert role"
            self.interaction_history = []
            self.user_preferences = {}
            self.adaptation_rules = {}

        def interact(self, query):
            """TODO: Process query and adapt role."""
            # TODO: Build adapted prompt
            prompt = self.base_role

            # TODO: Apply adaptations based on history
            if self.user_preferences:
                prompt += f"\nUser preferences: TODO"

            # TODO: Generate response
            response = "TODO: Generate response"

            # TODO: Learn from interaction
            self.learn_from_interaction(query, response)

            return response

        def learn_from_interaction(self, query, response):
            """TODO: Extract learnings and adapt role."""
            pass

    # Simulate evolving interactions
    role = EvolvingRole()

    interactions = [
        {"query": "Explain database indexing", "feedback": "too technical"},
        {"query": "How do indexes improve performance?", "feedback": "better"},
        {"query": "Can you give a real example?", "feedback": "perfect"},
        {"query": "What about compound indexes?", "feedback": "good"}
    ]

    print("Simulating role evolution through interactions:\n")

    for i, interaction in enumerate(interactions, 1):
        print(f"Interaction {i}:")
        print(f"Query: {interaction['query']}")

        # TODO: Process with evolving role
        response = role.interact(interaction['query'])
        print(f"Response: TODO")

        # TODO: Apply feedback
        print(f"Feedback: {interaction['feedback']}")
        print("Role adapted: TODO\n")


# ===== Challenge: Build a Role Management System =====

def challenge_role_management_system():
    """
    Challenge: Build a complete role management system.

    Requirements:
    1. Role definition and storage
    2. Role selection based on context
    3. Role consistency enforcement
    4. Multi-role collaboration
    5. Performance tracking

    TODO: Complete the implementation
    """
    client = LLMClient("openai")

    print("Challenge: Role Management System")
    print("=" * 50)

    class RoleManager:
        def __init__(self):
            self.roles = {}
            self.active_role = None
            self.context = []
            self.performance_metrics = {}

        def define_role(self, name, definition):
            """TODO: Store role definition."""
            self.roles[name] = {
                "definition": definition,
                "usage_count": 0,
                "avg_quality": 0.0
            }

        def select_role(self, query, context=None):
            """TODO: Select best role for query."""
            # TODO: Implement role selection logic
            pass

        def apply_role(self, role_name, query):
            """TODO: Apply role to generate response."""
            # TODO: Build prompt with role
            # TODO: Track performance
            pass

        def collaborate(self, roles, task):
            """TODO: Multiple roles collaborate on task."""
            # TODO: Orchestrate multi-role collaboration
            pass

        def evaluate_performance(self, role_name, response, expected):
            """TODO: Track role performance."""
            # TODO: Calculate quality metrics
            pass

    # TODO: Test the role management system
    manager = RoleManager()

    # Define roles
    print("TODO: Define multiple roles")

    # Test role selection
    print("TODO: Test automatic role selection")

    # Test collaboration
    print("TODO: Test multi-role collaboration")

    # Evaluate performance
    print("TODO: Show performance metrics")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 06: Role-Based Exercises")
    parser.add_argument("--exercise", type=int, help="Run specific exercise (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge")
    parser.add_argument("--all", action="store_true", help="Run all exercises")

    args = parser.parse_args()

    exercises = {
        1: exercise_1_design_personas,
        2: exercise_2_role_consistency,
        3: exercise_3_multi_perspective,
        4: exercise_4_dynamic_role_selection,
        5: exercise_5_role_evolution
    }

    if args.all:
        for ex_num, ex_func in exercises.items():
            print(f"\n{'='*60}")
            ex_func()
            print(f"\n{'='*60}\n")
        challenge_role_management_system()
    elif args.challenge:
        challenge_role_management_system()
    elif args.exercise and args.exercise in exercises:
        exercises[args.exercise]()
    else:
        print("Module 06: Role-Based Prompting - Exercises")
        print("\nUsage:")
        print("  python exercises.py --exercise N  # Run exercise N")
        print("  python exercises.py --challenge    # Run challenge")
        print("  python exercises.py --all          # Run all exercises")
        print("\nAvailable exercises:")
        print("  1: Design Effective Personas")
        print("  2: Role Consistency")
        print("  3: Multi-Perspective Analysis")
        print("  4: Dynamic Role Selection")
        print("  5: Role Evolution")
        print("  Challenge: Role Management System")