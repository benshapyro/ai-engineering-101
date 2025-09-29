"""
Module 06: Role-Based Prompting - Solutions

Complete solutions for all exercises demonstrating mastery of role-based prompting.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from typing import Dict, List, Optional
import re


# ===== Exercise 1 Solution: Design Effective Personas =====

def solution_1_design_personas():
    """
    Solution: Create detailed personas with distinct expertise and personalities.
    """
    client = LLMClient("openai")

    print("Solution 1: Design Effective Personas")
    print("=" * 50)

    test_question = "How should we handle data validation in our application?"

    print(f"Test Question: {test_question}\n")

    # Well-designed personas with complete details
    personas = [
        {
            "domain": "Security",
            "role": "Senior Security Engineer",
            "background": "10 years in cybersecurity, former penetration tester",
            "expertise": "OWASP, input sanitization, SQL injection prevention, XSS protection",
            "personality": "Cautious, detail-oriented, always considers worst-case scenarios",
            "constraints": "Zero tolerance for security vulnerabilities, compliance-focused"
        },
        {
            "domain": "User Experience",
            "role": "Lead UX Designer",
            "background": "Background in cognitive psychology, 8 years in product design",
            "expertise": "User research, accessibility standards, interaction design, A/B testing",
            "personality": "Empathetic, user-advocate, values simplicity and clarity",
            "constraints": "Must maintain usability for non-technical users, WCAG compliance"
        },
        {
            "domain": "Performance",
            "role": "Principal Performance Engineer",
            "background": "Former game engine developer, specialized in optimization",
            "expertise": "Caching strategies, database optimization, async processing, profiling",
            "personality": "Pragmatic, data-driven, obsessed with metrics",
            "constraints": "Sub-100ms response times, horizontal scalability requirements"
        }
    ]

    for persona in personas:
        print(f"\n{persona['domain']} Expert:")
        print("-" * 40)

        # Build comprehensive persona prompt
        prompt = f"""You are a {persona['role']} with the following profile:

Background: {persona['background']}
Expertise: {persona['expertise']}
Personality: {persona['personality']}
Constraints: {persona['constraints']}

Answer from your unique perspective, emphasizing your domain concerns.

Question: {test_question}"""

        response = client.complete(prompt, temperature=0.3, max_tokens=200)
        print(f"{response.strip()}")

    print("\n💡 Each persona provides domain-specific insights")


# ===== Exercise 2 Solution: Role Consistency =====

def solution_2_role_consistency():
    """
    Solution: Maintain consistent role characteristics across interactions.
    """
    client = LLMClient("openai")

    print("\nSolution 2: Role Consistency")
    print("=" * 50)

    # Detailed, consistent role definition
    role_definition = """You are Dr. Eleanor Martinez, PhD in Computer Science from Stanford.

Title: Chief Technology Officer at TechVision Inc.
Experience: 15 years in distributed systems and machine learning

Communication Style:
- Always starts responses with concrete examples
- Uses the phrase "In my experience..." frequently
- Explains complex topics with analogies to cooking (personal hobby)
- Ends responses with actionable next steps

Expertise:
- Distributed systems architecture
- Machine learning at scale
- Team leadership and mentoring
- Strategic technology planning

Knowledge Boundaries:
- Admits when something is outside expertise
- Refers to specific team members for specialized topics
- Values empirical evidence over speculation"""

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

        prompt = f"""{role_definition}

Maintain your consistent personality and expertise.

Question: {question}"""

        response = client.complete(prompt, temperature=0.3, max_tokens=150)
        print(f"A: {response.strip()}")
        responses.append(response)

    # Consistency checker
    def check_consistency(responses):
        """Check if responses maintain consistent role characteristics."""
        consistency_markers = {
            "examples": 0,
            "experience_phrase": 0,
            "cooking_analogies": 0,
            "actionable_steps": 0
        }

        for response in responses:
            if "example" in response.lower() or "instance" in response.lower():
                consistency_markers["examples"] += 1
            if "in my experience" in response.lower():
                consistency_markers["experience_phrase"] += 1
            if any(word in response.lower() for word in ["recipe", "ingredient", "cook", "dish", "kitchen"]):
                consistency_markers["cooking_analogies"] += 1
            if any(phrase in response.lower() for phrase in ["next step", "recommend", "start by", "try"]):
                consistency_markers["actionable_steps"] += 1

        return consistency_markers

    print("\nConsistency Analysis:")
    print("-" * 40)
    markers = check_consistency(responses)
    for marker, count in markers.items():
        consistency = "✓" if count >= 3 else "⚠"
        print(f"{consistency} {marker}: {count}/{len(responses)} responses")


# ===== Exercise 3 Solution: Multi-Perspective Analysis =====

def solution_3_multi_perspective():
    """
    Solution: Analyze problems from multiple expert perspectives.
    """
    client = LLMClient("openai")

    print("\nSolution 3: Multi-Perspective Analysis")
    print("=" * 50)

    problem = """Our e-commerce platform is experiencing:
    - 20% cart abandonment increase
    - Customer complaints about checkout process
    - Payment failures during high traffic
    - Mobile users reporting UI issues"""

    print(f"Problem:\n{problem}\n")

    experts = [
        {
            "role": "Senior UX Researcher",
            "focus": "User behavior patterns and pain points",
            "approach": "Data-driven user research and testing"
        },
        {
            "role": "Principal Backend Engineer",
            "focus": "System reliability and performance",
            "approach": "Technical architecture and infrastructure"
        },
        {
            "role": "Lead Data Analyst",
            "focus": "Metrics, trends, and statistical patterns",
            "approach": "Quantitative analysis and correlation"
        },
        {
            "role": "VP of Business Strategy",
            "focus": "Revenue impact and competitive position",
            "approach": "Business metrics and market analysis"
        }
    ]

    analyses = []

    for expert in experts:
        print(f"\n{expert['role']}:")
        print("-" * 30)

        prompt = f"""You are a {expert['role']} focusing on {expert['focus']}.
Your approach: {expert['approach']}

Analyze this problem from your perspective, providing:
1. Root cause hypothesis
2. Immediate recommendations
3. Long-term solutions

Problem: {problem}

Your analysis:"""

        response = client.complete(prompt, temperature=0.3, max_tokens=200)
        analysis = response.strip()
        print(analysis)
        analyses.append({"expert": expert["role"], "analysis": analysis})

    # Synthesize all perspectives
    print("\nSynthesis:")
    print("-" * 30)

    synthesis_prompt = f"""As a Chief Product Officer, synthesize these expert analyses into a comprehensive action plan:

{chr(10).join([f"{a['expert']}: {a['analysis'][:100]}..." for a in analyses])}

Create a prioritized action plan that addresses all perspectives:"""

    synthesis = client.complete(synthesis_prompt, temperature=0.3, max_tokens=200)
    print(synthesis.strip())


# ===== Exercise 4 Solution: Dynamic Role Selection =====

def solution_4_dynamic_role_selection():
    """
    Solution: Implement intelligent role selection based on query analysis.
    """
    client = LLMClient("openai")

    print("\nSolution 4: Dynamic Role Selection")
    print("=" * 50)

    # Comprehensive role registry
    role_registry = {
        "technical": {
            "triggers": ["bug", "error", "code", "performance", "api", "database", "algorithm"],
            "role": "Senior Software Engineer with 10+ years experience in debugging and optimization",
            "style": "Technical, precise, includes code examples"
        },
        "business": {
            "triggers": ["roi", "cost", "budget", "revenue", "profit", "market", "customer"],
            "role": "Business Analyst specializing in tech ROI and market analysis",
            "style": "Business-focused, metrics-driven, strategic"
        },
        "security": {
            "triggers": ["vulnerability", "breach", "authentication", "encryption", "hack", "password"],
            "role": "Security Expert certified in CISSP and ethical hacking",
            "style": "Security-first, paranoid, compliance-aware"
        },
        "data": {
            "triggers": ["analytics", "metrics", "dashboard", "reporting", "visualization", "insights"],
            "role": "Data Scientist with expertise in analytics and ML",
            "style": "Data-driven, statistical, visualization-focused"
        }
    }

    queries = [
        "We found a SQL injection vulnerability",
        "What's the ROI of this new feature?",
        "The dashboard is loading slowly",
        "How do we implement OAuth2?",
        "Analyze user engagement metrics"
    ]

    def select_role(query, registry):
        """Select appropriate role based on query content."""
        query_lower = query.lower()
        scores = {}

        for role_type, config in registry.items():
            score = sum(1 for trigger in config["triggers"] if trigger in query_lower)
            scores[role_type] = score

        # Select role with highest score, default to technical
        selected = max(scores, key=scores.get)
        if scores[selected] == 0:
            selected = "technical"

        return selected, registry[selected]

    for query in queries:
        print(f"\nQuery: {query}")

        role_type, role_config = select_role(query, role_registry)
        print(f"Selected Role: {role_type.capitalize()} Expert")

        prompt = f"""You are a {role_config['role']}.
Communication style: {role_config['style']}

Provide expert guidance for: {query}"""

        response = client.complete(prompt, temperature=0.3, max_tokens=150)
        print(f"Response: {response.strip()}")


# ===== Exercise 5 Solution: Role Evolution =====

def solution_5_role_evolution():
    """
    Solution: Implement adaptive role evolution based on interactions.
    """
    client = LLMClient("openai")

    print("\nSolution 5: Role Evolution")
    print("=" * 50)

    class EvolvingRole:
        def __init__(self):
            self.base_role = "Technical Expert specializing in databases and system architecture"
            self.interaction_history = []
            self.user_preferences = {
                "detail_level": "medium",
                "use_examples": True,
                "technical_depth": "intermediate"
            }
            self.adaptation_rules = {
                "too technical": {"technical_depth": "beginner", "use_examples": True},
                "too simple": {"technical_depth": "advanced", "detail_level": "high"},
                "perfect": {"keep_current": True},
                "needs examples": {"use_examples": True, "detail_level": "high"}
            }

        def interact(self, query, client):
            """Process query with adapted role."""
            # Build adapted prompt
            prompt = f"""{self.base_role}

User preferences:
- Detail level: {self.user_preferences['detail_level']}
- Include examples: {self.user_preferences['use_examples']}
- Technical depth: {self.user_preferences['technical_depth']}

Adapt your response accordingly.

Question: {query}"""

            response = client.complete(prompt, temperature=0.3, max_tokens=150)
            return response.strip()

        def learn_from_interaction(self, query, response, feedback):
            """Extract learnings and adapt role based on feedback."""
            self.interaction_history.append({
                "query": query,
                "response": response[:50],
                "feedback": feedback
            })

            # Apply adaptation rules
            if feedback in self.adaptation_rules:
                rules = self.adaptation_rules[feedback]
                if "keep_current" not in rules:
                    self.user_preferences.update(rules)

            return f"Adapted: {', '.join([f'{k}={v}' for k, v in rules.items() if k != 'keep_current'])}"

    role = EvolvingRole()

    interactions = [
        {"query": "Explain database indexing", "feedback": "too technical"},
        {"query": "How do indexes improve performance?", "feedback": "perfect"},
        {"query": "Can you give a real example?", "feedback": "needs examples"},
        {"query": "What about compound indexes?", "feedback": "perfect"}
    ]

    print("Simulating role evolution through interactions:\n")

    for i, interaction in enumerate(interactions, 1):
        print(f"Interaction {i}:")
        print(f"Query: {interaction['query']}")

        response = role.interact(interaction['query'], client)
        print(f"Response: {response[:100]}...")

        adaptation = role.learn_from_interaction(
            interaction['query'],
            response,
            interaction['feedback']
        )
        print(f"Feedback: {interaction['feedback']}")
        print(f"Adaptation: {adaptation}\n")

    print(f"Final preferences: {role.user_preferences}")


# ===== Challenge Solution: Build a Role Management System =====

def challenge_solution_role_management_system():
    """
    Challenge Solution: Complete role management system with all features.
    """
    client = LLMClient("openai")

    print("\nChallenge: Role Management System")
    print("=" * 50)

    class RoleManager:
        def __init__(self, client):
            self.client = client
            self.roles = {}
            self.active_role = None
            self.context = []
            self.performance_metrics = {}

        def define_role(self, name, definition):
            """Store role definition with metadata."""
            self.roles[name] = {
                "definition": definition,
                "usage_count": 0,
                "avg_quality": 0.0,
                "total_quality": 0.0
            }
            print(f"✓ Defined role: {name}")

        def select_role(self, query, context=None):
            """Select best role for query using intelligent matching."""
            if not self.roles:
                return None

            # Score each role based on keyword matching
            scores = {}
            for role_name, role_data in self.roles.items():
                definition_lower = role_data["definition"].lower()
                query_lower = query.lower()

                # Calculate relevance score
                score = 0
                query_words = query_lower.split()
                for word in query_words:
                    if word in definition_lower:
                        score += 1

                # Boost score based on past performance
                if role_data["usage_count"] > 0:
                    score += role_data["avg_quality"] * 0.5

                scores[role_name] = score

            # Select highest scoring role
            best_role = max(scores, key=scores.get)
            if scores[best_role] > 0:
                self.active_role = best_role
                return best_role
            return None

        def apply_role(self, role_name, query):
            """Apply role to generate response."""
            if role_name not in self.roles:
                return "Role not found"

            role = self.roles[role_name]
            prompt = f"""{role['definition']}

Question: {query}"""

            response = self.client.complete(prompt, temperature=0.3, max_tokens=200)

            # Update usage statistics
            role["usage_count"] += 1
            self.context.append({"role": role_name, "query": query})

            return response.strip()

        def collaborate(self, roles, task):
            """Multiple roles collaborate on task."""
            if not all(r in self.roles for r in roles):
                return "Some roles not found"

            print(f"\nCollaborative Analysis: {', '.join(roles)}")
            print("-" * 40)

            responses = {}
            for role_name in roles:
                print(f"\n{role_name}:")
                response = self.apply_role(role_name, task)
                responses[role_name] = response
                print(response[:150] + "...")

            # Synthesize responses
            synthesis_prompt = f"""Synthesize these expert opinions into a unified recommendation:

{chr(10).join([f"{role}: {resp[:100]}..." for role, resp in responses.items()])}

Unified recommendation:"""

            synthesis = self.client.complete(synthesis_prompt, temperature=0.3, max_tokens=200)
            return synthesis.strip()

        def evaluate_performance(self, role_name, response, rating):
            """Track role performance with quality metrics."""
            if role_name not in self.roles:
                return

            role = self.roles[role_name]
            role["total_quality"] += rating
            role["avg_quality"] = role["total_quality"] / role["usage_count"]

            self.performance_metrics[role_name] = {
                "latest_rating": rating,
                "avg_quality": role["avg_quality"],
                "usage_count": role["usage_count"]
            }

        def get_metrics(self):
            """Return performance metrics for all roles."""
            return self.performance_metrics

    # Test the role management system
    manager = RoleManager(client)

    # Define multiple roles
    roles_to_define = [
        ("Security Expert", "You are a cybersecurity specialist focusing on secure coding practices, vulnerability assessment, and compliance."),
        ("Performance Engineer", "You are a performance optimization expert specializing in profiling, caching, and scalability."),
        ("UX Designer", "You are a user experience designer focusing on usability, accessibility, and user satisfaction."),
        ("Data Architect", "You are a data systems architect specializing in database design, ETL pipelines, and data governance.")
    ]

    print("1. DEFINING ROLES:")
    for name, definition in roles_to_define:
        manager.define_role(name, definition)

    # Test role selection
    print("\n2. AUTOMATIC ROLE SELECTION:")
    test_queries = [
        "How do we prevent SQL injection attacks?",
        "Our queries are running too slowly",
        "Users can't find the search button"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        selected = manager.select_role(query)
        print(f"Selected: {selected}")
        if selected:
            response = manager.apply_role(selected, query)
            print(f"Response: {response[:100]}...")
            # Simulate quality rating
            manager.evaluate_performance(selected, response, 4.5)

    # Test collaboration
    print("\n3. MULTI-ROLE COLLABORATION:")
    collaborative_task = "Design a secure, fast, and user-friendly login system"
    result = manager.collaborate(
        ["Security Expert", "Performance Engineer", "UX Designer"],
        collaborative_task
    )
    print(f"\nSynthesis: {result[:200]}...")

    # Show performance metrics
    print("\n4. PERFORMANCE METRICS:")
    metrics = manager.get_metrics()
    for role, data in metrics.items():
        print(f"{role}: Avg Quality={data['avg_quality']:.1f}, Uses={data['usage_count']}")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 06: Solutions")
    parser.add_argument("--solution", type=int, help="Run specific solution (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge solution")
    parser.add_argument("--all", action="store_true", help="Run all solutions")

    args = parser.parse_args()

    solutions = {
        1: solution_1_design_personas,
        2: solution_2_role_consistency,
        3: solution_3_multi_perspective,
        4: solution_4_dynamic_role_selection,
        5: solution_5_role_evolution
    }

    if args.all:
        for sol_num, sol_func in solutions.items():
            print(f"\n{'='*60}")
            sol_func()
            print(f"\n{'='*60}\n")
        challenge_solution_role_management_system()
    elif args.challenge:
        challenge_solution_role_management_system()
    elif args.solution and args.solution in solutions:
        solutions[args.solution]()
    else:
        print("Module 06: Role-Based Prompting - Solutions")
        print("\nUsage:")
        print("  python solutions.py --solution N  # Run solution N")
        print("  python solutions.py --challenge    # Run challenge solution")
        print("  python solutions.py --all          # Run all solutions")
        print("\nAvailable solutions:")
        print("  1: Design Effective Personas")
        print("  2: Role Consistency")
        print("  3: Multi-Perspective Analysis")
        print("  4: Dynamic Role Selection")
        print("  5: Role Evolution")
        print("  Challenge: Role Management System")