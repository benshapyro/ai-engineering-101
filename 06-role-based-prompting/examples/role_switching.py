"""
Module 06: Dynamic Role Switching

Techniques for changing roles during conversations and adapting personas.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from typing import Dict, List, Optional


def example_1_conversation_role_switch():
    """Switch roles during a conversation flow."""
    print("=" * 60)
    print("Example 1: Conversation Role Switching")
    print("=" * 60)

    client = LLMClient("openai")

    conversation_steps = [
        {
            "role": "Business Analyst",
            "question": "What are the requirements for the new feature?",
            "prompt": """You are a Business Analyst gathering requirements.
You focus on user needs, business value, and acceptance criteria.

Question: What are the requirements for the new customer dashboard feature?"""
        },
        {
            "role": "Technical Architect",
            "question": "How should we design the system?",
            "prompt": """You are now a Technical Architect. Based on the requirements discussed,
you need to design the technical solution. Focus on scalability and maintainability.

Previous context: We need a real-time customer dashboard with analytics.

Question: How should we architect this system?"""
        },
        {
            "role": "Project Manager",
            "question": "What's the timeline and resource plan?",
            "prompt": """You are now a Project Manager. Given the requirements and technical design,
create a realistic timeline and resource allocation plan.

Previous context: Real-time dashboard with microservices architecture.

Question: What's the project timeline and resource needs?"""
        }
    ]

    print("DYNAMIC ROLE SWITCHING IN CONVERSATION:\n")

    for i, step in enumerate(conversation_steps, 1):
        print(f"Step {i} - {step['role']}:")
        print(f"Question: {step['question']}")
        print("-" * 40)

        response = client.complete(step["prompt"], temperature=0.3, max_tokens=150)
        print(f"{response.strip()}\n")

    print("💡 Roles change to match conversation needs")


def example_2_context_aware_switching():
    """Switch roles based on detected context."""
    print("\n" + "=" * 60)
    print("Example 2: Context-Aware Role Switching")
    print("=" * 60)

    client = LLMClient("openai")

    queries = [
        "My code is throwing a NullPointerException",
        "How much will this feature cost to develop?",
        "Is this approach GDPR compliant?",
        "How can we make this more user-friendly?"
    ]

    print("AUTOMATIC ROLE SELECTION BASED ON QUERY:\n")

    for query in queries:
        # Detect appropriate role
        role_detection_prompt = f"""Determine the best expert role for this query:
Query: {query}

Choose from: Software Engineer, Business Analyst, Legal Advisor, UX Designer
Return only the role name."""

        detected_role = client.complete(role_detection_prompt, temperature=0.1, max_tokens=20).strip()

        # Switch to detected role
        role_prompts = {
            "Software Engineer": f"You are a Senior Software Engineer. Debug this issue: {query}",
            "Business Analyst": f"You are a Business Analyst. Analyze this: {query}",
            "Legal Advisor": f"You are a Legal Compliance Expert. Advise on: {query}",
            "UX Designer": f"You are a UX Designer. Improve this: {query}"
        }

        print(f"Query: '{query}'")
        print(f"Detected Role: {detected_role}")

        if detected_role in role_prompts:
            response = client.complete(
                role_prompts[detected_role],
                temperature=0.3,
                max_tokens=100
            )
            print(f"Response: {response.strip()}\n")
            print("-" * 40 + "\n")

    print("💡 Context determines the appropriate expert role")


def example_3_gradual_expertise_adjustment():
    """Gradually adjust expertise level based on user understanding."""
    print("\n" + "=" * 60)
    print("Example 3: Gradual Expertise Adjustment")
    print("=" * 60)

    client = LLMClient("openai")

    # Simulate a learning conversation
    exchanges = [
        {
            "user": "What is an API?",
            "detected_level": "beginner",
            "role": "You are a patient teacher explaining to someone new to programming."
        },
        {
            "user": "Oh, so it's like a menu at a restaurant. How do I make one?",
            "detected_level": "beginner-intermediate",
            "role": "You are a teacher who sees the student understands concepts. Add slightly more technical detail."
        },
        {
            "user": "I see. Can you show me a REST API example with authentication?",
            "detected_level": "intermediate",
            "role": "You are a technical mentor. The student grasps basics, so you can use technical terms and code."
        },
        {
            "user": "How would I implement OAuth2 with refresh tokens?",
            "detected_level": "advanced",
            "role": "You are a senior engineer discussing with a peer. Use advanced concepts freely."
        }
    ]

    print("ADAPTIVE EXPERTISE LEVEL:\n")

    for exchange in exchanges:
        print(f"User: {exchange['user']}")
        print(f"Detected Level: {exchange['detected_level']}")
        print("-" * 40)

        prompt = f"{exchange['role']}\n\nQuestion: {exchange['user']}"
        response = client.complete(prompt, temperature=0.3, max_tokens=150)
        print(f"Response: {response.strip()}\n")

    print("💡 Expertise adapts to user's demonstrated knowledge")


def example_4_role_inheritance():
    """Build complex roles by inheriting from base roles."""
    print("\n" + "=" * 60)
    print("Example 4: Role Inheritance and Composition")
    print("=" * 60)

    client = LLMClient("openai")

    # Base role
    base_role = """Base: You are a technology professional with strong communication skills."""

    # Specialized roles that inherit
    specialized_roles = [
        {
            "title": "DevOps Engineer",
            "extension": "Specialization: CI/CD, Kubernetes, cloud infrastructure, monitoring.",
            "task": "How do we improve our deployment pipeline?"
        },
        {
            "title": "Security Engineer",
            "extension": "Specialization: Penetration testing, encryption, compliance, threat modeling.",
            "task": "How do we improve our deployment pipeline?"
        },
        {
            "title": "Site Reliability Engineer",
            "extension": "Specialization: System reliability, incident response, SLOs, chaos engineering.",
            "task": "How do we improve our deployment pipeline?"
        }
    ]

    print("ROLE INHERITANCE - Same base, different specializations:\n")
    print(f"Base Role: {base_role}\n")

    task = "How do we improve our deployment pipeline?"
    print(f"Task for all: '{task}'\n")

    for role in specialized_roles:
        print(f"{role['title'].upper()}:")
        print(f"Extension: {role['extension']}")
        print("-" * 40)

        full_prompt = f"""{base_role}
{role['extension']}

{role['task']}"""

        response = client.complete(full_prompt, temperature=0.3, max_tokens=150)
        print(f"{response.strip()}\n")

    print("💡 Inheritance creates consistent yet specialized roles")


def example_5_role_state_management():
    """Maintain role state across interactions."""
    print("\n" + "=" * 60)
    print("Example 5: Role State Management")
    print("=" * 60)

    client = LLMClient("openai")

    class RoleState:
        def __init__(self):
            self.current_role = None
            self.context = []
            self.preferences = {}

        def update(self, role, context_item=None, preference=None):
            self.current_role = role
            if context_item:
                self.context.append(context_item)
            if preference:
                self.preferences.update(preference)

        def get_prompt(self, question):
            prompt_parts = [f"You are a {self.current_role}."]

            if self.context:
                prompt_parts.append(f"Previous context: {'; '.join(self.context[-3:])}")

            if self.preferences:
                pref_str = ', '.join([f"{k}: {v}" for k, v in self.preferences.items()])
                prompt_parts.append(f"User preferences: {pref_str}")

            prompt_parts.append(f"\nQuestion: {question}")
            return '\n'.join(prompt_parts)

    # Simulate stateful conversation
    state = RoleState()

    interactions = [
        {
            "role": "Python Developer",
            "question": "Should I use Flask or Django?",
            "context": "Building a REST API",
            "preference": {"style": "pragmatic"}
        },
        {
            "role": "Python Developer",  # Same role, accumulating context
            "question": "What about database choices?",
            "context": "Chose Flask for simplicity",
            "preference": {"scale": "startup"}
        },
        {
            "role": "DevOps Engineer",  # Role switch, keeping context
            "question": "How should I deploy this?",
            "context": "Flask API with PostgreSQL",
            "preference": {"cloud": "AWS"}
        }
    ]

    print("STATEFUL ROLE MANAGEMENT:\n")

    for interaction in interactions:
        state.update(
            interaction["role"],
            interaction.get("context"),
            interaction.get("preference")
        )

        print(f"Role: {interaction['role']}")
        print(f"Question: {interaction['question']}")
        print(f"State: {len(state.context)} context items, {len(state.preferences)} preferences")
        print("-" * 40)

        prompt = state.get_prompt(interaction["question"])
        response = client.complete(prompt, temperature=0.3, max_tokens=150)
        print(f"Response: {response.strip()}\n")

    print("💡 State management maintains continuity across role changes")


def example_6_conditional_role_switching():
    """Switch roles based on specific conditions or triggers."""
    print("\n" + "=" * 60)
    print("Example 6: Conditional Role Switching")
    print("=" * 60)

    client = LLMClient("openai")

    # Define role switching rules
    def determine_role(query, current_role=None):
        """Determine role based on query content and current context."""
        query_lower = query.lower()

        # Priority triggers
        if any(word in query_lower for word in ['budget', 'cost', 'roi', 'investment']):
            return "Financial Analyst"
        elif any(word in query_lower for word in ['bug', 'error', 'crash', 'exception']):
            return "Debug Specialist"
        elif any(word in query_lower for word in ['scale', 'performance', 'optimize']):
            return "Performance Engineer"
        elif any(word in query_lower for word in ['user', 'interface', 'design', 'experience']):
            return "UX Designer"
        else:
            return current_role or "General Tech Advisor"

    # Conversation with automatic role switching
    conversation = [
        "How should we structure our new application?",
        "We're seeing performance issues with database queries",
        "What would this cost to implement?",
        "Users are complaining about the interface",
        "There's a bug in the payment processing"
    ]

    print("CONDITIONAL ROLE SWITCHING:\n")

    current_role = None
    for query in conversation:
        new_role = determine_role(query, current_role)

        if new_role != current_role:
            print(f"🔄 Switching from {current_role} to {new_role}")
            current_role = new_role

        print(f"\nQuery: {query}")
        print(f"Active Role: {current_role}")
        print("-" * 40)

        prompt = f"""You are a {current_role}.
Provide focused expertise for: {query}"""

        response = client.complete(prompt, temperature=0.3, max_tokens=100)
        print(f"{response.strip()}\n")

    print("💡 Conditions trigger appropriate role transitions")


def example_7_role_blending():
    """Blend multiple roles for complex situations."""
    print("\n" + "=" * 60)
    print("Example 7: Role Blending")
    print("=" * 60)

    client = LLMClient("openai")

    # Situations requiring blended expertise
    scenarios = [
        {
            "situation": "Design a secure payment system",
            "primary_role": "Software Architect",
            "secondary_role": "Security Expert",
            "blend_ratio": "60/40",
            "prompt": """You are primarily a Software Architect (60%) with Security Expert knowledge (40%).
Design a payment system balancing architecture elegance with security requirements.

Task: Design a secure payment system for our e-commerce platform."""
        },
        {
            "situation": "Create user-friendly data visualizations",
            "primary_role": "Data Scientist",
            "secondary_role": "UX Designer",
            "blend_ratio": "50/50",
            "prompt": """You are equally a Data Scientist and UX Designer.
Balance statistical accuracy with user experience in your approach.

Task: Create dashboards for non-technical executives to understand ML model performance."""
        }
    ]

    print("BLENDED ROLES FOR COMPLEX TASKS:\n")

    for scenario in scenarios:
        print(f"Situation: {scenario['situation']}")
        print(f"Blend: {scenario['primary_role']} ({scenario['blend_ratio'].split('/')[0]}%) + "
              f"{scenario['secondary_role']} ({scenario['blend_ratio'].split('/')[1]}%)")
        print("-" * 40)

        response = client.complete(scenario["prompt"], temperature=0.3, max_tokens=200)
        print(f"{response.strip()}\n")

    print("💡 Role blending combines complementary expertise")


def run_all_examples():
    """Run all role switching examples."""
    examples = [
        example_1_conversation_role_switch,
        example_2_context_aware_switching,
        example_3_gradual_expertise_adjustment,
        example_4_role_inheritance,
        example_5_role_state_management,
        example_6_conditional_role_switching,
        example_7_role_blending
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

    parser = argparse.ArgumentParser(description="Module 06: Role Switching")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_conversation_role_switch,
            2: example_2_context_aware_switching,
            3: example_3_gradual_expertise_adjustment,
            4: example_4_role_inheritance,
            5: example_5_role_state_management,
            6: example_6_conditional_role_switching,
            7: example_7_role_blending
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 06: Dynamic Role Switching")
        print("\nUsage:")
        print("  python role_switching.py --all        # Run all examples")
        print("  python role_switching.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: Conversation Role Switch")
        print("  2: Context-Aware Switching")
        print("  3: Gradual Expertise Adjustment")
        print("  4: Role Inheritance")
        print("  5: Role State Management")
        print("  6: Conditional Switching")
        print("  7: Role Blending")