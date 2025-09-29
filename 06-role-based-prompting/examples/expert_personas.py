"""
Module 06: Expert Personas

Creating and managing specialized expert roles for different domains.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from typing import Dict, List, Optional


def example_1_basic_expert_role():
    """Create a basic expert persona."""
    print("=" * 60)
    print("Example 1: Basic Expert Role")
    print("=" * 60)

    client = LLMClient("openai")

    # Generic question
    generic_prompt = """How can I improve my website's performance?"""

    print("GENERIC RESPONSE (No Role):")
    generic_response = client.complete(generic_prompt, temperature=0.3, max_tokens=150)
    print(f"{generic_response.strip()}\n")

    # With expert role
    expert_prompt = """You are a Senior Performance Engineer with 15 years of experience
optimizing high-traffic websites for Fortune 500 companies. You specialize in
frontend optimization, caching strategies, and distributed systems.

How can I improve my website's performance?"""

    print("-" * 40)
    print("\nEXPERT RESPONSE (With Role):")
    expert_response = client.complete(expert_prompt, temperature=0.3, max_tokens=200)
    print(f"{expert_response.strip()}")

    print("\n💡 Notice: More specific, technical, and actionable advice with role")


def example_2_detailed_persona():
    """Create a detailed persona with background and constraints."""
    print("\n" + "=" * 60)
    print("Example 2: Detailed Persona with Background")
    print("=" * 60)

    client = LLMClient("openai")

    detailed_persona = """You are Dr. Sarah Mitchell, Chief Data Scientist at a leading fintech company.

Background:
- PhD in Statistics from MIT, 2010
- 12 years experience in financial data analysis
- Published 15 papers on risk modeling and fraud detection
- Led teams of 20+ data scientists
- Specializes in real-time anomaly detection

Personality:
- Direct and pragmatic
- Values evidence-based decisions
- Emphasizes production readiness over theoretical perfection
- Mentors junior team members

Constraints:
- Always considers regulatory compliance (GDPR, CCPA)
- Prioritizes explainable AI for stakeholder trust
- Budget-conscious, focuses on ROI

Current Context:
- Working on reducing false positives in fraud detection
- Team uses Python, Spark, and cloud infrastructure

Question: How should we approach building a new fraud detection model?"""

    print("DETAILED PERSONA RESPONSE:")
    response = client.complete(detailed_persona, temperature=0.3, max_tokens=300)
    print(f"\nDr. Mitchell's Response:\n{response.strip()}")

    print("\n💡 Rich personas provide consistent, contextual responses")


def example_3_technical_expert_panel():
    """Create multiple technical experts for comprehensive analysis."""
    print("\n" + "=" * 60)
    print("Example 3: Technical Expert Panel")
    print("=" * 60)

    client = LLMClient("openai")

    problem = "Our microservices architecture is experiencing intermittent latency spikes."

    experts = [
        {
            "role": "DevOps Engineer",
            "focus": "infrastructure and deployment",
            "prompt": f"""You are a Senior DevOps Engineer specializing in Kubernetes and cloud infrastructure.
Focus on infrastructure-related causes and solutions.

Problem: {problem}

Your infrastructure-focused analysis:"""
        },
        {
            "role": "Software Architect",
            "focus": "system design and patterns",
            "prompt": f"""You are a Principal Software Architect with expertise in distributed systems.
Focus on architectural patterns and design issues.

Problem: {problem}

Your architectural analysis:"""
        },
        {
            "role": "Performance Engineer",
            "focus": "optimization and monitoring",
            "prompt": f"""You are a Performance Engineering Lead specializing in system optimization.
Focus on performance metrics and optimization strategies.

Problem: {problem}

Your performance analysis:"""
        }
    ]

    print(f"Problem: {problem}\n")
    print("EXPERT PANEL ANALYSIS:")
    print("-" * 40)

    for expert in experts:
        print(f"\n{expert['role']} ({expert['focus']}):")
        response = client.complete(expert["prompt"], temperature=0.3, max_tokens=150)
        print(f"{response.strip()}\n")

    print("💡 Multiple perspectives provide comprehensive problem analysis")


def example_4_domain_specific_experts():
    """Create experts for different domains."""
    print("\n" + "=" * 60)
    print("Example 4: Domain-Specific Experts")
    print("=" * 60)

    client = LLMClient("openai")

    domains = {
        "Healthcare": """You are Dr. James Chen, a Healthcare IT Specialist with both MD and CS degrees.
You understand HIPAA compliance, medical workflows, and clinical decision support systems.

Question: How can we implement AI in our hospital's diagnostic process?""",

        "Finance": """You are Alexandra Kumar, a Quantitative Analyst at a hedge fund.
You specialize in algorithmic trading, risk modeling, and regulatory compliance.

Question: How can we implement AI for portfolio optimization?""",

        "E-commerce": """You are Marcus Thompson, Head of Data Science at a major online retailer.
You focus on recommendation systems, inventory optimization, and customer behavior.

Question: How can we implement AI to reduce cart abandonment?"""
    }

    for domain, prompt in domains.items():
        print(f"\n{domain.upper()} EXPERT:")
        print("-" * 30)
        response = client.complete(prompt, temperature=0.3, max_tokens=150)
        print(f"{response.strip()}\n")

    print("💡 Domain expertise shapes vocabulary, priorities, and solutions")


def example_5_personality_driven_roles():
    """Demonstrate how personality affects communication style."""
    print("\n" + "=" * 60)
    print("Example 5: Personality-Driven Communication")
    print("=" * 60)

    client = LLMClient("openai")

    question = "Explain how machine learning models work"

    personalities = [
        {
            "type": "Academic Professor",
            "prompt": f"""You are a tenured Computer Science professor who loves teaching.
You use formal language, cite research, and build understanding from first principles.

{question}"""
        },
        {
            "type": "Startup CTO",
            "prompt": f"""You are a pragmatic startup CTO who values speed and practical results.
You communicate concisely, focus on implementation, and care about time-to-market.

{question}"""
        },
        {
            "type": "Patient Mentor",
            "prompt": f"""You are a patient senior engineer who mentors junior developers.
You use analogies, provide examples, and check for understanding frequently.

{question}"""
        }
    ]

    print(f"Question: {question}\n")

    for personality in personalities:
        print(f"\n{personality['type'].upper()}:")
        print("-" * 30)
        response = client.complete(personality["prompt"], temperature=0.4, max_tokens=150)
        print(f"{response.strip()}\n")

    print("💡 Personality shapes communication style and approach")


def example_6_role_with_examples():
    """Combine role-based prompting with few-shot examples."""
    print("\n" + "=" * 60)
    print("Example 6: Role + Few-Shot Examples")
    print("=" * 60)

    client = LLMClient("openai")

    role_with_examples = """You are a Senior Code Reviewer at Google with 10 years of experience.
You focus on readability, maintainability, and following best practices.

Here are examples of your code review style:

Code: def calc(x,y): return x+y
Your Review: "Function name 'calc' is not descriptive. Consider 'calculate_sum'.
Also add type hints: def calculate_sum(x: float, y: float) -> float:"

Code: if user.is_admin == True:
Your Review: "Redundant comparison with True. Use: if user.is_admin:
This is more Pythonic and cleaner."

Now review this code:
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
    return result

Your Review:"""

    print("CODE REVIEW WITH ROLE + EXAMPLES:")
    response = client.complete(role_with_examples, temperature=0.3, max_tokens=200)
    print(f"\nReview:\n{response.strip()}")

    print("\n💡 Examples reinforce role behavior patterns")


def example_7_adaptive_expertise_level():
    """Adjust expertise level based on audience."""
    print("\n" + "=" * 60)
    print("Example 7: Adaptive Expertise Level")
    print("=" * 60)

    client = LLMClient("openai")

    question = "How does database indexing improve query performance?"

    audience_levels = [
        {
            "level": "Beginner",
            "prompt": f"""You are a database expert explaining to a junior developer who just started.
Use simple language, avoid jargon, and relate to everyday concepts.

{question}"""
        },
        {
            "level": "Intermediate",
            "prompt": f"""You are a database expert explaining to a developer with 2 years experience.
You can use technical terms but should still explain complex concepts clearly.

{question}"""
        },
        {
            "level": "Expert",
            "prompt": f"""You are a database expert discussing with a senior architect.
Use precise technical language, discuss internals, and mention advanced optimizations.

{question}"""
        }
    ]

    print(f"Question: {question}\n")

    for audience in audience_levels:
        print(f"\n{audience['level'].upper()} EXPLANATION:")
        print("-" * 30)
        response = client.complete(audience["prompt"], temperature=0.3, max_tokens=150)
        print(f"{response.strip()}\n")

    print("💡 Expertise adapts to audience knowledge level")


def example_8_collaborative_experts():
    """Experts that reference and build on each other's insights."""
    print("\n" + "=" * 60)
    print("Example 8: Collaborative Expert Dialogue")
    print("=" * 60)

    client = LLMClient("openai")

    topic = "Should we migrate our monolithic application to microservices?"

    # First expert
    architect_prompt = f"""You are a Software Architect evaluating a migration decision.

{topic}

Your architectural assessment:"""

    print(f"Topic: {topic}\n")
    print("SOFTWARE ARCHITECT:")
    architect_response = client.complete(architect_prompt, temperature=0.3, max_tokens=150).strip()
    print(f"{architect_response}\n")

    # Second expert builds on first
    devops_prompt = f"""You are a DevOps Lead. The Software Architect said:

"{architect_response[:100]}..."

Building on their points, what are the operational considerations?

Your operational assessment:"""

    print("-" * 40)
    print("\nDEVOPS LEAD (building on Architect):")
    devops_response = client.complete(devops_prompt, temperature=0.3, max_tokens=150).strip()
    print(f"{devops_response}\n")

    # Third expert synthesizes
    cto_prompt = f"""You are the CTO. You've heard from:

Software Architect: "{architect_response[:75]}..."
DevOps Lead: "{devops_response[:75]}..."

What's your executive decision?"""

    print("-" * 40)
    print("\nCTO (synthesizing both views):")
    cto_response = client.complete(cto_prompt, temperature=0.3, max_tokens=150)
    print(f"{cto_response.strip()}")

    print("\n💡 Experts can build on each other's contributions")


def run_all_examples():
    """Run all expert persona examples."""
    examples = [
        example_1_basic_expert_role,
        example_2_detailed_persona,
        example_3_technical_expert_panel,
        example_4_domain_specific_experts,
        example_5_personality_driven_roles,
        example_6_role_with_examples,
        example_7_adaptive_expertise_level,
        example_8_collaborative_experts
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

    parser = argparse.ArgumentParser(description="Module 06: Expert Personas")
    parser.add_argument("--example", type=int, help="Run specific example (1-8)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_basic_expert_role,
            2: example_2_detailed_persona,
            3: example_3_technical_expert_panel,
            4: example_4_domain_specific_experts,
            5: example_5_personality_driven_roles,
            6: example_6_role_with_examples,
            7: example_7_adaptive_expertise_level,
            8: example_8_collaborative_experts
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 06: Expert Personas")
        print("\nUsage:")
        print("  python expert_personas.py --all        # Run all examples")
        print("  python expert_personas.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: Basic Expert Role")
        print("  2: Detailed Persona")
        print("  3: Technical Expert Panel")
        print("  4: Domain-Specific Experts")
        print("  5: Personality-Driven Roles")
        print("  6: Role with Examples")
        print("  7: Adaptive Expertise Level")
        print("  8: Collaborative Experts")