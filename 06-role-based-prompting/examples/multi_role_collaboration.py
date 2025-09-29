"""
Module 06: Multi-Role Collaboration

Multiple roles working together for comprehensive problem-solving.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from typing import Dict, List, Optional
import time


def example_1_panel_discussion():
    """Simulate a panel discussion with multiple experts."""
    print("=" * 60)
    print("Example 1: Expert Panel Discussion")
    print("=" * 60)

    client = LLMClient("openai")

    topic = "Should we adopt a microservices architecture for our growing e-commerce platform?"

    panelists = [
        {
            "name": "Sarah Chen",
            "role": "Software Architect",
            "stance": "technical feasibility"
        },
        {
            "name": "Mike Rodriguez",
            "role": "DevOps Lead",
            "stance": "operational complexity"
        },
        {
            "name": "Lisa Park",
            "role": "Product Manager",
            "stance": "business value and timeline"
        }
    ]

    print(f"TOPIC: {topic}\n")
    print("PANEL DISCUSSION:")
    print("-" * 40)

    discussion_context = []

    for panelist in panelists:
        # Build prompt with previous discussion context
        prompt = f"""You are {panelist['name']}, a {panelist['role']} focusing on {panelist['stance']}.

Topic: {topic}"""

        if discussion_context:
            prompt += f"\n\nPrevious points made:\n"
            for point in discussion_context:
                prompt += f"- {point}\n"

        prompt += f"\n\nYour perspective (be concise, 2-3 key points):"

        response = client.complete(prompt, temperature=0.4, max_tokens=150)
        response_text = response.strip()

        print(f"\n{panelist['name']} ({panelist['role']}):")
        print(response_text)

        # Add to context for next panelist
        discussion_context.append(f"{panelist['name']}: {response_text[:100]}...")

    # Moderator summary
    summary_prompt = f"""You are the panel moderator. Summarize the key insights from:

{chr(10).join(discussion_context)}

Provide a balanced summary and recommendation:"""

    print("\n" + "-" * 40)
    print("\nMODERATOR SUMMARY:")
    summary = client.complete(summary_prompt, temperature=0.3, max_tokens=150)
    print(summary.strip())

    print("\n💡 Multiple perspectives enrich decision-making")


def example_2_role_debate():
    """Two roles debate different approaches."""
    print("\n" + "=" * 60)
    print("Example 2: Role-Based Debate")
    print("=" * 60)

    client = LLMClient("openai")

    debate_topic = "SQL vs NoSQL for our new analytics platform"

    # Initial positions
    sql_advocate = """You are a Database Architect who strongly advocates for SQL databases.
You value ACID compliance, mature tooling, and proven reliability.

Topic: {topic}
Your opening argument (be persuasive but technical):"""

    nosql_advocate = """You are a Big Data Engineer who champions NoSQL solutions.
You value scalability, flexibility, and modern architectures.

Topic: {topic}
Your opening argument (be persuasive but technical):"""

    print(f"DEBATE: {debate_topic}\n")

    # Opening arguments
    print("SQL ADVOCATE:")
    print("-" * 30)
    sql_opening = client.complete(
        sql_advocate.format(topic=debate_topic),
        temperature=0.4,
        max_tokens=150
    ).strip()
    print(sql_opening)

    print("\n" + "=" * 30 + "\n")

    print("NOSQL ADVOCATE:")
    print("-" * 30)
    nosql_opening = client.complete(
        nosql_advocate.format(topic=debate_topic),
        temperature=0.4,
        max_tokens=150
    ).strip()
    print(nosql_opening)

    # Rebuttal round
    print("\n" + "=" * 40)
    print("REBUTTAL ROUND:\n")

    sql_rebuttal_prompt = f"""You are the SQL Database Architect.
The NoSQL advocate argued: "{nosql_opening[:150]}..."

Your rebuttal (address their points directly):"""

    print("SQL ADVOCATE REBUTTAL:")
    sql_rebuttal = client.complete(sql_rebuttal_prompt, temperature=0.4, max_tokens=100).strip()
    print(sql_rebuttal)

    print("\n" + "-" * 30 + "\n")

    nosql_rebuttal_prompt = f"""You are the Big Data Engineer.
The SQL advocate argued: "{sql_opening[:150]}..."

Your rebuttal (address their points directly):"""

    print("NOSQL ADVOCATE REBUTTAL:")
    nosql_rebuttal = client.complete(nosql_rebuttal_prompt, temperature=0.4, max_tokens=100).strip()
    print(nosql_rebuttal)

    print("\n💡 Debates reveal strengths and weaknesses of each approach")


def example_3_collaborative_problem_solving():
    """Multiple roles collaborate to solve a complex problem."""
    print("\n" + "=" * 60)
    print("Example 3: Collaborative Problem Solving")
    print("=" * 60)

    client = LLMClient("openai")

    problem = "Our application is experiencing intermittent timeouts during peak hours"

    # Define collaborative workflow
    workflow = [
        {
            "role": "Site Reliability Engineer",
            "task": "diagnose",
            "prompt": f"""You are an SRE investigating an issue.

Problem: {problem}

Your diagnosis (check metrics, logs, patterns):"""
        },
        {
            "role": "Backend Developer",
            "task": "identify_code_issues",
            "depends_on": "diagnose",
            "prompt": """You are a Backend Developer.

Based on the SRE's diagnosis: [PREVIOUS]

Identify potential code-level issues:"""
        },
        {
            "role": "Database Administrator",
            "task": "check_database",
            "depends_on": "diagnose",
            "prompt": """You are a DBA.

Based on the SRE's diagnosis: [PREVIOUS]

Check database-related factors:"""
        },
        {
            "role": "Solutions Architect",
            "task": "propose_solution",
            "depends_on": ["identify_code_issues", "check_database"],
            "prompt": """You are a Solutions Architect.

Based on findings from:
- Backend Developer: [PREVIOUS_1]
- DBA: [PREVIOUS_2]

Propose a comprehensive solution:"""
        }
    ]

    print(f"PROBLEM: {problem}\n")
    print("COLLABORATIVE ANALYSIS:")
    print("-" * 40)

    results = {}

    for step in workflow:
        print(f"\n{step['role']} - {step['task'].upper()}:")

        # Build prompt with dependencies
        prompt = step["prompt"]

        if "depends_on" in step:
            if isinstance(step["depends_on"], str):
                prev_result = results.get(step["depends_on"], "")
                prompt = prompt.replace("[PREVIOUS]", prev_result[:150] + "...")
            elif isinstance(step["depends_on"], list):
                for i, dep in enumerate(step["depends_on"], 1):
                    prev_result = results.get(dep, "")
                    prompt = prompt.replace(f"[PREVIOUS_{i}]", prev_result[:100] + "...")

        response = client.complete(prompt, temperature=0.3, max_tokens=150).strip()
        print(response)

        results[step["task"]] = response

    print("\n💡 Collaboration combines specialized expertise effectively")


def example_4_consensus_building():
    """Multiple roles work toward consensus."""
    print("\n" + "=" * 60)
    print("Example 4: Consensus Building")
    print("=" * 60)

    client = LLMClient("openai")

    decision = "Choosing a cloud provider for our startup"

    stakeholders = [
        {"role": "CTO", "priority": "technical capabilities and innovation"},
        {"role": "CFO", "priority": "cost optimization and predictability"},
        {"role": "Security Officer", "priority": "compliance and security features"},
        {"role": "DevOps Manager", "priority": "ease of use and tool ecosystem"}
    ]

    print(f"DECISION: {decision}\n")

    # Round 1: Initial positions
    print("ROUND 1 - Initial Positions:")
    print("-" * 40)

    positions = {}
    for stakeholder in stakeholders:
        prompt = f"""You are the {stakeholder['role']} prioritizing {stakeholder['priority']}.

Decision: {decision}

Your recommendation (which provider and why):"""

        response = client.complete(prompt, temperature=0.3, max_tokens=100).strip()
        positions[stakeholder["role"]] = response

        print(f"\n{stakeholder['role']}:")
        print(response)

    # Round 2: Consider others' views
    print("\n" + "=" * 40)
    print("ROUND 2 - Considering Other Perspectives:")
    print("-" * 40)

    revised_positions = {}
    for stakeholder in stakeholders:
        others_views = "\n".join([
            f"{role}: {pos[:50]}..."
            for role, pos in positions.items()
            if role != stakeholder["role"]
        ])

        prompt = f"""You are the {stakeholder['role']}.

Other stakeholders said:
{others_views}

Considering their perspectives, your revised position:"""

        response = client.complete(prompt, temperature=0.3, max_tokens=100).strip()
        revised_positions[stakeholder["role"]] = response

        print(f"\n{stakeholder['role']} (revised):")
        print(response)

    # Final consensus
    print("\n" + "=" * 40)
    print("CONSENSUS RECOMMENDATION:")
    print("-" * 40)

    consensus_prompt = f"""As a neutral facilitator, synthesize these positions into a consensus:

{chr(10).join([f"{role}: {pos[:75]}..." for role, pos in revised_positions.items()])}

Final consensus recommendation:"""

    consensus = client.complete(consensus_prompt, temperature=0.3, max_tokens=150)
    print(consensus.strip())

    print("\n💡 Iterative discussion builds consensus")


def example_5_hierarchical_review():
    """Hierarchical review process with multiple roles."""
    print("\n" + "=" * 60)
    print("Example 5: Hierarchical Review Process")
    print("=" * 60)

    client = LLMClient("openai")

    code_to_review = """
def calculate_discount(price, customer_type):
    if customer_type == "premium":
        return price * 0.8
    elif customer_type == "regular":
        return price * 0.9
    else:
        return price
    """

    review_hierarchy = [
        {
            "level": 1,
            "role": "Junior Developer",
            "focus": "basic issues and readability"
        },
        {
            "level": 2,
            "role": "Senior Developer",
            "focus": "design patterns and best practices"
        },
        {
            "level": 3,
            "role": "Tech Lead",
            "focus": "architecture and maintainability"
        }
    ]

    print("CODE TO REVIEW:")
    print("-" * 40)
    print(code_to_review)
    print("\nHIERARCHICAL REVIEW:")
    print("-" * 40)

    accumulated_feedback = []

    for reviewer in review_hierarchy:
        prompt = f"""You are a {reviewer['role']} focusing on {reviewer['focus']}.

Review this code:
```python
{code_to_review}
```"""

        if accumulated_feedback:
            prompt += f"\n\nPrevious reviews:\n"
            for feedback in accumulated_feedback:
                prompt += f"- {feedback[:100]}...\n"

            prompt += "\nAdd your higher-level insights:"
        else:
            prompt += "\n\nYour review:"

        response = client.complete(prompt, temperature=0.3, max_tokens=150).strip()

        print(f"\nLEVEL {reviewer['level']} - {reviewer['role']}:")
        print(response)

        accumulated_feedback.append(f"{reviewer['role']}: {response}")

    print("\n💡 Hierarchical review provides comprehensive feedback")


def example_6_role_orchestration():
    """Orchestrate multiple roles in a complex workflow."""
    print("\n" + "=" * 60)
    print("Example 6: Role Orchestration")
    print("=" * 60)

    client = LLMClient("openai")

    project = "Launch a new mobile app feature"

    # Define orchestrated workflow
    phases = [
        {
            "phase": "Discovery",
            "parallel_roles": [
                {"role": "User Researcher", "task": "Identify user needs"},
                {"role": "Market Analyst", "task": "Analyze competition"},
                {"role": "Technical Lead", "task": "Assess technical feasibility"}
            ]
        },
        {
            "phase": "Planning",
            "sequential_roles": [
                {"role": "Product Manager", "task": "Define requirements based on discovery"},
                {"role": "UX Designer", "task": "Create design based on requirements"},
                {"role": "Engineer", "task": "Estimate effort based on design"}
            ]
        },
        {
            "phase": "Decision",
            "consensus_roles": [
                {"role": "Product Owner", "task": "Business priority"},
                {"role": "Tech Lead", "task": "Technical priority"},
                {"role": "Project Manager", "task": "Resource priority"}
            ]
        }
    ]

    print(f"PROJECT: {project}\n")
    print("ORCHESTRATED WORKFLOW:")
    print("=" * 40)

    context = {"project": project}

    for phase_def in phases:
        print(f"\n{phase_def['phase'].upper()} PHASE:")
        print("-" * 30)

        if "parallel_roles" in phase_def:
            print("(Parallel execution)")
            phase_results = []

            for role_def in phase_def["parallel_roles"]:
                prompt = f"""You are a {role_def['role']}.
Project: {project}
Your task: {role_def['task']}
Be concise (2-3 points):"""

                response = client.complete(prompt, temperature=0.3, max_tokens=100).strip()
                print(f"\n{role_def['role']}: {response}")
                phase_results.append(response)

            context[phase_def["phase"]] = phase_results

        elif "sequential_roles" in phase_def:
            print("(Sequential execution)")
            prev_output = ""

            for role_def in phase_def["sequential_roles"]:
                prompt = f"""You are a {role_def['role']}.
Project: {project}"""

                if prev_output:
                    prompt += f"\nBased on: {prev_output[:100]}..."

                prompt += f"\nYour task: {role_def['task']}"

                response = client.complete(prompt, temperature=0.3, max_tokens=100).strip()
                print(f"\n{role_def['role']}: {response}")
                prev_output = response

            context[phase_def["phase"]] = prev_output

        elif "consensus_roles" in phase_def:
            print("(Consensus building)")
            priorities = []

            for role_def in phase_def["consensus_roles"]:
                prompt = f"""You are a {role_def['role']}.
Project: {project}
Previous findings: {str(context.get('Discovery', ''))[:100]}...
Your {role_def['task']} (one line):"""

                response = client.complete(prompt, temperature=0.3, max_tokens=50).strip()
                print(f"\n{role_def['role']}: {response}")
                priorities.append(response)

            context[phase_def["phase"]] = priorities

    print("\n💡 Orchestration coordinates complex multi-role workflows")


def example_7_adversarial_collaboration():
    """Roles that challenge each other for better outcomes."""
    print("\n" + "=" * 60)
    print("Example 7: Adversarial Collaboration")
    print("=" * 60)

    client = LLMClient("openai")

    proposal = "Implement automatic code deployment on every commit to main branch"

    roles = [
        {
            "name": "Advocate",
            "role": "DevOps Engineer",
            "stance": "supporting the proposal"
        },
        {
            "name": "Critic",
            "role": "QA Manager",
            "stance": "finding potential issues"
        },
        {
            "name": "Improver",
            "role": "Senior Architect",
            "stance": "suggesting improvements"
        }
    ]

    print(f"PROPOSAL: {proposal}\n")
    print("ADVERSARIAL COLLABORATION:")
    print("-" * 40)

    # Round 1: Initial positions
    positions = {}

    for role_def in roles:
        prompt = f"""You are a {role_def['role']} {role_def['stance']}.

Proposal: {proposal}

Your perspective:"""

        response = client.complete(prompt, temperature=0.4, max_tokens=150).strip()
        positions[role_def["name"]] = response

        print(f"\n{role_def['name'].upper()} ({role_def['role']}):")
        print(response)

    # Round 2: Address challenges
    print("\n" + "=" * 40)
    print("ADDRESSING CHALLENGES:")
    print("-" * 40)

    improved_proposal_prompt = f"""You are a Solutions Architect.

Original proposal: {proposal}

Advocate said: {positions['Advocate'][:100]}...
Critic raised: {positions['Critic'][:100]}...
Improver suggested: {positions['Improver'][:100]}...

Create an improved proposal addressing all concerns:"""

    improved = client.complete(improved_proposal_prompt, temperature=0.3, max_tokens=200)
    print("\nIMPROVED PROPOSAL:")
    print(improved.strip())

    print("\n💡 Adversarial collaboration strengthens proposals")


def run_all_examples():
    """Run all multi-role collaboration examples."""
    examples = [
        example_1_panel_discussion,
        example_2_role_debate,
        example_3_collaborative_problem_solving,
        example_4_consensus_building,
        example_5_hierarchical_review,
        example_6_role_orchestration,
        example_7_adversarial_collaboration
    ]

    for example in examples:
        try:
            example()
            print("\n" + "=" * 60 + "\n")
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 06: Multi-Role Collaboration")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_panel_discussion,
            2: example_2_role_debate,
            3: example_3_collaborative_problem_solving,
            4: example_4_consensus_building,
            5: example_5_hierarchical_review,
            6: example_6_role_orchestration,
            7: example_7_adversarial_collaboration
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 06: Multi-Role Collaboration")
        print("\nUsage:")
        print("  python multi_role_collaboration.py --all        # Run all examples")
        print("  python multi_role_collaboration.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: Panel Discussion")
        print("  2: Role Debate")
        print("  3: Collaborative Problem Solving")
        print("  4: Consensus Building")
        print("  5: Hierarchical Review")
        print("  6: Role Orchestration")
        print("  7: Adversarial Collaboration")