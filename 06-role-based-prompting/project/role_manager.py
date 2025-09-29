"""
Module 06: Role Manager Project

A comprehensive role management system for dynamic expert consultation.
This project demonstrates advanced role-based prompting patterns including
role selection, collaboration, state management, and performance tracking.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
from enum import Enum


class ExpertiseLevel(Enum):
    """Expertise levels for adaptive communication."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class RoleCategory(Enum):
    """Categories for organizing roles."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    STRATEGIC = "strategic"


class Expert:
    """Represents an expert role with full context and capabilities."""

    def __init__(self, name: str, title: str, category: RoleCategory):
        self.name = name
        self.title = title
        self.category = category
        self.background = ""
        self.expertise = []
        self.personality = ""
        self.communication_style = ""
        self.constraints = []
        self.usage_count = 0
        self.quality_scores = []
        self.last_used = None

    def to_prompt(self) -> str:
        """Convert expert to a prompt string."""
        prompt_parts = [
            f"You are {self.name}, {self.title}.",
            f"Background: {self.background}" if self.background else "",
            f"Expertise: {', '.join(self.expertise)}" if self.expertise else "",
            f"Personality: {self.personality}" if self.personality else "",
            f"Communication Style: {self.communication_style}" if self.communication_style else "",
            f"Constraints: {', '.join(self.constraints)}" if self.constraints else ""
        ]
        return "\n".join([p for p in prompt_parts if p])

    def update_metrics(self, quality_score: float):
        """Update usage metrics for the expert."""
        self.usage_count += 1
        self.quality_scores.append(quality_score)
        self.last_used = datetime.now()

    def get_average_quality(self) -> float:
        """Calculate average quality score."""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)


class RoleManager:
    """Advanced role management system for dynamic expert consultation."""

    def __init__(self, client: LLMClient):
        self.client = client
        self.experts: Dict[str, Expert] = {}
        self.active_expert: Optional[Expert] = None
        self.conversation_context = []
        self.user_expertise_level = ExpertiseLevel.INTERMEDIATE
        self.collaboration_history = []

    def create_expert(self, name: str, title: str, category: RoleCategory,
                     background: str = "", expertise: List[str] = None,
                     personality: str = "", communication_style: str = "",
                     constraints: List[str] = None) -> Expert:
        """Create and register a new expert."""
        expert = Expert(name, title, category)
        expert.background = background
        expert.expertise = expertise or []
        expert.personality = personality
        expert.communication_style = communication_style
        expert.constraints = constraints or []

        self.experts[name] = expert
        return expert

    def setup_default_experts(self):
        """Initialize the system with a comprehensive set of experts."""
        # Technical Experts
        self.create_expert(
            "Alex Chen",
            "Principal Software Architect",
            RoleCategory.TECHNICAL,
            "15 years building distributed systems at scale, author of 'Microservices in Practice'",
            ["System design", "Microservices", "Cloud architecture", "Performance optimization"],
            "Pragmatic, focused on practical solutions over theoretical perfection",
            "Uses diagrams and code examples, explains trade-offs clearly",
            ["Must consider scalability", "Production-ready solutions only"]
        )

        self.create_expert(
            "Sarah Mitchell",
            "Chief Security Officer",
            RoleCategory.TECHNICAL,
            "Former ethical hacker, CISSP certified, led security at 3 Fortune 500 companies",
            ["Cybersecurity", "Penetration testing", "Compliance", "Threat modeling"],
            "Cautious, detail-oriented, assumes breach mentality",
            "Direct warnings about risks, provides mitigation strategies",
            ["Zero tolerance for vulnerabilities", "Compliance is non-negotiable"]
        )

        # Business Experts
        self.create_expert(
            "Marcus Johnson",
            "VP of Product Strategy",
            RoleCategory.BUSINESS,
            "Harvard MBA, 12 years in product management, launched 5 successful products",
            ["Product strategy", "Market analysis", "User research", "Roadmap planning"],
            "Customer-obsessed, data-driven decision maker",
            "Starts with user problems, backs up with metrics",
            ["ROI must be demonstrable", "User value is paramount"]
        )

        self.create_expert(
            "Lisa Wang",
            "Financial Analyst",
            RoleCategory.BUSINESS,
            "CFA charterholder, specializes in tech company valuations",
            ["Financial modeling", "Cost analysis", "ROI calculation", "Budget planning"],
            "Analytical, risk-aware, focused on bottom line",
            "Uses precise numbers, creates financial scenarios",
            ["Must justify with financial metrics", "Consider opportunity costs"]
        )

        # Creative Expert
        self.create_expert(
            "Jordan Taylor",
            "Head of User Experience",
            RoleCategory.CREATIVE,
            "Psychology background, designed interfaces used by millions",
            ["UX design", "User research", "Accessibility", "Design systems"],
            "Empathetic, user advocate, values simplicity",
            "Uses user stories, emphasizes emotional design",
            ["Accessibility is mandatory", "Cognitive load must be minimal"]
        )

        # Analytical Expert
        self.create_expert(
            "Dr. Raj Patel",
            "Lead Data Scientist",
            RoleCategory.ANALYTICAL,
            "PhD in Statistics from MIT, published 20+ papers on ML",
            ["Machine learning", "Statistical analysis", "Data visualization", "A/B testing"],
            "Methodical, hypothesis-driven, skeptical of assumptions",
            "Explains statistical significance, questions data quality",
            ["Correlation is not causation", "Sample size must be adequate"]
        )

        # Strategic Expert
        self.create_expert(
            "Catherine Brooks",
            "Chief Strategy Officer",
            RoleCategory.STRATEGIC,
            "McKinsey alum, led digital transformation at 3 companies",
            ["Strategic planning", "Change management", "Digital transformation", "Competitive analysis"],
            "Visionary yet practical, thinks in systems",
            "Connects tactics to strategy, uses frameworks",
            ["Align with company vision", "Consider long-term implications"]
        )

        print(f"✓ Initialized {len(self.experts)} default experts")

    def detect_expertise_level(self, query: str) -> ExpertiseLevel:
        """Detect user's expertise level from their query."""
        beginner_indicators = ["what is", "how do i", "explain", "new to", "beginner", "start"]
        advanced_indicators = ["optimize", "architecture", "pattern", "best practice", "scale", "performance"]
        expert_indicators = ["trade-off", "implications", "edge case", "distributed", "consensus", "CAP theorem"]

        query_lower = query.lower()

        expert_count = sum(1 for ind in expert_indicators if ind in query_lower)
        advanced_count = sum(1 for ind in advanced_indicators if ind in query_lower)
        beginner_count = sum(1 for ind in beginner_indicators if ind in query_lower)

        if expert_count > 0:
            return ExpertiseLevel.EXPERT
        elif advanced_count > beginner_count:
            return ExpertiseLevel.ADVANCED
        elif beginner_count > 0:
            return ExpertiseLevel.BEGINNER
        else:
            return ExpertiseLevel.INTERMEDIATE

    def select_expert(self, query: str, category_filter: Optional[RoleCategory] = None) -> Optional[Expert]:
        """Intelligently select the best expert for a query."""
        if not self.experts:
            return None

        # Detect user expertise level
        self.user_expertise_level = self.detect_expertise_level(query)

        # Score each expert
        scores = {}
        query_lower = query.lower()

        for name, expert in self.experts.items():
            # Skip if category filter doesn't match
            if category_filter and expert.category != category_filter:
                continue

            score = 0

            # Check expertise match
            for skill in expert.expertise:
                if skill.lower() in query_lower:
                    score += 3

            # Check title relevance
            title_words = expert.title.lower().split()
            for word in title_words:
                if word in query_lower:
                    score += 2

            # Boost based on past performance
            if expert.usage_count > 0:
                score += expert.get_average_quality()

            # Slight penalty for overuse (encourage diversity)
            if expert.usage_count > 5:
                score -= 0.5

            scores[name] = score

        # Select highest scoring expert
        if scores:
            best_expert_name = max(scores, key=scores.get)
            if scores[best_expert_name] > 0:
                self.active_expert = self.experts[best_expert_name]
                return self.active_expert

        return None

    def consult_expert(self, expert: Expert, query: str) -> str:
        """Get response from a specific expert."""
        # Build expertise-aware prompt
        prompt = f"""{expert.to_prompt()}

User Expertise Level: {self.user_expertise_level.value}
Adjust your response complexity accordingly.

Question: {query}

Response:"""

        response = self.client.complete(prompt, temperature=0.3, max_tokens=300)

        # Update metrics
        expert.update_metrics(4.0)  # Default quality score

        # Store context
        self.conversation_context.append({
            "expert": expert.name,
            "query": query,
            "response": response[:100],
            "timestamp": datetime.now()
        })

        return response.strip()

    def multi_expert_collaboration(self, query: str, expert_names: List[str]) -> Dict[str, str]:
        """Get multiple experts to collaborate on a complex problem."""
        if not all(name in self.experts for name in expert_names):
            return {"error": "Some experts not found"}

        responses = {}
        print(f"\n🤝 Multi-Expert Collaboration on: {query[:50]}...")
        print("-" * 50)

        # First round: Individual expert opinions
        for name in expert_names:
            expert = self.experts[name]
            print(f"\n{expert.name} ({expert.title}):")

            response = self.consult_expert(expert, query)
            responses[name] = response
            print(f"{response[:200]}...")

        # Second round: Synthesis
        synthesis_prompt = f"""You are a Chief Technology Officer synthesizing expert opinions.

The following experts have provided their perspectives on: {query}

{chr(10).join([f"{name} ({self.experts[name].title}): {resp[:150]}..."
               for name, resp in responses.items()])}

Synthesize these perspectives into a comprehensive recommendation that:
1. Identifies consensus points
2. Highlights important disagreements
3. Provides a balanced final recommendation

Synthesis:"""

        synthesis = self.client.complete(synthesis_prompt, temperature=0.3, max_tokens=300)
        responses["synthesis"] = synthesis.strip()

        # Store collaboration
        self.collaboration_history.append({
            "query": query,
            "experts": expert_names,
            "timestamp": datetime.now()
        })

        return responses

    def debate_mode(self, query: str, expert1_name: str, expert2_name: str, rounds: int = 2) -> List[Dict]:
        """Have two experts debate different perspectives."""
        if expert1_name not in self.experts or expert2_name not in self.experts:
            return [{"error": "Experts not found"}]

        expert1 = self.experts[expert1_name]
        expert2 = self.experts[expert2_name]

        debate_log = []
        print(f"\n⚖️ Debate: {expert1.name} vs {expert2.name}")
        print(f"Topic: {query}")
        print("-" * 50)

        for round_num in range(rounds):
            print(f"\nRound {round_num + 1}:")

            # Expert 1's turn
            if round_num == 0:
                prompt1 = f"{expert1.to_prompt()}\n\nProvide your perspective on: {query}"
            else:
                prev_response = debate_log[-1]["response"]
                prompt1 = f"""{expert1.to_prompt()}

The other expert said: "{prev_response[:200]}..."

Respond with your perspective, addressing their points: {query}"""

            response1 = self.client.complete(prompt1, temperature=0.4, max_tokens=200).strip()
            print(f"\n{expert1.name}: {response1[:150]}...")

            debate_log.append({
                "round": round_num + 1,
                "expert": expert1.name,
                "response": response1
            })

            # Expert 2's turn
            prompt2 = f"""{expert2.to_prompt()}

The other expert said: "{response1[:200]}..."

Respond with your perspective, addressing their points: {query}"""

            response2 = self.client.complete(prompt2, temperature=0.4, max_tokens=200).strip()
            print(f"\n{expert2.name}: {response2[:150]}...")

            debate_log.append({
                "round": round_num + 1,
                "expert": expert2.name,
                "response": response2
            })

        return debate_log

    def get_expert_roster(self) -> str:
        """Display all available experts."""
        if not self.experts:
            return "No experts registered"

        roster = "📋 Expert Roster:\n" + "=" * 50 + "\n"

        for category in RoleCategory:
            category_experts = [e for e in self.experts.values() if e.category == category]
            if category_experts:
                roster += f"\n{category.value.upper()} EXPERTS:\n"
                for expert in category_experts:
                    quality = expert.get_average_quality()
                    roster += f"  • {expert.name} - {expert.title}\n"
                    roster += f"    Expertise: {', '.join(expert.expertise[:3])}\n"
                    roster += f"    Usage: {expert.usage_count} | Avg Quality: {quality:.1f}\n"

        return roster

    def analyze_collaboration_patterns(self) -> Dict:
        """Analyze patterns in expert collaboration."""
        if not self.collaboration_history:
            return {"message": "No collaboration history yet"}

        analysis = {
            "total_collaborations": len(self.collaboration_history),
            "expert_participation": {},
            "common_pairs": {},
            "average_team_size": 0
        }

        team_sizes = []
        pair_counts = {}

        for collab in self.collaboration_history:
            team_sizes.append(len(collab["experts"]))

            # Count expert participation
            for expert in collab["experts"]:
                analysis["expert_participation"][expert] = \
                    analysis["expert_participation"].get(expert, 0) + 1

            # Count pairs
            for i, expert1 in enumerate(collab["experts"]):
                for expert2 in collab["experts"][i + 1:]:
                    pair = tuple(sorted([expert1, expert2]))
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

        analysis["average_team_size"] = sum(team_sizes) / len(team_sizes) if team_sizes else 0

        # Get top pairs
        if pair_counts:
            top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            analysis["common_pairs"] = {f"{p[0]} & {p[1]}": count for (p, count) in top_pairs}

        return analysis


def interactive_demo():
    """Interactive demonstration of the role management system."""
    client = LLMClient("openai")
    manager = RoleManager(client)

    print("=" * 60)
    print("ADVANCED ROLE MANAGEMENT SYSTEM")
    print("=" * 60)

    # Setup experts
    manager.setup_default_experts()
    print(manager.get_expert_roster())

    # Demo scenarios
    scenarios = [
        {
            "type": "single",
            "query": "How should we handle authentication in our microservices?",
            "description": "Single Expert Consultation"
        },
        {
            "type": "collaboration",
            "query": "Should we migrate from monolith to microservices?",
            "experts": ["Alex Chen", "Marcus Johnson", "Lisa Wang"],
            "description": "Multi-Expert Collaboration"
        },
        {
            "type": "debate",
            "query": "SQL vs NoSQL for our new application",
            "expert1": "Alex Chen",
            "expert2": "Dr. Raj Patel",
            "description": "Expert Debate"
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'=' * 60}")
        print(f"SCENARIO {i}: {scenario['description']}")
        print(f"{'=' * 60}")

        if scenario["type"] == "single":
            print(f"Query: {scenario['query']}")
            expert = manager.select_expert(scenario["query"])
            if expert:
                print(f"Selected Expert: {expert.name} ({expert.title})")
                response = manager.consult_expert(expert, scenario["query"])
                print(f"Response: {response[:300]}...")

        elif scenario["type"] == "collaboration":
            responses = manager.multi_expert_collaboration(
                scenario["query"],
                scenario["experts"]
            )
            if "synthesis" in responses:
                print(f"\nFinal Synthesis:")
                print(responses["synthesis"][:300] + "...")

        elif scenario["type"] == "debate":
            debate_log = manager.debate_mode(
                scenario["query"],
                scenario["expert1"],
                scenario["expert2"],
                rounds=2
            )

    # Show analytics
    print(f"\n{'=' * 60}")
    print("COLLABORATION ANALYTICS")
    print(f"{'=' * 60}")
    analysis = manager.analyze_collaboration_patterns()
    print(json.dumps(analysis, indent=2, default=str))


def main():
    """Main execution with different modes."""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Role Management System")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--query", type=str, help="Query for expert consultation")
    parser.add_argument("--expert", type=str, help="Specific expert to consult")
    parser.add_argument("--collaborate", nargs="+", help="Experts for collaboration")
    parser.add_argument("--debate", nargs=2, help="Two experts for debate")

    args = parser.parse_args()

    client = LLMClient("openai")
    manager = RoleManager(client)
    manager.setup_default_experts()

    if args.demo:
        interactive_demo()
    elif args.query:
        if args.expert:
            # Consult specific expert
            if args.expert in manager.experts:
                expert = manager.experts[args.expert]
                print(f"Consulting {expert.name}...")
                response = manager.consult_expert(expert, args.query)
                print(f"\nResponse:\n{response}")
            else:
                print("Expert not found")
        elif args.collaborate:
            # Multi-expert collaboration
            responses = manager.multi_expert_collaboration(args.query, args.collaborate)
            if "synthesis" in responses:
                print(f"\nSynthesis:\n{responses['synthesis']}")
        elif args.debate:
            # Expert debate
            debate_log = manager.debate_mode(args.query, args.debate[0], args.debate[1])
            for entry in debate_log:
                if "error" not in entry:
                    print(f"\n{entry['expert']} (Round {entry['round']}):")
                    print(entry['response'][:200] + "...")
        else:
            # Auto-select expert
            expert = manager.select_expert(args.query)
            if expert:
                print(f"Selected Expert: {expert.name}")
                response = manager.consult_expert(expert, args.query)
                print(f"\nResponse:\n{response}")
            else:
                print("No suitable expert found")
    else:
        print("Advanced Role Management System")
        print("\nUsage:")
        print("  python role_manager.py --demo")
        print("  python role_manager.py --query 'Your question' [--expert 'Expert Name']")
        print("  python role_manager.py --query 'Your question' --collaborate Expert1 Expert2")
        print("  python role_manager.py --query 'Your question' --debate Expert1 Expert2")
        print("\nAvailable Experts:")
        print(manager.get_expert_roster())


if __name__ == "__main__":
    main()