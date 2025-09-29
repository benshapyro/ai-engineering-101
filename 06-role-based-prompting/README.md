# Module 06: Role-Based Prompting

## Learning Objectives
By the end of this module, you will:
- Master creating and managing AI personas for specialized tasks
- Understand how role assignment affects model behavior and output quality
- Learn to combine multiple roles for complex problem-solving
- Implement dynamic role switching based on context
- Build systems with persistent expert personas

## Key Concepts

### 1. What is Role-Based Prompting?
Role-based prompting assigns specific personas, expertise, or perspectives to the LLM, enabling it to generate responses aligned with particular domains, styles, or viewpoints. This technique leverages the model's ability to simulate different expert behaviors.

### 2. Core Benefits
- **Domain Expertise**: Access specialized knowledge and terminology
- **Consistent Voice**: Maintain character across interactions
- **Perspective Diversity**: Explore problems from multiple angles
- **Quality Improvement**: Role-appropriate responses often have higher quality
- **User Trust**: Clear expertise builds confidence

### 3. Role Definition Components

#### Basic Role Structure
```python
role = {
    "identity": "Senior Data Engineer",
    "expertise": ["SQL optimization", "ETL pipelines", "distributed systems"],
    "personality": "Detail-oriented, practical, efficiency-focused",
    "constraints": "Avoid over-engineering, consider cost",
    "output_style": "Technical but accessible"
}
```

#### Advanced Role Attributes
- **Background**: Professional history and experience
- **Knowledge Domains**: Specific areas of expertise
- **Communication Style**: Tone, formality, verbosity
- **Decision Framework**: How the role makes choices
- **Limitations**: What the role doesn't know or won't do

### 4. Role Patterns

#### Single Expert
```python
prompt = """You are a Senior DevOps Engineer with 10 years of experience
in cloud infrastructure and Kubernetes. You prioritize reliability and
cost-efficiency.

Task: Review this deployment configuration..."""
```

#### Multiple Perspectives
```python
roles = ["Security Expert", "Product Manager", "Data Scientist"]
for role in roles:
    prompt = f"As a {role}, evaluate this feature proposal..."
```

#### Role Hierarchies
```python
lead_role = "Chief Architect"
supporting_roles = ["Backend Dev", "Frontend Dev", "QA Engineer"]
```

### 5. Common Pitfalls
- **Role Confusion**: Inconsistent or contradictory attributes
- **Over-Specification**: Too many constraints limiting creativity
- **Under-Specification**: Vague roles producing generic output
- **Role Bleeding**: Previous role affecting next interaction
- **Unrealistic Expertise**: Claiming impossible knowledge

## Module Structure

### Examples
1. `expert_personas.py` - Creating specialized expert roles
2. `role_switching.py` - Dynamic role changes during conversation
3. `multi_role_collaboration.py` - Multiple roles working together

### Exercises
Practice problems focusing on:
- Designing effective personas
- Role consistency maintenance
- Multi-perspective analysis
- Dynamic role selection
- Role-based quality improvement

### Project: Role Management System
Build a system that:
- Defines reusable role templates
- Manages role switching and context
- Combines multiple roles for decisions
- Maintains role consistency
- Evaluates role effectiveness

## Best Practices

### 1. Role Definition
```python
effective_role = """You are Dr. Sarah Chen, a Machine Learning Engineer
with a PhD from Stanford and 8 years of experience at top tech companies.

Expertise:
- Deep learning architectures (transformers, CNNs)
- Production ML systems
- Model optimization and deployment

Communication style:
- Clear, structured explanations
- Uses examples from real projects
- Balances theory with practical application

Current focus: Making ML accessible to data engineers"""
```

### 2. Context Preservation
```python
# Maintain role across interactions
context = {
    "role": role_definition,
    "history": previous_responses,
    "consistency_checks": validation_rules
}
```

### 3. Role Validation
```python
def validate_role_response(response, role):
    checks = {
        "uses_appropriate_terminology": check_terminology(response, role),
        "maintains_expertise_level": check_expertise(response, role),
        "consistent_personality": check_personality(response, role)
    }
    return all(checks.values())
```

## Production Considerations

### Role Performance
- **Token Usage**: Role definitions add to prompt length
- **Caching**: Store role definitions separately
- **Reusability**: Create role libraries for common needs

### Quality Assurance
```python
class RoleQualityChecker:
    def assess_role_adherence(self, response, role):
        metrics = {
            "vocabulary_match": self.check_vocabulary(response, role),
            "expertise_demonstration": self.check_expertise(response, role),
            "personality_consistency": self.check_personality(response, role)
        }
        return sum(metrics.values()) / len(metrics)
```

### Dynamic Role Selection
```python
def select_role_for_task(task_description):
    task_analysis = analyze_task_requirements(task_description)

    role_scores = {}
    for role in available_roles:
        score = calculate_role_fit(role, task_analysis)
        role_scores[role.id] = score

    return max(role_scores, key=role_scores.get)
```

## Common Role Categories

### 1. Technical Experts
- Software Architect
- Data Scientist
- Security Analyst
- DevOps Engineer
- Database Administrator

### 2. Business Roles
- Product Manager
- Business Analyst
- Project Manager
- Strategic Consultant
- Financial Analyst

### 3. Creative Roles
- UX Designer
- Content Strategist
- Technical Writer
- Innovation Consultant

### 4. Specialized Domains
- Healthcare Professional
- Legal Advisor
- Academic Researcher
- Industry Specialist

## Role Interaction Patterns

### 1. Panel Discussion
```python
# Multiple experts discuss
moderator = "Technical Lead"
panelists = ["Security Expert", "Performance Engineer", "UX Designer"]
```

### 2. Mentor-Student
```python
mentor = "Senior Developer with 15 years experience"
student = "Junior Developer seeking guidance"
```

### 3. Devil's Advocate
```python
primary_role = "Solution Architect"
challenger_role = "Critical Reviewer finding potential issues"
```

### 4. Collaborative Team
```python
team_roles = {
    "lead": "Project Coordinator",
    "members": ["Developer", "Tester", "Analyst"],
    "stakeholder": "Product Owner"
}
```

## Exercises Overview

1. **Role Design Workshop**: Create effective personas for different domains
2. **Consistency Challenge**: Maintain role across extended conversations
3. **Multi-Role Analysis**: Same problem from different expert perspectives
4. **Dynamic Role Switching**: Adapt roles based on conversation flow
5. **Role Effectiveness**: Measure and optimize role performance

## Success Metrics
- **Role Adherence**: 90%+ consistency with defined persona
- **Quality Improvement**: 25%+ better domain-specific responses
- **User Satisfaction**: Higher trust and engagement
- **Efficiency**: Reduced prompting for specialized tasks

## Next Steps
After mastering role-based prompting, you'll move to Module 07: Context Window Management, where you'll learn to optimize the use of limited context space, especially important when maintaining complex roles and conversation history.