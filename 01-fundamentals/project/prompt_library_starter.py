"""
Module 01: Project - Build Your Own Prompt Library

This starter file provides the foundation for building a reusable prompt library.
Complete the TODOs to create a fully functional prompt management system.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any


class PromptTemplate:
    """
    A reusable prompt template with variable substitution.
    """

    def __init__(
        self,
        name: str,
        template: str,
        variables: List[str],
        description: str = "",
        category: str = "general",
        optimal_settings: Dict[str, Any] = None
    ):
        self.name = name
        self.template = template
        self.variables = variables
        self.description = description
        self.category = category
        self.optimal_settings = optimal_settings or {"temperature": 0.7}
        self.created_at = datetime.now().isoformat()
        self.usage_history = []

    def format(self, **kwargs) -> str:
        """
        TODO: Implement template formatting with variable substitution.

        Requirements:
        1. Validate all required variables are provided
        2. Substitute variables in template
        3. Record usage in history
        4. Return formatted prompt
        """
        pass

    def validate_variables(self, **kwargs) -> bool:
        """
        TODO: Check if all required variables are provided.
        """
        pass

    def to_dict(self) -> Dict:
        """Convert template to dictionary for storage."""
        return {
            "name": self.name,
            "template": self.template,
            "variables": self.variables,
            "description": self.description,
            "category": self.category,
            "optimal_settings": self.optimal_settings,
            "created_at": self.created_at,
            "usage_count": len(self.usage_history)
        }


class PromptLibrary:
    """
    A library to manage and organize prompt templates.
    """

    def __init__(self, storage_path: str = "prompt_library.json"):
        self.storage_path = storage_path
        self.templates: Dict[str, PromptTemplate] = {}
        self.load()

    def add_template(self, template: PromptTemplate) -> None:
        """
        TODO: Add a new template to the library.

        Requirements:
        1. Check if name already exists
        2. Add template to library
        3. Save to storage
        """
        pass

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        TODO: Retrieve a template by name.
        """
        pass

    def search(self, category: str = None, keyword: str = None) -> List[PromptTemplate]:
        """
        TODO: Search templates by category or keyword.

        Requirements:
        1. Filter by category if provided
        2. Search in name/description if keyword provided
        3. Return matching templates
        """
        pass

    def list_categories(self) -> List[str]:
        """
        TODO: Get all unique categories in the library.
        """
        pass

    def save(self) -> None:
        """Save library to JSON file."""
        data = {
            name: template.to_dict()
            for name, template in self.templates.items()
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load library from JSON file."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for name, template_data in data.items():
                    # Reconstruct PromptTemplate objects
                    self.templates[name] = PromptTemplate(
                        name=template_data['name'],
                        template=template_data['template'],
                        variables=template_data['variables'],
                        description=template_data.get('description', ''),
                        category=template_data.get('category', 'general'),
                        optimal_settings=template_data.get('optimal_settings', {})
                    )


class PromptOptimizer:
    """
    Helper class to optimize prompts based on performance metrics.
    """

    def __init__(self):
        self.metrics = []

    def record_performance(
        self,
        template_name: str,
        response_quality: float,  # 0-1 score
        response_time: float,
        token_count: int,
        cost: float
    ) -> None:
        """
        TODO: Record performance metrics for a prompt execution.
        """
        pass

    def get_recommendations(self, template_name: str) -> Dict[str, Any]:
        """
        TODO: Analyze metrics and provide optimization recommendations.

        Should return:
        - Average quality score
        - Average response time
        - Average cost
        - Suggested temperature adjustments
        - Other optimization tips
        """
        pass


# ===== Example Templates to Get Started =====

def create_default_templates() -> List[PromptTemplate]:
    """
    Create some default templates to populate the library.
    """
    templates = []

    # Email template
    templates.append(PromptTemplate(
        name="professional_email",
        template="""Write a professional email with these details:
To: {recipient}
Subject: {subject}
Key Points: {key_points}
Tone: {tone}

Email:""",
        variables=["recipient", "subject", "key_points", "tone"],
        description="Generate professional emails",
        category="communication",
        optimal_settings={"temperature": 0.7, "max_tokens": 300}
    ))

    # Code documentation template
    templates.append(PromptTemplate(
        name="code_documentation",
        template="""Document this {language} code:

```{language}
{code}
```

Include:
- Purpose and functionality
- Parameters and return values
- Usage example
- Any important notes

Documentation:""",
        variables=["language", "code"],
        description="Generate code documentation",
        category="development",
        optimal_settings={"temperature": 0.3, "max_tokens": 500}
    ))

    # Data analysis template
    templates.append(PromptTemplate(
        name="data_analysis",
        template="""Analyze this dataset information:

Dataset: {dataset_description}
Columns: {columns}
Size: {size}
Goal: {analysis_goal}

Provide:
1. Initial observations
2. Suggested analyses
3. Potential visualizations
4. Key questions to explore

Analysis:""",
        variables=["dataset_description", "columns", "size", "analysis_goal"],
        description="Guide data analysis approach",
        category="data_science",
        optimal_settings={"temperature": 0.5, "max_tokens": 600}
    ))

    return templates


# ===== Main Application =====

def main():
    """
    Main application to demonstrate the prompt library.
    """
    print("Prompt Library Manager")
    print("=" * 50)

    # TODO: Implement the main application logic
    # 1. Create or load library
    # 2. Add default templates if library is empty
    # 3. Provide menu for user interactions:
    #    - List all templates
    #    - Search templates
    #    - Use a template
    #    - Add new template
    #    - View performance metrics
    #    - Export/Import templates

    library = PromptLibrary()

    # If library is empty, add defaults
    if not library.templates:
        print("Initializing library with default templates...")
        for template in create_default_templates():
            # TODO: Add each template to library
            pass

    while True:
        print("\nOptions:")
        print("1. List all templates")
        print("2. Search templates")
        print("3. Use a template")
        print("4. Add new template")
        print("5. View categories")
        print("6. Exit")

        choice = input("\nSelect option (1-6): ")

        if choice == "1":
            # TODO: List all templates
            print("TODO: Implement listing")

        elif choice == "2":
            # TODO: Search templates
            print("TODO: Implement search")

        elif choice == "3":
            # TODO: Use a template
            print("TODO: Implement template usage")

        elif choice == "4":
            # TODO: Add new template
            print("TODO: Implement template creation")

        elif choice == "5":
            # TODO: View categories
            print("TODO: Implement category view")

        elif choice == "6":
            print("Saving library...")
            library.save()
            print("Goodbye!")
            break

        else:
            print("Invalid option, please try again")


if __name__ == "__main__":
    main()