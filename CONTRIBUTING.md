# Contributing to Prompt Engineering 101

Thank you for your interest in contributing to this curriculum! This document outlines our contribution guidelines and policies.

## 🎯 Our Mission

This curriculum aims to teach prompt engineering through hands-on practice. We prioritize:
1. **Learning by doing** - Exercises before solutions
2. **Production readiness** - Real-world applicable patterns
3. **Progressive difficulty** - Building from fundamentals to advanced topics
4. **Quality over quantity** - Well-tested, documented code

## 🤝 How to Contribute

### Types of Contributions Welcome

1. **Bug Fixes** - Fix typos, errors, or broken code
2. **Examples** - Add new working examples to modules
3. **Exercises** - Create new practice problems
4. **Documentation** - Improve READMEs, add clarifications
5. **Tests** - Add or improve autograder tests
6. **Tooling** - Enhance development workflows

### Contribution Process

1. **Fork the repository** and create a feature branch
2. **Make your changes** following our coding standards
3. **Test your changes** - Ensure all tests pass
4. **Submit a Pull Request** with a clear description
5. **Respond to feedback** during code review

## 📋 Coding Standards

### Python Code Style

- **PEP 8** compliance (use `black` for formatting)
- **Type hints** for function signatures
- **Docstrings** for all public functions/classes
- **Error handling** with informative messages

Example:
```python
def calculate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Calculate token count for text using specified model.

    Args:
        text: Input text to tokenize
        model: Model name for tokenization (default: "gpt-4")

    Returns:
        int: Number of tokens

    Raises:
        ValueError: If model is not supported
    """
    # Implementation
    pass
```

### Module Structure

Each module must include:
- **README.md** - Learning objectives, concepts, exercises
- **examples/** - 2-3 working code examples
- **exercises/** - Practice problems with clear instructions
- **solutions/** - Reference implementations (access-controlled)
- **project/** - Integrative mini-project (optional)

### Testing Requirements

- All new code must include tests
- Tests should use pytest markers appropriately
- Integration tests gated behind API key checks
- Aim for >80% code coverage on shared utilities

## 🔒 Solutions Policy

### Philosophy

Solutions are hidden by default to encourage independent learning. Students learn more effectively by attempting exercises before viewing solutions.

### Implementation

All solution files must use the solutions access control system:

```python
from shared.solutions import require_solutions, solutions_enabled

@require_solutions
def get_solution():
    """This function is only accessible when ALLOW_SOLUTIONS=1."""
    return "Solution content"

# Or check directly
if solutions_enabled():
    print("Solution: ...")
else:
    from shared.solutions import get_solution_message
    print(get_solution_message())
```

### Access Control Rules

1. **Solutions require `ALLOW_SOLUTIONS=1`** environment variable
2. **Never commit solutions to exercises** in public-facing documentation
3. **Solution files live in `solutions/` directories** within each module
4. **Tests may check solution structure** but should not expose answers
5. **Instructors may provide solutions** via separate distribution channels

### For Contributors

When adding new exercises:
- ✅ **DO** create a solution file in the `solutions/` directory
- ✅ **DO** use the `@require_solutions` decorator
- ✅ **DO** provide clear exercise instructions
- ❌ **DON'T** include solutions in README or example files
- ❌ **DON'T** hardcode solutions in autograder tests
- ❌ **DON'T** bypass the access control system

### For Learners

- **Attempt exercises independently first** - You'll learn more!
- **Set `ALLOW_SOLUTIONS=1`** only after genuine attempts
- **Use solutions for verification** - Not as a crutch
- **Understand, don't copy** - Typing out solutions builds muscle memory

### For Instructors

If you're teaching a course using this curriculum:
- Solutions can be shared with students after assignment deadlines
- Consider using a private fork with solutions enabled
- Encourage students to attempt exercises before revealing solutions
- Use autograder tests to check understanding without exposing answers

## 🧪 Testing Guidelines

### Test Organization

Tests use pytest markers for organization:
- `@pytest.mark.unit` - Fast tests, no external dependencies
- `@pytest.mark.integration` - Tests that call APIs (gated behind API key checks)
- `@pytest.mark.exercise` - Autograder tests for exercises
- `@pytest.mark.module01` through `@pytest.mark.module14` - Module-specific tests
- `@pytest.mark.slow` - Tests taking >1 second
- `@pytest.mark.requires_api` - Tests requiring API keys

### Running Tests

```bash
# Run all tests
pytest

# Run specific module tests
pytest -m module04

# Run only unit tests (fast)
pytest -m unit

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=shared --cov-report=html
```

### Writing Tests

Example autograder test:
```python
import pytest

@pytest.mark.module04
@pytest.mark.exercise
@pytest.mark.unit
def test_student_understands_chain_of_thought():
    """Test that student's CoT prompt includes reasoning steps."""
    # Import student's solution (with access control)
    from solutions.exercise1 import create_cot_prompt

    prompt = create_cot_prompt("What is 15% of 80?")

    # Validate structure, not exact answer
    assert "step" in prompt.lower(), "CoT prompt should include explicit steps"
    assert len(prompt) > 50, "CoT prompt should be detailed"
```

## 📚 Documentation Standards

### README Structure

Module READMEs should follow this template:
1. **Learning Objectives** - What you'll learn
2. **Key Concepts** - Core ideas introduced
3. **Prerequisites** - What to know before starting
4. **Examples** - Brief overview of example files
5. **Exercises** - Practice problems to complete
6. **Project** - Integrative mini-project
7. **Additional Resources** - Links to further reading

### Code Comments

- **Explain WHY, not WHAT** - Code shows what, comments explain why
- **Document assumptions** - What conditions must be true
- **Highlight gotchas** - Non-obvious behavior or edge cases
- **Provide context** - Why this approach vs. alternatives

## 🚀 Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd prompting-101

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run tests to verify setup
pytest -m unit
```

### Making Changes

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# ... edit files ...

# Format code
black .

# Run tests
pytest

# Commit with conventional commits
git commit -m "feat: Add new example for chain-of-thought"
```

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions or modifications
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

Examples:
- `feat: Add hybrid retrieval example to module 11`
- `fix: Correct token counting in shared/utils.py`
- `docs: Update README with Azure OpenAI setup instructions`
- `test: Add integration tests for RAG pipeline`

## 🔍 Code Review Process

### What We Look For

1. **Correctness** - Does it work as intended?
2. **Clarity** - Is the code easy to understand?
3. **Completeness** - Are tests and documentation included?
4. **Consistency** - Does it follow existing patterns?
5. **Performance** - Are there obvious optimizations?

### Review Criteria

- ✅ Code follows PEP 8 and project conventions
- ✅ Tests pass locally
- ✅ Documentation is clear and complete
- ✅ No secrets or API keys committed
- ✅ Solutions use access control properly
- ✅ Examples are well-commented
- ✅ Error messages are helpful

## 🎓 Educational Philosophy

### Design Principles

1. **Progressive Complexity** - Start simple, build gradually
2. **Hands-On Learning** - Code first, theory second
3. **Real-World Focus** - Production-applicable patterns
4. **Conceptual Understanding** - Not just "how" but "why"
5. **Safe Experimentation** - Encourage trying and failing

### Exercise Design

Good exercises should:
- **Have clear objectives** - What skill is being practiced?
- **Provide context** - Why is this useful?
- **Include constraints** - What are the requirements?
- **Allow creativity** - Multiple valid approaches
- **Build incrementally** - Each harder than the last

Example exercise structure:
```markdown
## Exercise 3: Implement Chain-of-Thought Reasoning

**Objective**: Practice breaking down complex problems into steps.

**Context**: Many reasoning tasks benefit from explicit step-by-step thinking.

**Task**: Create a function that generates a chain-of-thought prompt for math word problems.

**Requirements**:
- Prompt should include "Let's think step by step"
- Break problem into at least 3 steps
- Include verification step at the end

**Hints**:
- Review example_cot.py for patterns
- Consider what intermediate steps are needed
- Think about how to structure the reasoning

**Success Criteria**:
- Function returns a valid prompt string
- Prompt includes explicit reasoning steps
- Test cases pass
```

## 🐛 Reporting Issues

When reporting bugs or issues:

1. **Check existing issues** - Has it been reported already?
2. **Provide context** - What were you trying to do?
3. **Include details** - Python version, OS, error messages
4. **Share reproduction steps** - How can we recreate the issue?
5. **Suggest solutions** - Have ideas for fixing it?

Issue template:
```markdown
**Description**: Brief description of the issue

**Environment**:
- OS: macOS 13.5
- Python: 3.11.4
- Module: 04-chain-of-thought

**Steps to Reproduce**:
1. Run `python examples/example_cot.py`
2. See error: ...

**Expected Behavior**: What should happen

**Actual Behavior**: What actually happened

**Error Message**:
```
[paste error here]
```

**Possible Solution**: (optional)
```

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as this project (see LICENSE file).

## 🙏 Thank You!

Your contributions make this curriculum better for everyone. Whether you're fixing a typo or adding a new module, your help is appreciated!

For questions or discussions, feel free to open an issue or reach out to the maintainers.

---

**Happy learning and contributing! 🚀**
