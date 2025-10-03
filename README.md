# Prompt Engineering Learning Path

A comprehensive, hands-on curriculum for mastering prompt engineering and context engineering with Large Language Models (LLMs).

## ✅ Status: Production Ready (October 2025)

- **14 Modules** | **100+ Python Files** | **81+ Tests** | **Full CI/CD Pipeline**
- All modules include examples, exercises, solutions, and production projects
- **Recent Enhancements** ([PR-10-02-25](docs/PR-10-02-25.md) - ✅ COMPLETED):
  - ✅ Comprehensive pytest test suite with 81+ autograder tests
  - ✅ Docker containerization with multi-stage builds
  - ✅ GitHub Actions CI/CD pipeline (lint, test, security scan)
  - ✅ End-to-end capstone RAG project with evaluation harness
  - ✅ Run tracking system for reproducibility and cost analysis
  - ✅ Solutions access control to encourage independent learning
  - ✅ Rich CLI formatting for enhanced user experience
  - ✅ Makefile with 20+ development commands
- **Models**: GPT-5 family (gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-codex) and Claude Sonnet 4.5
- **Production Features**: Testing, Docker, CI/CD, monitoring, cost tracking, safety patterns

## 🎯 Learning Objectives

By completing this curriculum, you will:
- Master fundamental prompt engineering techniques
- Understand context window management and optimization
- Build production-ready AI applications using RAG
- Implement advanced techniques like function calling and structured outputs
- Work effectively with the latest models: GPT-5, Claude Sonnet 4.5
- Design and deploy AI agents and workflows

## 📚 Curriculum Overview

The curriculum is organized into 14 progressive modules, each building upon the previous:

### Foundation (Modules 01-03)
- **01-fundamentals**: Core prompt engineering principles
- **02-zero-shot-prompting**: Working without examples
- **03-few-shot-learning**: Learning from examples

### Core Techniques (Modules 04-07)
- **04-chain-of-thought**: Step-by-step reasoning
- **05-prompt-chaining**: Multi-step workflows
- **06-role-based-prompting**: Persona-driven interactions
- **07-context-management**: Optimizing context usage

### Advanced Methods (Modules 08-11)
- **08-structured-outputs**: JSON mode and schema validation
- **09-function-calling**: Tool use and API integration
- **10-rag-basics**: Retrieval Augmented Generation
- **11-advanced-rag**: Production RAG patterns

### Expert Level (Modules 12-14)
- **12-prompt-optimization**: Performance tuning
- **13-agent-design**: Building AI agents
- **14-production-patterns**: Enterprise deployment

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ installed
- OpenAI API key (get one at https://platform.openai.com) - for GPT-5 models
- Anthropic API key (optional, for Claude models)
- Basic programming knowledge
- Familiarity with JSON and APIs

### Setup Instructions

1. Clone this repository:
```bash
git clone <repository-url>
cd prompting-101
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API keys:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Quick Start with Makefile

We provide a comprehensive Makefile for common tasks:

```bash
# Complete setup (creates venv, installs deps, creates .env)
make setup

# Run tests
make test              # All tests
make test-unit         # Fast unit tests only
make test-module M=04  # Test specific module

# Run examples
make run-example E=01-fundamentals/examples/basic_prompting.py --all

# Start Jupyter
make jupyter

# Code quality
make lint              # Run linting
make format            # Format code with black
make type-check        # Run mypy

# See all commands
make help
```

### Docker Quick Start

Use Docker for a consistent development environment:

```bash
# Build and run tests
docker compose up test

# Interactive development shell
docker compose run dev

# Start Jupyter notebook (port 8888)
docker compose up jupyter

# Run specific module examples
docker compose run app python 01-fundamentals/examples/basic_prompting.py --all
```

## 📖 How to Use This Curriculum

### Recommended Learning Path

1. **Start with Module 01**: Read the README, complete exercises
2. **Practice with examples**: Each module has working code examples
3. **Complete the exercises**: Hands-on practice reinforces concepts
4. **Build the project**: Apply what you learned in a mini-project
5. **Review and reflect**: Check your solution against provided examples

### Time Investment
- **Per module**: 2-4 hours
- **Total curriculum**: 30-50 hours
- **Pace**: 1-2 modules per week recommended

### Learning Tips
- Run all code examples yourself
- Experiment with variations
- Keep a learning journal
- Join the discussions (if available)
- Apply concepts to your own projects

## 🛠️ Tools and Technologies

### Primary Tools
- **OpenAI GPT-5**: Latest flagship model (also: GPT-5-mini, GPT-5-nano, GPT-5-codex)
- **Anthropic Claude Sonnet 4.5**: High-performance model with excellent cost/quality ratio
- **LangChain**: For complex workflows and RAG
- **Python 3.8+**: Primary programming language
- **FastAPI**: For production applications
- **Jupyter Notebooks**: For interactive experimentation

### Included Dependencies
All required packages are in `requirements.txt`:
- Core: `openai`, `anthropic`, `tiktoken`, `pydantic`
- RAG: `langchain`, `chromadb`, `faiss-cpu`
- ML/Data: `numpy`, `pandas`, `scikit-learn`
- Production: `fastapi`, `uvicorn`, `sqlalchemy`, `redis`
- Utilities: `aiofiles`, `PyJWT`, `PyYAML`

### What You'll Build
- **Module 01-03**: Prompt libraries and templates
- **Module 04-07**: Reasoning systems and workflows
- **Module 08-09**: JSON APIs and tool integrations
- **Module 10-11**: Production RAG systems with vector DBs
- **Module 12**: Performance optimization platforms
- **Module 13**: Autonomous agent systems with memory
- **Module 14**: Full production LLM platform with monitoring

## 📂 Repository Structure

```
prompting-101/
├── README.md                    # This file
├── CONTRIBUTING.md              # Contribution guidelines and solutions policy
├── requirements.txt             # Python dependencies (all verified)
├── requirements-dev.txt         # Development dependencies (pytest, black, etc.)
├── .env.example                # API key template
├── Dockerfile                   # Multi-stage Docker build
├── compose.yaml                 # Docker Compose orchestration
├── Makefile                     # Common development tasks
├── pytest.ini                   # Pytest configuration with module markers
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD pipeline
│
├── shared/                      # Shared utilities
│   ├── utils.py                # LLM client, token counting, cost estimation
│   ├── prompts.py              # Common prompt templates
│   ├── solutions.py            # Solutions access control
│   ├── runs.py                 # Run tracking for reproducibility
│   └── printing.py             # Rich CLI formatting utilities
│
├── llm/                         # Modern LLM client (Responses API)
│   ├── __init__.py
│   └── client.py               # Responses API wrapper with structured outputs
│
├── tests/                       # Pytest test suite
│   ├── test_module_01.py       # Autograders for fundamentals
│   ├── test_module_02.py       # Zero-shot prompting tests
│   ├── test_module_03.py       # Few-shot learning tests
│   ├── test_module_04.py       # Chain-of-thought tests
│   ├── test_module_05.py       # Prompt chaining tests
│   └── test_modules_06_14.py   # Advanced modules tests
│
├── capstone/                    # End-to-end RAG capstone project
│   ├── README.md               # Comprehensive project guide
│   ├── RUBRIC.md               # Grading criteria (100 points)
│   ├── data/                   # Sample corpus and test queries
│   ├── src/                    # RAG system implementation
│   └── tests/                  # Capstone test suite
│
├── docs/                        # Documentation
│   ├── PR-10-02-25.md          # Original implementation spec
│   └── PLAN-PR-10-02-25.md     # Detailed implementation plan
│
├── 01-fundamentals/             # Module structure (repeated for all 14)
│   ├── README.md               # Module guide with learning objectives
│   ├── examples/               # 3 working code examples
│   ├── exercises/              # Practice exercises
│   ├── solutions/              # Solutions (access-controlled)
│   └── project/                # Production-ready project
│
├── [02-14 modules...]           # 13 additional modules
│
└── runs/                        # Run logs for reproducibility (gitignored)
```

## 🎓 Learning Outcomes

### After Module 07
You'll be able to:
- Write effective prompts for any LLM
- Manage context windows efficiently
- Implement basic prompt patterns
- Debug common prompting issues

### After Module 11
You'll be able to:
- Build RAG applications
- Implement function calling
- Generate structured outputs
- Design multi-step workflows

### After Module 14
You'll be able to:
- Design production AI systems
- Optimize for cost and performance
- Implement evaluation frameworks
- Deploy enterprise solutions

## 🎯 Capstone Project

The curriculum culminates in a comprehensive **capstone project** that integrates all 14 modules:

### What You'll Build
A production-ready RAG (Retrieval Augmented Generation) system with:
- **Hybrid retrieval** (dense + sparse methods with fusion)
- **Reranking** for improved relevance
- **Query processing** (expansion, correction, classification)
- **Structured outputs** with JSON validation
- **Citation tracking** for source attribution
- **FastAPI REST API** with proper error handling
- **Automated evaluation** (precision, recall, F1, latency, cost)
- **Caching and cost optimization**
- **Comprehensive testing**

### Dataset
Python programming documentation corpus (100 documents) covering:
- Language fundamentals (loops, functions, data types)
- Standard library modules (datetime, collections, itertools)
- Advanced topics (decorators, generators, async/await)
- Best practices (PEP 8, testing, packaging)

### Assessment
Graded on 100-point rubric:
- **Functionality (40%)**: Core RAG pipeline + advanced features
- **Code Quality (20%)**: Organization, documentation, testing
- **RAG Performance (20%)**: Retrieval and answer quality metrics
- **Module Integration (10%)**: Concepts from all 14 modules
- **Production Readiness (10%)**: API, monitoring, robustness

See [capstone/README.md](capstone/README.md) and [capstone/RUBRIC.md](capstone/RUBRIC.md) for complete details.

## 🧪 Testing & Quality Assurance

### Test Suite
Comprehensive pytest test suite with 81+ tests:

```bash
# Run all tests
pytest

# Run specific module tests
pytest -m module04

# Run only fast unit tests
pytest -m unit

# Run with coverage report
pytest --cov=shared --cov-report=html
```

### Test Organization
- **Module 01-03**: Basic prompting, zero-shot, few-shot
- **Module 04-05**: Chain-of-thought, prompt chaining
- **Module 06-14**: Advanced techniques (RAG, agents, production)
- **Capstone**: End-to-end RAG system tests

### CI/CD Pipeline
GitHub Actions workflow runs on every push:
- ✅ Multi-version Python testing (3.9, 3.10, 3.11)
- ✅ Linting (flake8, black, pylint)
- ✅ Type checking (mypy)
- ✅ Security scanning (bandit, safety)
- ✅ Docker build validation
- ✅ Documentation checks

### Solutions Access Control
Solutions are hidden by default to encourage independent learning:

```bash
# Solutions require environment variable
export ALLOW_SOLUTIONS=1

# Or add to .env file
echo "ALLOW_SOLUTIONS=1" >> .env
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete solutions policy.

## 📚 Additional Resources

### Documentation
- [OpenAI Documentation](https://platform.openai.com/docs)
- [Anthropic Claude Docs](https://docs.claude.com)
- [Prompt Engineering Guide](https://www.promptingguide.ai)

### Courses
- DeepLearning.AI Prompt Engineering
- Coursera RAG Specialization
- Fast.ai Practical Deep Learning

### Communities
- OpenAI Developer Forum
- r/LocalLLaMA Reddit
- Discord AI Communities

## 📊 Quality Assurance

This curriculum has undergone comprehensive review and testing:
- ✅ All 86 Python files syntax-validated
- ✅ All imports and dependencies verified
- ✅ All file references in READMEs accurate
- ✅ Complete examples, exercises, and solutions for all modules
- ✅ Production-ready project in each module

See [ISSUES.md](ISSUES.md) for the complete quality assurance report.

## 🤝 Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

### Ways to Contribute
- Add new examples
- Improve explanations
- Fix bugs in code
- Suggest new modules
- Share your projects

## 📄 License

This curriculum is provided for educational purposes. See LICENSE for details.

## 🙏 Acknowledgments

This curriculum incorporates best practices from:
- OpenAI's Prompt Engineering Guide
- Anthropic's Claude Documentation
- Academic research papers
- Community contributions

---

**Start your journey with [Module 01: Fundamentals →](01-fundamentals/README.md)**