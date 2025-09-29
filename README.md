# Prompt Engineering Learning Path

A comprehensive, hands-on curriculum for mastering prompt engineering and context engineering with Large Language Models (LLMs).

## 🎯 Learning Objectives

By completing this curriculum, you will:
- Master fundamental prompt engineering techniques
- Understand context window management and optimization
- Build production-ready AI applications using RAG
- Implement advanced techniques like function calling and structured outputs
- Work effectively with both OpenAI and Anthropic models
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
- **07-context-window-management**: Optimizing context usage

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
- OpenAI API key (get one at https://platform.openai.com)
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
- **OpenAI GPT-4**: Primary model for exercises
- **Anthropic Claude**: Alternative model examples
- **LangChain**: For complex workflows
- **Python**: Primary programming language

### Optional Tools
- **Jupyter Notebooks**: For experimentation
- **Vector Databases**: Pinecone, Weaviate for RAG
- **Evaluation Tools**: For testing prompt effectiveness

## 📂 Repository Structure

```
prompting-101/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                # API key template
├── shared/                     # Shared utilities
│   ├── utils.py               # Helper functions
│   └── prompts.py             # Common prompt templates
├── 01-fundamentals/            # First module
│   ├── README.md              # Module guide
│   ├── examples/              # Code examples
│   ├── exercises/             # Practice problems
│   ├── solutions/             # Exercise solutions
│   └── project/               # Module project
└── [02-14 modules...]          # Additional modules
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