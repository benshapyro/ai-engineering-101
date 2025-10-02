# AI Digest Configuration

This directory contains configuration files for [ai-digest](https://www.npmjs.com/package/ai-digest), a tool that aggregates the codebase into a single Markdown file for use with Claude Projects or ChatGPT Custom GPTs.

## Files

- **`.aidigestignore`** - Files and patterns to exclude from the digest
- **`.aidigestminify`** - Files to include as placeholders (reduces token count)
- **Generated digest files** (`.md`) - Not committed to git, regenerate as needed

## Quick Start

### Generate Digest

```bash
# From project root
npx ai-digest --whitespace-removal -o ai-digest/prompting-101-digest.md --ignore-file ai-digest/.aidigestignore --minify-file ai-digest/.aidigestminify
```

### View Statistics

```bash
npx ai-digest --show-output-files sort --ignore-file ai-digest/.aidigestignore --minify-file ai-digest/.aidigestminify
```

### Watch Mode (Auto-regenerate)

```bash
npx ai-digest --watch --whitespace-removal -o ai-digest/prompting-101-digest.md --ignore-file ai-digest/.aidigestignore --minify-file ai-digest/.aidigestminify
```

## Usage with AI Models

### Claude Projects
1. Generate the digest: `npx ai-digest --whitespace-removal -o ai-digest/prompting-101-digest.md --ignore-file ai-digest/.aidigestignore --minify-file ai-digest/.aidigestminify`
2. Upload `ai-digest/prompting-101-digest.md` to your Claude Project knowledge base
3. Re-upload before major sessions to keep context fresh

### ChatGPT Custom GPT
1. Generate the digest (same command as above)
2. Upload `ai-digest/prompting-101-digest.md` to your Custom GPT's knowledge base
3. Update periodically as the curriculum evolves

## Current Stats

- **~148 files included** (~420K tokens with whitespace removal)
- **16 files minified** (solutions preserved as placeholders)
- **661 files ignored** (venv, cache, IDE files, etc.)

## Customization

### Adding Files to Ignore

Edit `.aidigestignore`:
```
# Add patterns like .gitignore
*.log
temp/
```

### Adding Files to Minify

Edit `.aidigestminify`:
```
# Files to include as placeholders only
*/solutions/*.py
large-data-file.json
```

## Token Optimization

The digest is optimized for AI context windows:
- Whitespace removal enabled (except for Python/YAML)
- Solution files minified to placeholders
- Binary files noted but not included
- Secrets and environment files excluded

Estimated tokens: ~420K (Claude) / ~419K (GPT-4)
