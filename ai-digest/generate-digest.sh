#!/bin/bash
# Generate AI digest for the prompting-101 curriculum

# Change to project root
cd "$(dirname "$0")/.."

# Generate digest with optimized settings
npx ai-digest \
  --whitespace-removal \
  -o ai-digest/prompting-101-digest.md \
  --ignore-file ai-digest/.aidigestignore \
  --minify-file ai-digest/.aidigestminify

echo ""
echo "✅ Digest generated at: ai-digest/prompting-101-digest.md"
echo "📤 Ready to upload to Claude Project or ChatGPT Custom GPT"
