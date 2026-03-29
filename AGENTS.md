# AGENTS.md - AI Agent + Local GPT System

This document provides guidance for AI agents working in this repository.

## Project Overview

This project combines a cloud-based AI agent with a local lightweight Transformer-based GPT model. The system refines queries, improves responses, and adapts to user feedback using both local and cloud AI components.

## Build/Lint/Test Commands

### Running Tests
```bash
# Run all Transformer tests
cd Transformer && python -m pytest

# Run a specific test file
python -m pytest Transformer/test_tokenizer.py

# Run a specific test function
python -m pytest Transformer/test_tokenizer.py::test_word_collection

# Run tokenizer tests directly
cd Transformer && python test_tokenizer.py

# Run dataset tests directly
cd Transformer && python test_dataset.py

# Run similarity tests directly
cd Transformer && python test_similarity.py

# Run performance tests
cd Performance_Tests && python test_performance.py
cd Performance_Tests && python benchmark.py
```

### Testing Single Components
```bash
# Tokenizer
cd Transformer && python -c "from tokenizer import WordCollection; tc = WordCollection(); print('OK')"

# Dataset
cd Transformer && python -c "from dataset import TextDataset; print('OK')"

# Model
cd Transformer && python -c "from model import MiniTransformer; print('OK')"

# AI agent
python -c "from AI_agent.agent import agent; print('OK')"

# CLI
python main.py --help
```

### Linting and Formatting
```bash
# Format code
black .

# Lint
flake8 .

# Type checking
mypy . --strict

# Syntax check
python -m py_compile Transformer/*.py AI_agent/*.py config/*.py
```

### No Build System Required
This is a pure Python project with minimal dependencies (NumPy, OpenAI, pytest).

## Code Style Guidelines

### General Style
- **Indentation**: 4 spaces (no tabs)
- **Line length**: ~80-100 chars, readable but not enforced
- **Naming**:
  - Classes: `PascalCase` (e.g., `TextDataset`, `WordCollection`, `MiniTransformer`)
  - Functions/variables: `snake_case` (e.g., `encode_word`, `vocab_size`, `update_knowledge`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `MIN_TRAIN_DATA`, `WEIGHTS_PATH`)
- **File structure**: One class per file typically

### Imports Organization
```python
import os
import re

import numpy as np

from AI_agent.memory import save_log, get_high_quality_logs
from config.config import OPENAI_API_KEY
```
- Standard library imports first
- Third-party (numpy, openai) next
- Local imports last
- No wildcard imports (`from module import *`)
- Group imports with blank lines between groups

### Type Hints (Recommended)
```python
def encode(self, word: str) -> int:
    """Encode a word to token index."""
```

### Error Handling
- Use `assert` statements for development/testing invariants (e.g., `assert d_model % num_heads == 0`)
- No production error handling (educational project)
- Input validation: normalize inputs (lowercase) in tokenizer
- Document preconditions in docstrings

### Documentation
- Brief header comment on each file: `# file_name.py`
- Class docstrings: Triple-quoted with description
- Function docstrings: Include parameters and return values
- Inline comments: Explain complex NumPy operations or mathematical formulas

### Testing Conventions
- Test files: `test_*.py` naming
- Test functions: `test_*` naming
- Use `assert` for validation
- Mock external dependencies when needed (see `test_dataset.py`)
- Run tests from Transformer directory for imports to work

### Mathematical Code Style
For Transformer operations:
- Use descriptive variable names: `position`, `div_term`, `attention_scores`
- Comment mathematical formulas: `# positional encoding: sin/cos formula`
- Shape annotations in comments: `# [seq_len, d_model]`
- Use explicit shapes: `np.zeros((seq_len, d_model))`
- Use `keepdims=True` when needed for broadcasting

## Project Structure

```
AI_Summary_Prompts/
├── Transformer/           # Local GPT implementation
│   ├── config.py          # Model hyperparameters
│   ├── tokenizer.py       # Word tokenization (WordCollection)
│   ├── dataset.py         # Text dataset (TextDataset)
│   ├── model.py           # Transformer (Embedding, MultiHeadSelfAttention, etc.)
│   ├── train.py           # Training loop
│   ├── generate.py        # Text generation
│   ├── loss.py            # Cross-entropy loss
│   ├── similarity.py      # Similarity metrics
│   ├── test_tokenizer.py # Tokenizer tests
│   ├── test_dataset.py    # Dataset tests
│   └── test_similarity.py # Similarity tests
├── AI_agent/              # AI agent system
│   ├── agent.py           # Main agent + cloud agent
│   ├── cli.py             # CLI entry point
│   ├── memory.py          # SQLite logging
│   ├── router.py          # Query routing
│   └── rag.py             # RAG implementation
├── config/                # Configuration
│   ├── config.py          # Settings (OPENAI_API_KEY, MIN_TRAIN_DATA, SIMILARITY_THRESHOLD, etc.)
│   └── .env               # Environment variables
├── Data/                  # Generated data
├── Performance_Tests/     # Performance benchmarking
│   ├── test_performance.py
│   ├── benchmark.py
│   └── README.md
├── main.py                # Entry point
└── README.md
```

## AI Agent Guidelines

### When Working on This Codebase
1. **Understand the educational purpose**: For learning Transformer internals, not production
2. **Keep it minimal**: Avoid unnecessary abstractions or dependencies
3. **Follow existing patterns**: Match the simple, direct style
4. **Add tests for new functionality**: Create `test_*.py` files
5. **Validate NumPy operations**: Check shapes match expected dimensions

### What to Avoid
- Adding PyTorch, TensorFlow, or complex frameworks
- Over-engineering solutions
- Premature optimization
- Breaking simplicity of educational examples

### Recommended Workflow
1. Test existing code: `cd Transformer && python test_*.py`
2. Add new functionality with matching tests
3. Verify imports: `python -c "import new_module"`
4. Run affected tests
5. Update docs if API changes

### Common Tasks
- **Adding model components**: Follow `model.py` patterns (forward/backward, layer structure)
- **Extending tokenizer**: Keep `WordCollection` simple, lowercase normalization
- **Adding datasets**: Follow `TextDataset` pattern (`__len__`, `__getitem__`)
- **Writing tests**: Use `test_*.py` as templates

## Environment
- Python 3.13.5+
- NumPy 2.1.3+
- OpenAI SDK
- pytest
- Run from `Transformer/` directory for relative imports to work

## Key Implementation Details

### Transformer Model
- Pure NumPy implementation (no ML frameworks)
- Manual backpropagation in each layer
- Components: Embedding, MultiHeadSelfAttention, FeedForward, LayerNorm, TransformerBlock, OutputLayer
- Uses causal masking for autoregressive generation

### AI Agent Flow (via AI_agent/agent.py)
1. `agent(query)` calls `generate_response()` from similarity.py
2. Each question (including sub-questions) goes through:
   - Check DB records ≥ MIN_SIMILARITY_DATA (100) → else direct cloud call
   - Check DB similarity ≥ 0.90 → return DB answer directly
   - Else: weighted average length estimation
   - If estimated > 200 words → decompose into sub-questions (recursive, max depth 10)
   - Else → call cloud_agent() with English prompt
3. Sub-questions each go through the same flow recursively
4. All answers combined into coherent response
5. User feedback (1-5 rating) → SQLite → Training data

### Similarity Module (Transformer/similarity.py)
- `generate_response(query, tokenizer, model, depth=0)`: Main routing function
- `predict_response_length()`: Check DB similarity + weighted length estimation
- `decompose_question()`: Split complex questions into sub-questions
- `_answer_sub_questions()`: Recursively answer each sub-question
- `_combine_answers()`: Merge into coherent response
- `cloud_agent()`: English prompts for OpenAI API

### Config Constants (config/config.py)
- `MIN_SIMILARITY_DATA = 100`: Min DB records before using similarity (skip if insufficient)
- `SIMILARITY_THRESHOLD = 0.90`: Min similarity for direct DB response
- `MAX_RESPONSE_LENGTH = 200`: Max answer length before decomposition
- `MAX_DECOMPOSE_DEPTH = 10`: Max recursion depth for question decomposition
- `MAX_DB_RECORDS = 5000`: Max DB records to load (memory protection)
