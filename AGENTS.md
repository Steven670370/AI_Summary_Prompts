# AGENTS.md - Mini LLM From Scratch

This document provides guidance for AI agents working in this repository.

## Project Overview
This is a minimal Transformer-based LLM built from scratch in Python/NumPy to understand modern language model internals. The model learns to convert natural user input into structured AI prompt instructions.

## Build/Lint/Test Commands

### Running Tests
- Run all tests: `cd Transformer && python -m pytest`
- Run tokenizer tests: `cd Transformer && python test_tokenizer.py`
- Run dataset tests: `cd Transformer && python test_dataset.py`

### Testing Single Components
- Tokenizer: `cd Transformer && python -c "from tokenizer import WordCollection; tc = WordCollection(); print('Tokenizer works')"`
- Dataset: `cd Transformer && python -c "from dataset import TextDataset; print('Dataset works')"`
- Model: `cd Transformer && python -c "from model import Embedding; print('Model imports OK')"`

### No Build System Required
This is a pure Python/NumPy project with no external dependencies beyond NumPy. No compilation or build steps are needed.

### Code Validation
- Check Python syntax: `python -m py_compile Transformer/*.py`
- Run type hints check (if added): `python -m mypy Transformer/ --strict`
- Import validation: `cd Transformer && python -c "import config, tokenizer, dataset, model"`

## Code Style Guidelines

### General Style
- **Indentation**: 4 spaces (no tabs)
- **Line length**: No strict limit, but keep readable (typically 80-100 chars)
- **Naming**:
  - Classes: `PascalCase` (e.g., `TextDataset`, `WordCollection`)
  - Functions/variables: `snake_case` (e.g., `encode_word`, `vocab_size`)
  - Constants: `UPPER_SNAKE_CASE` (in config.py)
- **File structure**: One class per file typically, except config.py

### Imports Organization
```python
import numpy as np
from config import config
from tokenizer import WordCollection
```

- Standard library imports first
- Third-party imports (numpy) next
- Local imports last
- No wildcard imports (`from module import *`)
- Group imports with blank lines between groups

### Type Hints (Recommended but not required)
```python
def encode(self, word: str) -> int:
    """Encode a word to token index."""
```

### Error Handling
- Use `assert` statements for development/testing invariants
- No production error handling needed (educational project)
- Input validation: normalize inputs (lowercase) in tokenizer
- Document preconditions in docstrings

### Documentation
- Each file should have a brief header comment: `# file_name.py`
- Class docstrings: Triple-quoted with brief description
- Function docstrings: Include parameters and return values
- Inline comments: Explain complex NumPy operations or mathematical formulas

### Testing Conventions
- Test files: `test_*.py` naming
- Test functions: `test_*` naming
- Use `assert` for test validation
- Mock dependencies when needed (see test_dataset.py)
- Run tests from Transformer directory

### Mathematical Code Style
For Transformer mathematical operations:
- Use descriptive variable names: `position`, `div_term`, `attention_scores`
- Comment mathematical formulas: `# positional encoding: sin/cos formula`
- Use NumPy broadcasting properly: `PE[:, 0::2] = np.sin(position * div_term)`
- Shape annotations in comments: `# [seq_len, d_model]`

### Project Structure Conventions
```
Transformer/
├── config.py          # Model hyperparameters
├── tokenizer.py       # Word tokenization
├── dataset.py         # Text dataset creation
├── model.py           # Transformer architecture
├── train.py           # Training loop
├── generate.py        # Text generation
├── test_tokenizer.py  # Tokenizer tests
└── test_dataset.py    # Dataset tests
```

### NumPy Usage Guidelines
- Use explicit shapes: `np.zeros((seq_len, d_model))`
- Use descriptive axis parameters: `axis=-1` for last dimension
- Use `keepdims=True` when needed for broadcasting
- Precompute values when possible (e.g., positional encoding)

### Git Practices
- Commits: Descriptive messages explaining "why" not just "what"
- Branches: Feature branches for major components
- No secrets in code (none needed for this project)

## AI Agent Guidelines

### When Working on This Codebase
1. **Understand the educational purpose**: This is for learning Transformer internals, not production
2. **Keep it minimal**: Avoid adding unnecessary abstractions or dependencies
3. **Follow existing patterns**: Match the simple, direct style of existing code
4. **Add tests for new functionality**: Create `test_*.py` files for new components
5. **Validate NumPy operations**: Check shapes match expected dimensions

### What to Avoid
- Adding complex frameworks (PyTorch, TensorFlow, etc.)
- Over-engineering solutions
- Premature optimization
- Breaking the simplicity of the educational examples

### Recommended Workflow
1. Test existing code: `cd Transformer && python test_*.py`
2. Add new functionality with matching tests
3. Verify imports work: `python -c "import new_module"`
4. Run any affected tests
5. Update documentation if API changes

### Common Tasks for Agents
- **Adding new model components**: Follow model.py patterns
- **Extending tokenizer**: Keep WordCollection simple
- **Adding datasets**: Follow TextDataset pattern
- **Writing tests**: Use existing test_*.py as templates
- **Fixing bugs**: Check shape mismatches, NumPy operations

## Environment
- Python 3.13.5+
- NumPy 2.1.3+
- No other dependencies
- Run from `Transformer/` directory for imports to work