# AI Agent + Local GPT System

## Overview
A hybrid AI assistant system that combines a cloud-based AI agent with a local lightweight Transformer-based GPT model. This architecture enables intelligent query routing, response refinement, and continuous learning from user interactions.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (CLI)                     │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                Query Router & Decision Engine               │
│                     (AI_agent/router.py)                    │
└─────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────────┐   ┌───────────────┐
│  Local GPT      │     │  Cloud AI Agent     │   │   Memory &    │
│  (Transformer)  │     │   (OpenAI GPT-4o)   │   │    Logging    │
├─────────────────┤     ├─────────────────────┤   ├───────────────┤
│• MiniTransformer│     │• API-based queries  │   │• SQLite DB    │
│• NumPy-based    │     │• Complex reasoning  │   │• Response logs│
│• On-device      │     │• Latest knowledge   │   │• User feedback│
└─────────────────┘     └─────────────────────┘   └───────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                Response Aggregation & Learning              │
│              (RAG + Feedback loop integration)              │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. **Local Transformer Model** (`Transformer/`)
- **Tokenizer** (`tokenizer.py`): Word-level tokenization with vocabulary management
- **Dataset** (`dataset.py`): Text preprocessing and sequence generation
- **Model** (`model.py`): MiniTransformer implementation with attention mechanisms
- **Training** (`train.py`): Gradient descent optimization loop
- **Generation** (`generate.py`): Text generation with sampling strategies
- **Loss** (`loss.py`): Cross-entropy loss implementation

### 2. **AI Agent System** (`AI_agent/`)
- **Agent** (`agent.py`): Main orchestration logic and cloud API integration
- **CLI** (`cli.py`): Command-line interface for user interaction
- **Router** (`router.py`): Intelligent query routing decisions
- **RAG** (`rag.py`): Retrieval-augmented generation from stored logs
- **Memory** (`memory.py`): SQLite-based logging and feedback storage

### 3. **Configuration** (`config/`)
- **API Configuration**: OpenAI API key management
- **Environment Variables**: Project settings and thresholds

## Key Features

### Intelligent Query Routing
- **Dynamic Decision Making**: Routes queries based on complexity, available training data, and query type
- **Fallback Strategies**: Graceful degradation when cloud API is unavailable
- **Threshold-based Routing**: Uses `MIN_TRAIN_DATA` threshold (1000 logs) to determine when to use local vs cloud
- **Local Prompt Optimization**: When conditions are met (query < 20 characters AND ≥ 1000 training logs), the local GPT generates optimized prompts for cloud processing, not direct answers

### Response Enhancement
- **Prompt Refinement**: Local GPT transforms vague queries into precise prompts
- **Context Preservation**: Maintains conversation history across interactions
- **Quality Filtering**: Filters and improves cloud responses using local intelligence

### Learning & Adaptation
- **Feedback Loop**: Stores user satisfaction scores for continuous improvement
- **RAG Enhancement**: Uses past high-quality responses for context in new queries
- **Progressive Improvement**: System becomes more accurate as training data accumulates

### Performance Optimization
- **Local Processing**: Simple queries handled entirely on-device for speed
- **Cloud Leverage**: Complex reasoning delegated to powerful cloud models
- **Caching**: Frequently used responses retrieved from local memory

## Getting Started

### Prerequisites
```bash
Python 3.13.5+
NumPy 2.1.3+
OpenAI Python SDK
SQLite3 (built-in)
```

### Installation
```bash
# Clone repository
git clone <repository-url>
cd AI_Summary_Prompts

# Set up environment variables
echo "OPENAI_API_KEY=your_api_key_here" > config/.env

# Verify installation
cd Transformer && python test_tokenizer.py
```

### Usage Examples

#### Basic CLI Interaction
```bash
python main.py
# or
python -m AI_agent.cli
```

#### Testing Components
```bash
# Run all tests
cd Transformer && python -m pytest

# Test specific components
python -c "from AI_agent.agent import cloud_agent; print('Agent OK')"
python -c "from Transformer.tokenizer import WordCollection; tc = WordCollection(); print('Tokenizer OK')"
```

#### Development Commands
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy . --strict

# Run full test suite
cd Transformer && python -m pytest && cd .. && python -m pytest AI_agent/
```

## Configuration

### Environment Variables
Create `config/.env` with:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Model Parameters (Transformer/config.py)
```python
VOCAB_SIZE = 10000      # Vocabulary size
D_MODEL = 32            # Embedding dimension
NUM_HEADS = 4           # Attention heads
SEQ_LEN = 10            # Maximum sequence length
D_FF = 64               # Feed-forward dimension
```

## Workflow Details

### 1. **Query Processing Pipeline**
```
User Query → Tokenization → Routing Decision → 
├─ Has high-quality history (rating ≥ 4) → RAG+Cloud (retrieve from memory)
├─ Short query (<20 chars) AND sufficient training data (≥1000 logs) → 
│  Local GPT prompt optimization → Cloud Agent → "LocalPrompt+Cloud" response
└─ Default → Direct Cloud processing
```

### 2. **Learning Cycle**
```
Response Generated → User Feedback Collected → Log Stored → 
Training Data Updated → Router Decisions Improved → Better Future Responses
```

### 3. **Quality Assurance**
- **Response Validation**: Checks for completeness and relevance
- **Fallback Mechanisms**: Local generation if cloud fails
- **Error Handling**: Graceful degradation for missing dependencies

## Project Structure
```
AI_Summary_Prompts/
├── Transformer/          # Local GPT implementation
│   ├── config.py        # Model hyperparameters
│   ├── tokenizer.py     # Word tokenization
│   ├── dataset.py       # Text dataset creation
│   ├── model.py         # Transformer architecture
│   ├── train.py         # Training loop
│   ├── generate.py      # Text generation
│   ├── loss.py          # Loss functions
│   └── test_*.py        # Unit tests
├── AI_agent/            # AI agent system
│   ├── agent.py         # Main agent logic
│   ├── cli.py           # Command-line interface
│   ├── memory.py        # Memory/logging system
│   ├── router.py        # Query routing
│   ├── rag.py           # Retrieval-augmented generation
│   └── __init__.py
├── config/              # Configuration
│   ├── config.py        # Project settings
│   └── .env             # Environment variables
├── Data/                # Generated data
│   └── logs.db          # SQLite database
├── main.py              # Entry point
└── README.md            # This file
```

## Key Technical Insights

### How the System Actually Works

**Important Discovery**: The local GPT model **does not directly answer questions** - it only optimizes prompts for the cloud AI.

**Routing Logic Details**:
- **Initial State**: System starts with 0 training logs → always uses cloud (`MIN_TRAIN_DATA = 1000`)
- **Local Usage Condition**: Query length < 20 characters **AND** training logs ≥ 1000
- **Local Function**: Generates short, optimized prompts (5 tokens max) to guide cloud responses
- **Response Types**: 
  - `RAG+Cloud`: Uses retrieved high-quality history (rating ≥ 4)
  - `LocalPrompt+Cloud`: Uses local GPT-optimized prompts
  - `CloudFallback`: Direct cloud processing (fallback)
  - `CLOUD`: Default cloud processing

**Progressive Learning**:
1. User asks question → cloud answers → user rates response (1-5)
2. High-rated responses (≥4) stored in SQLite database
3. When training logs reach 1000, system activates local prompt optimization
4. Future short queries trigger `LocalPrompt+Cloud` flow for enhanced responses

## Benefits & Advantages

### Technical Advantages
1. **Reduced Latency**: Local processing for simple queries
2. **Cost Efficiency**: Minimizes API calls for trivial requests
3. **Privacy Preservation**: Sensitive queries can stay local
4. **Offline Capability**: Basic functionality without internet
5. **Smart Routing**: Intelligent decision between local optimization vs direct cloud processing

### User Experience Benefits
1. **Faster Responses**: Immediate answers for common questions
2. **Higher Quality**: Professionally refined outputs
3. **Personalization**: Adapts to user preferences over time
4. **Consistency**: Maintains tone and style across interactions

### Development Benefits
1. **Educational Value**: Learn Transformer internals hands-on
2. **Modular Design**: Easy to extend or replace components
3. **Minimal Dependencies**: Pure Python/NumPy core
4. **Test Coverage**: Comprehensive unit tests

## Roadmap & Future Enhancements

### Planned Features
- [ ] Multi-modal support (text + code generation)
- [ ] Fine-tuning capabilities for domain specialization
- [ ] Advanced caching strategies
- [ ] Web interface in addition to CLI
- [ ] Plugin system for extensibility

### Research Directions
- Improving local model quality while maintaining efficiency
- Advanced routing algorithms using reinforcement learning
- Federated learning for privacy-preserving improvement
- Cross-platform deployment (mobile, web, desktop)

## Contributing

See `AGENTS.md` for detailed coding guidelines and workflow instructions for AI agents working on this project.

## License
MIT License - See LICENSE file for details.
