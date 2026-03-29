# AI Agent + Local GPT System

## Overview

A hybrid AI assistant that combines semantic similarity-based routing with a local Transformer model and cloud AI. The system analyzes queries, finds similar past responses, estimates answer complexity, and generates or decomposes responses accordingly.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface (CLI)                   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│              generate_response() - Similarity Module        │
│                    (Transformer/similarity.py)              │
└─────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌───────────────┐
│  SQLite DB    │     │  Local Model    │     │  Cloud AI     │
│  (Memory)     │     │  (Transformer)  │     │  (OpenAI)     │
├───────────────┤     ├─────────────────┤     ├───────────────┤
│• Query/Answer │     │• Embedding      │     │• Complex      │
│  storage      │     │  computation    │     │  reasoning    │
│• Similarity   │     │• Semantic       │     │• Long-form    │
│  matching     │     │  similarity     │     │  responses    │
└───────────────┘     └─────────────────┘     └───────────────┘
```

## Query Processing Flowchart

```
                    ┌──────────────────┐
                    │   User Query     │
                    └────────┬─────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ DB Records ≥ 100?            │
              │ (MIN_SIMILARITY_DATA)        │
              └──────────────┬───────────────┘
                    ┌────────┴────────┐
                    │                 │
                   YES                NO
                    │                 │
                    ▼                 ▼
        ┌───────────────────┐  ┌──────────────────┐
        │ Compute semantic  │  │ cloud_agent()    │
        │ similarity with   │  │ Direct answer    │
        │ all DB records    │  │ source:          │
        └─────────┬─────────┘  │ "cloud_direct"   │
                  │            └──────────────────┘
                  ▼
    ┌─────────────────────────────┐
    │ Max Similarity ≥ 0.90?      │
    │ (SIMILARITY_THRESHOLD)      │
    └─────────────┬───────────────┘
          ┌───────┴───────┐
          │               │
         YES              NO
          │               │
          ▼               ▼
  ┌──────────────┐  ┌─────────────────────────────┐
  │ Return DB    │  │ Weighted avg of similar     │
  │ answer       │  │ responses → estimate        │
  │ directly     │  │ answer length               │
  │ source:      │  └─────────────┬───────────────┘
  │ "db_direct"  │                │
  └──────────────┘                ▼
                   ┌─────────────────────────────┐
                   │ Estimated Length > 200?     │
                   │ (MAX_RESPONSE_LENGTH)       │
                   └─────────────┬───────────────┘
                         ┌───────┴───────┐
                         │               │
                        YES              NO
                         │               │
                         ▼               ▼
           ┌─────────────────────┐  ┌──────────────────┐
           │ decompose_question()│  │ cloud_agent()    │
           │ Split into 3-8      │  │ Direct answer    │
           │ sub-questions       │  │ source:          │
           └──────────┬──────────┘  │ "cloud_direct"   │
                      │             └──────────────────┘
                      ▼
          ┌─────────────────────────────┐
          │ Recursively answer each     │
          │ sub-question (depth + 1)    │
          │                             │
          │ Each sub-question goes      │
          │ through the same flowchart  │
          └──────────────┬──────────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
      depth < 10               depth ≥ 10
           │                           │
           ▼                           ▼
  ┌───────────────────┐    ┌────────────────────────┐
  │ _combine_answers()│    │ "Max depth reached"    │
  │ Merge into        │    │ Return error message   │
  │ coherent article  │    │ source:                │
  │ source:           │    │ "max_depth_reached"    │
  │ "decomposed"      │    └────────────────────────┘
  └───────────────────┘
```

## Response Sources

| Source | Condition | Description |
|--------|-----------|-------------|
| `db_direct` | Similarity ≥ 0.90 | Direct DB lookup |
| `cloud_direct` | Insufficient data OR estimated length ≤ 200 | Direct AI response |
| `decomposed` | Estimated length > 200 AND successfully decomposed | Recursive sub-answers combined |
| `max_depth_reached` | Recursion depth ≥ 10 | Decomposition stopped |

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_SIMILARITY_DATA` | 100 | Min DB records before using similarity |
| `SIMILARITY_THRESHOLD` | 0.90 | Min similarity for direct DB response |
| `MAX_RESPONSE_LENGTH` | 200 | Max answer length before decomposition |
| `MAX_DECOMPOSE_DEPTH` | 10 | Max recursion depth |
| `MAX_DB_RECORDS` | 5000 | Max DB records to load (memory protection) |

## Core Components

### 1. Transformer Module (`Transformer/`)
- **tokenizer.py**: Word-level tokenization (WordCollection)
- **dataset.py**: Text dataset for training (TextDataset)
- **model.py**: MiniTransformer with multi-head attention
- **similarity.py**: Semantic similarity and response generation
- **train.py**: Training loop with gradient descent
- **loss.py**: Cross-entropy loss implementation

### 2. AI Agent Module (`AI_agent/`)
- **agent.py**: Main entry point, calls `generate_response()`
- **cli.py**: Command-line interface
- **memory.py**: SQLite logging and retrieval
- **router.py**: Query routing decisions
- **rag.py**: Retrieval-augmented generation

### 3. Configuration (`config/`)
- **config.py**: All threshold and parameter settings
- **.env**: Environment variables (OPENAI_API_KEY)

## Getting Started

### Prerequisites
```bash
Python 3.13.5+
NumPy 2.1.3+
OpenAI SDK
pytest
```

### Installation
```bash
# Clone and setup
cd AI_Summary_Prompts
echo "OPENAI_API_KEY=your_key" > config/.env

# Run tests
cd Transformer && python test_similarity.py
```

### Usage
```bash
python main.py
```

## Project Structure

```
AI_Summary_Prompts/
├── Transformer/
│   ├── config.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── model.py
│   ├── similarity.py      # Query routing logic
│   ├── loss.py
│   ├── train.py
│   ├── generate.py
│   └── test_*.py
├── AI_agent/
│   ├── agent.py           # Entry point
│   ├── cli.py
│   ├── memory.py
│   ├── router.py
│   └── rag.py
├── config/
│   ├── config.py
│   └── .env
├── Data/
├── Performance_Tests/
│   ├── test_performance.py
│   ├── benchmark.py
│   └── README.md
├── main.py
├── AGENTS.md
└── README.md
```

## How It Works

1. **User Feedback Loop**
   ```
   User Query → AI Response → User Rating (1-5) → SQLite Storage
                                         ↓
                              High-rated (≥4) used for similarity matching
   ```

2. **Similarity-Based Routing**
   - Find semantically similar past queries using Transformer embeddings
   - If very similar (≥0.90), return the stored answer directly
   - Otherwise, estimate answer length from similar queries

3. **Intelligent Decomposition**
   - Complex questions (>200 words estimated) are split into sub-questions
   - Each sub-question is answered recursively
   - All sub-answers are combined into a coherent response

4. **Progressive Learning**
   - System becomes smarter as more high-rated responses accumulate
   - With <100 records: always uses cloud directly
   - With ≥100 records: uses similarity-based routing

## Testing

```bash
# Run all Transformer tests
cd Transformer && python -m pytest

# Run specific test file
python -m pytest Transformer/test_similarity.py

# Run with direct execution
cd Transformer && python test_similarity.py

# Run performance tests
cd Performance_Tests && python test_performance.py
cd Performance_Tests && python benchmark.py
```

## License

MIT License
