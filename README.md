# Mini LLM From Scratch – Transformer-Based Prompt Generator

## Project Overview
This project is a minimal Large Language Model (LLM) built from scratch using the Transformer architecture.
The goal is not to build a powerful production model, but to deeply understand how modern LLMs work internally.
The model is trained to:
`Convert natural user input into structured AI prompt instructions.`

- *Example:*
  - `I feel stressed and cannot sleep at night.`
- *Output:*
  - `Generate a supportive response that comforts a stressed user and provides sleep improvement advice.`

---

## Learning Objectives
- How language models predict the next token
- How self-attention works mathematically
- Why masking is required in causal language models
- How embeddings represent words in vector space
- How Transformer blocks are structured
- How conditional text generation works
- Why LLMs are fundamentally probability models

---

## Core Idea
The model learns the conditional probability:

$$
P(\text{output} \mid \text{input})
$$

Which expands into next-token prediction:

$$
P(y_t \mid x, y_{1:t-1})
$$

---

## Architecture
- Character-level tokenizer (for simplicity)
- Embedding layer
- Positional encoding
- Masked self-attention
- Multi-head attention
- Feed-forward network
- Layer normalization
- Residual connections
- Linear output projection
- Cross-entropy loss
