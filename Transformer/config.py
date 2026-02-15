# config.py
config = {
    "vocab_size": 10000,       # Vocabulary size, total number of unique tokens
    "d_model": 512,            # Embedding size / hidden dimension of the model
    "num_heads": 8,            # Number of attention heads in multi-head attention
    "d_ff": 2048,              # Dimension of the feed-forward network
    "num_layers": 6,           # Number of Transformer layers (Encoder/Decoder layers)
    "max_seq_len": 128,        # Maximum sequence length the model can handle
    "dropout": 0.1,            # Dropout probability for general layers
    "pad_token_id": 0,         # ID of the padding token
    "bos_token_id": 1,         # ID of the beginning-of-sequence token
    "eos_token_id": 2          # ID of the end-of-sequence token
}