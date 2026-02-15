import numpy as np

class Embedding:
    def __init__(self, vocab_size, d_model, seq_len, mean=0, std=0.1):
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        # initializing vector
        self.embeddings = mean + std * np.random.randn(vocab_size, d_model)

        # initializing pos codes
        self.positional_encoding = self._generate_positional_encoding(seq_len, d_model)

    def _generate_positional_encoding(self, seq_len, d_model):
        PE = np.zeros((seq_len, d_model))
        position = np.arange(seq_len).reshape(seq_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        PE[:, 0::2] = np.sin(position * div_term)
        PE[:, 1::2] = np.cos(position * div_term)
        return PE

    def get_embeddings(self, token_ids):
        x_emb = self.embeddings[token_ids]  # [seq_len, d_model]
        x_emb += self.positional_encoding[:len(token_ids), :]  # adding pos codes
        return x_emb

# Mixing    → Attention
# Transform → FFN

# Normalization
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class MultiHeadSelfAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # random weights
        self.Wq = np.random.randn(d_model, d_model)
        self.Wk = np.random.randn(d_model, d_model)
        self.Wv = np.random.randn(d_model, d_model)
        self.Wo = np.random.randn(d_model, d_model)

    def split_heads(self, X):
        # [seq_len, d_model] → [num_heads, seq_len, d_k]
        seq_len = X.shape[0]
        X = X.reshape(seq_len, self.num_heads, self.d_k)
        return X.transpose(1, 0, 2)

    def combine_heads(self, X):
        # [num_heads, seq_len, d_k] → [seq_len, d_model]
        X = X.transpose(1, 0, 2)
        seq_len = X.shape[0]
        return X.reshape(seq_len, self.d_model)

    def forward(self, X):
        Q = X @ self.Wq # Query: the information that token wants (filter)
        K = X @ self.Wk # Key: features of each token (keys)
        V = X @ self.Wv # Information that each token contains
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        scores = Q @ K.transpose(0, 2, 1) # How much information is available at the current location
        scores = scores / np.sqrt(self.d_k) # Used to control gradient explosion
        
        attention = softmax(scores)
        
        # The current word is defined using the vectors of other words in the semantic space
        output = attention @ V
        
        output = self.combine_heads(output)
        
        return output @ self.Wo
    
class FeedForward:
    
    def __init__(self, d_model, d_ff):
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # first layer
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        
        # second layer
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros(d_model)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, X):
        """
        Input:
            X shape = [seq_len, d_model]
        
        Output:
            [seq_len, d_model]
        """
        
        hidden = X @ self.W1 + self.b1
        
        # Relu
        hidden = self.relu(hidden)
        
        output = hidden @ self.W2 + self.b2
        
        return output
    
class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps

        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, X):
        """
        X: [seq_len, d_model]
        Normalize each token
        """
        
        mean = np.mean(X, axis=1, keepdims=True)
        
        var = np.var(X, axis=1, keepdims=True)
        
        X_norm = (X - mean) / np.sqrt(var + self.eps)
        
        output = self.gamma * X_norm + self.beta
        
        return output
    
# x = x + f(x)
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, X):
        
        # Attention
        attn_out = self.attention.forward(self.ln1.forward(X))
        X = X + attn_out
        
        # FFN
        ffn_out = self.ffn.forward(self.ln2.forward(X))
        X = X + ffn_out
        
        return X
    
class OutputLayer:
    def __init__(self, d_model, vocab_size):
        self.W = np.random.randn(d_model, vocab_size)

    def forward(self, X):
        """
        X: [seq_len, d_model]
        Output: [seq_len, vocab_size]
        """
        return X @ self.W