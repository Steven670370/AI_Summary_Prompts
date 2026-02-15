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
        self.token_ids = token_ids  # save token indices for backward
        self.x_emb = x_emb  # save forward output for backward
        return x_emb
    
    def backward(self, dX, token_ids, lr=1e-3):
        """
        dX: [seq_len, d_model] gradients
        token_ids: index of token
        lr: learning rate
        """
        self.embeddings[token_ids] -= lr * dX
        return None  # no gradient needs to pass back

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

        # save inputs for backward
        self.X = X
        self.Q = Q
        self.K = K
        self.V = V

        self.Q_split = self.split_heads(Q)
        self.K_split = self.split_heads(K)
        self.V_split = self.split_heads(V)

        scores = self.Q_split @ self.K_split.transpose(0, 2, 1)
        scores = scores / np.sqrt(self.d_k)

        self.attention = softmax(scores)  # save attention matrix
        self.attention_out = self.attention @ self.V_split
        self.output = self.combine_heads(self.attention_out) @ self.Wo

        return self.output
    
    def backward(self, d_output, lr=1e-3):
        # simplified backward (ignoring softmax internal gradient)
        # d_output: [seq_len, d_model]
        dWo = self.combine_heads(self.attention_out).T @ d_output
        d_attention_out = d_output @ self.Wo.T

        dXq = d_attention_out @ self.Wq.T
        dXk = d_attention_out @ self.Wk.T
        dXv = d_attention_out @ self.Wv.T
        dX = dXq + dXk + dXv

        self.Wq -= lr * (self.X.T @ dXq)
        self.Wk -= lr * (self.X.T @ dXk)
        self.Wv -= lr * (self.X.T @ dXv)
        self.Wo -= lr * dWo

        return dX

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

    def relu_backward(self, grad, x):
        return grad * (x > 0)

    def forward(self, X):
        """
        Input:
            X shape = [seq_len, d_model]
        
        Output:
            [seq_len, d_model]
        """
        self.X = X  # save for backward
        self.hidden_linear = X @ self.W1 + self.b1  # save pre-activation
        self.hidden = self.relu(self.hidden_linear)  # save post-activation
        self.output = self.hidden @ self.W2 + self.b2  # save output
        return self.output
    
    def backward(self, d_output, lr=1e-3):
        dW2 = self.hidden.T @ d_output
        db2 = np.sum(d_output, axis=0)

        d_hidden = d_output @ self.W2.T
        d_hidden = self.relu_backward(d_hidden, self.hidden_linear)

        dW1 = self.X.T @ d_hidden
        db1 = np.sum(d_hidden, axis=0)

        dX = d_hidden @ self.W1.T

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        return dX

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
        self.X = X  # save input
        self.mean = np.mean(X, axis=1, keepdims=True)
        self.var = np.var(X, axis=1, keepdims=True)
        self.X_norm = (X - self.mean) / np.sqrt(self.var + self.eps)  # save normalized X
        self.output = self.gamma * self.X_norm + self.beta
        return self.output
    
    def backward(self, d_output, lr=1e-3):
        seq_len = self.X.shape[0]
        dgamma = np.sum(d_output * self.X_norm, axis=0)
        dbeta = np.sum(d_output, axis=0)

        dX_norm = d_output * self.gamma
        var_eps = self.var + self.eps
        dX = (1. / np.sqrt(var_eps)) * (dX_norm - np.mean(dX_norm, axis=1, keepdims=True)
                                       - self.X_norm * np.mean(dX_norm * self.X_norm, axis=1, keepdims=True))

        # update parameters
        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta

        return dX

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
        self.X1 = X  # save for backward
        X = X + attn_out

        # FFN
        ffn_out = self.ffn.forward(self.ln2.forward(X))
        self.X2 = X  # save for backward
        X = X + ffn_out

        return X
    
    def backward(self, dX, lr=1e-3):
        dX_ffn = self.ffn.backward(dX, lr)  # FFN backward
        dX_ln2 = self.ln2.backward(dX_ffn, lr)

        dX_attn = self.attention.backward(dX_ln2, lr)
        dX_ln1 = self.ln1.backward(dX_attn, lr)

        dX_total = dX_ln1 + dX
        return dX_total

class OutputLayer:
    def __init__(self, d_model, vocab_size):
        self.W = np.random.randn(d_model, vocab_size)

    def forward(self, X):
        """
        X: [seq_len, d_model]
        Output: [seq_len, vocab_size]
        """
        self.X = X  # save input
        logits = X @ self.W
        return logits
    
    def backward(self, d_logits, lr=1e-3):
        """
        d_logits: [seq_len, vocab_size]
        self.X: [seq_len, d_model]
        """
        dW = self.X.T @ d_logits
        dX = d_logits @ self.W.T

        self.W -= lr * dW

        return dX
