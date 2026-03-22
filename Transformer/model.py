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
    
    def backward(self, dX, lr=1e-3):
        """
        dX: [seq_len, d_model]
        """

        np.add.at(self.embeddings, self.token_ids, -lr * dX)

        return None
    

# -----------------------
# utils
# -----------------------

# Mixing    → Attention
# Transform → FFN
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def softmax_backward(dout, out):
    # out = softmax(x)
    return out * (dout - np.sum(dout * out, axis=-1, keepdims=True))


# -----------------------
# Multi-Head Attention
# -----------------------

class MultiHeadSelfAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = np.random.randn(d_model, d_model) * 0.02
        self.Wk = np.random.randn(d_model, d_model) * 0.02
        self.Wv = np.random.randn(d_model, d_model) * 0.02
        self.Wo = np.random.randn(d_model, d_model) * 0.02

    def split_heads(self, X):
        S = X.shape[0]
        X = X.reshape(S, self.num_heads, self.d_k)
        return X.transpose(1, 0, 2)  # (H, S, d_k)

    def combine_heads(self, X):
        X = X.transpose(1, 0, 2)  # (S, H, d_k)
        S = X.shape[0]
        return X.reshape(S, self.d_model)
    
    def causal_mask(self, S):
        mask = np.triu(np.ones((S, S)), k=1) * -1e9
        return mask

    def forward(self, X):
        self.X = X

        self.Q = X @ self.Wq
        self.K = X @ self.Wk
        self.V = X @ self.Wv

        self.Qh = self.split_heads(self.Q)
        self.Kh = self.split_heads(self.K)
        self.Vh = self.split_heads(self.V)

        self.scores = self.Qh @ self.Kh.transpose(0, 2, 1)
        self.scores /= np.sqrt(self.d_k)

        S = self.scores.shape[-1]
        self.scores += self.causal_mask(S)

        self.attn = softmax(self.scores)

        self.Zh = self.attn @ self.Vh
        self.Z = self.combine_heads(self.Zh)

        self.out = self.Z @ self.Wo
        return self.out

    def backward(self, dout, lr=1e-3):
        # -------- Wo --------
        dWo = self.Z.T @ dout
        dZ = dout @ self.Wo.T

        # -------- split heads --------
        dZh = self.split_heads(dZ)

        # -------- Z = attn @ V --------
        dAttn = dZh @ self.Vh.transpose(0, 2, 1)
        dVh = self.attn.transpose(0, 2, 1) @ dZh

        # -------- softmax --------
        dScores = softmax_backward(dAttn, self.attn)

        # -------- scaling --------
        dScores /= np.sqrt(self.d_k)

        # -------- scores = QK^T --------
        dQh = dScores @ self.Kh
        dKh = dScores.transpose(0, 2, 1) @ self.Qh

        # -------- combine heads --------
        dQ = self.combine_heads(dQh)
        dK = self.combine_heads(dKh)
        dV = self.combine_heads(dVh)

        # -------- Q = XW --------
        dWq = self.X.T @ dQ
        dWk = self.X.T @ dK
        dWv = self.X.T @ dV

        dXq = dQ @ self.Wq.T
        dXk = dK @ self.Wk.T
        dXv = dV @ self.Wv.T

        dX = dXq + dXk + dXv

        # -------- update --------
        self.Wq -= lr * dWq
        self.Wk -= lr * dWk
        self.Wv -= lr * dWv
        self.Wo -= lr * dWo

        return dX


# -----------------------
# Feed Forward
# -----------------------

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)

        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_backward(self, grad, x):
        return grad * (x > 0)

    def forward(self, X):
        self.X = X
        self.h1 = X @ self.W1 + self.b1
        self.h2 = self.relu(self.h1)
        self.out = self.h2 @ self.W2 + self.b2
        return self.out

    def backward(self, dout, lr=1e-3):
        dW2 = self.h2.T @ dout
        db2 = np.sum(dout, axis=0)

        dh = dout @ self.W2.T
        dh = self.relu_backward(dh, self.h1)

        dW1 = self.X.T @ dh
        db1 = np.sum(dh, axis=0)

        dX = dh @ self.W1.T

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        return dX


# -----------------------
# LayerNorm
# -----------------------

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, X):
        self.X = X
        self.mean = X.mean(axis=1, keepdims=True)
        self.var = X.var(axis=1, keepdims=True)

        self.X_norm = (X - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.X_norm + self.beta

    def backward(self, dout, lr=1e-3):
        dgamma = np.sum(dout * self.X_norm, axis=0)
        dbeta = np.sum(dout, axis=0)

        dX_norm = dout * self.gamma
        var_eps = self.var + self.eps

        dX = (1. / np.sqrt(var_eps)) * (
            dX_norm
            - np.mean(dX_norm, axis=1, keepdims=True)
            - self.X_norm * np.mean(dX_norm * self.X_norm, axis=1, keepdims=True)
        )

        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta

        return dX


# -----------------------
# Transformer Block
# -----------------------

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)

        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, X):
        self.X = X

        self.ln1_out = self.ln1.forward(X)
        self.attn_out = self.attn.forward(self.ln1_out)
        self.res1 = X + self.attn_out

        self.ln2_out = self.ln2.forward(self.res1)
        self.ffn_out = self.ffn.forward(self.ln2_out)
        self.out = self.res1 + self.ffn_out

        return self.out

    def backward(self, dout, lr=1e-3):
        # ----- FFN residual -----
        d_res1 = dout.copy()
        d_ffn = dout.copy()

        d_ffn = self.ffn.backward(d_ffn, lr)
        d_ln2 = self.ln2.backward(d_ffn, lr)

        d_res1 += d_ln2

        # ----- Attention residual -----
        d_attn = d_res1.copy()
        d_attn = self.attn.backward(d_attn, lr)
        d_ln1 = self.ln1.backward(d_attn, lr)

        dX = d_res1 + d_ln1

        return dX


# -----------------------
# Output Layer
# -----------------------

class OutputLayer:
    def __init__(self, d_model, vocab_size):
        self.W = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, X):
        self.X = X
        return X @ self.W

    def backward(self, dlogits, lr=1e-3):
        dW = self.X.T @ dlogits
        dX = dlogits @ self.W.T

        self.W -= lr * dW
        return dX
    

# -----------------------
# Mini Transformer
# -----------------------

class MiniTransformer:
    def __init__(self, vocab_size, d_model=32, num_heads=4, d_ff=64, seq_len=4):
        self.embedding = Embedding(vocab_size, d_model, seq_len)
        self.block = TransformerBlock(d_model, num_heads, d_ff)
        self.output = OutputLayer(d_model, vocab_size)

    def forward(self, x_ids):
        """
        x_ids: [seq_len]
        """
        x = self.embedding.get_embeddings(x_ids)
        x = self.block.forward(x)
        logits = self.output.forward(x)
        return logits

    def backward(self, dlogits, lr):
        dX = self.output.backward(dlogits, lr)
        dX = self.block.backward(dX, lr)
        self.embedding.backward(dX, lr)
