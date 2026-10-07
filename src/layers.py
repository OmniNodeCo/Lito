"""Neural network layers built from scratch."""

import numpy as np
from .tensor import Tensor
from typing import List, Optional


class Layer:
    """Base layer class."""
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        return []

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class Linear(Layer):
    """Fully connected layer."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor.he_normal(in_features, out_features, requires_grad=True)
        self.bias_tensor = None
        if bias:
            self.bias_tensor = Tensor(np.zeros((1, out_features)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias_tensor is not None:
            out = out + self.bias_tensor
        return out

    def parameters(self) -> List[Tensor]:
        params = [self.weight]
        if self.bias_tensor is not None:
            params.append(self.bias_tensor)
        return params


class Embedding(Layer):
    """Embedding lookup table."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim) * 0.02,
            requires_grad=True
        )

    def forward(self, indices) -> Tensor:
        if isinstance(indices, Tensor):
            idx = indices.data.astype(int)
        else:
            idx = np.array(indices, dtype=int)

        embedded = self.weight.data[idx]
        out = Tensor(embedded, requires_grad=True, _children=(self.weight,), _op='embedding')

        def _backward():
            grad = np.zeros_like(self.weight.data)
            np.add.at(grad, idx, out.grad)
            self.weight.grad += grad

        out._backward = _backward
        return out

    def parameters(self) -> List[Tensor]:
        return [self.weight]


class LayerNorm(Layer):
    """Layer normalization."""
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = Tensor(np.ones((1, normalized_shape)), requires_grad=True)
        self.beta = Tensor(np.zeros((1, normalized_shape)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        x_norm_data = (x.data - mean) / np.sqrt(var + self.eps)

        out_data = self.gamma.data * x_norm_data + self.beta.data
        out = Tensor(out_data, requires_grad=True, _children=(x, self.gamma, self.beta), _op='layernorm')

        def _backward():
            N = x.data.shape[-1]
            dout = out.grad

            self.gamma.grad += np.sum(dout * x_norm_data, axis=0, keepdims=True)
            if len(self.gamma.grad.shape) > 2:
                self.gamma.grad = self.gamma.grad.sum(axis=tuple(range(len(self.gamma.grad.shape) - 2)), keepdims=False)
                if len(self.gamma.grad.shape) == 1:
                    self.gamma.grad = self.gamma.grad.reshape(1, -1)
            self.beta.grad += np.sum(dout, axis=0, keepdims=True)
            if len(self.beta.grad.shape) > 2:
                self.beta.grad = self.beta.grad.sum(axis=tuple(range(len(self.beta.grad.shape) - 2)), keepdims=False)
                if len(self.beta.grad.shape) == 1:
                    self.beta.grad = self.beta.grad.reshape(1, -1)

            dx_norm = dout * self.gamma.data
            std_inv = 1.0 / np.sqrt(var + self.eps)
            dx = (1.0 / N) * std_inv * (
                N * dx_norm - np.sum(dx_norm, axis=-1, keepdims=True) -
                x_norm_data * np.sum(dx_norm * x_norm_data, axis=-1, keepdims=True)
            )
            x.grad += dx

        out._backward = _backward
        return out

    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]


class Dropout(Layer):
    """Dropout regularization."""
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float64)
        scale = 1.0 / (1.0 - self.p)
        out = Tensor(x.data * mask * scale, requires_grad=x.requires_grad,
                     _children=(x,), _op='dropout')
        def _backward():
            x.grad += out.grad * mask * scale
        out._backward = _backward
        return out


class MultiHeadAttention(Layer):
    """Multi-head self-attention mechanism."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        batch_size = x.data.shape[0] if len(x.data.shape) == 3 else 1
        seq_len = x.data.shape[-2]

        # Linear projections
        Q = self.W_q(x)  # (batch, seq, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head: (batch, seq, n_heads, d_k) -> (batch, n_heads, seq, d_k)
        def reshape_heads(t: Tensor) -> np.ndarray:
            data = t.data.reshape(batch_size, seq_len, self.n_heads, self.d_k)
            return data.transpose(0, 2, 1, 3)

        q = reshape_heads(Q)
        k = reshape_heads(K)
        v = reshape_heads(V)

        # Attention scores
        scale = np.sqrt(self.d_k)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale  # (batch, n_heads, seq, seq)

        # Causal mask
        if mask is not None:
            scores = scores + mask * (-1e9)
        else:
            # Auto causal mask
            causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            scores = scores + causal_mask * (-1e9)

        # Softmax
        scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)

        # Apply attention to values
        attn_output = attn_weights @ v  # (batch, n_heads, seq, d_k)

        # Reshape back: (batch, seq, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        attn_tensor = Tensor(attn_output, requires_grad=True,
                            _children=(Q, K, V), _op='attention')

        # Store for backward
        _q, _k, _v, _attn_weights = q, k, v, attn_weights

        def _backward():
            dout = attn_tensor.grad.reshape(batch_size, seq_len, self.n_heads, self.d_k)
            dout = dout.transpose(0, 2, 1, 3)

            dv = _attn_weights.transpose(0, 1, 3, 2) @ dout
            dattn = dout @ _v.transpose(0, 1, 3, 2)

            # Softmax backward
            ds = _attn_weights * (dattn - np.sum(dattn * _attn_weights, axis=-1, keepdims=True))
            ds = ds / scale

            dq = ds @ _k
            dk = ds.transpose(0, 1, 3, 2) @ _q

            # Reshape back
            dq = dq.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
            dk = dk.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
            dv = dv.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

            Q.grad += dq
            K.grad += dk
            V.grad += dv

        attn_tensor._backward = _backward

        out = self.W_o(attn_tensor)
        return out

    def parameters(self) -> List[Tensor]:
        params = []
        params.extend(self.W_q.parameters())
        params.extend(self.W_k.parameters())
        params.extend(self.W_v.parameters())
        params.extend(self.W_o.parameters())
        return params


class FeedForward(Layer):
    """Position-wise feed-forward network."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.linear1(x)
        # GELU approximation
        h_data = h.data
        gelu_data = 0.5 * h_data * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (h_data + 0.044715 * h_data ** 3)
        ))
        out = Tensor(gelu_data, requires_grad=True, _children=(h,), _op='gelu')

        def _backward():
            x_val = h.data
            cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_val + 0.044715 * x_val ** 3)))
            pdf = np.exp(-0.5 * x_val ** 2) / np.sqrt(2.0 * np.pi)
            h.grad += out.grad * (cdf + x_val * pdf)
        out._backward = _backward

        out = self.dropout(out)
        out = self.linear2(out)
        return out

    def parameters(self) -> List[Tensor]:
        return self.linear1.parameters() + self.linear2.parameters()


class TransformerBlock(Layer):
    """A single transformer block."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Pre-norm architecture
        normed = self.ln1(x)
        attn_out = self.attention(normed)
        attn_out = self.dropout1(attn_out)

        # Residual connection
        residual1_data = x.data + attn_out.data
        residual1 = Tensor(residual1_data, requires_grad=True,
                          _children=(x, attn_out), _op='residual')
        def _backward1():
            x.grad += residual1.grad
            attn_out.grad += residual1.grad
        residual1._backward = _backward1

        normed2 = self.ln2(residual1)
        ff_out = self.ff(normed2)
        ff_out = self.dropout2(ff_out)

        # Second residual
        residual2_data = residual1.data + ff_out.data
        residual2 = Tensor(residual2_data, requires_grad=True,
                          _children=(residual1, ff_out), _op='residual')
        def _backward2():
            residual1.grad += residual2.grad
            ff_out.grad += residual2.grad
        residual2._backward = _backward2

        return residual2

    def parameters(self) -> List[Tensor]:
        params = []
        params.extend(self.attention.parameters())
        params.extend(self.ff.parameters())
        params.extend(self.ln1.parameters())
        params.extend(self.ln2.parameters())
        return params

    def train(self):
        self.training = True
        self.dropout1.training = True
        self.dropout2.training = True
        self.attention.dropout.training = True
        self.ff.dropout.training = True

    def eval(self):
        self.training = False
        self.dropout1.training = False
        self.dropout2.training = False
        self.attention.dropout.training = False
        self.ff.dropout.training = False