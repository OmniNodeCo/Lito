"""The main transformer language model."""

import numpy as np
import json
import os
from typing import List, Optional
from .tensor import Tensor
from .layers import (
    Layer, Linear, Embedding, LayerNorm,
    TransformerBlock, Dropout
)


class TransformerLM(Layer):
    """A GPT-style transformer language model built entirely from scratch."""
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, d_ff: int = 512, max_seq_len: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        self.token_embedding = Embedding(vocab_size, d_model)
        self.position_embedding = Embedding(max_seq_len, d_model)

        self.blocks = []
        for _ in range(n_layers):
            self.blocks.append(TransformerBlock(d_model, n_heads, d_ff, dropout))

        self.ln_f = LayerNorm(d_model)
        self.output_proj = Linear(d_model, vocab_size, bias=False)
        self.dropout = Dropout(dropout)

        self._param_count = sum(p.data.size for p in self.parameters())
        print(f"Model initialized with {self._param_count:,} parameters")

    def forward(self, input_ids: np.ndarray) -> Tensor:
        if len(input_ids.shape) == 1:
            input_ids = input_ids.reshape(1, -1)

        batch_size, seq_len = input_ids.shape
        positions = np.arange(seq_len).reshape(1, -1)

        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)

        h = tok_emb + pos_emb
        h = self.dropout(h)

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        logits = self.output_proj(h)
        return logits

    def parameters(self) -> List[Tensor]:
        params = []
        params.extend(self.token_embedding.parameters())
        params.extend(self.position_embedding.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.ln_f.parameters())
        params.extend(self.output_proj.parameters())
        return params

    def train_mode(self):
        self.training = True
        self.dropout.training = True
        for block in self.blocks:
            block.train()

    def eval_mode(self):
        self.training = False
        self.dropout.training = False
        for block in self.blocks:
            block.eval()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        params = self.parameters()
        config = {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
        }
        np.savez_compressed(
            path,
            **{f'w{i}': p.data for i, p in enumerate(params)},
            config=json.dumps(config)
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        loaded = np.load(path, allow_pickle=True)
        params = self.parameters()
        for i, p in enumerate(params):
            key = f'w{i}'
            if key in loaded:
                p.data = loaded[key].astype(np.float64)
        print(f"Model loaded from {path}")

    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> np.ndarray:
        self.eval_mode()

        if len(input_ids.shape) == 1:
            input_ids = input_ids.reshape(1, -1)

        generated = input_ids.copy()

        for _ in range(max_new_tokens):
            context = generated[:, -self.max_seq_len:]
            logits = self.forward(context)
            next_logits = logits.data[0, -1, :]
            next_logits = next_logits / max(temperature, 1e-8)

            if top_k > 0:
                top_k_idx = np.argsort(next_logits)[-top_k:]
                mask = np.full_like(next_logits, -1e10)
                mask[top_k_idx] = next_logits[top_k_idx]
                next_logits = mask

            shifted = next_logits - np.max(next_logits)
            probs = np.exp(shifted)
            probs = probs / (np.sum(probs) + 1e-12)

            if top_p < 1.0:
                sorted_idx = np.argsort(probs)[::-1]
                cumulative = np.cumsum(probs[sorted_idx])
                cutoff_idx = np.searchsorted(cumulative, top_p) + 1
                keep_idx = sorted_idx[:cutoff_idx]
                filtered_probs = np.zeros_like(probs)
                filtered_probs[keep_idx] = probs[keep_idx]
                probs = filtered_probs / (np.sum(filtered_probs) + 1e-12)

            probs = np.clip(probs, 0, None)
            probs = probs / (np.sum(probs) + 1e-12)
            next_token = np.random.choice(len(probs), p=probs)

            generated = np.concatenate([generated, np.array([[next_token]])], axis=1)

        return generated[0]