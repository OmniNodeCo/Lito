"""
A from-scratch numpy inference engine for GGUF llama-architecture models.

This is what runs Lito's chat: it loads a `lito-chat.gguf` file (written
by export_gguf.py, trainable with train_big.py - or any small GGUF of
the same family) and runs the transformer forward pass in pure numpy,
with a KV cache, sampling and a ChatML chat loop. No ML frameworks, no
pretrained weights, no external processes - the same "everything from
scratch" rule the rest of Lito follows.

The forward pass mirrors llama.cpp's "llama" architecture exactly:
RMSNorm (pre-norm), RoPE on pairs of consecutive head values
(GGML_ROPE_TYPE_NORMAL), SwiGLU FFN, multi-head attention and tied
input/output embeddings - the conventions train_big.py trains with.
"""

import math

import numpy as np

from .bpe import ByteBPE
from .gguf import read_gguf


def _rms_norm(x, weight, eps):
    # x: (T, d) - matches llama.cpp rms_norm (f32 math)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * weight


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class LlamaEngine:
    """GGUF llama-arch model + GPT-2 byte-level BPE, inference only."""

    def __init__(self, path, verbose=False):
        meta, tensors = read_gguf(path)
        self.path = path
        self.name = meta.get('general.name', 'model')
        arch = meta.get('general.architecture')
        if arch and arch != 'llama':
            raise ValueError(f'{path}: architecture {arch!r} is not "llama"')

        # ---- architecture (llama.* metadata) --------------------------
        self.n_ctx = int(meta.get('llama.context_length', 512))
        self.d_model = int(meta.get('llama.embedding_length', 256))
        self.n_layers = int(meta.get('llama.block_count', 6))
        self.n_heads = int(meta.get('llama.attention.head_count', 8))
        self.n_kv_heads = int(meta.get('llama.attention.head_count_kv',
                                       self.n_heads))
        self.head_dim = self.d_model // self.n_heads
        self.eps = float(meta.get('llama.attention.layer_norm_rms_epsilon',
                                  1e-5))
        self.rope_base = float(meta.get('llama.rope.freq_base', 10000.0))

        self.emb = tensors['token_embd.weight']        # (vocab, d)
        self.out_w = tensors.get('output.weight', self.emb)
        self.out_norm = tensors['output_norm.weight']
        self.blocks = []
        for i in range(self.n_layers):
            p = f'blk.{i}.'
            self.blocks.append({
                'attn_norm': tensors[p + 'attn_norm.weight'],
                'wq': tensors[p + 'attn_q.weight'],     # (d, d)
                'wk': tensors[p + 'attn_k.weight'],
                'wv': tensors[p + 'attn_v.weight'],
                'wo': tensors[p + 'attn_output.weight'],
                'ffn_norm': tensors[p + 'ffn_norm.weight'],
                'gate': tensors[p + 'ffn_gate.weight'],  # (d, ffn)
                'up': tensors[p + 'ffn_up.weight'],
                'down': tensors[p + 'ffn_down.weight'],  # (ffn, d)
            })
        self.vocab_size = self.emb.shape[0]

        # ---- tokenizer (tokenizer.ggml.*) -----------------------------
        if meta.get('tokenizer.ggml.model') != 'gpt2':
            raise ValueError(f'{path}: only gpt2 byte-level BPE vocabs '
                             'are supported')
        self.tokenizer = ByteBPE.from_vocab(
            meta['tokenizer.ggml.tokens'],
            meta['tokenizer.ggml.merges'],
            meta.get('tokenizer.ggml.token_type'))
        self.eot = int(meta.get('tokenizer.ggml.eot_token_id',
                                self.tokenizer.special_ids.get('<|im_end|>',
                                                               0)))
        self.im_start = self.tokenizer.special_ids.get('<|im_start|>', -1)
        self.im_end = self.tokenizer.special_ids.get('<|im_end|>', -1)

        # RoPE frequencies: pair j rotates by base^(-2j/head_dim)
        inv = 1.0 / (self.rope_base ** (np.arange(0, self.head_dim, 2)
                                        / self.head_dim))
        self._inv_freq = inv.astype(np.float32)

        self.reset()
        self._param_count = int(sum(t.size for t in tensors.values()))
        if verbose:
            print(f'GGUF model loaded: {self.name} '
                  f'({self._param_count:,} parameters, '
                  f'vocab {self.vocab_size}, ctx {self.n_ctx})')

    # ------------------------------------------------------------------
    # Tokenizer helpers
    # ------------------------------------------------------------------
    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, ids):
        return self.tokenizer.decode(list(ids))

    # ------------------------------------------------------------------
    # KV cache
    # ------------------------------------------------------------------
    def reset(self):
        """Clear the KV cache (start a fresh context)."""
        self.cache = [None] * self.n_layers   # per layer: (k, v) (T,H,hd)
        self.n_past = 0

    def _rope(self, x, positions):
        """Rotate pairs of consecutive head values.

        x: (T, H, hd) -> (T, H, hd); positions: (T,) absolute positions.
        """
        cos = np.cos(positions[:, None] * self._inv_freq)  # (T, hd/2)
        sin = np.sin(positions[:, None] * self._inv_freq)
        x1 = x[..., 0::2]                                   # (T, H, hd/2)
        x2 = x[..., 1::2]
        r1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
        r2 = x1 * sin[:, None, :] + x2 * cos[:, None, :]
        out = np.empty_like(x)
        out[..., 0::2] = r1
        out[..., 1::2] = r2
        return out

    def _forward(self, ids):
        """Run ids (list/array of new tokens) through the model, using
        and extending the KV cache. Returns logits for the LAST token,
        shape (vocab,)."""
        ids = np.asarray(ids, dtype=np.int64)
        T = len(ids)
        total = self.n_past + T
        positions = np.arange(self.n_past, total, dtype=np.float32)

        x = self.emb[ids]                                   # (T, d)
        H, hd = self.n_heads, self.head_dim
        scale = 1.0 / math.sqrt(hd)

        for i, blk in enumerate(self.blocks):
            h = _rms_norm(x, blk['attn_norm'], self.eps)
            q = (h @ blk['wq'].T).reshape(T, H, hd)
            k = (h @ blk['wk'].T).reshape(T, H, hd)
            v = (h @ blk['wv'].T).reshape(T, H, hd)
            q = self._rope(q, positions)
            k = self._rope(k, positions)
            # (T, H, hd) -> (H, T, hd) for batched attention
            q = np.transpose(q, (1, 0, 2))
            k = np.transpose(k, (1, 0, 2))
            v = np.transpose(v, (1, 0, 2))

            cached = self.cache[i]
            if cached is not None:
                ck, cv = cached
                k = np.concatenate([ck, k], axis=1)
                v = np.concatenate([cv, v], axis=1)
            self.cache[i] = (k, v)
            S = k.shape[1]                                  # total tokens

            att = q @ np.transpose(k, (0, 2, 1)) * scale     # (H, T, S)
            # causal mask: new token t (absolute n_past+t) sees keys <= it
            key_pos = np.arange(S)
            query_pos = np.arange(self.n_past, total)
            mask = key_pos[None, :] > query_pos[:, None]     # (T, S)
            att[:, mask] = -np.inf
            att = _softmax(att, axis=-1)
            y = att @ v                                      # (H, T, hd)
            y = np.transpose(y, (1, 0, 2)).reshape(T, self.d_model)
            x = x + y @ blk['wo'].T

            h = _rms_norm(x, blk['ffn_norm'], self.eps)
            gate = h @ blk['gate'].T
            up = h @ blk['up'].T
            x = x + (_silu(gate) * up) @ blk['down'].T

        x = _rms_norm(x[-1:], self.out_norm, self.eps)       # (1, d)
        logits = (x @ self.out_w.T)[0]                       # (vocab,)
        self.n_past = total
        return logits

    def forward_logits(self, ids):
        """Full-recompute logits for every position (no cache use);
        returns (T, vocab) - row j is the distribution after ids[:j+1].
        Used for tests / perplexity scoring."""
        ids = list(ids)
        saved = (self.cache, self.n_past)
        self.reset()
        acc = [self._forward([t]) for t in ids]
        self.cache, self.n_past = saved
        return np.stack(acc) if acc else np.zeros((0, self.vocab_size),
                                                  dtype=np.float32)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sample(self, logits, temperature, top_p, repeat_penalty,
                recent_ids, rng):
        logits = logits.astype(np.float64).copy()
        if repeat_penalty and repeat_penalty != 1.0 and recent_ids:
            for t in set(recent_ids):
                if 0 <= t < len(logits):
                    if logits[t] > 0:
                        logits[t] /= repeat_penalty
                    else:
                        logits[t] *= repeat_penalty
        if temperature <= 0:
            return int(np.argmax(logits))
        logits /= temperature
        probs = _softmax(logits)
        if top_p and top_p < 1.0:
            order = np.argsort(-probs)
            sorted_probs = probs[order]
            cum = np.cumsum(sorted_probs)
            # keep tokens until the cumulative mass passes top_p
            keep = cum - sorted_probs < top_p
            keep[0] = True
            mask = np.zeros_like(probs, dtype=bool)
            mask[order[keep]] = True
            probs = np.where(mask, probs, 0.0)
            probs /= probs.sum()
        return int(rng.choice(len(probs), p=probs))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, ids, max_new_tokens=96, temperature=0.8, top_p=0.9,
                 repeat_penalty=1.1, stop_ids=None, seed=None):
        """Continue `ids`, generating up to max_new_tokens tokens.
        Returns only the newly generated ids. Uses/extends the KV cache -
        call reset() between independent generations."""
        ids = list(ids)
        if not ids:
            return []
        rng = np.random.default_rng(seed)
        stop = set(stop_ids or [])
        stop.discard(-1)
        stop.add(self.eot)
        if self.im_start >= 0:
            stop.add(self.im_start)
        if self.im_end >= 0:
            stop.add(self.im_end)

        recent = list(ids[-64:])
        new_ids = []
        logits = self._forward(ids)          # prefill -> next-token dist
        for _ in range(max_new_tokens):
            tok = self._sample(logits, temperature, top_p, repeat_penalty,
                               recent, rng)
            new_ids.append(tok)
            recent.append(tok)
            if tok in stop:
                break
            logits = self._forward([tok])
        return new_ids

    # ------------------------------------------------------------------
    # Chat (ChatML, the format the models are trained on)
    # ------------------------------------------------------------------
    def chat(self, messages, max_new_tokens=160, temperature=0.8,
             top_p=0.9, repeat_penalty=1.1, seed=None):
        """messages: list of {'role': ..., 'content': ...} dicts.
        Generates the assistant reply for the conversation so far and
        returns it as text (chat markers stripped)."""
        prompt = ''
        for m in messages:
            prompt += (f'<|im_start|>{m["role"]}\n{m["content"]}'
                       '<|im_end|>\n')
        prompt += '<|im_start|>assistant\n'
        ids = self.tokenize(prompt)
        self.reset()
        out = self.generate(ids, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_p=top_p,
                            repeat_penalty=repeat_penalty, seed=seed)
        text = self.detokenize(out)
        for marker in ('<|im_end|>', '<|im_start|>assistant'):
            text = text.replace(marker, '')
        return text.strip()

    # ------------------------------------------------------------------
    # Scoring (tests / perplexity)
    # ------------------------------------------------------------------
    def score(self, ids):
        """Mean negative log-likelihood of ids[1:] given ids[:-1]."""
        logits = self.forward_logits(ids[:-1])
        if not len(logits):
            return float('inf')
        logp = logits - self.logsumexp(logits, axis=1, keepdims=True)
        targets = np.asarray(ids[1:], dtype=np.int64)
        picked = logp[np.arange(len(targets)), targets]
        return float(-picked.mean())

    @staticmethod
    def logsumexp(x, axis=None, keepdims=False):
        m = np.max(x, axis=axis, keepdims=True)
        out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
        return out if keepdims else np.squeeze(out, axis=axis)
