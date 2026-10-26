#!/usr/bin/env python3
"""
Train a real from-scratch chat model (llama architecture) and export it
to GGUF for llama.cpp / LM Studio / Ollama.

    python train_big.py                  # default: ~6M params, CPU-sized
    python train_big.py --steps 2000     # train longer for better quality
    python export_gguf.py                # checkpoint -> lito-chat.gguf

The model is a genuine llama-architecture transformer trained from random
initialization on instruction data (Stanford Alpaca, 52k instruction/
response pairs, formatted as ChatML) plus public-domain prose for
diversity. Everything - tokenizer (GPT-2-style byte-level BPE), training,
GGUF export - is implemented in this repo, no pretrained weights.

The architecture deliberately matches llama.cpp's "llama" arch exactly
(RMSNorm, RoPE, SwiGLU, tied embeddings), so the export runs offline in
any GGUF-capable app with the ChatML chat template.

The default settings train in about an hour on two CPU cores and
produce a competent tiny assistant: it learns the chat format, spelling
and short coherent answers. The same script scales - run it with more
steps on a GPU box and it just gets better.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.bpe import ByteBPE  # noqa: E402

DATA_DIR = os.path.join(ROOT, 'data', 'bigcorpus')
MODEL_DIR = os.path.join(ROOT, 'checkpoints-big')

CORPUS_SOURCES = [
    # (url of a tarball, member to extract, local name)
    ('https://codeload.github.com/tatsu-lab/stanford_alpaca/tar.gz/HEAD',
     'alpaca_data.json', 'alpaca_data.json'),
    ('https://codeload.github.com/karpathy/char-rnn/tar.gz/HEAD',
     'data/tinyshakespeare/input.txt', 'shakespeare.txt'),
]


# ----------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------

def fetch_corpus():
    """Download the training corpus (once) into data/bigcorpus/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    import tarfile
    for url, member, local in CORPUS_SOURCES:
        local_path = os.path.join(DATA_DIR, local)
        if os.path.exists(local_path):
            continue
        print(f'fetching {local} from {url.split("/")[2]} ...')
        tgz_path = local_path + '.tgz'
        urllib.request.urlretrieve(url, tgz_path)
        with tarfile.open(tgz_path) as tf:
            found = False
            for cand in tf.getnames():
                if cand.endswith(member):
                    with open(local_path, 'wb') as out:
                        out.write(tf.extractfile(cand).read())
                    found = True
                    break
            if not found:
                raise RuntimeError(f'{member} not found in {url}')
        os.remove(tgz_path)
    # corpus license/attribution notes
    notes = os.path.join(DATA_DIR, 'SOURCES.md')
    if not os.path.exists(notes):
        with open(notes, 'w') as f:
            f.write(
                '# Training corpus sources\n\n'
                '- `alpaca_data.json` - Stanford Alpaca, 52,000 instruction/\n'
                '  response pairs (tatsu-lab/stanford_alpaca). Intended for\n'
                '  research purposes; see the repo\'s DATA_LICENSE.\n'
                '- `shakespeare.txt` - tiny shakespeare (karpathy/char-rnn),\n'
                '  public domain.\n')


def build_documents():
    """All training documents as ChatML-formatted strings."""
    with open(os.path.join(DATA_DIR, 'alpaca_data.json')) as f:
        alpaca = json.load(f)
    docs = []
    for row in alpaca:
        instruction = (row.get('instruction') or '').strip()
        extra = (row.get('input') or '').strip()
        output = (row.get('output') or '').strip()
        if not instruction or not output:
            continue
        user = instruction + (f'\n{extra}' if extra else '')
        docs.append(
            f'<|im_start|>user\n{user}<|im_end|>\n'
            f'<|im_start|>assistant\n{output}<|im_end|>')
    # public-domain prose, chunked into ~1KB paragraphs as neutral documents
    prose = open(os.path.join(DATA_DIR, 'shakespeare.txt'),
                 encoding='utf-8', errors='ignore').read()
    chunk = []
    size = 0
    for para in prose.split('\n\n'):
        para = para.strip()
        if not para:
            continue
        chunk.append(para)
        size += len(para)
        if size > 1000:
            docs.append('<|im_start|>system\nContinue this text.<|im_end|>\n'
                        f'<|im_start|>assistant\n{para}<|im_end|>')
            chunk, size = [], 0
    random.Random(13).shuffle(docs)
    return docs


# ----------------------------------------------------------------------
# Model (llama architecture, trained from scratch)
# ----------------------------------------------------------------------

def build_model(config, vocab_size):
    """The exact architecture llama.cpp's 'llama' arch expects:
    RMSNorm (pre-norm), RoPE, SwiGLU FFN, tied embeddings."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            dtype = x.dtype
            x = x.float()
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return (self.weight * x).to(dtype)

    def rope_cache(seq_len, head_dim, base=10000.0):
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(seq_len).float()
        freqs = torch.outer(t, inv)
        return torch.cos(freqs), torch.sin(freqs)

    def apply_rope(x, cos, sin):
        # x: (B, heads, T, head_dim); rotation on pairs of CONSECUTIVE
        # values (llama.cpp "normal" RoPE, GGML_ROPE_TYPE_NORMAL, layout
        # [c s c s ...]) - the same convention llama.cpp applies to the
        # llama architecture at inference, so no permutation is needed
        # on export.
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        cos = cos[None, None, :x.shape[-2], :]
        sin = sin[None, None, :x.shape[-2], :]
        r1 = x1 * cos - x2 * sin
        r2 = x1 * sin + x2 * cos
        return torch.stack((r1, r2), dim=-1).flatten(-2)

    class Block(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            d = cfg['d_model']
            self.n_heads = cfg['n_heads']
            self.attn_norm = RMSNorm(d)
            self.wq = nn.Linear(d, d, bias=False)
            self.wk = nn.Linear(d, d, bias=False)
            self.wv = nn.Linear(d, d, bias=False)
            self.wo = nn.Linear(d, d, bias=False)
            self.ffn_norm = RMSNorm(d)
            self.w_gate = nn.Linear(d, cfg['n_ff'], bias=False)
            self.w_up = nn.Linear(d, cfg['n_ff'], bias=False)
            self.w_down = nn.Linear(cfg['n_ff'], d, bias=False)

        def forward(self, x, cos, sin, mask):
            B, T, C = x.shape
            hd = C // self.n_heads
            h = self.attn_norm(x)
            q = apply_rope(self.wq(h).view(B, T, self.n_heads, hd).transpose(1, 2), cos, sin)
            k = apply_rope(self.wk(h).view(B, T, self.n_heads, hd).transpose(1, 2), cos, sin)
            v = self.wv(h).view(B, T, self.n_heads, hd).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            att = att + mask
            att = F.softmax(att, dim=-1)
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
            x = x + self.wo(y)
            h = self.ffn_norm(x)
            x = x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
            return x

    class Llama(nn.Module):
        def __init__(self, cfg, vocab_size):
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Embedding(vocab_size, cfg['d_model'])
            self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg['n_layers']))
            self.norm = RMSNorm(cfg['d_model'])
            self.head = nn.Linear(cfg['d_model'], vocab_size, bias=False)
            self.head.weight = self.embed.weight  # tied
            self._cos, self._sin = rope_cache(cfg['ctx'], cfg['d_model'] // cfg['n_heads'])
            self.register_buffer('mask',
                                 torch.triu(torch.full((cfg['ctx'], cfg['ctx']),
                                                       float('-inf')), diagonal=1))

        def forward(self, ids):
            B, T = ids.shape
            x = self.embed(ids)
            cos = self._cos.to(x.device)
            sin = self._sin.to(x.device)
            mask = self.mask[:T, :T]
            for b in self.blocks:
                x = b(x, cos, sin, mask)
            return self.head(self.norm(x))

    return Llama(config, vocab_size)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train the from-scratch llama-architecture chat model')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_ff', type=int, default=672)
    parser.add_argument('--ctx', type=int, default=512)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true',
                        help='continue training the existing checkpoint '
                             '(loads weights, runs --steps more steps)')
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))

    os.makedirs(args.model_dir, exist_ok=True)

    # 1. Corpus ------------------------------------------------------
    fetch_corpus()
    docs = build_documents()
    print(f'corpus: {len(docs)} documents')

    # 2. Tokenizer ---------------------------------------------------
    tok_path = os.path.join(args.model_dir, 'bpe_tokenizer.json')
    if os.path.exists(tok_path):
        tok = ByteBPE.load(tok_path)
        print(f'tokenizer: loaded ({tok.vocab_size} tokens)')
    else:
        print('training the byte-level BPE tokenizer...')
        tok = ByteBPE(vocab_size=args.vocab_size)
        tok.train(docs, verbose=True)
        tok.save(tok_path)
        print(f'tokenizer: {tok.vocab_size} tokens -> {tok_path}')

    # 3. Tokenize ----------------------------------------------------
    bin_path = os.path.join(args.model_dir, 'tokens.bin')
    if not os.path.exists(bin_path):
        print('tokenizing the corpus...')
        ids = []
        for i, doc in enumerate(docs):
            ids.extend(tok.encode(doc))
            ids.append(tok.encode('<|im_end|>')[0])
            if (i + 1) % 10000 == 0:
                print(f'  {i + 1}/{len(docs)} docs')
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(bin_path)
        print(f'tokens: {len(arr):,} -> {bin_path}')
    tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')

    # 4. Model -------------------------------------------------------
    config = {
        'd_model': args.d_model, 'n_layers': args.n_layers,
        'n_heads': args.n_heads, 'n_ff': args.n_ff, 'ctx': args.ctx,
        'vocab_size': tok.vocab_size,
    }
    model = build_model(config, tok.vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'model: {n_params:,} parameters '
          f'(d={args.d_model} L={args.n_layers} h={args.n_heads} ff={args.n_ff} ctx={args.ctx})')

    resumed_from = 0
    ckpt_path = os.path.join(args.model_dir, 'checkpoint.pt')
    if args.resume and os.path.exists(ckpt_path):
        import torch
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if ckpt['config'] != config:
            raise SystemExit(
                f'checkpoint config {ckpt["config"]} != current {config}; '
                'resume needs identical model settings')
        model.load_state_dict(ckpt['state_dict'])
        resumed_from = ckpt.get('step', 0)
        print(f'resumed weights from step {resumed_from}')

    # 5. Train -------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1, fused=False)
    warmup = max(10, args.steps // 50)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        t = (step - warmup) / max(1, args.steps - warmup)
        return 0.1 * args.lr + 0.45 * args.lr * (1 + math.cos(math.pi * t))

    print(f'training {args.steps} steps (batch {args.batch} x ctx {args.ctx})...')
    model.train()
    started = time.time()
    tokens_seen = 0
    for step in range(args.steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        # random windows from the token stream
        ix = np.random.randint(0, len(tokens) - args.ctx - 1, size=args.batch)
        x = np.stack([tokens[i:i + args.ctx] for i in ix]).astype(np.int64)
        y = np.stack([tokens[i + 1:i + args.ctx + 1] for i in ix]).astype(np.int64)
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        logits = model(xt)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), yt.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens_seen += args.batch * args.ctx

        if step % 20 == 0 or step == args.steps - 1:
            elapsed = time.time() - started
            print(f'step {step:5d} | loss {loss.item():.4f} | lr {lr_at(step):.2e} '
                  f'| {tokens_seen / elapsed:,.0f} tok/s | '
                  f'eta {elapsed / (step + 1) * (args.steps - step - 1) / 60:.1f} min',
                  flush=True)

        if (step + 1) % 200 == 0 or step == args.steps - 1:
            save_checkpoint(model, config, args.model_dir,
                            resumed_from + step, tok_path)
            print(f'checkpoint saved (step {resumed_from + step + 1})')

    save_checkpoint(model, config, args.model_dir,
                    resumed_from + args.steps - 1, tok_path)
    total = time.time() - started
    print(f'done: {args.steps} more steps ({resumed_from + args.steps} total), '
          f'{tokens_seen:,} tokens in {total / 60:.1f} min')
    print(f'next: python export_gguf.py --model_dir {args.model_dir}')


def save_checkpoint(model, config, model_dir, step, tok_path):
    import torch
    ckpt = {
        'config': config,
        'step': step + 1,
        'tokenizer': os.path.basename(tok_path),
        'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
    }
    path = os.path.join(model_dir, 'checkpoint.pt')
    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


if __name__ == '__main__':
    main()
