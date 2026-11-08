#!/usr/bin/env python3
"""
Train a real from-scratch chat model (llama architecture) and export it
to GGUF for llama.cpp / LM Studio / Ollama.

    python train_big.py                  # default: ~6.8M params, Muon
    python train_big.py --steps 2000     # train longer for better quality
    python export_gguf.py                # checkpoint -> lito-chat.gguf

The model is a genuine llama-architecture transformer trained from random
initialization on a quality corpus: ~52k instruction/response pairs with
GPT-4-quality answers (Stanford Alpaca prompts re-answered by GPT-4,
formatted as ChatML), TinyStories (GPT-4 written simple-English stories -
the data that famously lets tiny models speak coherently) and public-
domain prose for diversity. Everything - tokenizer (GPT-2-style byte-
level BPE), training, GGUF export - is implemented in this repo, no
pretrained weights.

Fast training: hidden matrices are optimized with Muon (momentum +
Newton-Schulz orthogonalization - roughly 1.5-2x more learning per token
than AdamW at this scale); embeddings and norms use AdamW. A small
held-out split reports validation loss while training.

The architecture deliberately matches llama.cpp's "llama" arch exactly
(RMSNorm, RoPE, SwiGLU, tied embeddings), so the export runs offline in
any GGUF-capable app with the ChatML chat template. The default settings
train in about an hour on two CPU cores and scale - run with more steps
or a bigger --d_model on a GPU box and it just gets better.
"""

import argparse
import base64
import json
import math
import os
import random
import re
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.bpe import ByteBPE  # noqa: E402

DATA_DIR = os.path.join(ROOT, 'data', 'bigcorpus')
MODEL_DIR = os.path.join(ROOT, 'checkpoints-big')

# Tarball sources: (url, member to extract, local name)
TARBALL_SOURCES = [
    # fallback instruction data if the GPT-4-answered set is unreachable
    ('https://codeload.github.com/tatsu-lab/stanford_alpaca/tar.gz/HEAD',
     'alpaca_data.json', 'alpaca_data.json'),
    ('https://codeload.github.com/karpathy/char-rnn/tar.gz/HEAD',
     'data/tinyshakespeare/input.txt', 'shakespeare.txt'),
]

# Single-file sources fetched through the GitHub blobs API (content is
# addressed by an immutable SHA, so these are pinned exactly).
# (owner/repo, blob sha, local name)
BLOB_SOURCES = [
    ('Instruction-Tuning-with-GPT-4/GPT-4-LLM',
     '25508714b7879a1e5a6764ba7f979a980f549f1a',
     'alpaca_gpt4_data.json'),
    ('datawhalechina/diy-llm',
     '485b1c3ece0e5c88adb9d4a59f2dc7b5863c21a3',
     'tinystories.txt'),
]


# ----------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------

def _fetch_blob(repo, sha, dest):
    """Fetch one file from GitHub by its immutable blob SHA. Uses a
    GitHub token from the environment when available (CI) for the
    friendlier rate limit."""
    url = f'https://api.github.com/repos/{repo}/git/blobs/{sha}'
    headers = {'Accept': 'application/vnd.github+json',
               'User-Agent': 'lito-trainer'}
    token = (os.environ.get('LITO_GH_TOKEN')
             or os.environ.get('GITHUB_TOKEN'))
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        meta = json.load(r)
    raw = base64.b64decode(meta['content'])
    with open(dest, 'wb') as f:
        f.write(raw)
    return len(raw)


def fetch_corpus():
    """Download the training corpus (once) into data/bigcorpus/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    import tarfile
    for url, member, local in TARBALL_SOURCES:
        local_path = os.path.join(DATA_DIR, local)
        if os.path.exists(local_path):
            continue
        print(f'fetching {local} from {url.split("/")[2]} ...', flush=True)
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
    for repo, sha, local in BLOB_SOURCES:
        local_path = os.path.join(DATA_DIR, local)
        if os.path.exists(local_path):
            continue
        try:
            print(f'fetching {local} from {repo} (blob API) ...', flush=True)
            size = _fetch_blob(repo, sha, local_path)
            print(f'  {size:,} bytes', flush=True)
        except Exception as e:  # optional data: train on what's there
            print(f'  WARNING: could not fetch {local} ({e}); '
                  f'training without it', flush=True)
    # corpus license/attribution notes
    notes = os.path.join(DATA_DIR, 'SOURCES.md')
    if not os.path.exists(notes):
        with open(notes, 'w') as f:
            f.write(
                '# Training corpus sources\n\n'
                '- `alpaca_gpt4_data.json` - the 52,000 Stanford Alpaca\n'
                '  prompts re-answered by GPT-4\n'
                '  (Instruction-Tuning-with-GPT-4/GPT-4-LLM). Intended for\n'
                '  research purposes; see the repo\'s license notes.\n'
                '- `tinystories.txt` - TinyStoriesV2-GPT4 valid split\n'
                '  (roneneldan/TinyStories, GPT-4 generated simple-English\n'
                '  stories; CDLA-Sharing-1.0), via the mirror committed in\n'
                '  datawhalechina/diy-llm.\n'
                '- `alpaca_data.json` - the original Stanford Alpaca data\n'
                '  (tatsu-lab/stanford_alpaca), used as a fallback when the\n'
                '  GPT-4-answered set is unreachable. Research purposes; see\n'
                '  the repo\'s DATA_LICENSE.\n'
                '- `shakespeare.txt` - tiny shakespeare (karpathy/char-rnn),\n'
                '  public domain.\n')


STORY_PROMPTS = [
    'Tell me a story.',
    'Tell me a short story.',
    'Tell me a bedtime story.',
    'Tell me another story.',
    'I would like to hear a story.',
    'Can you tell me a story?',
    'Please tell me a story.',
    'Make up a story for me.',
]
_NAME_RE = re.compile(r'\bnamed ([A-Z][a-z]+)')


def _story_docs(text, rng):
    """TinyStories -> ChatML docs: some as story requests, some as plain
    continuation, so the model both follows 'tell me a story' and keeps
    narrative English flowing."""
    docs = []
    for story in text.split('<|endoftext|>'):
        story = story.strip()
        if len(story) < 80:
            continue
        roll = rng.random()
        if roll < 0.45:
            prompt = rng.choice(STORY_PROMPTS)
            m = _NAME_RE.search(story[:220])
            if m and rng.random() < 0.6:
                prompt = f'Tell me a story about {m.group(1)}.'
            docs.append(f'<|im_start|>user\n{prompt}<|im_end|>\n'
                        f'<|im_start|>assistant\n{story}<|im_end|>')
        else:
            docs.append('<|im_start|>system\nContinue this text.<|im_end|>\n'
                        f'<|im_start|>assistant\n{story}<|im_end|>')
    return docs


def build_documents():
    """All training documents as ChatML-formatted strings."""
    rng = random.Random(13)

    # instruction data: prefer the GPT-4-answered set
    gpt4_path = os.path.join(DATA_DIR, 'alpaca_gpt4_data.json')
    alpaca_path = os.path.join(DATA_DIR, 'alpaca_data.json')
    src = gpt4_path if os.path.exists(gpt4_path) else alpaca_path
    if os.path.exists(src):
        print(f'instruction data: {os.path.basename(src)}')
        with open(src) as f:
            rows = json.load(f)
        docs = []
        for row in rows:
            instruction = (row.get('instruction') or '').strip()
            extra = (row.get('input') or '').strip()
            output = (row.get('output') or '').strip()
            if not instruction or not output:
                continue
            user = instruction + (f'\n{extra}' if extra else '')
            docs.append(
                f'<|im_start|>user\n{user}<|im_end|>\n'
                f'<|im_start|>assistant\n{output}<|im_end|>')
    else:
        docs = []

    # TinyStories: simple, coherent English (the TinyStories paper's data)
    ts_path = os.path.join(DATA_DIR, 'tinystories.txt')
    if os.path.exists(ts_path):
        text = open(ts_path, encoding='utf-8', errors='ignore').read()
        story_docs = _story_docs(text, rng)
        print(f'story data: {len(story_docs):,} TinyStories docs')
        docs.extend(story_docs)
    else:
        print('WARNING: no TinyStories - coherence will suffer')

    # public-domain prose, chunked into ~1KB paragraphs as neutral documents
    shk_path = os.path.join(DATA_DIR, 'shakespeare.txt')
    if os.path.exists(shk_path):
        prose = open(shk_path, encoding='utf-8', errors='ignore').read()
        chunk, size = [], 0
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

    rng.shuffle(docs)
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
# Muon: momentum + orthogonalized updates (fast training)
# ----------------------------------------------------------------------

def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate the orthogonal factor U of G = U S V^T with a quintic
    Newton-Schulz iteration (no SVD needed). Same coefficients as the
    modded-nanogpt Muon."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon:
    """Muon for 2-D hidden matrices: momentum buffer, then the update
    direction is orthogonalized so every singular direction moves with
    equal strength. At tiny scale this converges markedly faster than
    AdamW on the same tokens. Embeddings/norms stay on AdamW."""

    def __init__(self, params, lr=0.04, momentum=0.95,
                 nesterov=True, ns_steps=5):
        import torch
        self.torch = torch
        self.params = [p for p in params]
        self.defaults = dict(lr=lr, momentum=momentum,
                             nesterov=nesterov, ns_steps=ns_steps)
        self.state = {}
        # a compatible param_groups shim so the LR schedule works
        self.param_groups = [dict(lr=lr, momentum=momentum,
                                  nesterov=nesterov, ns_steps=ns_steps,
                                  params=self.params)]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def step(self):
        torch = self.torch
        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                momentum = group['momentum']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state.setdefault(p, {})
                    if 'buf' not in st:
                        st['buf'] = torch.zeros_like(g)
                    buf = st['buf']
                    buf.lerp_(g, 1.0 - momentum)
                    update = g.lerp(buf, momentum) if group['nesterov'] else buf
                    u = zeropower_via_newtonschulz5(update, group['ns_steps'])
                    # keep the update RMS shape-independent
                    scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                    p.add_(u, alpha=-lr * scale)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def eval_val(model, val_tokens, ctx, batches=8):
    """Mean loss over fixed, non-overlapping windows of the held-out
    split (deterministic, comparable across runs)."""
    import torch
    torch.set_grad_enabled(False)
    losses = []
    pos = 0
    for _ in range(batches):
        if pos + ctx + 1 >= len(val_tokens):
            break
        x = torch.from_numpy(val_tokens[pos:pos + ctx].astype(np.int64))[None]
        y = torch.from_numpy(val_tokens[pos + 1:pos + ctx + 1]
                             .astype(np.int64))[None]
        logits = model(x)
        losses.append(torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), y.view(-1)).item())
        pos += ctx
    torch.set_grad_enabled(True)
    return sum(losses) / len(losses) if losses else float('nan')


def main():
    parser = argparse.ArgumentParser(
        description='Train the from-scratch llama-architecture chat model')
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_ff', type=int, default=672)
    parser.add_argument('--ctx', type=int, default=512)
    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--accum', type=int, default=1,
                        help='gradient accumulation steps (effective '
                             'batch = batch x accum; use to fit small '
                             'RAM boxes)')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='AdamW lr for embeddings and norms')
    parser.add_argument('--muon_lr', type=float, default=0.04,
                        help='Muon lr for the hidden matrices')
    parser.add_argument('--optimizer', choices=('muon', 'adamw'),
                        default='muon')
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
    print(f'corpus: {len(docs):,} documents')

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

    # 3. Tokenize (train + held-out val split) ------------------------
    bin_path = os.path.join(args.model_dir, 'tokens.bin')
    val_path = os.path.join(args.model_dir, 'tokens_val.bin')
    if not (os.path.exists(bin_path) and os.path.exists(val_path)):
        print('tokenizing the corpus...')
        rng = random.Random(7)
        val_docs = set(rng.sample(range(len(docs)),
                                  max(1, int(0.005 * len(docs)))))
        ids, val_ids = [], []
        for i, doc in enumerate(docs):
            out = val_ids if i in val_docs else ids
            out.extend(tok.encode(doc))
            out.append(tok.encode('<|im_end|>')[0])
            if (i + 1) % 20000 == 0:
                print(f'  {i + 1}/{len(docs)} docs', flush=True)
        np.array(ids, dtype=np.uint16).tofile(bin_path)
        np.array(val_ids, dtype=np.uint16).tofile(val_path)
        print(f'tokens: {len(ids):,} train, {len(val_ids):,} val')
    tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')
    val_tokens = np.memmap(val_path, dtype=np.uint16, mode='r')

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
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if ckpt['config'] != config:
            raise SystemExit(
                f'checkpoint config {ckpt["config"]} != current {config}; '
                'resume needs identical model settings')
        model.load_state_dict(ckpt['state_dict'])
        resumed_from = ckpt.get('step', 0)
        print(f'resumed weights from step {resumed_from}')

    # 5. Optimizers --------------------------------------------------
    # hidden 2-D matrices -> Muon; embeddings + norms -> AdamW
    hidden, others = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and 'embed' not in name:
            hidden.append(p)
        else:
            others.append(p)
    optimizers = []
    if args.optimizer == 'muon':
        optimizers.append(Muon(hidden, lr=args.muon_lr))
        optimizers.append(torch.optim.AdamW(others, lr=args.lr,
                                            betas=(0.9, 0.95),
                                            weight_decay=0.1, fused=False))
        print(f'optimizer: Muon (lr {args.muon_lr}) on {len(hidden)} hidden '
              f'matrices + AdamW (lr {args.lr}) on {len(others)} tensors')
    else:
        optimizers.append(torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.95),
            weight_decay=0.1, fused=False))
        print(f'optimizer: AdamW (lr {args.lr})')

    warmup = max(10, args.steps // 50)

    def lr_mult(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, args.steps - warmup)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * t))

    def set_lrs(step):
        m = lr_mult(step)
        for opt in optimizers:
            base = (args.muon_lr if (isinstance(opt, Muon))
                    else (args.lr if len(optimizers) == 1 else args.lr))
            for g in opt.param_groups:
                g['lr'] = base * m

    # 6. Train -------------------------------------------------------
    print(f'training {args.steps} steps (batch {args.batch} x {args.accum} '
          f'x ctx {args.ctx})...')
    model.train()
    started = time.time()
    tokens_seen = 0
    for step in range(args.steps):
        set_lrs(step)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        # accumulate micro-batches of random windows from the token stream
        loss_val = 0.0
        for _ in range(args.accum):
            ix = np.random.randint(0, len(tokens) - args.ctx - 1,
                                   size=args.batch)
            x = np.stack([tokens[i:i + args.ctx] for i in ix]).astype(np.int64)
            y = np.stack([tokens[i + 1:i + args.ctx + 1]
                          for i in ix]).astype(np.int64)
            xt, yt = torch.from_numpy(x), torch.from_numpy(y)
            logits = model(xt)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]), yt.view(-1)) / args.accum
            loss.backward()
            loss_val = loss.item() * args.accum
            tokens_seen += args.batch * args.ctx
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

        if step % 20 == 0 or step == args.steps - 1:
            elapsed = time.time() - started
            muon_lr = optimizers[0].param_groups[0]['lr']
            print(f'step {step:5d} | loss {loss_val:.4f} | lr {muon_lr:.2e} '
                  f'| {tokens_seen / elapsed:,.0f} tok/s | '
                  f'eta {elapsed / (step + 1) * (args.steps - step - 1) / 60:.1f} min',
                  flush=True)
        if (step + 1) % 100 == 0 or step == args.steps - 1:
            v = eval_val(model, val_tokens, args.ctx)
            print(f'  val loss {v:.4f} @ step {resumed_from + step + 1}',
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
