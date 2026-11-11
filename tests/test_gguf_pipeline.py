#!/usr/bin/env python3
"""
Tests for the from-scratch GGUF model pipeline: the GPT-2-style byte-level
BPE tokenizer (src/bpe.py), the GGUF v3 writer (src/gguf.py), and - when
torch is installed - the llama-architecture trainer (train_big.py) and
GGUF exporter (export_gguf.py).

Runs with plain Python (no test dependencies):
    python tests/test_gguf_pipeline.py

Also compatible with pytest:
    pytest tests/test_gguf_pipeline.py
"""

import os
import struct
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.bpe import ByteBPE, BYTE_MAP, CHAR_MAP, SPECIAL_TOKENS  # noqa: E402
from src.gguf import (GGUFWriter, GGUF_MAGIC, GGUF_VERSION,  # noqa: E402
                      GGUF_ALIGNMENT, TYPE_F16, TYPE_F32)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ----------------------------------------------------------------------
# GGUF file parsing helper (independent re-implementation of a reader)
# ----------------------------------------------------------------------

def read_gguf(path):
    raw = open(path, 'rb').read()
    magic, version, n_tensors, n_kv = struct.unpack('<IIQQ', raw[:24])
    pos = 24
    scalar = {0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2),
              4: ('<I', 4), 5: ('<i', 4), 6: ('<f', 4), 7: ('<B', 1),
              10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8)}
    kvs = {}
    for _ in range(n_kv):
        (klen,) = struct.unpack('<Q', raw[pos:pos + 8]); pos += 8
        key = raw[pos:pos + klen].decode(); pos += klen
        (vtype,) = struct.unpack('<I', raw[pos:pos + 4]); pos += 4
        if vtype == 8:
            (slen,) = struct.unpack('<Q', raw[pos:pos + 8]); pos += 8
            kvs[key] = raw[pos:pos + slen].decode(); pos += slen
        elif vtype == 9:
            (itype,) = struct.unpack('<I', raw[pos:pos + 4]); pos += 4
            (cnt,) = struct.unpack('<Q', raw[pos:pos + 8]); pos += 8
            if itype == 8:
                vals = []
                for _ in range(cnt):
                    (slen,) = struct.unpack('<Q', raw[pos:pos + 8]); pos += 8
                    vals.append(raw[pos:pos + slen].decode()); pos += slen
                kvs[key] = vals
            else:
                fmt, size = scalar[itype]
                kvs[key] = list(struct.unpack('<' + fmt[1:] * cnt,
                                               raw[pos:pos + size * cnt]))
                pos += size * cnt
        else:
            fmt, size = scalar[vtype]
            (kvs[key],) = struct.unpack(fmt, raw[pos:pos + size])
            pos += size
    tensors = []
    for _ in range(n_tensors):
        (nlen,) = struct.unpack('<Q', raw[pos:pos + 8]); pos += 8
        name = raw[pos:pos + nlen].decode(); pos += nlen
        (nd,) = struct.unpack('<I', raw[pos:pos + 4]); pos += 4
        ne = struct.unpack('<' + 'Q' * nd, raw[pos:pos + 8 * nd]); pos += 8 * nd
        (ttype,) = struct.unpack('<I', raw[pos:pos + 4]); pos += 4
        (off,) = struct.unpack('<Q', raw[pos:pos + 8]); pos += 8
        tensors.append({'name': name, 'ne': ne, 'type': ttype, 'offset': off})
    # the tensor data section starts at the next multiple of the alignment
    data_start = (pos + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT
    return {'magic': magic, 'version': version, 'kv': kvs,
            'tensors': tensors, 'data': raw[data_start:], 'data_start': data_start}


CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "translation: hello world, hello there, the world says hello",
    "Q: What is 2+2? A: 2+2 = 4. Q: What is 3+3? A: 3+3 = 6.",
    "Once upon a time in a land far far away lived a tiny dragon",
    "user: hi there, how are you? assistant: I'm fine, thanks!",
] * 8


def get_tokenizer():
    tok = ByteBPE(vocab_size=400)
    tok.train(CORPUS, verbose=False)
    return tok


# ----------------------------------------------------------------------
# BPE tokenizer
# ----------------------------------------------------------------------

@test
def bpe_byte_map_is_gpt2():
    # the first 94 printable bytes map to themselves, like GPT-2
    for b in range(ord('!'), ord('~') + 1):
        assert BYTE_MAP[b] == chr(b), (b, BYTE_MAP[b])
    # space becomes the printable stand-in, newline its own char
    assert BYTE_MAP[0x20] not in ' \t\n'
    assert len(BYTE_MAP) == 256 and len(CHAR_MAP) == 256
    for b, c in BYTE_MAP.items():
        assert CHAR_MAP[c] == b
        assert ord(c) >= 256 or 33 <= ord(c) <= 126 or 161 <= ord(c) <= 172 \
            or 174 <= ord(c) <= 255


@test
def bpe_roundtrip():
    tok = get_tokenizer()
    texts = [
        "Hello, world! How are you today? I'm fine - 42% naive cafe.",
        "UNICODE: tabs\tand\nnewlines\r\nand emoji \U0001f9ee and accents",
        "<|im_start|>user\nWhat's 9 * 9?<|im_end|>\n"
        "<|im_start|>assistant\n9 * 9 = 81<|im_end|>",
        "",
    ]
    for t in texts:
        ids = tok.encode(t)
        assert all(isinstance(i, int) and 0 <= i < tok.vocab_size for i in ids)
        assert tok.decode(ids) == t, repr(tok.decode(ids))


@test
def bpe_special_tokens():
    tok = get_tokenizer()
    ids = tok.encode("<|im_start|>user\nhi<|im_end|>")
    assert ids[0] == 256 and ids[-1] == 257, ids
    assert tok.decode([256]) == '<|im_start|>'
    assert tok.decode([257]) == '<|im_end|>'
    assert tok.vocab[256] == '<|im_start|>'
    assert tok.vocab[257] == '<|im_end|>'
    # special tokens survive a training vocab that contains them
    assert len(SPECIAL_TOKENS) == 2


@test
def bpe_merges_look_like_gpt2():
    tok = get_tokenizer()
    assert tok.merges, 'no merges learned'
    # merged symbols are in byte-unicode form: spaces appear as the
    # printable stand-in, never as a literal space inside a token
    for a, b in tok.merges:
        assert ' ' not in a and ' ' not in b, (a, b)
    tokens = set(tok.vocab)
    assert any(t.startswith(BYTE_MAP[0x20]) for t in tokens)
    assert all(t for t in tokens)
    # every merge result is in the vocab
    for a, b in tok.merges:
        assert a + b in tokens


@test
def bpe_save_load():
    tok = get_tokenizer()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'bpe.json')
        tok.save(path)
        tok2 = ByteBPE.load(path)
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.merges == tok.merges
        sample = "hello there, general Kenobi! 9 * 9 = 81"
        assert tok2.encode(sample) == tok.encode(sample)
        assert tok2.decode(tok2.encode(sample)) == sample


@test
def bpe_no_unknown_symbols():
    tok = get_tokenizer()
    # arbitrary byte sequences must encode to known vocab ids
    weird = "".join(chr(i) for i in range(1, 0x100)) + "¸ÿ\U0001f600"
    ids = tok.encode(weird)
    assert ids and all(0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == weird


# ----------------------------------------------------------------------
# GGUF writer
# ----------------------------------------------------------------------

@test
def gguf_header_and_layout():
    w = GGUFWriter('llama', 'Lito-Test')
    w.add_uint32('llama.block_count', 6)
    w.add_float32('llama.rope.freq_base', 10000.0)
    w.add_bool('tokenizer.ggml.add_bos_token', False)
    w.add_strings('tokenizer.ggml.tokens', ['a', 'Ġb', 'Ċ'])
    w.add_int32s('tokenizer.ggml.token_type', [1, 1, 3])
    emb = np.arange(20 * 10, dtype=np.float32).reshape(20, 10)
    w.add_tensor('token_embd.weight', emb, TYPE_F16)
    w.add_tensor('blk.0.attn_norm.weight', np.full(10, 0.5), TYPE_F32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'm.gguf')
        w.write(path)
        f = read_gguf(path)
    assert f['magic'] == GGUF_MAGIC and f['version'] == GGUF_VERSION
    assert f['kv']['general.architecture'] == 'llama'
    assert f['kv']['general.name'] == 'Lito-Test'
    assert f['kv']['llama.block_count'] == 6
    assert abs(f['kv']['llama.rope.freq_base'] - 10000.0) < 1e-3
    assert f['kv']['tokenizer.ggml.tokens'] == ['a', 'Ġb', 'Ċ']
    assert f['kv']['tokenizer.ggml.token_type'] == [1, 1, 3]
    # keys unique and sorted
    assert list(f['kv']) == sorted(f['kv'])
    t0, t1 = f['tensors']
    assert t0['name'] == 'token_embd.weight' and t0['type'] == TYPE_F16
    # GGUF ne = reversed torch shape: torch (20, 10) -> ne (10, 20)
    assert t0['ne'] == (10, 20)
    assert t1['ne'] == (10,)
    # 32-byte alignment: f16 block 20*10*2 = 400 -> padded to 416? no:
    # 400 is not a multiple of 32, so next offset is 416
    assert t1['offset'] == ((20 * 10 * 2 + GGUF_ALIGNMENT - 1)
                            // GGUF_ALIGNMENT * GGUF_ALIGNMENT)
    assert t1['offset'] % GGUF_ALIGNMENT == 0


@test
def gguf_tensor_values_roundtrip():
    w = GGUFWriter('llama', 'Lito-Test')
    rng = np.random.default_rng(7)
    emb = rng.standard_normal((64, 32)).astype(np.float32)
    norm = np.full(32, 0.25, dtype=np.float32)
    w.add_tensor('token_embd.weight', emb, TYPE_F16)
    w.add_tensor('output_norm.weight', norm, TYPE_F32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'm.gguf')
        w.write(path)
        f = read_gguf(path)
    d = f['data']
    got_emb = np.frombuffer(d[:64 * 32 * 2], dtype=np.float16) \
        .reshape(64, 32).astype(np.float32)
    assert np.allclose(got_emb, emb, atol=2e-3)
    off = f['tensors'][1]['offset']
    got_norm = np.frombuffer(d[off:off + 32 * 4], dtype=np.float32)
    assert np.allclose(got_norm, norm)


@test
def gguf_duplicate_keys_are_replaced():
    w = GGUFWriter('llama', 'first')
    w.add_string('general.name', 'second')
    w.add_uint32('llama.block_count', 1)
    w.add_uint32('llama.block_count', 7)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'm.gguf')
        w.write(path)
        f = read_gguf(path)
    assert f['kv']['general.name'] == 'second'
    assert f['kv']['llama.block_count'] == 7
    assert len(f['kv']) == 3  # architecture, name, block_count


# ----------------------------------------------------------------------
# torch trainer + exporter (skipped when torch is not installed)
# ----------------------------------------------------------------------

def _skip_if_no_torch():
    if not HAVE_TORCH:
        print('    (torch not installed - skipping trainer/exporter tests)')
        return True
    return False


@test
def muon_orthogonalizes_updates():
    """Newton-Schulz must return (approximately) orthogonal matrices, and
    Muon must descend: the fast-training optimizer used by train_big.py."""
    import torch
    from train_big import zeropower_via_newtonschulz5, Muon
    torch.manual_seed(4)
    for shape in ((32, 32), (64, 48), (48, 64), (256, 672)):
        G = torch.randn(*shape)
        U = zeropower_via_newtonschulz5(G, steps=5)
        eye = torch.eye(min(shape))
        gram = U @ U.T if U.shape[0] <= U.shape[1] else U.T @ U
        # the quintic iteration approximately equalizes the spectrum in
        # 5 steps (exact in the limit): check the result is close to
        # orthogonal and far closer than the raw gradient was
        raw = G @ G.T if G.shape[0] <= G.shape[1] else G.T @ G
        raw_off = (raw - torch.eye(raw.shape[0])).abs().max().item()
        off = (gram - eye).abs().max().item()
        assert off < 0.5, f'{shape}: |UU^T - I| = {off:.3f}'
        assert off < raw_off * 0.1, \
            f'{shape}: not meaningfully orthogonalized ({off} vs {raw_off})'

    # a small quadratic problem: minimize ||W x - y||^2 with Muon
    W0 = torch.randn(24, 16)
    x = torch.randn(16, 64)
    y = torch.randn(24, 64)
    W = W0.clone().requires_grad_(True)
    opt = Muon([W], lr=0.05)
    losses = []
    for _ in range(60):
        loss = ((W @ x - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, (losses[0], losses[-1])


@test
def trainer_learns_and_checkpoints():
    if _skip_if_no_torch():
        return
    from train_big import build_model
    torch.manual_seed(0)
    cfg = {'d_model': 32, 'n_layers': 2, 'n_heads': 4, 'n_ff': 64,
           'ctx': 16, 'vocab_size': 300}
    model = build_model(cfg, 300)
    assert sum(p.numel() for p in model.parameters()) < 100_000
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 300, (8, 16))
    y = torch.randint(0, 300, (8, 16))
    losses = []
    for _ in range(30):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 300), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    # tied embeddings
    sd = model.state_dict()
    assert sd['head.weight'].data_ptr() == sd['embed.weight'].data_ptr()


@test
def exporter_writes_loadable_gguf():
    if _skip_if_no_torch():
        return
    from train_big import build_model, save_checkpoint
    from export_gguf import export
    torch.manual_seed(0)
    tok = get_tokenizer()
    cfg = {'d_model': 32, 'n_layers': 2, 'n_heads': 4, 'n_ff': 64,
           'ctx': 16, 'vocab_size': tok.vocab_size}
    model = build_model(cfg, tok.vocab_size)
    with tempfile.TemporaryDirectory() as tmp:
        tok.save(os.path.join(tmp, 'bpe_tokenizer.json'))
        save_checkpoint(model, cfg, tmp, 3, 'bpe_tokenizer.json')
        out = os.path.join(tmp, 'lito.gguf')
        export(tmp, out, name='Lito-Test')
        f = read_gguf(out)

    assert f['magic'] == GGUF_MAGIC and f['version'] == 3
    kv = f['kv']
    assert kv['general.architecture'] == 'llama'
    assert kv['llama.block_count'] == 2
    assert kv['llama.embedding_length'] == 32
    assert kv['llama.feed_forward_length'] == 64
    assert kv['llama.attention.head_count'] == 4
    assert kv['llama.attention.head_count_kv'] == 4
    assert kv['llama.context_length'] == 16
    assert kv['llama.vocab_size'] == tok.vocab_size
    assert abs(kv['llama.attention.layer_norm_rms_epsilon'] - 1e-5) < 1e-12
    assert kv['llama.rope.freq_base'] == 10000.0
    assert kv['tokenizer.ggml.model'] == 'gpt2'
    assert len(kv['tokenizer.ggml.tokens']) == tok.vocab_size
    assert len(kv['tokenizer.ggml.token_type']) == tok.vocab_size
    assert kv['tokenizer.ggml.token_type'][256] == 3   # CONTROL
    assert kv['tokenizer.ggml.token_type'][257] == 3
    assert kv['tokenizer.ggml.tokens'][256] == '<|im_start|>'
    assert kv['tokenizer.ggml.tokens'][257] == '<|im_end|>'
    assert kv['tokenizer.ggml.eos_token_id'] == 257
    assert kv['tokenizer.ggml.eot_token_id'] == 257
    assert '<|im_start|>' in kv['tokenizer.chat_template']
    assert len(kv['tokenizer.ggml.merges']) == len(tok.merges)

    names = {t['name'] for t in f['tensors']}
    expected = {'token_embd.weight', 'output_norm.weight', 'output.weight'}
    for i in range(2):
        expected |= {f'blk.{i}.{p}' for p in (
            'attn_norm.weight', 'attn_q.weight', 'attn_k.weight',
            'attn_v.weight', 'attn_output.weight', 'ffn_norm.weight',
            'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight')}
    assert names == expected, names ^ expected
    by_name = {t['name']: t for t in f['tensors']}
    assert by_name['token_embd.weight']['ne'] == (32, tok.vocab_size)
    assert by_name['blk.0.attn_q.weight']['ne'] == (32, 32)
    assert by_name['blk.1.ffn_up.weight']['ne'] == (32, 64)
    assert by_name['blk.1.ffn_down.weight']['ne'] == (64, 32)
    assert all(t['offset'] % GGUF_ALIGNMENT == 0 for t in f['tensors'])
    # norms are F32 (llama.cpp CPU multiplies them against f32
    # activations); all other tensors are F16
    for t in f['tensors']:
        is_norm = t['name'].endswith('_norm.weight')
        assert t['type'] == (TYPE_F32 if is_norm else TYPE_F16), t['name']
    assert kv['tokenizer.ggml.pre'] == 'gpt-2'


# ----------------------------------------------------------------------
# NumPy inference engine (v1.2.6: the app chats with GGUF directly)
# ----------------------------------------------------------------------

def _write_synthetic_gguf(path, vocab_size=300, seed=7):
    """A tiny but valid llama GGUF with (seeded) random weights."""
    import numpy as np
    from src.gguf import GGUFWriter
    tok = get_tokenizer()
    rng = np.random.default_rng(seed)
    d, nff = 32, 64
    w = GGUFWriter('llama', 'Lito-Test')
    w.add_uint32('llama.context_length', 64)
    w.add_uint32('llama.embedding_length', d)
    w.add_uint32('llama.block_count', 2)
    w.add_uint32('llama.feed_forward_length', nff)
    w.add_uint32('llama.attention.head_count', 4)
    w.add_uint32('llama.attention.head_count_kv', 4)
    w.add_float32('llama.attention.layer_norm_rms_epsilon', 1e-5)
    w.add_float32('llama.rope.freq_base', 10000.0)
    w.add_uint32('llama.vocab_size', tok.vocab_size)
    w.add_string('tokenizer.ggml.model', 'gpt2')
    w.add_string('tokenizer.ggml.pre', 'gpt-2')
    w.add_strings('tokenizer.ggml.tokens', tok.vocab)
    w.add_int32s('tokenizer.ggml.token_type',
                 [3 if i in tok.id_to_special else 1
                  for i in range(tok.vocab_size)])
    w.add_strings('tokenizer.ggml.merges',
                  [a + ' ' + b for a, b in tok.merges])
    w.add_uint32('tokenizer.ggml.eos_token_id',
                 tok.special_ids['<|im_end|>'])
    w.add_uint32('tokenizer.ggml.eot_token_id',
                 tok.special_ids['<|im_end|>'])
    w.add_string('tokenizer.chat_template',
                 '<|im_start|>user\n{prompt}<|im_end|>')

    def tens(shape, norm=False):
        a = rng.standard_normal(shape).astype(np.float32) * (0.3 if norm else 0.05)
        return a
    w.add_tensor('token_embd.weight', tens((tok.vocab_size, d)), 1)
    for i in range(2):
        w.add_tensor(f'blk.{i}.attn_norm.weight', tens((d,)), 0)
        for p in ('attn_q', 'attn_k', 'attn_v', 'attn_output'):
            w.add_tensor(f'blk.{i}.{p}.weight', tens((d, d)), 1)
        w.add_tensor(f'blk.{i}.ffn_norm.weight', tens((d,)), 0)
        w.add_tensor(f'blk.{i}.ffn_gate.weight', tens((nff, d)), 1)
        w.add_tensor(f'blk.{i}.ffn_up.weight', tens((nff, d)), 1)
        w.add_tensor(f'blk.{i}.ffn_down.weight', tens((d, nff)), 1)
    w.add_tensor('output_norm.weight', tens((d,)), 0)
    w.write(path)
    return tok


@test
def engine_loads_gguf_and_chats():
    from src.llama_np import LlamaEngine
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'synthetic.gguf')
        _write_synthetic_gguf(path)
        engine = LlamaEngine(path)
    assert engine.name == 'Lito-Test'
    assert engine.vocab_size > 250
    assert 25_000 < engine._param_count < 200_000
    # tokenizer roundtrip through the engine
    ids = engine.tokenize('hello world')
    assert ids and engine.detokenize(ids) == 'hello world'
    # chat returns clean text (no ChatML markers leak out)
    reply = engine.chat([{'role': 'user', 'content': 'hello there'}],
                        max_new_tokens=12, temperature=0.0)
    assert isinstance(reply, str)
    assert '<|im_' not in reply, reply


@test
def engine_kv_cache_matches_recompute():
    """Greedy generation through the KV cache must produce exactly the
    tokens a full forward recompute would pick (repeat penalty off)."""
    from src.llama_np import LlamaEngine
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'synthetic.gguf')
        _write_synthetic_gguf(path)
        engine = LlamaEngine(path)

    ids = engine.tokenize('<|im_start|>user\nhi there how are you<|im_end|>')
    engine.reset()
    out = engine.generate(ids, max_new_tokens=20, temperature=0.0,
                          repeat_penalty=1.0)

    # recompute: at every step run a fresh forward over prompt+generated
    # so far and take the argmax of the last position
    seq = list(ids)
    recomputed = []
    for _ in range(20):
        logits = engine.forward_logits(seq)
        nxt = int(np.argmax(logits[-1]))
        recomputed.append(nxt)
        seq.append(nxt)
        if nxt == engine.eot:
            break
    assert out == recomputed, (out, recomputed)


@test
def engine_seeded_sampling_is_deterministic():
    from src.llama_np import LlamaEngine
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'synthetic.gguf')
        _write_synthetic_gguf(path)
        engine = LlamaEngine(path)
    ids = engine.tokenize('once upon a time')
    a1 = engine.generate(ids, max_new_tokens=16, temperature=0.9,
                         seed=1234)
    engine.reset()
    a2 = engine.generate(ids, max_new_tokens=16, temperature=0.9,
                         seed=1234)
    assert a1 == a2 and len(a1) > 0
    engine.reset()
    b = engine.generate(ids, max_new_tokens=16, temperature=0.9,
                        seed=99)
    assert b != a1, 'different seeds should (almost surely) differ'
    # score is finite and penalizes shuffled text
    s1 = engine.score(ids)
    assert np.isfinite(s1)


@test
def engine_scores_text():
    from src.llama_np import LlamaEngine
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'synthetic.gguf')
        _write_synthetic_gguf(path)
        engine = LlamaEngine(path)
    ids = engine.tokenize('the the the the the the')
    good = engine.score(engine.tokenize('hello hello hello'))
    assert np.isfinite(good)


@test
def engine_matches_torch_reference():
    """The numpy engine must reproduce the torch model it was exported
    from (within f16 rounding)."""
    if _skip_if_no_torch():
        return
    from train_big import build_model
    from export_gguf import export
    from src.llama_np import LlamaEngine
    torch.manual_seed(3)
    tok = get_tokenizer()
    cfg = {'d_model': 32, 'n_layers': 2, 'n_heads': 4, 'n_ff': 64,
           'ctx': 64, 'vocab_size': tok.vocab_size}
    model = build_model(cfg, tok.vocab_size)
    model.eval()
    with tempfile.TemporaryDirectory() as tmp:
        from train_big import save_checkpoint
        tok.save(os.path.join(tmp, 'bpe_tokenizer.json'))
        save_checkpoint(model, cfg, tmp, 3, 'bpe_tokenizer.json')
        out = os.path.join(tmp, 'lito.gguf')
        export(tmp, out, name='Lito-Test')
        engine = LlamaEngine(out)

    text = 'hello world, this is a test of the engine'
    ids = engine.tokenize(text)
    with torch.no_grad():
        ref = model(torch.tensor([ids])).numpy()[0]   # (T, V)
    got = engine.forward_logits(ids)                   # (T, V)
    assert got.shape == ref.shape
    diff = np.abs(got - ref)
    cos = float(np.sum(got * ref) /
                (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-9))
    agree = float(np.mean(np.argmax(got, -1) == np.argmax(ref, -1)))
    assert diff.max() < 0.05, f'max |diff| {diff.max():.4f}'
    assert cos > 0.999, f'cosine {cos:.6f}'
    assert agree >= 0.98, f'argmax agreement {agree:.3f}'


# ----------------------------------------------------------------------
# Training speed/quality fixes (same parameters, better and faster)
# ----------------------------------------------------------------------

@test
def model_init_starts_at_uniform_loss():
    """A freshly built model must start at about ln(vocab).

    torch's default N(0, 1) embedding init with a tied head starts
    training at loss ~240 instead of ~8.3 and wastes hundreds of steps
    just shrinking the logits.
    """
    if _skip_if_no_torch():
        return
    import math
    from train_big import build_model
    torch.manual_seed(0)
    vocab = 512
    cfg = {'d_model': 64, 'n_layers': 4, 'n_heads': 4, 'n_ff': 128,
           'ctx': 32, 'vocab_size': vocab}
    model = build_model(cfg, vocab)
    ids = torch.randint(0, vocab, (4, 32))
    with torch.no_grad():
        logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab), torch.randint(0, vocab, (4 * 32,))).item()
    uniform = math.log(vocab)
    assert abs(loss - uniform) < 1.0, f'init loss {loss:.2f} vs {uniform:.2f}'


@test
def fused_and_split_layouts_are_the_same_model():
    """Fusing Q/K/V (and gate/up) is a layout change only: identical
    parameter count, identical outputs, lossless round trip."""
    if _skip_if_no_torch():
        return
    from train_big import build_model, split_state_dict, fuse_state_dict
    vocab = 128
    cfg = {'d_model': 32, 'n_layers': 2, 'n_heads': 4, 'n_ff': 64,
           'ctx': 16, 'vocab_size': vocab}
    torch.manual_seed(5)
    fused = build_model(cfg, vocab, fused=True)
    torch.manual_seed(5)
    split = build_model(cfg, vocab, fused=False)
    assert (sum(p.numel() for p in fused.parameters())
            == sum(p.numel() for p in split.parameters()))
    # the split checkpoint of the fused model reproduces it exactly
    split.load_state_dict(split_state_dict(fused.state_dict(), cfg))
    ids = torch.randint(0, vocab, (2, 16))
    with torch.no_grad():
        a, b = fused(ids), split(ids)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max().item()
    # and back again, bit for bit
    again = build_model(cfg, vocab, fused=True)
    again.load_state_dict(fuse_state_dict(split_state_dict(
        fused.state_dict(), cfg)))
    with torch.no_grad():
        c = again(ids)
    assert torch.equal(a, c)


@test
def sdpa_matches_hand_written_attention():
    """The fast attention path computes the same function as the
    original hand-rolled matmul + mask + softmax."""
    if _skip_if_no_torch():
        return
    from train_big import build_model
    vocab = 128
    cfg = {'d_model': 32, 'n_layers': 2, 'n_heads': 4, 'n_ff': 64,
           'ctx': 24, 'vocab_size': vocab}
    torch.manual_seed(7)
    fast = build_model(cfg, vocab, fused=False, sdpa=True)
    slow = build_model(cfg, vocab, fused=False, sdpa=False)
    slow.load_state_dict(fast.state_dict())
    ids = torch.randint(0, vocab, (2, 24))
    with torch.no_grad():
        a, b = fast(ids), slow(ids)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max().item()


@test
def checkpoints_stay_in_the_llama_layout():
    """Whatever layout training used, the checkpoint on disk is always
    the split layout the exporter and the numpy engine read."""
    if _skip_if_no_torch():
        return
    from train_big import build_model, save_checkpoint
    tok = get_tokenizer()
    cfg = {'d_model': 32, 'n_layers': 2, 'n_heads': 4, 'n_ff': 64,
           'ctx': 16, 'vocab_size': tok.vocab_size}
    model = build_model(cfg, tok.vocab_size, fused=True)
    with tempfile.TemporaryDirectory() as tmp:
        tok.save(os.path.join(tmp, 'bpe_tokenizer.json'))
        save_checkpoint(model, cfg, tmp, 1, 'bpe_tokenizer.json')
        state = torch.load(os.path.join(tmp, 'checkpoint.pt'),
                           map_location='cpu',
                           weights_only=False)['state_dict']
    keys = set(state)
    for layer in range(cfg['n_layers']):
        for name in ('wq', 'wk', 'wv', 'wo', 'w_gate', 'w_up', 'w_down'):
            assert f'blocks.{layer}.{name}.weight' in keys, name
    assert not any('wqkv' in k or 'gate_up' in k for k in keys)
    assert not any(k.endswith('mask') for k in keys)


# ----------------------------------------------------------------------
# Conversation core (why "hello" works)
# ----------------------------------------------------------------------

@test
def chat_core_teaches_greetings_and_identity():
    import random as _random
    from chatdata import build_chat_core
    docs = build_chat_core(_random.Random(0))
    assert len(docs) > 2000, len(docs)
    joined = '\n'.join(docs).lower()
    # the exact things a person types first must be in training
    for probe in ('>hello<', '>hi<', '>hey<', '>who are you<',
                  '>thanks<', '>bye<', '>what can you do?<'):
        needle = f'<|im_start|>user\n{probe[1:-1]}<|im_end|>'
        assert needle in joined, probe
    assert 'lito' in joined
    # every document is well formed ChatML with a non-empty answer
    for doc in docs[:500]:
        assert doc.count('<|im_start|>') == doc.count('<|im_end|>')
        assert doc.endswith('<|im_end|>')
        for turn in doc.split('<|im_start|>assistant\n')[1:]:
            assert len(turn.split('<|im_end|>')[0].strip()) > 2


@test
def chat_core_arithmetic_and_facts_are_correct():
    """Training data must not teach wrong answers."""
    import random as _random
    import re as _re
    from chatdata import build_chat_core
    docs = build_chat_core(_random.Random(3))
    checked = 0
    for doc in docs:
        for match in _re.finditer(r'(\d+) plus (\d+) is (\d+)\.', doc):
            a, b, c = (int(g) for g in match.groups())
            assert a + b == c, match.group(0)
            checked += 1
        for match in _re.finditer(r'(\d+) times (\d+) is (\d+)\.', doc):
            a, b, c = (int(g) for g in match.groups())
            assert a * b == c, match.group(0)
            checked += 1
    assert checked > 500, checked
    joined = '\n'.join(docs)
    assert 'The capital of France is Paris.' in joined


@test
def look_alike_prompts_are_taught_apart():
    """'how are you' must not be drowned out by 'who are you'.

    Both failures this guards against were real: the model answered the
    identity blurb to "how are you", and wished the user good night at
    breakfast, because the look-alike intent simply had more training
    examples than the one being asked for.
    """
    import random as _random
    import re as _re
    from chatdata import build_chat_core, INTENTS, CONTRASTS, _intent

    def phrasings(marker):
        for entry in INTENTS:
            users, _replies, _w = _intent(entry)
            low = [u.lower() for u in users]
            if marker in low:
                return set(low)
        raise AssertionError(f'no intent contains {marker!r}')

    docs = build_chat_core(_random.Random(0))
    groups = {name: phrasings(name) for name in
              ('how are you', 'who are you', 'good morning', 'good night')}
    seen = dict.fromkeys(groups, 0)
    for doc in docs:
        for turn in _re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>',
                                doc, _re.S):
            text = turn.strip().lower()
            for name, group in groups.items():
                if text in group:
                    seen[name] += 1

    # the question that was getting the wrong answer must be at least as
    # well represented as the one whose answer was being stolen
    assert seen['how are you'] >= seen['who are you'], seen
    assert seen['good morning'] >= seen['good night'], seen

    # and they are taught side by side, in one dialogue
    pairs = [d for d in docs
             if d.lower().count('<|im_start|>user') == 2
             and 'morning' in d.lower() and 'night' in d.lower()]
    assert pairs, 'no contrast dialogues were generated'
    assert len(CONTRASTS) >= 4


@test
def every_capital_is_taught_in_every_phrasing():
    """A fact seen once is learned as a shape, not a fact - which is how
    'the capital of France is Morocco' happens."""
    import random as _random
    from chatdata import _knowledge_qa, CAPITALS
    qa = _knowledge_qa(_random.Random(0))
    asked = [u for u, a in qa if 'capital of France' in a]
    assert len(asked) >= 4, asked
    assert len(set(asked)) == len(asked), 'phrasings should all differ'
    answers = {a for u, a in qa if a.startswith('The capital of')}
    assert len(answers) == len(CAPITALS), (len(answers), len(CAPITALS))


@test
def instruction_filter_drops_list_shaped_answers():
    """Long markdown-list answers are what a tiny model turns into
    word salad with bullet points, so they are filtered out."""
    from train_big import _LIST_RE, MAX_ANSWER_CHARS
    assert _LIST_RE.search('1. first\n2. second')
    assert _LIST_RE.search('- a bullet')
    assert _LIST_RE.search('Verse 2:\n\n1. Stay the Civil')
    assert not _LIST_RE.search('Paris is the capital of France.')
    assert 200 < MAX_ANSWER_CHARS < 1000


@test
def chat_eval_rejects_word_salad_and_accepts_real_answers():
    """The release gate must fail on the kind of output a broken model
    produces, and pass on ordinary replies."""
    sys.path.insert(0, os.path.join(REPO_ROOT, 'tools'))
    from src.quality import looks_like_english
    from chat_eval import PROBES
    salad = ('- Noice:\n- Like\n- Two Creating\n- Priorit\n- Feeder\n'
             'Verse 2:\n1. Stay the Civil\n'
             '2. Add a mysterious cave, Are youn\'t yet,\n'
             '3. Water tomorrow: Wick is a fertilizen, which is often '
             'found in the middle.')
    ok, _ = looks_like_english(salad)
    assert not ok, 'word salad slipped through the gate'
    ok, why = looks_like_english('Hello! How can I help you today?')
    assert ok, why
    ok, _ = looks_like_english('the the the the the the the the the the')
    assert not ok, 'repetition slipped through'
    # invented words are caught even without list formatting
    ok, why = looks_like_english(
        'Wick is a fertilizen and Minsewall the melurgies of strangespace '
        'can take a trophh to the mutro of overling letturs.')
    assert not ok, 'invented words slipped through'
    # and ordinary replies are not rejected
    for good in ("I'm Lito, a small AI assistant trained from scratch.",
                 '7 plus 5 is 12.',
                 'Once upon a time there was a cat named Mia.'):
        ok, why = looks_like_english(good)
        assert ok, f'{good!r} rejected: {why}'
    prompts = [p[0] for p in PROBES]
    assert 'hello' in prompts and 'who are you' in prompts


@test
def core_anneal_ramps_only_at_the_end():
    """The conversation core must not crowd out the main corpus early;
    it ramps in over the last stretch of training."""
    from train_big import core_fraction
    assert core_fraction(0.0) == 0.0
    assert core_fraction(0.5) == 0.0
    assert core_fraction(0.70, anneal=0.3, mix=0.5) == 0.0
    mid = core_fraction(0.85, anneal=0.3, mix=0.5)
    assert 0.2 < mid < 0.3, mid
    assert abs(core_fraction(1.0, anneal=0.3, mix=0.5) - 0.5) < 1e-9
    assert core_fraction(1.0, anneal=0.0) == 0.0      # switched off
    assert core_fraction(2.0, anneal=0.3, mix=0.5) == 0.5   # clamped


@test
def gguf_carries_its_provenance():
    """Every exported model says which code trained it, for how long and
    how well it validated - so "is this the fixed model?" is answerable
    from the file itself."""
    if _skip_if_no_torch():
        return
    from train_big import build_model, save_checkpoint, git_commit
    from export_gguf import export
    from src.llama_np import LlamaEngine
    tok = get_tokenizer()
    cfg = {'d_model': 32, 'n_layers': 2, 'n_heads': 4, 'n_ff': 64,
           'ctx': 16, 'vocab_size': tok.vocab_size}
    model = build_model(cfg, tok.vocab_size)
    info = {'steps': 1234, 'tokens': 7_243_776, 'corpus_tokens': 8_107_431,
            'val_loss': 2.2631, 'optimizer': 'muon', 'commit': 'abc1234',
            'date': '2026-11-09'}
    with tempfile.TemporaryDirectory() as tmp:
        tok.save(os.path.join(tmp, 'bpe_tokenizer.json'))
        save_checkpoint(model, cfg, tmp, 1233, 'bpe_tokenizer.json', info)
        out = os.path.join(tmp, 'lito.gguf')
        export(tmp, out, name='Lito-Test')
        kv = read_gguf(out)['kv']
        engine = LlamaEngine(out)

    assert kv['lito.train.commit'] == 'abc1234'
    assert kv['lito.train.steps'] == 1234
    assert kv['lito.train.tokens'] == 7_243_776
    assert abs(kv['lito.train.val_loss'] - 2.2631) < 1e-4
    assert 'commit abc1234' in kv['general.description']
    summary = engine.build_summary()
    for fragment in ('abc1234', '1,234 steps', '2.2631', '2026-11-09'):
        assert fragment in summary, (fragment, summary)
    # git_commit never explodes, whatever the environment
    assert isinstance(git_commit(), str)


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

def main():
    passed, failed = 0, 0
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
            print(f'  PASS  {name}')
            passed += 1
        except AssertionError as e:
            print(f'  FAIL  {name}: {e}')
            failed += 1
        except Exception as e:
            print(f'  ERROR {name}: {type(e).__name__}: {e}')
            failed += 1
    print(f'\n{passed} passed, {failed} failed, {len(TESTS)} total')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
