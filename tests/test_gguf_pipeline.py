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
