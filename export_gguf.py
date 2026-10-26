#!/usr/bin/env python3
"""
Export a train_big.py checkpoint to a GGUF file that runs offline in
llama.cpp, LM Studio, Ollama, etc.

    python export_gguf.py                     # checkpoints-big -> lito-chat.gguf

Tensor names and metadata follow the "llama" architecture in llama.cpp.
Weights are stored as f16. Because the model uses the same vocabulary
layout as GPT-2 byte-level BPE (byte-to-unicode trick, <|im_start|> /
<|im_end|> control tokens), the file self-describes as
tokenizer.ggml.model=gpt2 with the tokens + merges embedded, plus a
ChatML chat template so chat front-ends know how to prompt it.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.bpe import ByteBPE  # noqa: E402
from src.gguf import GGUFWriter, TYPE_F16, TYPE_F32  # noqa: E402

CHAT_TEMPLATE = (
    '{%- for message in messages %}'
    '{{ "<|im_start|>" + message["role"] + "\\n" + message["content"] + "<|im_end|>\\n" }}'
    '{%- endfor %}'
    '{%- if add_generation_prompt %}'
    '{{ "<|im_start|>assistant\\n" }}'
    '{%- endif %}'
)


def export(model_dir, out_path, name='Lito-Chat'):
    import torch

    ckpt = torch.load(os.path.join(model_dir, 'checkpoint.pt'),
                      map_location='cpu', weights_only=False)
    config = ckpt['config']
    state = ckpt['state_dict']
    step = ckpt.get('step', '?')

    tok = ByteBPE.load(os.path.join(model_dir, 'bpe_tokenizer.json'))
    n_vocab = max(tok.vocab_size, len(tok.vocab))

    # ----------------------------- tensors ---------------------------
    writer = GGUFWriter('llama', name)

    def tensor(key, torch_key, tensor_type=TYPE_F16):
        writer.add_tensor(key, state[torch_key].float().numpy(), tensor_type)

    # llama.cpp convention: normalization weights stay F32 (the CPU
    # backend multiplies them against f32 activations); everything else
    # is f16.
    def norm(key, torch_key):
        tensor(key, torch_key, TYPE_F32)

    writer.add_tensor('token_embd.weight', state['embed.weight'].float().numpy(), TYPE_F16)
    for i in range(config['n_layers']):
        p = f'blocks.{i}.'
        g = f'blk.{i}.'
        norm(g + 'attn_norm.weight', p + 'attn_norm.weight')
        tensor(g + 'attn_q.weight', p + 'wq.weight')
        tensor(g + 'attn_k.weight', p + 'wk.weight')
        tensor(g + 'attn_v.weight', p + 'wv.weight')
        tensor(g + 'attn_output.weight', p + 'wo.weight')
        norm(g + 'ffn_norm.weight', p + 'ffn_norm.weight')
        tensor(g + 'ffn_gate.weight', p + 'w_gate.weight')
        tensor(g + 'ffn_up.weight', p + 'w_up.weight')
        tensor(g + 'ffn_down.weight', p + 'w_down.weight')
    norm('output_norm.weight', 'norm.weight')
    # output head is tied to the embedding; llama.cpp likes it present
    writer.add_tensor('output.weight', state['embed.weight'].float().numpy(), TYPE_F16)

    # ---------------------------- metadata ---------------------------
    w = writer
    w.add_string('general.name', name)
    w.add_string('general.source', 'https://github.com/OmniNodeCo/Lito')
    w.add_string('general.description',
                 'A small llama-architecture chat model trained from scratch '
                 '(no pretrained weights) by the Lito project.')
    w.add_uint32('general.file_type', 1)  # mostly f16
    w.add_string('llama.tensor_data_layout', 'Meta AI original pth')

    w.add_uint32('llama.context_length', config['ctx'])
    w.add_uint32('llama.embedding_length', config['d_model'])
    w.add_uint32('llama.block_count', config['n_layers'])
    w.add_uint32('llama.feed_forward_length', config['n_ff'])
    w.add_uint32('llama.attention.head_count', config['n_heads'])
    w.add_uint32('llama.attention.head_count_kv', config['n_heads'])  # MHA
    w.add_uint32('llama.attention.key_length',
                 config['d_model'] // config['n_heads'])
    w.add_uint32('llama.attention.value_length',
                 config['d_model'] // config['n_heads'])
    w.add_float32('llama.attention.layer_norm_rms_epsilon', 1e-5)
    w.add_float32('llama.rope.freq_base', 10000.0)
    w.add_uint32('llama.vocab_size', n_vocab)

    # tokenizer (GPT-2 byte-level BPE, byte-unicode encoded)
    im_end = tok.encode('<|im_end|>')[0]
    tokens = []
    for i in range(n_vocab):
        if i in tok.id_to_special:
            tokens.append(tok.id_to_special[i])
        else:
            word = tok.vocab[i]
            tokens.append(word)  # already in byte-unicode form
    token_types = []
    for i in range(n_vocab):
        token_types.append(3 if i in tok.id_to_special else 1)

    w.add_string('tokenizer.ggml.model', 'gpt2')
    w.add_string('tokenizer.ggml.pre', 'gpt-2')
    w.add_strings('tokenizer.ggml.tokens', tokens)
    w.add_int32s('tokenizer.ggml.token_type', token_types)
    w.add_strings('tokenizer.ggml.merges', [a + ' ' + b for a, b in tok.merges])

    w.add_bool('tokenizer.ggml.add_bos_token', False)
    w.add_bool('tokenizer.ggml.add_eos_token', False)
    w.add_uint32('tokenizer.ggml.eos_token_id', im_end)
    w.add_uint32('tokenizer.ggml.bos_token_id', im_end)
    w.add_uint32('tokenizer.ggml.eot_token_id', im_end)
    w.add_uint32('tokenizer.ggml.unknown_token_id', 0)
    w.add_string('tokenizer.chat_template', CHAT_TEMPLATE)

    writer.write(out_path)
    mb = os.path.getsize(out_path) / 1e6
    n_tensors = len(writer.tensors)
    print(f'exported checkpoint (step {step}) -> {out_path}')
    print(f'  {n_tensors} tensors, {n_vocab} vocab, {mb:.1f} MB, f16')
    print(f'  vocab: {n_vocab} tokens, <|im_end|> id = {im_end}')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Export checkpoint -> GGUF')
    parser.add_argument('--model_dir', type=str,
                        default=os.path.join(ROOT, 'checkpoints-big'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(ROOT, 'lito-chat.gguf'))
    args = parser.parse_args()
    export(args.model_dir, args.out)


if __name__ == '__main__':
    main()
