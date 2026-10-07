#!/usr/bin/env python3
"""
Train the SmartAI model from scratch.

Usage:
    python train.py [--epochs N] [--lr RATE] [--d_model DIM] [--n_layers N]
                    [--n_heads N] [--batch_size N] [--seq_len N]
"""

import argparse
import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tokenizer import BPETokenizer
from src.model import TransformerLM
from src.trainer import Trainer
from src.dataset import TrainingData


def parse_args():
    parser = argparse.ArgumentParser(description='Train SmartAI from scratch')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length')
    parser.add_argument('--vocab_size', type=int, default=2048, help='Target vocabulary size')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Max sequence length')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--warmup_steps', type=int, default=50, help='Warmup steps')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  SmartAI Training Pipeline")
    print("  Built from Scratch - No Pretrained Models")
    print("=" * 60)

    # Seed for reproducibility
    np.random.seed(42)

    # Step 1: Build tokenizer
    print("\n[1/4] Building tokenizer...")
    corpus = TrainingData.get_training_corpus()
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.build_vocab(corpus)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Step 2: Create model
    print(f"\n[2/4] Creating model...")
    print(f"  Config: d_model={args.d_model}, n_layers={args.n_layers}, "
          f"n_heads={args.n_heads}, d_ff={args.d_ff}")

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )

    # Step 3: Train
    print(f"\n[3/4] Training for {args.epochs} epochs...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
    )

    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        log_interval=5,
        save_interval=50,
        warmup_steps=args.warmup_steps,
    )

    # Step 4: Verify
    print(f"\n[4/4] Verifying model...")
    test_prompts = [
        "Hello",
        "What is",
        "The sun",
        "Machine learning",
    ]

    model.eval_mode()
    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = np.array(tokens).reshape(1, -1)
        try:
            output_ids = model.generate(input_ids, max_new_tokens=30, temperature=0.7, top_k=30)
            text = tokenizer.decode(output_ids.tolist())
            print(f"  '{prompt}' -> '{text[:80]}'")
        except Exception as e:
            print(f"  '{prompt}' -> Error: {e}")

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Model saved to: {args.save_dir}/")
    print(f"  Run 'python main.py' to chat with your AI")
    print("=" * 60)


if __name__ == '__main__':
    main()