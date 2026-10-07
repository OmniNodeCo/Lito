#!/usr/bin/env python3
"""
SmartAI - An AI built entirely from scratch.

Usage:
    python main.py          # Start interactive chat
    python main.py --train  # Train the model first
    python main.py --dir checkpoints  # Specify model directory
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='SmartAI - AI from Scratch')
    parser.add_argument('--train', action='store_true', help='Train the model before chatting')
    parser.add_argument('--dir', type=str, default='checkpoints', help='Model directory')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--chat-only', action='store_true', help='Skip training, chat with KB only')
    args = parser.parse_args()

    if args.train:
        print("Starting training pipeline...")
        from train import main as train_main
        sys.argv = ['train.py', '--epochs', str(args.epochs), '--save_dir', args.dir]
        train_main()
        print("\nTraining complete! Starting chat...\n")

    from src.chat import ChatInterface
    chat = ChatInterface(model_dir=args.dir)
    chat.start()


if __name__ == '__main__':
    main()