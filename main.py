#!/usr/bin/env python3
"""
SmartAI - An AI built entirely from scratch.

Usage:
    python main.py                     # Start interactive chat (auto-detects model)
    python main.py --download          # Download latest model from GitHub release
    python main.py --repo owner/repo   # Specify repository to download model from
    python main.py --train             # Train the model locally first
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_model_exists(save_dir: str) -> bool:
    """Check if model weights and tokenizer exist."""
    model_path = os.path.join(save_dir, 'best_model.npz')
    if not os.path.exists(model_path):
        model_path = os.path.join(save_dir, 'final_model.npz')
    tokenizer_path = os.path.join(save_dir, 'tokenizer.json')
    return os.path.exists(model_path) and os.path.exists(tokenizer_path)


def prompt_missing_model(save_dir: str, repo: str = '') -> bool:
    """Prompt user when no model is found locally."""
    print("\n" + "=" * 60)
    print("  No trained model found in:", os.path.abspath(save_dir))
    print("=" * 60)
    print("  [1] Download latest trained model from GitHub Releases")
    print("  [2] Train model locally from scratch")
    print("  [3] Continue with built-in Knowledge Base only")
    print("  [4] Exit")
    print("=" * 60)

    try:
        choice = input("Select an option [1-4] (default: 1): ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        return False

    if choice == "1":
        from src.downloader import download_model_from_release
        print("\nDownloading model...")
        success = download_model_from_release(repo=repo, save_dir=save_dir)
        if not success:
            print("\nDownload failed or release not found.")
            retry = input("Train locally instead? (Y/n): ").strip().lower()
            if retry != 'n':
                return trigger_training(save_dir)
        return success

    elif choice == "2":
        return trigger_training(save_dir)

    elif choice == "3":
        return True

    else:
        sys.exit(0)


def trigger_training(save_dir: str) -> bool:
    """Trigger local model training."""
    from train import main as train_main
    print("\nStarting local training...")
    sys.argv = ['train.py', '--epochs', '30', '--save_dir', save_dir]
    try:
        train_main()
        return True
    except Exception as e:
        print(f"Training failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='SmartAI - AI from Scratch')
    parser.add_argument('--train', action='store_true', help='Train the model locally before chatting')
    parser.add_argument('--download', action='store_true', help='Force download latest model from GitHub')
    parser.add_argument('--model-url', type=str, default='', help='Direct URL to model.zip')
    parser.add_argument('--repo', type=str, default='', help='GitHub repository (owner/name) to download model from')
    parser.add_argument('--token', type=str, default='', help='GitHub personal access token (optional)')
    parser.add_argument('--dir', type=str, default='checkpoints', help='Model checkpoints directory')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs if training locally')
    parser.add_argument('--chat-only', action='store_true', help='Skip check and use knowledge base only')
    args = parser.parse_args()

    # Step 1: Handle manual download flag
    if args.download:
        from src.downloader import download_model_from_release, download_model_from_url
        if args.model_url:
            download_model_from_url(args.model_url, save_dir=args.dir)
        else:
            download_model_from_release(repo=args.repo, save_dir=args.dir, token=args.token)

    # Step 2: Handle local train flag
    elif args.train:
        from train import main as train_main
        sys.argv = ['train.py', '--epochs', str(args.epochs), '--save_dir', args.dir]
        train_main()
        print("\nTraining complete! Starting chat...\n")

    # Step 3: Check if model exists, prompt if not
    elif not args.chat_only and not check_model_exists(args.dir):
        prompt_missing_model(args.dir, repo=args.repo)

    # Step 4: Start Chat Interface
    from src.chat import ChatInterface
    chat = ChatInterface(model_dir=args.dir)
    chat.start()


if __name__ == '__main__':
    main()