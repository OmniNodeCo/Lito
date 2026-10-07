"""Interactive chat interface."""

import sys
import os
from .brain import AIBrain


class ChatInterface:
    """Terminal-based chat interface for the AI."""

    BANNER = r"""
  ____                       _      _    ___
 / ___| _ __ ___   __ _ _ __| |_   / \  |_ _|
 \___ \| '_ ` _ \ / _` | '__| __| / _ \  | |
  ___) | | | | | | (_| | |  | |_ / ___ \ | |
 |____/|_| |_| |_|\__,_|_|   \__/_/   \_\___|

    AI Built From Scratch - No Pretrained Models
    =============================================
    """

    COMMANDS = {
        '/quit': 'Exit the chat',
        '/reset': 'Clear conversation history',
        '/help': 'Show available commands',
        '/info': 'Show model information',
        '/generate': 'Generate text from a prompt',
    }

    def __init__(self, model_dir: str = 'checkpoints'):
        self.brain = AIBrain(model_dir=model_dir)

    def start(self):
        """Start the interactive chat."""
        print(self.BANNER)
        print("Initializing AI brain...")
        self.brain.initialize()

        print("\nType your message and press Enter. Type /help for commands.\n")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye! 👋")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                if self._handle_command(user_input):
                    continue
                else:
                    break

            # Get AI response
            response = self.brain.think(user_input)
            print(f"\n🤖 AI: {response}")

    def _handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns False to quit."""
        cmd = command.lower().split()[0]

        if cmd == '/quit' or cmd == '/exit':
            print("\nGoodbye! Thanks for chatting! 👋")
            return False

        elif cmd == '/reset':
            self.brain.reset_conversation()
            print("✅ Conversation reset.")

        elif cmd == '/help':
            print("\n📋 Available Commands:")
            for cmd_name, desc in self.COMMANDS.items():
                print(f"  {cmd_name:15s} - {desc}")

        elif cmd == '/info':
            self._show_info()

        elif cmd == '/generate':
            parts = command.split(' ', 1)
            if len(parts) > 1:
                prompt = parts[1]
                print(f"\n📝 Generating from: '{prompt}'")
                response = self.brain._generate_with_model(prompt, max_tokens=100)
                if response:
                    print(f"   Result: {response}")
                else:
                    print("   (Neural generation unavailable - model not loaded)")
            else:
                print("Usage: /generate <prompt text>")

        else:
            print(f"Unknown command: {cmd}. Type /help for available commands.")

        return True

    def _show_info(self):
        """Display model information."""
        print("\n📊 Model Information:")
        print(f"  Model loaded: {'Yes ✅' if self.brain.loaded else 'No ❌'}")

        if self.brain.model is not None:
            model = self.brain.model
            print(f"  Architecture: Transformer (GPT-style)")
            print(f"  Vocabulary size: {model.vocab_size:,}")
            print(f"  Embedding dim: {model.d_model}")
            print(f"  Attention heads: {model.n_heads}")
            print(f"  Transformer layers: {model.n_layers}")
            print(f"  Feed-forward dim: {model.d_ff}")
            print(f"  Max sequence length: {model.max_seq_len}")
            print(f"  Total parameters: {model._param_count:,}")

        if self.brain.tokenizer is not None:
            print(f"  Tokenizer vocab: {self.brain.tokenizer.vocab_size:,}")

        print(f"  Conversation turns: {len(self.brain.conversation_history)}")


def main():
    """Entry point for the chat interface."""
    model_dir = 'checkpoints'
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]

    chat = ChatInterface(model_dir=model_dir)
    chat.start()


if __name__ == '__main__':
    main()