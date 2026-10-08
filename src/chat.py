"""
Interactive chat interface.
"""

import sys
import os
from .brain import AIBrain
from .version import APP_NAME, __version__, format_version


class ChatInterface:
    """Terminal-based chat interface for the AI."""

    BANNER = rf"""
 _       ___    _____     ___
| |     |_ _|  |_   _|   / _ \
| |      | |     | |    | | | |
| |___   | |     | |    | |_| |
|_____|  |___|   |_|     \___/

    {APP_NAME} {format_version(__version__)}
    AI Built From Scratch - No Pretrained Models
    =============================================
    Dictionary: 370k words + WordNet definitions + learns new words
    Web search: Wikipedia + DuckDuckGo (no API keys)
    """

    COMMANDS = {
        '/quit': 'Exit the chat',
        '/reset': 'Clear conversation history',
        '/help': 'Show available commands',
        '/info': 'Show model, dictionary and version information',
        '/generate': 'Generate text from a prompt',
        '/search': '/search <query> - search the web; /search on|off - toggle auto-search',
        '/define': '/define <word> - full dictionary entry (definitions, examples, synonyms)',
        '/spell': '/spell <word> - check spelling and get suggestions',
        '/synonyms': '/synonyms <word> - list synonyms',
        '/words': '/words <prefix> - autocomplete words from the dictionary',
        '/learned': 'Show words and terms the AI learned from the web',
    }

    def __init__(self, model_dir: str = 'checkpoints', search_enabled: bool = True,
                 dictionary_online: bool = True):
        self.brain = AIBrain(model_dir=model_dir, search_enabled=search_enabled,
                             dictionary_online=dictionary_online)

    def start(self):
        """Start the interactive chat."""
        print(self.BANNER)
        print("Initializing AI brain...")
        self.brain.initialize()

        print("\nType your message and press Enter. Type /help for commands.")
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
        parts = command.split(' ', 1)
        argument = parts[1].strip() if len(parts) > 1 else ''

        if cmd in ('/quit', '/exit'):
            print("\nGoodbye! Thanks for chatting! 👋")
            return False

        elif cmd == '/reset':
            self.brain.reset_conversation()
            print("✅ Conversation reset.")

        elif cmd == '/help':
            print("\n📋 Available Commands:")
            for cmd_name, desc in self.COMMANDS.items():
                print(f"  {cmd_name:12s} - {desc}")

        elif cmd == '/info':
            self._show_info()

        elif cmd == '/generate':
            if argument:
                print(f"\n📝 Generating from: '{argument}'")
                response = self.brain._generate_with_model(argument, max_tokens=100)
                if response:
                    print(f"   Result: {response}")
                else:
                    print("   (Neural generation unavailable - model not loaded)")
            else:
                print("Usage: /generate <prompt text>")

        elif cmd == '/search':
            self._handle_search(argument)

        elif cmd == '/define':
            if argument:
                print(f"\n📖 Looking up '{argument}'...\n")
                print(self.brain.dictionary.format_entry(argument))
            else:
                print("Usage: /define <word>")

        elif cmd == '/spell':
            if argument:
                word = argument.split()[0].lower().strip('.,!?')
                if self.brain.dictionary.is_word(word):
                    print(f"\n✅ '{word}' is spelled correctly ({len(word)} letters).")
                else:
                    suggestions = self.brain.dictionary.suggest(word)
                    if suggestions:
                        print(f"\n❌ '{word}' is not in the dictionary. "
                              f"Did you mean: {', '.join(suggestions)}?")
                    else:
                        print(f"\n❌ '{word}' is not in the dictionary "
                              f"(no close matches found).")
            else:
                print("Usage: /spell <word>")

        elif cmd == '/synonyms':
            if argument:
                synonyms = self.brain.dictionary.synonyms(argument)
                if synonyms:
                    print(f"\n🔁 Synonyms for '{argument}': {', '.join(synonyms[:12])}")
                else:
                    print(f"\nNo synonyms found for '{argument}'.")
            else:
                print("Usage: /synonyms <word>")

        elif cmd == '/words':
            if argument:
                prefix = argument.split()[0].lower().strip()
                completions = self.brain.dictionary.complete(prefix, n=12)
                if completions:
                    print(f"\n🔤 Words starting with '{prefix}':")
                    for i, w in enumerate(completions, 1):
                        print(f"  {i:2d}. {w}")
                else:
                    print(f"\nNo words found starting with '{prefix}'.")
            else:
                print("Usage: /words <prefix>")

        elif cmd == '/learned':
            self._show_learned()

        else:
            print(f"Unknown command: {cmd}. Type /help for available commands.")

        return True

    def _show_learned(self):
        """Show the words and terms the AI has learned from the web."""
        dictionary = self.brain.dictionary
        words = dictionary.learned_words
        if not words:
            print("\n🧠 I haven't learned any new words yet. Ask me to define a word "
                  "or term I don't know (with web search on, '/search on') and I'll "
                  "look it up and remember it forever.")
            return
        print(f"\n🧠 Learned dictionary - {len(words)} word(s)/term(s), "
              f"knowledge at {format_version(dictionary.knowledge_version)}:")
        for word in words:
            entry = dictionary._learned_entry(word) or {}
            source = entry.get('source', 'web')
            url = entry.get('url', '')
            learned_at = entry.get('learned_at', '')
            line = f"  {word}"
            if url:
                line += f"  ({url})"
            elif source:
                line += f"  (via {source})"
            if learned_at:
                line += f"  [{learned_at[:10]}]"
            print(line)
        print("These are remembered offline in data/learned_dictionary.json.")

    def _handle_search(self, argument: str) -> None:
        if not argument:
            status = 'ON' if self.brain.search_enabled else 'OFF'
            print(f"\n🔎 Web search is {status}. Usage: /search <query>, "
                  f"/search on, /search off")
            return
        if argument.lower() in ('on', 'off'):
            self.brain.set_search_enabled(argument.lower() == 'on')
            state = 'enabled' if argument.lower() == 'on' else 'disabled'
            print(f"\n✅ Auto web search {state}.")
            return

        print(f"\n🔎 Searching the web for: '{argument}'...")
        results = self.brain.search.search(argument, limit=3)
        if not results:
            print("   No results (offline or nothing found). "
                  "Type '/search on' to enable search.")
            return
        for i, result in enumerate(results, 1):
            print(f"\n  {i}. {result.title}")
            print(f"     {result.url}")
            snippet = result.display
            if len(snippet) > 220:
                snippet = snippet[:220] + '...'
            print(f"     {snippet}")

    def _show_info(self):
        """Display model information."""
        print(f"\n📊 {APP_NAME} {format_version(__version__)}")
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

        print(f"\n📚 Dictionary:")
        try:
            stats = self.brain.dictionary.stats
            print(f"  Word list: {stats['word_list_size']:,} words")
            print(f"  WordNet lemmas with definitions: {stats['defined_lemmas']:,}")
            print(f"  Irregular forms: {stats['irregular_forms']:,}")
            print(f"  Frequency data: {stats['frequency_entries']:,} words")
            print(f"  Learned words/terms: {stats['learned_words']:,} "
                  f"(knowledge {format_version(self.brain.dictionary.knowledge_version)})")
        except Exception as e:
            print(f"  (unavailable: {e})")

        print(f"\n🔎 Web search: {'enabled' if self.brain.search_enabled else 'disabled'}")
        print(f"  Sources: Wikipedia, DuckDuckGo Instant Answers")

        print(f"\n🧠 Knowledge index: {len(self.brain._ensure_index())} documents")
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
