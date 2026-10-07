"""
The AI Brain - combines all components into an intelligent system.
Includes pattern matching, knowledge retrieval, and generation.
"""

import numpy as np
import os
import re
from typing import Optional, List, Dict
from .model import TransformerLM
from .tokenizer import BPETokenizer
from .dataset import TrainingData


class AIBrain:
    """
    The central AI system that combines:
    - Neural network language model
    - Pattern matching
    - Knowledge base
    - Conversation management
    """

    def __init__(self, model_dir: str = 'checkpoints'):
        self.model_dir = model_dir
        self.model: Optional[TransformerLM] = None
        self.tokenizer: Optional[BPETokenizer] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.knowledge_base = self._build_knowledge_base()
        self.loaded = False

    def _build_knowledge_base(self) -> Dict[str, List[str]]:
        """Build a structured knowledge base for retrieval."""
        kb = {
            'greetings': [
                "Hello! I'm SmartAI, an artificial intelligence built from scratch. How can I help you?",
                "Hi there! I'm ready to help you with questions, conversation, or just chatting.",
                "Hey! Nice to meet you. I'm an AI assistant. What would you like to talk about?",
            ],
            'identity': [
                "I'm SmartAI, a neural network-based AI built entirely from scratch in Python. "
                "I use a transformer architecture with self-attention, trained on a knowledge corpus.",
                "I'm an artificial intelligence created without using any pre-trained models. "
                "My neural network was designed and trained from the ground up.",
            ],
            'capabilities': [
                "I can answer questions, have conversations, explain concepts, help with reasoning, "
                "and generate text. I was trained on knowledge spanning science, math, technology, "
                "history, and more.",
                "My abilities include text generation, question answering, and conversation. "
                "I learn from patterns in my training data to provide helpful responses.",
            ],
            'science': {
                'physics': [
                    "Physics studies the fundamental laws of nature including forces, energy, and matter.",
                    "Key concepts include gravity, electromagnetism, quantum mechanics, and relativity.",
                ],
                'biology': [
                    "Biology is the study of living organisms and their interactions.",
                    "Key topics include genetics, evolution, ecology, and cell biology.",
                ],
                'chemistry': [
                    "Chemistry studies the composition, structure, and properties of matter.",
                    "The periodic table organizes elements by atomic number and chemical properties.",
                ],
            },
            'math': [
                "Mathematics provides the language for describing patterns and relationships in nature.",
                "Key branches include algebra, geometry, calculus, statistics, and number theory.",
            ],
            'technology': [
                "Technology encompasses tools and methods created to solve problems.",
                "Computing, AI, the internet, and biotechnology are transforming our world.",
            ],
            'fallback': [
                "That's an interesting topic. Let me share what I know about it.",
                "I'll do my best to help with that. Let me think about it.",
                "That's a great question. Here's my understanding of the topic.",
            ],
        }
        return kb

    def initialize(self):
        """Initialize or load the AI model."""
        tokenizer_path = os.path.join(self.model_dir, 'tokenizer.json')
        model_path = os.path.join(self.model_dir, 'best_model.npz')

        if not os.path.exists(model_path):
            model_path = os.path.join(self.model_dir, 'final_model.npz')

        if os.path.exists(tokenizer_path) and os.path.exists(model_path):
            print("Loading trained model...")
            self.tokenizer = BPETokenizer()
            self.tokenizer.load(tokenizer_path)

            # Load config from model
            loaded = np.load(model_path, allow_pickle=True)
            if 'config' in loaded:
                import json
                config = json.loads(str(loaded['config']))
            else:
                config = {
                    'vocab_size': self.tokenizer.vocab_size,
                    'd_model': 128,
                    'n_heads': 4,
                    'n_layers': 4,
                    'd_ff': 512,
                    'max_seq_len': 256,
                }

            self.model = TransformerLM(
                vocab_size=config['vocab_size'],
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                n_layers=config['n_layers'],
                d_ff=config['d_ff'],
                max_seq_len=config['max_seq_len'],
            )
            self.model.load(model_path)
            self.model.eval_mode()
            self.loaded = True
            print("Model loaded successfully!")
        else:
            print("No trained model found. Using knowledge base only.")
            print("Run 'python train.py' first to train the neural network.")
            self.loaded = False

    def _find_relevant_knowledge(self, query: str) -> str:
        """Find relevant knowledge from the knowledge base."""
        query_lower = query.lower().strip()

        # Greeting detection
        greetings = ['hello', 'hi', 'hey', 'greetings', 'howdy', 'good morning',
                     'good afternoon', 'good evening', 'what\'s up', 'sup']
        if any(g in query_lower for g in greetings):
            return np.random.choice(self.knowledge_base['greetings'])

        # Identity questions
        identity_keywords = ['who are you', 'what are you', 'your name', 'about yourself',
                           'tell me about you', 'introduce yourself']
        if any(k in query_lower for k in identity_keywords):
            return np.random.choice(self.knowledge_base['identity'])

        # Capability questions
        cap_keywords = ['what can you do', 'your capabilities', 'help me', 'what do you know',
                       'can you help', 'abilities']
        if any(k in query_lower for k in cap_keywords):
            return np.random.choice(self.knowledge_base['capabilities'])

        # Science
        physics_kw = ['physics', 'gravity', 'force', 'energy', 'light', 'speed',
                      'relativity', 'quantum', 'electron', 'atom', 'nuclear']
        biology_kw = ['biology', 'dna', 'cell', 'evolution', 'species', 'gene',
                      'organism', 'photosynthesis', 'plant', 'animal', 'brain']
        chemistry_kw = ['chemistry', 'element', 'molecule', 'chemical', 'reaction',
                       'periodic table', 'compound', 'acid', 'base']
        math_kw = ['math', 'equation', 'number', 'calculate', 'algebra', 'geometry',
                  'calculus', 'theorem', 'prime', 'pi', 'statistics']
        tech_kw = ['computer', 'programming', 'software', 'internet', 'ai',
                  'artificial intelligence', 'machine learning', 'neural network',
                  'algorithm', 'data', 'code', 'technology']

        if any(k in query_lower for k in physics_kw):
            return np.random.choice(self.knowledge_base['science']['physics'])
        if any(k in query_lower for k in biology_kw):
            return np.random.choice(self.knowledge_base['science']['biology'])
        if any(k in query_lower for k in chemistry_kw):
            return np.random.choice(self.knowledge_base['science']['chemistry'])
        if any(k in query_lower for k in math_kw):
            return np.random.choice(self.knowledge_base['math'])
        if any(k in query_lower for k in tech_kw):
            return np.random.choice(self.knowledge_base['technology'])

        # Search training corpus for relevant content
        corpus = TrainingData.get_training_corpus()
        query_words = set(query_lower.split())
        best_match = ""
        best_score = 0

        for text in corpus:
            text_lower = text.lower()
            text_words = set(text_lower.split())
            # Calculate word overlap
            overlap = len(query_words.intersection(text_words))
            # Bonus for exact substring match
            for word in query_words:
                if len(word) > 3 and word in text_lower:
                    overlap += 2

            if overlap > best_score:
                best_score = overlap
                best_match = text

        if best_score >= 2 and best_match:
            return best_match

        return np.random.choice(self.knowledge_base['fallback'])

    def _generate_with_model(self, prompt: str, max_tokens: int = 80) -> str:
        """Generate text using the neural network model."""
        if not self.loaded or self.model is None:
            return ""

        try:
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) == 0:
                return ""

            input_ids = np.array(tokens).reshape(1, -1)
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_k=40,
                top_p=0.9
            )
            generated_text = self.tokenizer.decode(output_ids.tolist())

            # Clean up
            generated_text = generated_text.strip()
            # Remove the prompt from the beginning if repeated
            if generated_text.lower().startswith(prompt.lower()):
                generated_text = generated_text[len(prompt):].strip()

            return generated_text
        except Exception as e:
            return ""

    def think(self, user_input: str) -> str:
        """
        Process user input and generate a response.
        Combines knowledge retrieval with neural generation.
        """
        # Store in conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_input
        })

        # Get knowledge-based response
        kb_response = self._find_relevant_knowledge(user_input)

        # Try neural generation
        neural_response = ""
        if self.loaded:
            # Create a prompt combining context and user input
            context = ""
            if len(self.conversation_history) > 1:
                recent = self.conversation_history[-3:]
                for msg in recent[:-1]:
                    if msg['role'] == 'user':
                        context += f"User: {msg['content']} "
                    else:
                        context += f"AI: {msg['content']} "

            prompt = f"{context}User: {user_input} AI:"
            neural_response = self._generate_with_model(prompt, max_tokens=60)

        # Combine responses intelligently
        if neural_response and len(neural_response) > 20:
            # If neural response is coherent, blend with KB response
            response = self._blend_responses(kb_response, neural_response, user_input)
        else:
            response = kb_response

        # Post-process
        response = self._post_process(response)

        # Store response
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })

        return response

    def _blend_responses(self, kb_response: str, neural_response: str, query: str) -> str:
        """Intelligently blend knowledge-based and neural responses."""
        # Use KB response as the main one, neural as supplement
        # Check if neural response adds new information
        kb_words = set(kb_response.lower().split())
        neural_words = set(neural_response.lower().split())
        new_words = neural_words - kb_words

        if len(new_words) > 5 and len(neural_response) > 30:
            # Neural response has new information
            # Clean neural response
            neural_clean = neural_response.split('.')[0] + '.'
            if len(neural_clean) > 20:
                return f"{kb_response} {neural_clean}"

        return kb_response

    def _post_process(self, response: str) -> str:
        """Clean up and improve response quality."""
        # Remove excessive whitespace
        response = ' '.join(response.split())

        # Ensure proper ending
        if response and response[-1] not in '.!?':
            # Find last complete sentence
            last_period = response.rfind('.')
            last_question = response.rfind('?')
            last_excl = response.rfind('!')
            last_end = max(last_period, last_question, last_excl)
            if last_end > len(response) * 0.3:  # If we have at least 30% of text
                response = response[:last_end + 1]
            else:
                response += '.'

        # Capitalize first letter
        if response:
            response = response[0].upper() + response[1:]

        return response

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")