"""
Training data generation and management.
Contains built-in knowledge for training.
"""

import numpy as np
from typing import List, Tuple


class TrainingData:
    """Generates and manages training data."""

    @staticmethod
    def get_training_corpus() -> List[str]:
        """
        Built-in training corpus with diverse knowledge.
        In practice you'd load much larger datasets - this is a demonstration.
        """
        corpus = [
            # Science
            "The Earth revolves around the Sun in approximately 365.25 days. This orbital period defines our calendar year. The Earth is the third planet from the Sun in our solar system.",
            "Water is composed of two hydrogen atoms and one oxygen atom, giving it the chemical formula H2O. Water exists in three states: solid ice, liquid water, and gaseous steam.",
            "Gravity is a fundamental force of nature that attracts objects with mass toward each other. Isaac Newton described gravity mathematically, and Albert Einstein later refined our understanding with general relativity.",
            "DNA stands for deoxyribonucleic acid. It is the molecule that carries genetic information in all living organisms. DNA has a double helix structure discovered by Watson and Crick.",
            "The speed of light in a vacuum is approximately 299,792,458 meters per second. Nothing with mass can travel at or exceed the speed of light according to Einstein's theory of special relativity.",
            "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This process is essential for life on Earth.",
            "Atoms are the basic building blocks of matter. They consist of protons and neutrons in a central nucleus, surrounded by orbiting electrons.",
            "Evolution by natural selection, proposed by Charles Darwin, explains how species change over time through variation, inheritance, and differential survival.",
            "The periodic table organizes chemical elements by their atomic number and chemical properties. It was first proposed by Dmitri Mendeleev in 1869.",
            "Electricity is the flow of electrons through a conductor. It can be generated through various means including solar, wind, nuclear, and fossil fuel power.",

            # Mathematics
            "Mathematics is the study of numbers, quantities, and shapes. It provides the foundation for science, engineering, and technology.",
            "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a squared plus b squared equals c squared.",
            "Pi is a mathematical constant approximately equal to 3.14159. It represents the ratio of a circle's circumference to its diameter.",
            "Calculus, developed by Newton and Leibniz, deals with rates of change and accumulation. Differentiation finds rates of change, while integration finds areas under curves.",
            "Prime numbers are natural numbers greater than one that have no positive divisors other than one and themselves. Examples include 2, 3, 5, 7, 11, and 13.",
            "Algebra uses symbols and letters to represent numbers and quantities in equations and formulas. It is a fundamental branch of mathematics.",
            "Statistics is the science of collecting, analyzing, and interpreting data. It helps us understand patterns and make predictions.",
            "Geometry studies shapes, sizes, and properties of space. Euclidean geometry deals with flat spaces, while non-Euclidean geometry deals with curved spaces.",

            # Technology
            "Artificial intelligence is the simulation of human intelligence by machines. Machine learning is a subset of AI that enables computers to learn from data.",
            "Neural networks are computing systems inspired by biological neural networks in the brain. They consist of interconnected nodes called neurons organized in layers.",
            "The internet is a global network of interconnected computers. It was developed from ARPANET in the late 1960s and has since transformed communication worldwide.",
            "Programming languages such as Python, JavaScript, and C++ are used to write software. Each language has its own syntax and is suited for different tasks.",
            "Computer memory comes in two main types: RAM (Random Access Memory) for temporary storage and ROM (Read Only Memory) for permanent storage.",
            "Algorithms are step-by-step instructions for solving problems or performing tasks. Efficient algorithms are crucial for computer science and software engineering.",
            "Machine learning models learn patterns from data. Supervised learning uses labeled data, unsupervised learning finds hidden patterns, and reinforcement learning learns through rewards.",
            "Cybersecurity protects computer systems and networks from unauthorized access and attacks. Encryption, firewalls, and authentication are key security measures.",
            "Cloud computing provides on-demand access to computing resources over the internet. Major providers include Amazon Web Services, Google Cloud, and Microsoft Azure.",
            "Blockchain is a distributed ledger technology that records transactions across many computers. It provides transparency and security without requiring a central authority.",

            # History
            "The Renaissance was a cultural movement that began in Italy in the 14th century. It marked a period of great advances in art, science, and learning.",
            "World War II lasted from 1939 to 1945 and involved most of the world's nations. It resulted in significant geopolitical changes and the formation of the United Nations.",
            "The Industrial Revolution began in Britain in the late 18th century. It transformed manufacturing, transportation, and society through mechanization and new technologies.",
            "Ancient Egypt was one of the earliest and longest-lasting civilizations. The Egyptians built the pyramids, developed hieroglyphic writing, and made advances in mathematics and medicine.",
            "The moon landing on July 20, 1969 was a historic achievement. Neil Armstrong became the first person to walk on the moon during the Apollo 11 mission.",

            # Philosophy and reasoning
            "Critical thinking involves analyzing information objectively and making reasoned judgments. It is an essential skill for problem-solving and decision-making.",
            "Logic is the study of valid reasoning. Deductive reasoning moves from general principles to specific conclusions, while inductive reasoning moves from specific observations to general principles.",
            "The scientific method involves observation, hypothesis formation, experimentation, and conclusion. It is the systematic approach used to investigate natural phenomena.",
            "Ethics is the branch of philosophy that deals with moral principles. It examines concepts of right and wrong, justice, and virtue.",
            "Philosophy means love of wisdom. Major branches include metaphysics, epistemology, ethics, logic, and aesthetics.",

            # Conversation patterns
            "Hello! How can I help you today? I'm an AI assistant ready to answer your questions and have a conversation.",
            "I'm doing well, thank you for asking! I'm here to help with any questions you might have.",
            "That's a great question! Let me think about it carefully and provide you with a helpful answer.",
            "I understand your concern. Let me try to explain this in a clear and helpful way.",
            "Thank you for sharing that with me. I appreciate your curiosity and willingness to learn.",
            "I don't know everything, but I'll do my best to help you find the answer you're looking for.",
            "Learning is a lifelong journey. Every question you ask brings you one step closer to understanding.",
            "Problem solving requires breaking complex issues into smaller, manageable parts. Let's work through this step by step.",
            "Creativity is the ability to generate novel and useful ideas. It combines imagination with knowledge and experience.",
            "Communication is key to human connection. Clear expression of ideas helps build understanding between people.",

            # More knowledge
            "The human body has 206 bones, over 600 muscles, and approximately 37 trillion cells. The brain alone contains about 86 billion neurons.",
            "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, especially burning fossil fuels, are the primary driver of recent climate change.",
            "Music is a universal form of expression found in every culture. It involves the organized arrangement of sounds in time, using elements like melody, harmony, and rhythm.",
            "Literature encompasses written works including novels, poetry, drama, and essays. Great literature explores the human condition and stands the test of time.",
            "Economics studies how societies allocate scarce resources. Supply and demand, market forces, and government policies all influence economic outcomes.",
            "Psychology is the scientific study of mind and behavior. It encompasses biological, social, and cognitive perspectives on human thought and action.",
            "Geography studies Earth's physical features, climate, and human populations. It helps us understand the relationship between people and their environment.",
            "Cooking combines science and art to transform raw ingredients into delicious meals. Understanding heat transfer, chemistry, and flavor profiles leads to better cooking.",
            "Exercise and physical activity are essential for maintaining good health. Regular movement strengthens the heart, muscles, and bones while improving mental well-being.",
            "Space exploration has expanded our understanding of the universe. From the Hubble telescope to Mars rovers, technology continues to reveal cosmic mysteries.",

            # Q&A patterns
            "Question: What is machine learning? Answer: Machine learning is a type of artificial intelligence where computers learn from data to make predictions and decisions without being explicitly programmed for every scenario.",
            "Question: How does the brain work? Answer: The brain works through networks of neurons that communicate via electrical and chemical signals. Different brain regions handle different functions like vision, movement, memory, and language.",
            "Question: What causes seasons? Answer: Seasons are caused by the tilt of Earth's axis at about 23.5 degrees. As Earth orbits the Sun, different hemispheres receive varying amounts of direct sunlight throughout the year.",
            "Question: Why is the sky blue? Answer: The sky appears blue because of Rayleigh scattering. Sunlight entering the atmosphere is scattered by gas molecules, and shorter blue wavelengths scatter more than longer red wavelengths.",
            "Question: What is gravity? Answer: Gravity is a fundamental force that attracts objects with mass toward each other. It keeps planets in orbit around stars and keeps us on the ground.",
            "Question: How do computers work? Answer: Computers process information using binary code, sequences of ones and zeros. A processor executes instructions, memory stores data, and input output devices allow interaction with users.",
            "Question: What is energy? Answer: Energy is the capacity to do work. It exists in many forms including kinetic, potential, thermal, electrical, and nuclear energy. Energy cannot be created or destroyed, only transformed.",
            "Question: How do plants grow? Answer: Plants grow through cell division and elongation. They need sunlight for photosynthesis, water and minerals from soil, and carbon dioxide from the air to produce food and build new cells.",

            # Extended conversations
            "User: Can you explain how neural networks learn? AI: Neural networks learn by adjusting the weights of connections between neurons. During training, the network makes predictions, compares them to correct answers, and uses backpropagation to update weights to reduce errors. Over many iterations, the network learns to recognize patterns in the data.",
            "User: What makes a good question? AI: A good question is clear, specific, and thought-provoking. It shows curiosity and helps guide learning. Good questions often challenge assumptions and lead to deeper understanding of a topic.",
            "User: How can I improve my problem-solving skills? AI: To improve problem-solving, practice breaking complex problems into smaller parts. Learn to identify patterns, consider multiple approaches, and don't be afraid to try different solutions. Regular practice with puzzles, math problems, and real-world challenges builds these skills over time.",
            "User: Tell me something interesting about space. AI: Here's something fascinating: a teaspoon of neutron star material would weigh about 6 billion tons on Earth. Neutron stars are so dense that their gravity compresses matter to incredible densities. They are the remnants of massive stars that have exploded as supernovae.",
        ]

        # Augment with more patterns
        augmented = []
        for text in corpus:
            augmented.append(text)
            # Add variations
            sentences = text.split('. ')
            if len(sentences) > 1:
                for i in range(len(sentences)):
                    if len(sentences[i].strip()) > 20:
                        augmented.append(sentences[i].strip() + '.')

        return augmented

    @staticmethod
    def create_batches(token_ids_list: List[List[int]], batch_size: int,
                       seq_len: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create training batches with input/target pairs."""

        # Concatenate all tokens
        all_tokens = []
        for tokens in token_ids_list:
            all_tokens.extend(tokens)

        all_tokens = np.array(all_tokens, dtype=np.int64)

        # Create sequences
        sequences_input = []
        sequences_target = []

        for i in range(0, len(all_tokens) - seq_len - 1, seq_len // 2):  # Overlapping
            inp = all_tokens[i:i + seq_len]
            tgt = all_tokens[i + 1:i + seq_len + 1]
            if len(inp) == seq_len and len(tgt) == seq_len:
                sequences_input.append(inp)
                sequences_target.append(tgt)

        if not sequences_input:
            # Fallback: pad shorter sequences
            for tokens in token_ids_list:
                if len(tokens) > 2:
                    inp = tokens[:-1]
                    tgt = tokens[1:]
                    # Pad
                    pad_len = seq_len - len(inp)
                    if pad_len > 0:
                        inp = inp + [0] * pad_len
                        tgt = tgt + [0] * pad_len
                    inp = inp[:seq_len]
                    tgt = tgt[:seq_len]
                    sequences_input.append(np.array(inp))
                    sequences_target.append(np.array(tgt))

        # Shuffle
        combined = list(zip(sequences_input, sequences_target))
        np.random.shuffle(combined)
        sequences_input, sequences_target = zip(*combined)

        # Create batches
        batches = []
        for i in range(0, len(sequences_input), batch_size):
            batch_inp = np.array(sequences_input[i:i + batch_size])
            batch_tgt = np.array(sequences_target[i:i + batch_size])
            if len(batch_inp) == batch_size:
                batches.append((batch_inp, batch_tgt))

        # Handle remaining samples
        if len(sequences_input) % batch_size != 0:
            remaining_inp = list(sequences_input[-(len(sequences_input) % batch_size):])
            remaining_tgt = list(sequences_target[-(len(sequences_target) % batch_size):])
            # Pad batch
            while len(remaining_inp) < batch_size:
                remaining_inp.append(remaining_inp[0])
                remaining_tgt.append(remaining_tgt[0])
            batches.append((np.array(remaining_inp), np.array(remaining_tgt)))

        return batches