"""Training loop with logging and checkpointing."""

import numpy as np
import time
import os
from typing import Optional
from .model import TransformerLM
from .tokenizer import BPETokenizer
from .optimizer import Adam, CosineAnnealingLR
from .loss import cross_entropy_loss
from .dataset import TrainingData


class Trainer:
    """Handles the complete training pipeline."""

    def __init__(self, model: TransformerLM, tokenizer: BPETokenizer,
                 lr: float = 3e-4, weight_decay: float = 0.01,
                 save_dir: str = 'checkpoints'):
        self.model = model
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.train_losses = []
        self.best_loss = float('inf')

    def train(self, epochs: int = 50, batch_size: int = 4, seq_len: int = 64,
              log_interval: int = 10, save_interval: int = 100,
              warmup_steps: int = 50):
        """Run the training loop."""
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)

        # Get training data
        corpus = TrainingData.get_training_corpus()
        print(f"Training corpus: {len(corpus)} texts")

        # Tokenize
        print("Tokenizing corpus...")
        token_ids_list = []
        for text in corpus:
            tokens = self.tokenizer.encode(text)
            token_ids_list.append(tokens)

        total_tokens = sum(len(t) for t in token_ids_list)
        print(f"Total tokens: {total_tokens:,}")

        # Create batches
        batches = TrainingData.create_batches(token_ids_list, batch_size, seq_len)
        print(f"Training batches: {len(batches)}")
        print(f"Batch size: {batch_size}, Sequence length: {seq_len}")

        if not batches:
            print("ERROR: No training batches created. Check your data.")
            return

        # Learning rate scheduler
        total_steps = epochs * len(batches)
        scheduler = CosineAnnealingLR(
            self.optimizer, total_steps,
            warmup_steps=warmup_steps
        )

        self.model.train_mode()

        global_step = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_losses = []
            np.random.shuffle(batches)

            for batch_idx, (input_ids, target_ids) in enumerate(batches):
                global_step += 1

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                logits = self.model.forward(input_ids)

                # Compute loss
                loss = cross_entropy_loss(logits, target_ids)
                loss_val = float(loss.data[0])
                epoch_losses.append(loss_val)

                # Backward pass
                loss.backward()

                # Gradient clipping
                total_norm = 0.0
                for p in self.model.parameters():
                    total_norm += np.sum(p.grad ** 2)
                total_norm = np.sqrt(total_norm)
                max_norm = 1.0
                if total_norm > max_norm:
                    clip_coef = max_norm / (total_norm + 1e-6)
                    for p in self.model.parameters():
                        p.grad *= clip_coef

                # Update weights
                self.optimizer.step()
                scheduler.step()

                # Logging
                if global_step % log_interval == 0:
                    elapsed = time.time() - start_time
                    avg_loss = np.mean(epoch_losses[-log_interval:])
                    lr = self.optimizer.lr
                    tokens_per_sec = (global_step * batch_size * seq_len) / elapsed
                    perplexity = np.exp(min(avg_loss, 20))

                    print(f"Epoch {epoch}/{epochs} | Step {global_step} | "
                          f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
                          f"LR: {lr:.2e} | "
                          f"Tok/s: {tokens_per_sec:.0f} | "
                          f"GradNorm: {total_norm:.4f}")

                # Save checkpoint
                if global_step % save_interval == 0:
                    self._save_checkpoint(epoch, global_step, np.mean(epoch_losses))

            # End of epoch
            avg_epoch_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_epoch_loss)
            perplexity = np.exp(min(avg_epoch_loss, 20))
            print(f"\n--- Epoch {epoch} Complete | Avg Loss: {avg_epoch_loss:.4f} | "
                  f"PPL: {perplexity:.2f} ---")

            # Generate sample
            if epoch % 5 == 0 or epoch == epochs:
                self._generate_sample()

            # Save best model
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self.model.save(os.path.join(self.save_dir, 'best_model.npz'))
                print(f"New best model saved! Loss: {self.best_loss:.4f}")

        # Final save
        self.model.save(os.path.join(self.save_dir, 'final_model.npz'))
        self.tokenizer.save(os.path.join(self.save_dir, 'tokenizer.json'))

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Final perplexity: {np.exp(min(self.best_loss, 20)):.2f}")

    def _save_checkpoint(self, epoch, step, loss):
        path = os.path.join(self.save_dir, f'checkpoint_step{step}.npz')
        self.model.save(path)

    def _generate_sample(self):
        """Generate a text sample to monitor training progress."""
        print("\n--- Sample Generation ---")
        prompts = ["The Earth", "Machine learning", "Hello"]
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = np.array(tokens).reshape(1, -1)
            try:
                output_ids = self.model.generate(
                    input_ids, max_new_tokens=30,
                    temperature=0.8, top_k=30
                )
                text = self.tokenizer.decode(output_ids.tolist())
                print(f"  Prompt: '{prompt}' -> '{text[:100]}'")
            except Exception as e:
                print(f"  Generation error: {e}")
        print("--- End Sample ---\n")
        self.model.train_mode()