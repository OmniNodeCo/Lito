"""Loss functions."""

import numpy as np
from .tensor import Tensor


def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    """
    Cross entropy loss for language modeling.
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len) integer indices
    """
    if len(logits.data.shape) == 3:
        batch_size, seq_len, vocab_size = logits.data.shape
        logits_2d = logits.data.reshape(-1, vocab_size)
        targets_1d = targets.reshape(-1).astype(int)
    elif len(logits.data.shape) == 2:
        logits_2d = logits.data
        targets_1d = targets.reshape(-1).astype(int)
        batch_size = 1
        seq_len = logits_2d.shape[0]
        vocab_size = logits_2d.shape[1]
    else:
        raise ValueError(f"Unexpected logits shape: {logits.data.shape}")

    # Numerically stable softmax
    shifted = logits_2d - np.max(logits_2d, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)

    # Gather correct class probabilities
    n = logits_2d.shape[0]
    correct_probs = probs[np.arange(n), targets_1d]
    loss_val = -np.mean(np.log(correct_probs + 1e-12))

    loss = Tensor(np.array([loss_val]), requires_grad=True, _children=(logits,), _op='cross_entropy')

    def _backward():
        grad = probs.copy()
        grad[np.arange(n), targets_1d] -= 1.0
        grad /= n
        if len(logits.data.shape) == 3:
            logits.grad += grad.reshape(batch_size, seq_len, vocab_size)
        else:
            logits.grad += grad

    loss._backward = _backward
    return loss


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean squared error loss."""
    diff = predictions.data - targets.data
    loss_val = np.mean(diff ** 2)
    loss = Tensor(np.array([loss_val]), requires_grad=True,
                  _children=(predictions,), _op='mse')
    def _backward():
        n = predictions.data.size
        predictions.grad += (2.0 / n) * diff * loss.grad
    loss._backward = _backward
    return loss