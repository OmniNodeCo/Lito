"""Loss functions."""

import numpy as np
from .tensor import Tensor


def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    """
    Cross entropy loss for language modeling.
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len)
    """
    orig_shape = logits.data.shape
    vocab_size = orig_shape[-1]
    logits_2d = logits.data.reshape(-1, vocab_size)
    targets_1d = targets.reshape(-1).astype(int)
    n = logits_2d.shape[0]

    shifted = logits_2d - np.max(logits_2d, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)

    correct_probs = probs[np.arange(n), targets_1d]
    loss_val = -np.mean(np.log(np.clip(correct_probs, 1e-12, 1.0)))

    loss = Tensor(np.array([loss_val]), requires_grad=True, _children=(logits,), _op='cross_entropy')

    def _backward():
        if logits.requires_grad:
            grad = probs.copy()
            grad[np.arange(n), targets_1d] -= 1.0
            grad = (grad / n) * loss.grad[0]
            logits.grad += grad.reshape(orig_shape)

    loss._backward = _backward
    return loss