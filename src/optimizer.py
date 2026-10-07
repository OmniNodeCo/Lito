"""Optimizers with modern features."""

import numpy as np
from typing import List
from .tensor import Tensor


class Optimizer:
    """Base optimizer."""
    def __init__(self, parameters: List[Tensor], lr: float = 0.001):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)


class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum."""
    def __init__(self, parameters, lr=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            grad = p.grad.copy()
            if self.weight_decay > 0:
                grad += self.weight_decay * p.data
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            p.data -= self.lr * self.velocities[i]


class Adam(Optimizer):
    """Adam optimizer with weight decay (AdamW)."""
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.01):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            grad = p.grad

            # Gradient clipping
            grad_norm = np.sqrt(np.sum(grad ** 2))
            if grad_norm > 1.0:
                grad = grad / grad_norm

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Weight decay (AdamW style)
            if self.weight_decay > 0:
                p.data -= self.lr * self.weight_decay * p.data

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler with warmup."""
    def __init__(self, optimizer: Optimizer, total_steps: int,
                 warmup_steps: int = 100, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress))
        self.optimizer.lr = lr
        return lr