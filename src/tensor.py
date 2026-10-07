"""
Custom tensor operations built on top of numpy.
Implements automatic differentiation (autograd).
"""

import numpy as np
from typing import List, Optional, Tuple, Union


class Tensor:
    """A tensor with automatic differentiation support."""

    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        if isinstance(data, (int, float)):
            data = np.array([data], dtype=np.float64)
        elif isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float64)
        else:
            data = np.array(data, dtype=np.float64)

        self.data = data
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        out = Tensor(self.data.T, requires_grad=self.requires_grad, _children=(self,), _op='transpose')
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, _children=(self,), _op='reshape')
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad, _children=(self,), _op='sum')
        def _backward():
            if axis is None:
                self.grad += np.ones_like(self.data) * out.grad
            else:
                grad = out.grad
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.ones_like(self.data) * grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad, _children=(self,), _op='mean')
        def _backward():
            if axis is None:
                self.grad += np.ones_like(self.data) * out.grad / n
            else:
                grad = out.grad
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.ones_like(self.data) * grad / n
        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.full_like(self.data, other))
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='+')
        def _backward():
            g = out.grad
            # Handle broadcasting
            self_grad = g
            other_grad = g
            # Reduce along broadcast dimensions
            while len(self_grad.shape) > len(self.data.shape):
                self_grad = self_grad.sum(axis=0)
            for i, (s, gs) in enumerate(zip(self.data.shape, self_grad.shape)):
                if s == 1 and gs != 1:
                    self_grad = self_grad.sum(axis=i, keepdims=True)

            while len(other_grad.shape) > len(other.data.shape):
                other_grad = other_grad.sum(axis=0)
            for i, (s, gs) in enumerate(zip(other.data.shape, other_grad.shape)):
                if s == 1 and gs != 1:
                    other_grad = other_grad.sum(axis=i, keepdims=True)

            self.grad += self_grad
            other.grad += other_grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.full_like(self.data, other))
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='*')
        def _backward():
            self_grad = other.data * out.grad
            other_grad = self.data * out.grad
            while len(self_grad.shape) > len(self.data.shape):
                self_grad = self_grad.sum(axis=0)
            for i, (s, gs) in enumerate(zip(self.data.shape, self_grad.shape)):
                if s == 1 and gs != 1:
                    self_grad = self_grad.sum(axis=i, keepdims=True)
            while len(other_grad.shape) > len(other.data.shape):
                other_grad = other_grad.sum(axis=0)
            for i, (s, gs) in enumerate(zip(other.data.shape, other_grad.shape)):
                if s == 1 and gs != 1:
                    other_grad = other_grad.sum(axis=i, keepdims=True)
            self.grad += self_grad
            other.grad += other_grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        """Matrix multiplication with autograd."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='@')
        def _backward():
            if len(self.data.shape) == 1 and len(other.data.shape) == 1:
                self.grad += out.grad * other.data
                other.grad += out.grad * self.data
            elif len(self.data.shape) == 2 and len(other.data.shape) == 2:
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
            elif len(self.data.shape) >= 2 and len(other.data.shape) >= 2:
                self.grad += out.grad @ np.swapaxes(other.data, -2, -1)
                other.grad += np.swapaxes(self.data, -2, -1) @ out.grad
            else:
                self.grad += out.grad @ other.data.T if len(other.data.shape) == 2 else out.grad * other.data
                other.grad += self.data.T @ out.grad if len(self.data.shape) == 2 else self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power):
        out = Tensor(self.data ** power, requires_grad=self.requires_grad,
                     _children=(self,), _op=f'**{power}')
        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * Tensor(np.full_like(self.data, -1.0))

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return self * (other ** -1)
        return self * (1.0 / other)

    def __rtruediv__(self, other):
        return Tensor(np.full_like(self.data, other)) * (self ** -1)

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], requires_grad=self.requires_grad,
                     _children=(self,), _op='getitem')
        def _backward():
            grad = np.zeros_like(self.data)
            grad[idx] = out.grad
            self.grad += grad
        out._backward = _backward
        return out

    def exp(self):
        clipped = np.clip(self.data, -500, 500)
        out = Tensor(np.exp(clipped), requires_grad=self.requires_grad,
                     _children=(self,), _op='exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-12), requires_grad=self.requires_grad,
                     _children=(self,), _op='log')
        def _backward():
            self.grad += (1.0 / (self.data + 1e-12)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _children=(self,), _op='tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad,
                     _children=(self,), _op='relu')
        def _backward():
            self.grad += (self.data > 0).astype(np.float64) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        clipped = np.clip(self.data, -500, 500)
        s = 1.0 / (1.0 + np.exp(-clipped))
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op='sigmoid')
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exps = np.exp(shifted)
        sm = exps / np.sum(exps, axis=axis, keepdims=True)
        out = Tensor(sm, requires_grad=self.requires_grad, _children=(self,), _op='softmax')
        def _backward():
            # Jacobian-vector product for softmax
            g = out.grad
            dot = np.sum(g * sm, axis=axis, keepdims=True)
            self.grad += sm * (g - dot)
        out._backward = _backward
        return out

    def backward(self):
        """Run backpropagation."""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def detach(self):
        return Tensor(self.data.copy())

    def numpy(self):
        return self.data.copy()

    def __repr__(self):
        return f"Tensor({self.data}, grad={self.requires_grad})"

    def __len__(self):
        return len(self.data)

    @staticmethod
    def zeros(*shape, requires_grad=False):
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape, requires_grad=False):
        return Tensor(np.ones(shape), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, requires_grad=False):
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)

    @staticmethod
    def xavier_uniform(fan_in, fan_out, requires_grad=True):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        data = np.random.uniform(-limit, limit, (fan_in, fan_out))
        return Tensor(data, requires_grad=requires_grad)

    @staticmethod
    def he_normal(fan_in, fan_out, requires_grad=True):
        std = np.sqrt(2.0 / fan_in)
        data = np.random.randn(fan_in, fan_out) * std
        return Tensor(data, requires_grad=requires_grad)