"""Activation functions."""

import numpy as np
from .tensor import Tensor


def relu(x: Tensor) -> Tensor:
    return x.relu()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def softmax(x: Tensor, axis=-1) -> Tensor:
    return x.softmax(axis=axis)


def gelu(x: Tensor) -> Tensor:
    """Gaussian Error Linear Unit."""
    h_data = x.data
    gelu_data = 0.5 * h_data * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (h_data + 0.044715 * h_data ** 3)
    ))
    out = Tensor(gelu_data, requires_grad=x.requires_grad, _children=(x,), _op='gelu')

    def _backward():
        x_val = x.data
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_val + 0.044715 * x_val ** 3)))
        pdf = np.exp(-0.5 * x_val ** 2) / np.sqrt(2.0 * np.pi)
        x.grad += out.grad * (cdf + x_val * pdf)
    out._backward = _backward
    return out