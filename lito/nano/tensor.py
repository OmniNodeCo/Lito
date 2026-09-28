"""Tiny tensor ops on nested lists — stdlib only, optimized for small dims."""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence, Union

Tensor = List  # nested lists of float


def zeros(shape: Sequence[int]) -> Tensor:
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [zeros(shape[1:]) for _ in range(shape[0])]


def randn(shape: Sequence[int], scale: float = 0.02) -> Tensor:
    if len(shape) == 1:
        return [random.gauss(0.0, scale) for _ in range(shape[0])]
    return [randn(shape[1:], scale) for _ in range(shape[0])]


def xavier(in_f: int, out_f: int) -> Tensor:
    scale = math.sqrt(2.0 / (in_f + out_f))
    return [[random.gauss(0.0, scale) for _ in range(out_f)] for _ in range(in_f)]


def deepcopy_t(t: Tensor) -> Tensor:
    if not t or not isinstance(t[0], list):
        return list(t)
    return [deepcopy_t(r) for r in t]


def shape_of(t: Tensor) -> tuple:
    if not isinstance(t, list):
        return ()
    if not t:
        return (0,)
    if not isinstance(t[0], list):
        return (len(t),)
    return (len(t),) + shape_of(t[0])


def matvec(A: Tensor, x: Sequence[float]) -> List[float]:
    """A is [m,n], x is [n] -> [m]."""
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def vecadd(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def vecscale(a: Sequence[float], s: float) -> List[float]:
    return [x * s for x in a]


def matmul(A: Tensor, B: Tensor) -> Tensor:
    """A [m,k] @ B [k,n] -> [m,n]."""
    n = len(B[0])
    Bt = list(zip(*B))  # [n,k]
    out = []
    for row in A:
        out.append([sum(x * y for x, y in zip(row, col)) for col in Bt])
    return out


def transpose(A: Tensor) -> Tensor:
    return [list(row) for row in zip(*A)]


def softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits) if logits else 0.0
    ex = [math.exp(x - m) for x in logits]
    s = sum(ex) or 1.0
    return [e / s for e in ex]


def log_softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits) if logits else 0.0
    ex = [math.exp(x - m) for x in logits]
    log_z = math.log(sum(ex) or 1.0) + m
    return [x - log_z for x in logits]


def gelu(x: float) -> float:
    # tanh approximation
    return 0.5 * x * (1.0 + math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def layer_norm(x: Sequence[float], weight: Sequence[float], bias: Sequence[float], eps: float = 1e-5) -> List[float]:
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n
    inv = 1.0 / math.sqrt(var + eps)
    return [((v - mean) * inv) * w + b for v, w, b in zip(x, weight, bias)]


def embedding_lookup(table: Tensor, idx: int) -> List[float]:
    return list(table[idx])


def outer(a: Sequence[float], b: Sequence[float]) -> Tensor:
    return [[x * y for y in b] for x in a]


def add_inplace_mat(A: Tensor, B: Tensor, scale: float = 1.0) -> None:
    for i, row in enumerate(B):
        Ai = A[i]
        for j, v in enumerate(row):
            Ai[j] += scale * v


def add_inplace_vec(a: List[float], b: Sequence[float], scale: float = 1.0) -> None:
    for i, v in enumerate(b):
        a[i] += scale * v


def clip_vec(a: List[float], lo: float, hi: float) -> None:
    for i, v in enumerate(a):
        if v < lo:
            a[i] = lo
        elif v > hi:
            a[i] = hi
