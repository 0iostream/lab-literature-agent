"""Day 2 加分：面试手撕 self-attention 的最小演示。

- softmax: 数值稳定的 softmax
- scaled_dot_product_attention: softmax(QK^T/sqrt(d))V，形状 (batch, seq, d)
- main: 造随机 QKV 跑一遍并打印形状
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """q,k,v: (batch, seq, d) -> (batch, seq, d)"""
    d = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    weights = softmax(scores, axis=-1)
    return weights @ v


def main() -> None:
    rng = np.random.default_rng(0)
    batch, seq, d = 1, 4, 8
    q = rng.normal(size=(batch, seq, d))
    k = rng.normal(size=(batch, seq, d))
    v = rng.normal(size=(batch, seq, d))
    out = scaled_dot_product_attention(q, k, v)
    print("q/k/v shape:", q.shape)
    print("output shape:", out.shape)
    print("output[0,0,:4]:", np.round(out[0, 0, :4], 3))


if __name__ == "__main__":
    main()
