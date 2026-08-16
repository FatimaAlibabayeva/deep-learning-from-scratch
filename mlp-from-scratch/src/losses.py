"""
STEP 3 --- Numerically stable softmax cross-entropy
=====================================================
Implements the max-subtraction stabilization (Problem 4b) and the fused
backward dL/dz2 = (p - y) / N (Problem 2b), special-cased on the output
Tensor's _backward for numerical stability as the statement allows.
"""

import numpy as np
from src.autograd.tensor import Tensor


def softmax_cross_entropy(logits: Tensor, y: np.ndarray) -> Tensor:
    z = logits.data
    n = z.shape[0]

    z_stable = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    p = exp_z / exp_z.sum(axis=1, keepdims=True)

    log_probs_true = np.log(p[np.arange(n), y])
    loss_value = -log_probs_true.mean()

    out = Tensor(loss_value, (logits,), "softmax_cross_entropy")

    def _backward():
        one_hot = np.zeros_like(p)
        one_hot[np.arange(n), y] = 1.0
        # dL/dz2 = (p - y) / N, scaled by the incoming gradient (usually 1.0)
        logits.grad += (p - one_hot) / n * out.grad

    out._backward = _backward
    return out
