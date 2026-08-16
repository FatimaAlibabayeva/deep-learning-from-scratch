"""
STEP 2 (cont.) --- Activation functions
=========================================
relu is already a Tensor method; tanh/sigmoid are composed from existing
Tensor ops so autograd handles their gradients automatically.

Numerical stability
--------------------
sigmoid(x) = 1 / (1 + exp(-x)) computes exp(-x), which overflows float64
once x is a few hundred below zero (exp(-x) -> inf, and downstream
divisions/multiplications involving that inf turn into nan). tanh, built
here as 2*sigmoid(2x) - 1, would inherit the same problem.

Fix: clip the argument that actually goes into exp() to [-CLIP, CLIP]
before exponentiating. exp(60) is already ~1e26, so both branches of
sigmoid are numerically saturated to 0.0 or 1.0 at that magnitude anyway
-- clipping changes nothing about the *value* of sigmoid there, it only
stops exp() from being asked to produce a number bigger than float64 can
hold. The clip() primitive's backward is 1 inside the bounds and 0
outside, which matches the true (~0) gradient of an already-saturated
sigmoid/tanh.
"""

from src.autograd.tensor import Tensor

_EXP_CLIP = 60.0


def relu(x: Tensor) -> Tensor:
    return x.relu()


def sigmoid(x: Tensor) -> Tensor:
    # sigma(z) = 1 / (1 + e^-z)
    neg_x = x * Tensor(-1.0)
    neg_x = neg_x.clip(-_EXP_CLIP, _EXP_CLIP)
    e_neg_z = neg_x.exp()
    return Tensor(1.0) / (Tensor(1.0) + e_neg_z)


def tanh(x: Tensor) -> Tensor:
    # tanh(x) = 2*sigmoid(2x) - 1 (derivation in the report); reusing the
    # stable sigmoid above means tanh is stable "for free".
    s = sigmoid(x * Tensor(2.0))
    return s * Tensor(2.0) - Tensor(1.0)
