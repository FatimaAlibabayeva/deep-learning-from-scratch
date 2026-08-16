"""
STEP 7 --- Tests
==================
B3's requirement: gradient_check on a Linear weight, a hidden activation,
and the loss, all with rel-err <= 1e-4.
"""

import numpy as np
from src.autograd.tensor import Tensor
from src.layers import Linear
from src import activations
from src.losses import softmax_cross_entropy
from src.utils import gradient_check, set_seed

TOL = 1e-4


def _toy_chain(rng):
    """A small Linear -> relu -> Linear -> softmax_cross_entropy chain."""
    n, d, h, c = 4, 5, 3, 2
    X = rng.normal(size=(n, d))
    y = rng.integers(0, c, size=n)
    lin1 = Linear(d, h, rng)
    lin2 = Linear(h, c, rng)
    return X, y, lin1, lin2


def test_gradient_check_linear_weight():
    rng = set_seed(1)
    X, y, lin1, lin2 = _toy_chain(rng)

    def f(W):
        lin1.W = W
        out = activations.relu(lin1(Tensor(X)))
        out = lin2(out)
        return softmax_cross_entropy(out, y)

    err = gradient_check(f, lin1.W, h=1e-5)
    assert err <= TOL, f"relative error too high: {err}"


def test_gradient_check_hidden_activation():
    rng = set_seed(2)
    X, y, lin1, lin2 = _toy_chain(rng)
    z1 = lin1(Tensor(X))

    def f(h_pre):
        h = activations.relu(h_pre)
        out = lin2(h)
        return softmax_cross_entropy(out, y)

    err = gradient_check(f, z1, h=1e-5)
    assert err <= TOL, f"relative error too high: {err}"


def test_gradient_check_loss():
    rng = set_seed(3)
    n, c = 6, 4
    logits_data = rng.normal(size=(n, c))
    y = rng.integers(0, c, size=n)

    def f(logits):
        return softmax_cross_entropy(logits, y)

    logits = Tensor(logits_data)
    err = gradient_check(f, logits, h=1e-5)
    assert err <= TOL, f"relative error too high: {err}"

    # also directly confirm dL/dz2 == (p - onehot(y)) / n
    logits.zero_grad()
    loss = f(logits)
    loss.backward()
    z = logits.data
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    one_hot = np.zeros_like(p)
    one_hot[np.arange(n), y] = 1.0
    expected = (p - one_hot) / n
    assert np.allclose(logits.grad, expected, atol=1e-8)
