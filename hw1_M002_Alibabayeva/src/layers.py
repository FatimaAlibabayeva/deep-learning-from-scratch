"""
STEP 2 --- Layers: Module, Linear, Dropout
============================================
Thin wrappers over Tensor ops. None of these compute gradients themselves
-- that's entirely Tensor's job (step 1).
"""

import numpy as np
from src.autograd.tensor import Tensor


class Module:
    """Base class for anything with (optionally) trainable parameters."""

    def parameters(self) -> list[Tensor]:
        return []

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Module):
    """Fully-connected layer: y = xW + b"""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        # He initialization -- suited to ReLU nets since it keeps the
        # variance of activations roughly constant through depth when
        # half the units are zeroed by ReLU.
        std = np.sqrt(2.0 / in_features)
        W = rng.normal(0.0, std, size=(in_features, out_features))
        b = np.zeros(out_features)
        self.W = Tensor(W)
        self.b = Tensor(b)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b

    def parameters(self) -> list[Tensor]:
        return [self.W, self.b]


class BatchNorm1d(Module):
    """BONUS 2 --- BatchNorm for 2D (N, D) inputs.

    Built ENTIRELY from existing Tensor ops (+, *, /, sum, sqrt) so
    gradients flow through the normal autograd graph -- no hand-written
    backward here. Running mean/var are plain numpy bookkeeping (not part
    of the graph), used only at eval time.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = Tensor(np.ones(num_features))
        self.beta = Tensor(np.zeros(num_features))
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta]

    def forward(self, x: Tensor) -> Tensor:
        n = x.data.shape[0]
        if self.training:
            # mean/var computed WITH autograd so backprop covers them
            mean = x.sum(axis=0) * Tensor(1.0 / n)                      # (D,)
            centered = x - mean                                        # broadcast (N, D)
            var = (centered * centered).sum(axis=0) * Tensor(1.0 / n)   # (D,)

            # bookkeeping copy (not part of the graph), used at eval time
            unbiased_var = var.data * n / max(n - 1, 1)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbiased_var
        else:
            mean = Tensor(self.running_mean)
            centered = x - mean
            var = Tensor(self.running_var)

        std = (var + Tensor(self.eps)).sqrt()
        x_hat = centered / std
        return x_hat * self.gamma + self.beta


class Dropout(Module):
    """Inverted dropout: random zeroing at train time, no-op at eval time."""

    def __init__(self, p: float, rng: np.random.Generator):
        assert 0.0 <= p < 1.0
        self.p = p
        self.rng = rng
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = (self.rng.random(x.data.shape) < keep_prob).astype(np.float64) / keep_prob
        # mask is a constant w.r.t. autograd -- wrap it as a Tensor with no
        # further parents so elementwise mul just scales/gates the gradient.
        return x * Tensor(mask)
