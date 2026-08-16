"""STEP 5(c) --- Adam (Problem 5b: moments + bias correction)"""

import numpy as np
from src.optimizers.base import Optimizer
from src.autograd.tensor import Tensor


class Adam(Optimizer):
    def __init__(self, params: list[Tensor], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1  # increment before bias correction, so t starts at 1
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad + self.weight_decay * p.data
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * (g ** 2)
            # bias correction (Problem 5b) -- without it, m and v are
            # biased toward 0 in the first few steps since they start at 0
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
