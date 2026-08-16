"""BONUS 1 --- RMSProp (same Optimizer interface as SGD/Momentum/Adam)"""

import numpy as np
from src.optimizers.base import Optimizer
from src.autograd.tensor import Tensor


class RMSProp(Optimizer):
    def __init__(self, params: list[Tensor], lr: float = 1e-3, decay: float = 0.9,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.decay = decay
        self.eps = eps
        self.sq_avg = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for p, s in zip(self.params, self.sq_avg):
            g = p.grad + self.weight_decay * p.data
            s[:] = self.decay * s + (1 - self.decay) * (g ** 2)
            p.data -= self.lr * g / (np.sqrt(s) + self.eps)
