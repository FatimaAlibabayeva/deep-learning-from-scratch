"""STEP 5(b) --- SGD + Momentum (Problem 5a: velocity form)"""

import numpy as np
from src.optimizers.base import Optimizer
from src.autograd.tensor import Tensor


class Momentum(Optimizer):
    def __init__(self, params: list[Tensor], lr: float, momentum: float = 0.9,
                 weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for p, v in zip(self.params, self.velocities):
            g = p.grad + self.weight_decay * p.data
            v[:] = self.momentum * v + g
            p.data -= self.lr * v
