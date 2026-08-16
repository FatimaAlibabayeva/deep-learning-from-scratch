"""STEP 5(a) --- Plain SGD"""

from src.optimizers.base import Optimizer


class SGD(Optimizer):
    def step(self) -> None:
        for p in self.params:
            p.data -= self.lr * (p.grad + self.weight_decay * p.data)
