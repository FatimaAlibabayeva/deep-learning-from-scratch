"""STEP 5 --- Optimizer base class"""

from abc import ABC, abstractmethod
from src.autograd.tensor import Tensor


class Optimizer(ABC):
    def __init__(self, params: list[Tensor], lr: float, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    @abstractmethod
    def step(self) -> None:
        ...

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()
