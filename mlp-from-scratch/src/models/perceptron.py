"""
STEP 4 --- MLP (Multi-Layer Perceptron)
=========================================
Stacks Linear -> relu -> (Dropout) blocks, with a final bare Linear that
outputs raw logits (softmax happens inside the loss, not here).
"""

import numpy as np
from src.layers import Module, Linear, Dropout, BatchNorm1d
from src import activations
from src.autograd.tensor import Tensor


class MLP(Module):
    def __init__(self, sizes: list[int], rng: np.random.Generator, dropout: float = 0.0):
        self.linears: list[Linear] = []
        self.dropouts: list[Dropout] = []
        for i in range(len(sizes) - 1):
            self.linears.append(Linear(sizes[i], sizes[i + 1], rng))
            is_last = i == len(sizes) - 2
            # a Dropout after every hidden layer, but not after the output
            self.dropouts.append(None if is_last or dropout == 0.0 else Dropout(dropout, rng))

    def forward(self, x: Tensor) -> Tensor:
        h = x
        n_layers = len(self.linears)
        for i, lin in enumerate(self.linears):
            h = lin(h)
            is_last = i == n_layers - 1
            if not is_last:
                h = activations.relu(h)
                if self.dropouts[i] is not None:
                    h = self.dropouts[i](h)
        return h

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for lin in self.linears:
            params.extend(lin.parameters())
        return params

    def train(self) -> None:
        for d in self.dropouts:
            if d is not None:
                d.training = True

    def eval(self) -> None:
        for d in self.dropouts:
            if d is not None:
                d.training = False


class MLPBatchNorm(Module):
    """BONUS 2 --- same shape as MLP, but with BatchNorm1d after each
    hidden Linear (before the ReLU). Kept as a separate class so the
    required MLP API/signature in the spec stays untouched."""

    def __init__(self, sizes: list[int], rng: np.random.Generator):
        self.linears: list[Linear] = []
        self.bns: list[BatchNorm1d] = []
        for i in range(len(sizes) - 1):
            self.linears.append(Linear(sizes[i], sizes[i + 1], rng))
            is_last = i == len(sizes) - 2
            self.bns.append(None if is_last else BatchNorm1d(sizes[i + 1]))

    def forward(self, x: Tensor) -> Tensor:
        h = x
        n_layers = len(self.linears)
        for i, lin in enumerate(self.linears):
            h = lin(h)
            if self.bns[i] is not None:
                h = self.bns[i](h)
                h = activations.relu(h)
        return h

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for lin in self.linears:
            params.extend(lin.parameters())
        for bn in self.bns:
            if bn is not None:
                params.extend(bn.parameters())
        return params

    def train(self) -> None:
        for bn in self.bns:
            if bn is not None:
                bn.training = True

    def eval(self) -> None:
        for bn in self.bns:
            if bn is not None:
                bn.training = False
