"""
STEP 1 --- Tensor autograd engine
==================================
Everything else in this project (Linear, MLP, losses, optimizers) is built
on top of this one class. Each op returns a NEW Tensor and installs a
local `_backward` closure on it; `backward()` walks the graph in reverse
topological order and lets each node push its gradient to its parents.
"""

from __future__ import annotations

import numpy as np


class Tensor:
    def __init__(self, data, _children: tuple = (), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------------
    # helper: sum away broadcasted axes so a gradient matches a shape
    # ------------------------------------------------------------------
    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
        # 1. drop leading axes that were broadcast in (ndim grew)
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        # 2. any axis that was size-1 in `shape` but >1 in grad got
        #    broadcast there too -- sum it back down, keepdims so the
        #    shapes still line up positionally
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(shape)

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def __add__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "add")

        def _backward():
            self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "mul")

        def _backward():
            self.grad += Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis: int | None = None) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis), (self,), "sum")

        def _backward():
            grad = out.grad
            if axis is None:
                self.grad += np.ones_like(self.data) * grad
            else:
                self.grad += np.ones_like(self.data) * np.expand_dims(grad, axis)

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        out = Tensor(np.sqrt(self.data), (self,), "sqrt")

        def _backward():
            self.grad += (0.5 / out.data) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def clip(self, min_val: float, max_val: float) -> "Tensor":
        """Elementwise clamp to [min_val, max_val].

        Added as a genuine autograd primitive (not composed from other ops)
        so sigmoid/tanh can clip the argument of exp() *before* it overflows,
        instead of computing inf/nan and hoping later ops clean it up.
        Gradient is the local derivative of clipping: 1 where the input was
        inside the bounds (untouched), 0 where it was clamped -- which is
        exactly right, since a clamped input is, to floating-point precision,
        already in the fully-saturated flat part of sigmoid/tanh anyway.
        """
        out = Tensor(np.clip(self.data, min_val, max_val), (self,), "clip")
        mask = (self.data >= min_val) & (self.data <= max_val)

        def _backward():
            self.grad += mask * out.grad

        out._backward = _backward
        return out

    # convenience (not required by the spec, but harmless / useful)
    def __neg__(self) -> "Tensor":
        return self * Tensor(-np.ones_like(self.data))

    def __sub__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other._reciprocal()

    def _reciprocal(self) -> "Tensor":
        # d(1/x)/dx = -1/x^2 -- kept as a real graph node (not a detached
        # constant) so division still backprops through `other` correctly.
        out = Tensor(1.0 / self.data, (self,), "reciprocal")

        def _backward():
            self.grad += (-1.0 / (self.data ** 2)) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # ------------------------------------------------------------------
    # walk the graph
    # ------------------------------------------------------------------

    def backward(self) -> None:
        topo: list["Tensor"] = []
        visited: set = set()

        def build(v: "Tensor"):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"
