from __future__ import annotations

from typing import Callable

import numpy.typing as npt

# thanks to https://sidsite.com/posts/autodiff/

type Operation = tuple[Tensor, str, Callable]


class Tensor:
    def __init__(self, value: npt.NDArray, operations: list[Operation] | None = None):
        self.value = value
        if operations is None:
            self.operations = []
        else:
            self.operations = operations

    def compute_gradient(self) -> dict["Tensor", float]:
        gradients = {}

        # FIXME: I guess one downside to this is that, since I don't store the
        # backwards pass anywhere, it's going to be recomputed in a lot of
        # places if I need multiple gradients.
        stack: list[tuple[Tensor, float]] = [(self, 1)]
        while stack:
            current_variable, current_value = stack.pop()
            for child, name, op in current_variable.operations:
                child_grad = gradients.get(child, 0)
                child_value = op(current_value)
                gradients[child] = child_grad + child_value
                stack.append((child, child_value))

        return gradients

    def __add__(self: Tensor, b: Tensor) -> Tensor:
        value = self.value + b.value
        operations = [
            (self, "add", lambda acc: acc),  # conceptually, 1*acc
            (b, "add", lambda acc: acc),
        ]
        return Tensor(value, operations)

    def __sub__(self: Tensor, b: Tensor) -> Tensor:
        value = self.value - b.value
        operations = [(self, "sub", lambda acc: acc), (b, "sub", lambda acc: -acc)]
        return Tensor(value, operations)

    def __mul__(self: Tensor, b: Tensor) -> Tensor:
        # elementwise
        value = self.value * b.value
        operations = [
            (self, "mul", lambda acc: acc * b.value),
            (b, "mul", lambda acc: acc * self.value),
        ]
        return Tensor(value, operations)

    def __neg__(self: Tensor) -> Tensor:
        value = -self.value
        operations = [(self, "neg", lambda acc: -acc)]
        return Tensor(value, operations)

    def __truediv__(self: Tensor, b: Tensor) -> Tensor:
        # elementwise
        value = self.value / b.value
        operations = [
            (self, "div", lambda acc: acc / b.value),
            (b, "div", lambda acc: -acc * self.value / (b.value**2)),
        ]
        return Tensor(value, operations)
