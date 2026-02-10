from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

# thanks to https://sidsite.com/posts/autodiff/

type Operation = tuple[Tensor, str, Callable]


class Tensor:
    def __init__(
        self, value: npt.NDArray, operations: list[Operation] | None = None, frozen=True
    ):
        self.value = value
        self.frozen = frozen
        if operations is None:
            self.operations = []
        else:
            self.operations = operations

    def compute_gradient(self) -> dict["Tensor", npt.NDArray[np.floating]]:
        gradients = {}

        # FIXME: I guess one downside to this is that, since I don't store the
        # backwards pass anywhere, it's going to be recomputed in a lot of
        # places if I need multiple gradients.
        stack: list[tuple[Tensor, float]] = [(self, np.array([1]))]
        while stack:
            current_variable, current_value = stack.pop()
            for child, name, op in current_variable.operations:
                child_grad = gradients.get(child, 0)
                child_value = op(current_value)
                gradients[child] = child_grad + child_value
                stack.append((child, child_value))

        return gradients

    def __add__(self, b: Tensor) -> Tensor:
        value = self.value + b.value
        operations = [
            (self, "add", lambda acc: acc),  # conceptually, 1*acc
            (b, "add", lambda acc: acc),
        ]
        return Tensor(value, operations)

    def __sub__(self, b: Tensor) -> Tensor:
        value = self.value - b.value
        operations = [(self, "sub", lambda acc: acc), (b, "sub", lambda acc: -acc)]
        return Tensor(value, operations)

    def __mul__(self, b: Tensor) -> Tensor:
        # elementwise
        value = self.value * b.value
        operations = [
            (self, "mul", lambda acc: acc * b.value),
            (b, "mul", lambda acc: acc * self.value),
        ]
        return Tensor(value, operations)

    def __neg__(self) -> Tensor:
        value = -self.value
        operations = [(self, "neg", lambda acc: -acc)]
        return Tensor(value, operations)

    def __truediv__(self, b: Tensor) -> Tensor:
        # elementwise
        value = self.value / b.value
        operations = [
            (self, "div", lambda acc: acc / b.value),
            (b, "div", lambda acc: -acc * self.value / (b.value**2)),
        ]
        return Tensor(value, operations)

    def matmul(self, matrix: Tensor) -> Tensor:
        # FIXME: this is where my broadcasting ignorance will really get me into trouble
        # hard assumption right now that self is a 1d vector
        # also that acc is a scalar
        # TODO: also i couldn't justify this local gradient term if asked, so work
        # through the math more closely
        value = np.matmul(self.value, matrix.value)
        operations = [
            (self, "matmul", lambda acc: np.matmul(acc.reshape(1, -1), matrix.value.T)),
            (
                matrix,
                "matmul",
                lambda acc: np.matmul(self.value.reshape(-1, 1), acc.reshape(1, -1)),
            ),
        ]
        return Tensor(value, operations)

    def exp(self) -> Tensor:
        value = np.exp(self.value)
        operations = [(self, "exp", lambda acc: acc * np.exp(self.value))]
        return Tensor(value, operations)
