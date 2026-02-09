from __future__ import annotations
from typing import Callable
import numpy as np
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
        # FIXME: I've lost track of the concept of the "path_value"
        stack = list(self.operations)
        while stack:
            print(stack)
            if len(stack) > 5:
                1/0
            op = stack.pop()
            target_tensor = op[0]
            for child in target_tensor.operations:
                stack.append(op)

            current = gradients.get(target_tensor, 0)
            gradients[target_tensor] = current + op[2](current)

        return gradients


    def __add__(self: Tensor, b: Tensor) -> Tensor:
        value = self.value + b.value
        operations = [
            (self, "add", lambda acc: acc), # conceptually, 1*acc
            (b, "add", lambda acc: acc)
        ]
        return Tensor(value, operations)


    def __sub__(self: Tensor, b: Tensor) -> Tensor:
        value = self.value - b.value
        operations = [
            (self, "sub", lambda acc: acc),
            (b, "sub", lambda acc: -acc)
        ]
        return Tensor(value, operations)

    def __mul__(self: Tensor, b: Tensor) -> Tensor:
        # elementwise
        value = self.value * b.value
        operations = [
            (self, "mul", lambda acc: acc * b.value),
            (b, "mul", lambda acc: acc * self.value)
        ]
        return Tensor(value, operations)

    def __neg__(self: Tensor) -> Tensor:
        value = -self.value
        operations = [
            (self, "neg", lambda acc: -acc)
        ]
        return Tensor(value, operations)

    def __truediv__(self: Tensor, b: Tensor) -> Tensor:
        # elementwise
        value = self.value / b.value
        operations = [
            (self, "div", lambda acc: acc * b.value),
            (b, "div", lambda acc: acc * self.value)
        ]
        return Tensor(value, operations)