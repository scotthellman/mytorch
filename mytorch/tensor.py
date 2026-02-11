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
        stack: list[tuple[Tensor, npt.NDArray[np.floating]]] = [
            (self, np.ones(self.value.shape, dtype=float))
        ]
        while stack:
            current_variable, current_value = stack.pop()
            for child, name, op in current_variable.operations:
                child_grad = gradients.get(child, 0)
                child_value = op(current_value)
                gradients[child] = child_grad + child_value
                stack.append((child, child_value))

        return gradients

    def __add__(self, b: Tensor) -> Tensor:
        result = self.value + b.value
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        b_broadcast_axes = compute_broadcast_axes(b.value.shape, result.shape)

        def local_grad_self(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return handle_broadcasting(acc, self_broadcast_axes)

        def local_grad_b(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return handle_broadcasting(acc, b_broadcast_axes)

        operations = [
            (self, "add", local_grad_self),  # conceptually, 1*acc
            (b, "add", local_grad_b),
        ]
        return Tensor(result, operations)

    def __sub__(self, b: Tensor) -> Tensor:
        result = self.value - b.value
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        b_broadcast_axes = compute_broadcast_axes(b.value.shape, result.shape)

        def local_grad_self(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return handle_broadcasting(acc, self_broadcast_axes)

        def local_grad_b(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return handle_broadcasting(-acc, b_broadcast_axes)

        operations = [(self, "sub", local_grad_self), (b, "sub", local_grad_b)]
        return Tensor(result, operations)

    def __mul__(self, b: Tensor) -> Tensor:
        # elementwise
        result = self.value * b.value
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        b_broadcast_axes = compute_broadcast_axes(b.value.shape, result.shape)

        def local_grad_self(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return handle_broadcasting(acc * b.value, self_broadcast_axes)

        def local_grad_b(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return handle_broadcasting(acc * self.value, b_broadcast_axes)

        operations = [
            (self, "mul", local_grad_self),
            (b, "mul", local_grad_b),
        ]
        return Tensor(result, operations)

    def __neg__(self) -> Tensor:
        value = -self.value
        operations = [(self, "neg", lambda acc: -acc)]
        return Tensor(value, operations)

    def __truediv__(self, b: Tensor) -> Tensor:
        # elementwise
        result = self.value / b.value
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        b_broadcast_axes = compute_broadcast_axes(b.value.shape, result.shape)

        def local_grad_self(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return handle_broadcasting(acc / b.value, self_broadcast_axes)

        def local_grad_b(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return handle_broadcasting(
                -acc * self.value / (b.value**2), b_broadcast_axes
            )

        operations = [
            (self, "div", local_grad_self),
            (b, "div", local_grad_b),
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


def compute_broadcast_axes(
    start_shape: tuple[int], broadcast_shape: tuple[int]
) -> tuple[int]:
    altered_axes = []
    for i in range(-1, -len(broadcast_shape) - 1, -1):
        current = broadcast_shape[i]
        if abs(i) > len(start_shape) or current != start_shape[i]:
            altered_axes.append(len(broadcast_shape) + i)

    return tuple(altered_axes)


def handle_broadcasting(
    path_gradient: npt.NDArray[np.floating], broadcast_axes: tuple[int] | None = None
) -> npt.NDArray[np.floating]:
    if broadcast_axes:
        return np.sum(path_gradient, axis=broadcast_axes, keepdims=True)
    return path_gradient
