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
        names = set()
        while stack:
            current_variable, current_value = stack.pop()
            for child, name, op in current_variable.operations:
                names.add(name)
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
            return handle_broadcasting(acc, self_broadcast_axes, self.value.shape)

        def local_grad_b(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return handle_broadcasting(acc, b_broadcast_axes, b.value.shape)

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
            return handle_broadcasting(acc, self_broadcast_axes, self.value.shape)

        def local_grad_b(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return handle_broadcasting(-acc, b_broadcast_axes, b.value.shape)

        operations = [(self, "sub", local_grad_self), (b, "sub", local_grad_b)]
        return Tensor(result, operations)

    def __mul__(self, b: Tensor) -> Tensor:
        # elementwise
        result = self.value * b.value
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        b_broadcast_axes = compute_broadcast_axes(b.value.shape, result.shape)

        def local_grad_self(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return handle_broadcasting(
                acc * b.value, self_broadcast_axes, self.value.shape
            )

        def local_grad_b(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return handle_broadcasting(
                acc * self.value, b_broadcast_axes, b.value.shape
            )

        operations = [
            (self, "mul", local_grad_self),
            (b, "mul", local_grad_b),
        ]
        return Tensor(result, operations)

    def __neg__(self) -> Tensor:
        value = -self.value

        def local_grad(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return -acc

        operations = [(self, "neg", local_grad)]
        return Tensor(value, operations)

    def __truediv__(self, b: Tensor) -> Tensor:
        # elementwise
        result = self.value / b.value
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        b_broadcast_axes = compute_broadcast_axes(b.value.shape, result.shape)

        def local_grad_self(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return handle_broadcasting(
                acc / b.value, self_broadcast_axes, self.value.shape
            )

        def local_grad_b(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return handle_broadcasting(
                -acc * self.value / (b.value**2), b_broadcast_axes, b.value.shape
            )

        operations = [
            (self, "div", local_grad_self),
            (b, "div", local_grad_b),
        ]
        return Tensor(result, operations)

    def __matmul__(self, b: Tensor) -> Tensor:
        result = np.matmul(self.value, b.value)
        self_broadcast_axes = compute_broadcast_axes(
            self.value.shape, result.shape, matmul=True
        )
        b_broadcast_axes = compute_broadcast_axes(
            b.value.shape, result.shape, matmul=True
        )

        def local_grad_self(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            grad = np.matmul(acc, np.swapaxes(b.value, -2, -1))
            return handle_broadcasting(grad, self_broadcast_axes, self.value.shape)

        def local_grad_b(acc: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            grad = np.matmul(np.swapaxes(self.value, -2, -1), acc)
            return handle_broadcasting(grad, b_broadcast_axes, b.value.shape)

        operations = [
            (self, "matmul", local_grad_self),
            (b, "matmul", local_grad_b),
        ]
        return Tensor(result, operations)

    def exp(self) -> Tensor:
        value = np.exp(self.value)
        operations = [(self, "exp", lambda acc: acc * np.exp(self.value))]
        return Tensor(value, operations)


def compute_broadcast_axes(
    start_shape: tuple[int], broadcast_shape: tuple[int], matmul: bool = False
) -> tuple[int]:
    altered_axes = []
    start = -3 if matmul else -1
    for i in range(start, -len(broadcast_shape) - 1, -1):
        current = broadcast_shape[i]
        if abs(i) > len(start_shape) or current != start_shape[i]:
            altered_axes.append(len(broadcast_shape) + i)

    return tuple(altered_axes)


def handle_broadcasting(
    path_gradient: npt.NDArray[np.floating],
    broadcast_axes: tuple[int],
    target_shape: tuple[int],
) -> npt.NDArray[np.floating]:
    if broadcast_axes:
        return np.sum(path_gradient, axis=broadcast_axes).reshape(target_shape)
    return path_gradient
