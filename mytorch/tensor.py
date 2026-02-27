from __future__ import annotations

import cupy as cp

from mytorch import kernels


class Tensor:
    def __init__(
        self,
        value: cp.ndarray,
        operations: list | None = None,
        frozen=True,
    ):
        self.value: cp.ndarray = value
        self.frozen = frozen
        if operations is None:
            self.operations = []
        else:
            self.operations = operations
        self.reset()

    def reset(self):
        self.grad = cp.zeros_like(self.value)

    def compute_gradient(self) -> dict["Tensor", cp.ndarray]:
        gradients = {}

        stack: list[tuple[Tensor, cp.ndarray]] = [
            (self, cp.ones(self.value.shape, dtype=cp.float32))
        ]
        while stack:
            current_variable, current_value = stack.pop()
            for child, name, op in current_variable.operations:
                child_grad = child.grad
                child_value = op(current_value)
                child.grad = child_value + child_grad
                stack.append((child, child_value))

        return gradients

    def __getitem__(self, key) -> Tensor:
        result = self.value[key]

        def local_grad(acc: cp.ndarray):
            grad = cp.zeros_like(self.value)
            grad[key] = acc
            return grad

        return Tensor(result, [(self, "index", local_grad)])

    def reshape(self, shape: tuple[int]) -> Tensor:
        result = self.value.reshape(shape)

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            return acc.reshape(self.value.shape)

        operations = [
            (self, "reshape", local_grad),
        ]

        return Tensor(result, operations)

    def __add__(self, b: Tensor) -> Tensor:
        result = kernels.add(self.value, b.value)
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        b_broadcast_axes = compute_broadcast_axes(b.value.shape, result.shape)

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting(acc, self_broadcast_axes, self.value.shape)

        def local_grad_b(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting(acc, b_broadcast_axes, b.value.shape)

        operations = [
            (self, "add", local_grad_self),  # conceptually, 1*acc
            (b, "add", local_grad_b),
        ]
        return Tensor(result, operations)

    def __sub__(self, b: Tensor) -> Tensor:
        result = kernels.sub(self.value, b.value)
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        b_broadcast_axes = compute_broadcast_axes(b.value.shape, result.shape)

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting(acc, self_broadcast_axes, self.value.shape)

        def local_grad_b(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting(
                kernels.neg(acc), b_broadcast_axes, b.value.shape
            )

        operations = [(self, "sub", local_grad_self), (b, "sub", local_grad_b)]
        return Tensor(result, operations)

    def __mul__(self, b: Tensor) -> Tensor:
        # elementwise
        result = kernels.mul(self.value, b.value)
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        b_broadcast_axes = compute_broadcast_axes(b.value.shape, result.shape)

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting(
                kernels.mul(acc, b.value), self_broadcast_axes, self.value.shape
            )

        def local_grad_b(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting(
                kernels.mul(acc, self.value), b_broadcast_axes, b.value.shape
            )

        operations = [
            (self, "mul", local_grad_self),
            (b, "mul", local_grad_b),
        ]
        return Tensor(result, operations)

    def __neg__(self) -> Tensor:
        result = kernels.neg(self.value)

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            return kernels.neg(acc)

        operations = [(self, "neg", local_grad)]
        return Tensor(result, operations)

    def __truediv__(self, b: Tensor) -> Tensor:
        # elementwise
        result = kernels.div(self.value, b.value)
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        b_broadcast_axes = compute_broadcast_axes(b.value.shape, result.shape)

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting(
                kernels.div(acc, b.value), self_broadcast_axes, self.value.shape
            )

        def local_grad_b(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting(
                kernels.div_local_grad(acc, self.value, b.value),
                b_broadcast_axes,
                b.value.shape,
            )

        operations = [
            (self, "div", local_grad_self),
            (b, "div", local_grad_b),
        ]
        return Tensor(result, operations)

    def __matmul__(self, b: Tensor) -> Tensor:
        result = kernels.matmul(self.value, b.value)
        self_broadcast_axes = compute_broadcast_axes(
            self.value.shape, result.shape, matmul=True
        )
        b_broadcast_axes = compute_broadcast_axes(
            b.value.shape, result.shape, matmul=True
        )

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            # TODO: arguably I should implement this myself
            # FIXME: especially now that i have to copy to make hte view concrete
            grad = kernels.matmul(acc, cp.swapaxes(b.value, -2, -1).copy())
            return handle_broadcasting(grad, self_broadcast_axes, self.value.shape)

        def local_grad_b(acc: cp.ndarray) -> cp.ndarray:
            grad = kernels.matmul(cp.swapaxes(self.value, -2, -1).copy(), acc)
            return handle_broadcasting(grad, b_broadcast_axes, b.value.shape)

        operations = [
            (self, "matmul", local_grad_self),
            (b, "matmul", local_grad_b),
        ]
        return Tensor(result, operations)

    def add_constant(self, c: float) -> Tensor:
        # FIXME: do this myself
        result = self.value + c
        operations = [(self, "+c", lambda acc: acc)]
        return Tensor(result, operations)

    def mult_constant(self, c: float) -> Tensor:
        # FIXME: do this myself
        result = self.value * c
        operations = [(self, "*c", lambda acc: c * acc)]
        return Tensor(result, operations)

    def exp(self) -> Tensor:
        result = kernels.exp(self.value)
        operations = [(self, "exp", lambda acc: acc * kernels.exp(self.value))]
        return Tensor(result, operations)

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        # FIXME: need to do this myself
        result = cp.sum(self.value, axis=axis, keepdims=keepdims)

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            grad = acc * cp.ones_like(self.value)
            return grad

        operations = [(self, "sum", local_grad_self)]
        return Tensor(result, operations)

    def sqrt(self) -> Tensor:
        result = kernels.sqrt(self.value)
        half = cp.ones_like(self.value) * 0.5
        operations = [(self, "sqrt", lambda acc: acc * half / kernels.sqrt(self.value))]
        return Tensor(result, operations)

    def mean(self, axis: int | None = None) -> Tensor:
        # FIXME: need to do this myself
        result = cp.mean(self.value, axis=axis, keepdims=True)

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            n = self.value.size if axis is None else self.value.shape[axis]
            grad = acc * cp.ones_like(self.value) / n
            return grad

        operations = [(self, "sum", local_grad_self)]
        return Tensor(result, operations)

    def var(self, axis: int) -> Tensor:
        # FIXME: need to do this myself
        result = cp.var(self.value, axis=axis, keepdims=True)
        mean = cp.mean(self.value, axis=axis, keepdims=True)
        n = self.value.shape[axis]

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            grad = acc * 2 / n * (self.value - mean)
            return grad

        operations = [(self, "sum", local_grad_self)]
        return Tensor(result, operations)

    def elu(self) -> Tensor:
        # FIXME: need to do this myself
        mask = self.value < 0
        result = cp.copy(self.value)
        result[mask] = cp.exp(result[mask]) - 1

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            # nb assuming alpha is 1
            grad = cp.ones_like(acc)
            grad[mask] += result[mask]
            return acc * grad

        operations = [(self, "elu", local_grad_self)]

        return Tensor(result, operations)

    def cumsum(self, axis: int) -> Tensor:
        # FIXME: need to do this myself
        result = cp.cumsum(self.value, axis=axis)

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            return cp.flip(cp.cumsum(cp.flip(acc, axis=axis), axis=axis), axis=axis)

        operations = [(self, "cumsum", local_grad_self)]

        return Tensor(result, operations)

    def transpose_last(self) -> Tensor:
        # NOTE: I give myself permission to not do this myself,
        # I've been leaving indexing stuff to cupy
        return self.transpose(-1, -2)

    def transpose(self, i: int, j: int) -> Tensor:
        # NOTE: I give myself permission to not do this myself,
        # I've been leaving indexing stuff to cupy
        result = self.value.swapaxes(i, j).copy()

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            return acc.swapaxes(i, j).copy()

        operations = [(self, "T", local_grad_self)]

        return Tensor(result, operations)

    def permute(self, indices) -> Tensor:
        # NOTE: I give myself permission to not do this myself,
        # I've been leaving indexing stuff to cupy
        result = self.value[..., indices].copy()

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            return acc[..., indices].copy()

        operations = [(self, "swapaxes", local_grad_self)]

        return Tensor(result, operations)


# FIXME: yet another symptom of my messy cpu/gpu divide


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
    path_gradient: cp.ndarray,
    broadcast_axes: tuple[int],
    target_shape: tuple[int],
) -> cp.ndarray:
    if broadcast_axes:
        return cp.sum(path_gradient, axis=broadcast_axes).reshape(target_shape)
    return path_gradient
