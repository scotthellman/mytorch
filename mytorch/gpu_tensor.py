from __future__ import annotations

import cupy as cp

from mytorch import kernels

# TODO: for now, I'm just mirroing the code in tensor.py. In the long run
# it would be nice to unify these in some way

# type Operation = tuple[Tensor, str, Callable]


class GpuTensor:
    def __init__(
        self,
        value: cp.ndarray[cp.float32],
        operations: list | None = None,
        frozen=True,
    ):
        self.value = cp.asarray(value, dtype=cp.float32)
        self.frozen = frozen
        if operations is None:
            self.operations = []
        else:
            self.operations = operations

    def compute_gradient(self) -> dict["GpuTensor", cp.ndarray]:
        # FIXME: literally the same code as in tensor.py, so get rid of the duplication
        gradients = {}

        stack: list[tuple[GpuTensor, cp.ndarray]] = [
            (self, cp.ones(self.value.shape, dtype=cp.float32))
        ]
        while stack:
            current_variable, current_value = stack.pop()
            for child, name, op in current_variable.operations:
                child_grad = gradients.get(child, 0)
                child_value = op(current_value)
                gradients[child] = child_grad + child_value
                stack.append((child, child_value))

        return gradients

    def __add__(self, b: GpuTensor) -> GpuTensor:
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
        return GpuTensor(result, operations)

    def __sub__(self, b: GpuTensor) -> GpuTensor:
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
        return GpuTensor(result, operations)

    def __mul__(self, b: GpuTensor) -> GpuTensor:
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
        return GpuTensor(result, operations)

    def __neg__(self) -> GpuTensor:
        result = kernels.neg(self.value)

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            return kernels.neg(acc)

        operations = [(self, "neg", local_grad)]
        return GpuTensor(result, operations)

    def __truediv__(self, b: GpuTensor) -> GpuTensor:
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
        return GpuTensor(result, operations)

    def __matmul__(self, b: GpuTensor) -> GpuTensor:
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
        return GpuTensor(result, operations)

    def exp(self) -> GpuTensor:
        result = kernels.exp(self.value)
        operations = [(self, "exp", lambda acc: acc * kernels.exp(self.value))]
        return GpuTensor(result, operations)

    def sum(self, axis: int | None = None) -> GpuTensor:
        # FIXME: need to do this myself
        result = cp.sum(self.value, axis=axis)

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            grad = acc * cp.ones_like(self.value)
            return grad

        operations = [(self, "sum", local_grad_self)]
        return GpuTensor(result, operations)

    def sqrt(self) -> GpuTensor:
        result = kernels.sqrt(self.value)
        half = cp.ones_like(self.value) * 0.5
        operations = [(self, "sqrt", lambda acc: acc * half / kernels.sqrt(self.value))]
        return GpuTensor(result, operations)

    def mean(self, axis: int) -> GpuTensor:
        # FIXME: need to do this myself
        result = cp.mean(self.value, axis=axis, keepdims=True)
        n = self.value.shape[axis]

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            grad = acc * cp.ones_like(self.value) / n
            return grad

        operations = [(self, "sum", local_grad_self)]
        return GpuTensor(result, operations)

    def var(self, axis: int) -> GpuTensor:
        # FIXME: need to do this myself
        result = cp.var(self.value, axis=axis, keepdims=True)
        mean = cp.mean(self.value, axis=axis, keepdims=True)
        n = self.value.shape[axis]

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            grad = acc * 2 / n * (self.value - mean)
            return grad

        operations = [(self, "sum", local_grad_self)]
        return GpuTensor(result, operations)


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
