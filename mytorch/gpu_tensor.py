from __future__ import annotations

import cupy as cp

from mytorch import kernels

# TODO: for now, I'm just mirroing the code in tensor.py. In the long run
# it would be nice to unify these in some way

# type Operation = tuple[Tensor, str, Callable]

add_code = r"""
extern "C" __global__
void add_kernel(const float* a, const float* b, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride){
        out[i] = a[i] + b[i];
    }
}
"""
add_kernel = cp.RawKernel(add_code, "add_kernel")


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

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            return acc

        def local_grad_b(acc: cp.ndarray) -> cp.ndarray:
            return acc

        operations = [
            (self, "add", local_grad_self),  # conceptually, 1*acc
            (b, "add", local_grad_b),
        ]
        return GpuTensor(result, operations)

    def __sub__(self, b: GpuTensor) -> GpuTensor:
        result = kernels.sub(self.value, b.value)

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            return acc

        def local_grad_b(acc: cp.ndarray) -> cp.ndarray:
            return kernels.neg(acc)

        operations = [(self, "sub", local_grad_self), (b, "sub", local_grad_b)]
        return GpuTensor(result, operations)

    def __mul__(self, b: GpuTensor) -> GpuTensor:
        # elementwise
        result = kernels.mul(self.value, b.value)

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            return kernels.mul(acc, b.value)

        def local_grad_b(acc: cp.ndarray) -> cp.ndarray:
            return kernels.mul(acc, self.value)

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

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            return kernels.div(acc, b.value)

        def local_grad_b(acc: cp.ndarray) -> cp.ndarray:
            return kernels.div_local_grad(acc, self.value, b.value)

        operations = [
            (self, "div", local_grad_self),
            (b, "div", local_grad_b),
        ]
        return GpuTensor(result, operations)

    def __matmul__(self, b: GpuTensor) -> GpuTensor:
        # FIXME: broadcasting
        result = kernels.matmul(self.value, b.value)

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            # FIXME: i should implement swapaxes myself
            grad = kernels.matmul(acc, cp.swapaxes(b.value, -2, -1))
            return grad

        def local_grad_b(acc: cp.ndarray) -> cp.ndarray:
            grad = kernels.matmul(cp.swapaxes(self.value, -2, -1), acc)
            return grad

        operations = [
            (self, "matmul", local_grad_self),
            (b, "matmul", local_grad_b),
        ]
        return GpuTensor(result, operations)
