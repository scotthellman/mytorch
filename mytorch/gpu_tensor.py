from __future__ import annotations

import cupy as cp
import cupy.typing as cpt

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
        value: cpt.NDArray[cp.float32],
        operations: list | None = None,
        frozen=True,
    ):
        self.value = cp.asarray(value, dtype=cp.float32)
        self.frozen = frozen
        if operations is None:
            self.operations = []
        else:
            self.operations = operations

    def __add__(self, b: GpuTensor) -> GpuTensor:
        # FIXME: hey the ghost of broadcasting is back! just ignoring it for now
        result_shape = self.value.shape
        result = cp.empty(result_shape, dtype=cp.float32)
        n = result.size
        add_kernel(
            (4,),
            (256,),
            (self.value, b.value, result, n),
        )

        def local_grad_self(acc: cpt.NDArray[cp.floating]) -> cpt.NDArray[cp.floating]:
            return acc

        def local_grad_b(acc: cpt.NDArray[cp.floating]) -> cpt.NDArray[cp.floating]:
            return acc

        operations = [
            (self, "add", local_grad_self),  # conceptually, 1*acc
            (b, "add", local_grad_b),
        ]
        return GpuTensor(result, operations)
