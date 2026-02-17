import cupy as cp

from mytorch import kernels
from mytorch.gpu_tensor import GpuTensor


class Linear:
    # y = xA.T + b
    def __init__(self, in_size: int, out_size: int, bias: bool):
        # TODO: We probably want to be more flexible about how we do this
        weight_data = cp.random.normal(0, 0.2, (in_size, out_size))
        # FIXME: this is just a hack to keep things from blowing up on big sizes
        self.weights = GpuTensor(weight_data, frozen=False)
        self.bias = GpuTensor(cp.array([[0.0] * out_size]), frozen=False)

    def forward(self, input: GpuTensor) -> GpuTensor:
        result = input @ self.weights + self.bias
        return result


class Sigmoid:
    def forward(self, input: GpuTensor) -> GpuTensor:
        # TODO: broadcasting sure would make this part easier
        p = kernels.logistic(input.value)
        operations = [(input, "sigmoid", lambda acc: acc * (p * (1 - p)))]

        return GpuTensor(value=p, operations=operations)
