import cupy as cp

from mytorch.gpu_tensor import GpuTensor


class Linear:
    # y = xA.T + b
    def __init__(self, in_size: int, out_size: int, bias: bool):
        # FIXME: need to worry more about how i initialize weights
        weight_data = cp.random.random((in_size, out_size))
        # FIXME: this is just a hack to keep things from blowing up on big sizes
        weight_data = weight_data / weight_data.size
        self.weights = GpuTensor(weight_data, frozen=False)
        self.bias = GpuTensor(cp.array([[0.0] * out_size]), frozen=False)

    def forward(self, input: GpuTensor) -> GpuTensor:
        result = input @ self.weights + self.bias
        return result


class Sigmoid:
    def forward(self, input: GpuTensor) -> GpuTensor:
        # TODO: broadcasting sure would make this part easier
        one = GpuTensor(cp.ones_like(input.value))
        result = one / (one + (-input).exp())
        return result
