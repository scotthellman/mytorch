import cupy as cp

from mytorch import kernels
from mytorch.gpu_tensor import GpuTensor


class Linear:
    # y = xA.T + b
    def __init__(self, in_size: int, out_size: int, bias: bool):
        # TODO: We probably want to be more flexible about how we do this
        weight_data = cp.random.normal(0, 0.02, (in_size, out_size))
        # FIXME: this is just a hack to keep things from blowing up on big sizes
        self.weights = GpuTensor(weight_data, frozen=False)
        if bias:
            self.bias = GpuTensor(cp.array([[0.0] * out_size]), frozen=False)
        else:
            self.bias = None

    def forward(self, input: GpuTensor) -> GpuTensor:
        if self.bias:
            result = input @ self.weights + self.bias
        else:
            result = input @ self.weights
        return result


class Sigmoid:
    def forward(self, input: GpuTensor) -> GpuTensor:
        p = kernels.logistic(input.value)
        operations = [(input, "sigmoid", lambda acc: acc * (p * (1 - p)))]

        return GpuTensor(value=p, operations=operations)


class LayerNorm:
    def __init__(self, eps: float = 1e-5):
        # TODO: pytorch lets this learn an affine transform
        self.eps = eps

    def forward(self, input: GpuTensor) -> GpuTensor:
        # So, for this to work, we need:
        # mean, variance, sqrt
        # FIXME: We can worry about a fused kernel after, let's just get it working
        # FIXME: this will only work after linear layers, it's not as general as pytorch's
        # (assumes input is 2d)
        eps_vec = GpuTensor(cp.ones_like(input.value) * self.eps)
        demeaned = input - input.mean(axis=-1)
        variance = input.var(axis=-1)
        normed = demeaned / (variance + eps_vec).sqrt()
        return normed
