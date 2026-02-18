import cupy as cp

from mytorch import kernels
from mytorch.gpu_tensor import GpuTensor


class Linear:
    # y = xA.T + b
    def __init__(self, in_size: int, out_size: int, bias: bool):
        # TODO: We probably want to be more flexible about how we do this
        weight_data = cp.random.normal(0, 0.02, (in_size, out_size), dtype=cp.float32)
        self.weights = GpuTensor(weight_data, frozen=False)
        if bias:
            self.bias = GpuTensor(
                cp.array([[0.0] * out_size], dtype=cp.float32), frozen=False
            )
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


class Embedding:
    def __init__(self, vocab_size: int, embedding_size: int):
        weight_data = cp.random.normal(
            0, 0.02, (vocab_size, embedding_size), dtype=cp.float32
        )
        self.weights = weight_data

    def forward(self, input: GpuTensor) -> GpuTensor:
        # making a hard assumption here: input is an int tensor
        embedded = self.weights[input.value]

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            # so. the local grad here is just selecting the relevant rows
            # so we need to expand acc out, 0ing any rows that weren't selected
            # TODO: shame we have to make this big matrix. Something to think about
            out = cp.zeros_like(self.weights)
            out[input.value] = acc
            return out

        return GpuTensor(value=embedded, operations=[self, "embed", local_grad])
