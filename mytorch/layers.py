import numpy as np

from mytorch.tensor import Tensor


class Linear:
    # y = xA.T + b
    def __init__(self, in_size: int, out_size: int, bias: bool):
        # FIXME: need to worry more about how i initialize weights
        weight_data = np.random.random((in_size, out_size))
        self.weights = Tensor(weight_data, frozen=False)
        self.bias = Tensor(np.array([0.0]), frozen=False)

    def forward(self, input: Tensor) -> Tensor:
        result = input.matmul(self.weights) + self.bias
        return result


class Sigmoid:
    def forward(self, input: Tensor) -> Tensor:
        # TODO: broadcasting sure would make this part easier
        one = Tensor(np.ones_like(input.value))
        result = one / (one + -input.exp())
        return result
