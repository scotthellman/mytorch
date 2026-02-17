import cupy as cp

from mytorch import gpu_layers
from mytorch.gpu_tensor import GpuTensor


def test_linear():
    layer = gpu_layers.Linear(3, 2, False)
    vec = GpuTensor(cp.array([[1, 2, 3, 4, 5, 6]], dtype=cp.float32).reshape(2, 1, 3))

    expected = vec.value @ layer.weights.value
    actual = layer.forward(vec)

    assert cp.allclose(expected, actual.value)

    layer = gpu_layers.Linear(3, 2, True)
    layer.bias = GpuTensor(cp.array([[1.0] * 2], dtype=cp.float32))

    expected = vec.value @ layer.weights.value + layer.bias.value
    actual = layer.forward(vec)

    assert cp.allclose(expected, actual.value)


def test_sigmoid():
    layer = gpu_layers.Sigmoid()
    vec = GpuTensor(cp.array([[0, 1, 2, 3, 4, 5]], dtype=cp.float32).reshape(2, 1, 3))

    expected = 1 / (1 + cp.exp(-vec.value))
    actual = layer.forward(vec)

    assert cp.allclose(expected, actual.value)

    ONE = cp.array([1.0], dtype=cp.float32)
    assert len(actual.operations) == 1
    backpropped = actual.operations[0][2](ONE)
    expected = expected * (1 - expected)
    assert cp.allclose(expected, backpropped)
