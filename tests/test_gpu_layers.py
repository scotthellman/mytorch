import cupy as cp
from utils import evaluate_empirical_grad

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


def test_layer_norm():
    layer = gpu_layers.LayerNorm(eps=0)
    tensor = GpuTensor(cp.array([[2, 1, 0], [1, 1, 0]], dtype=cp.float32))

    expected = cp.array(
        [[1.2247448, 0.0, -1.2247448], [0.7071067, 0.7071067, -1.4142135]],
        dtype=cp.float32,
    )

    result = layer.forward(tensor)

    assert cp.allclose(expected, result.value)


def test_self_attention_grad():
    layer = gpu_layers.SelfAttention(4, 1)
    tensor = GpuTensor(
        cp.array([0.2, 0.4, 0.1, 0.0], dtype=cp.float32).reshape((1, 1, 4))
    )

    def loss_func(x):
        result = layer.forward(tensor).sum()
        return result

    evaluate_empirical_grad(layer.weights, loss_func)


def test_linear_grad(two_d_tensor):
    layer = gpu_layers.Linear(two_d_tensor.value.shape[-1], 1, False)

    def loss_func(x):
        result = layer.forward(two_d_tensor).sum()
        return result

    evaluate_empirical_grad(two_d_tensor, loss_func)


def test_sigmoid_grad(two_d_tensor):
    layer = gpu_layers.Sigmoid()

    def loss_func(x):
        result = layer.forward(two_d_tensor).sum()
        return result

    evaluate_empirical_grad(two_d_tensor, loss_func)


def test_layer_norm_grad(two_d_tensor):
    layer = gpu_layers.LayerNorm(eps=0)

    def loss_func(x):
        result = layer.forward(two_d_tensor).sum()
        return result

    evaluate_empirical_grad(two_d_tensor, loss_func)
