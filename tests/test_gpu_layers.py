import cupy as cp
from utils import evaluate_empirical_grad

from mytorch import layers
from mytorch.tensor import Tensor


def test_linear():
    layer = layers.Linear(3, 2, False)
    vec = Tensor(cp.array([[1, 2, 3, 4, 5, 6]], dtype=cp.float32).reshape(2, 1, 3))

    expected = vec.value @ layer.weights.value
    actual = layer.forward(vec)

    assert cp.allclose(expected, actual.value)

    layer = layers.Linear(3, 2, True)
    layer.bias = Tensor(cp.array([[1.0] * 2], dtype=cp.float32))

    expected = vec.value @ layer.weights.value + layer.bias.value
    actual = layer.forward(vec)

    assert cp.allclose(expected, actual.value)


def test_sigmoid():
    layer = layers.Sigmoid()
    vec = Tensor(cp.array([[0, 1, 2, 3, 4, 5]], dtype=cp.float32).reshape(2, 1, 3))

    expected = 1 / (1 + cp.exp(-vec.value))
    actual = layer.forward(vec)

    assert cp.allclose(expected, actual.value)

    ONE = cp.array([1.0], dtype=cp.float32)
    assert len(actual.operations) == 1
    backpropped = actual.operations[0][2](ONE)
    expected = expected * (1 - expected)
    assert cp.allclose(expected, backpropped)


def test_layer_norm():
    layer = layers.LayerNorm(eps=0)
    tensor = Tensor(cp.array([[2, 1, 0], [1, 1, 0]], dtype=cp.float32))

    expected = cp.array(
        [[1.2247448, 0.0, -1.2247448], [0.7071067, 0.7071067, -1.4142135]],
        dtype=cp.float32,
    )

    result = layer.forward(tensor)

    assert cp.allclose(expected, result.value)


def test_self_attention_grad():
    layer = layers.SelfAttention(4, 1)
    tensor = Tensor(cp.array([0.2, 0.4, 0.1, 0.0], dtype=cp.float32).reshape((1, 1, 4)))
    layer.forward(tensor)

    def loss_func(x):
        result = layer.forward(tensor).sum()
        return result

    evaluate_empirical_grad(layer.weights, loss_func)


def test_linear_grad(two_d_tensor):
    layer = layers.Linear(two_d_tensor.value.shape[-1], 1, False)

    def loss_func(x):
        result = layer.forward(two_d_tensor).sum()
        return result

    evaluate_empirical_grad(two_d_tensor, loss_func)


def test_sigmoid_grad(two_d_tensor):
    layer = layers.Sigmoid()

    def loss_func(x):
        result = layer.forward(two_d_tensor).sum()
        return result

    evaluate_empirical_grad(two_d_tensor, loss_func)


def test_layer_norm_grad(two_d_tensor):
    layer = layers.LayerNorm(eps=0)

    def loss_func(x):
        result = layer.forward(two_d_tensor).sum()
        return result

    evaluate_empirical_grad(two_d_tensor, loss_func)
