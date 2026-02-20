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


def test_self_attention():
    layer = gpu_layers.SelfAttention(3, 1)
    # force the weights into a known state
    layer.Q.value = cp.array([[0.0], [0.0], [-0.0]], dtype=cp.float32)
    layer.K.value = cp.array([[0.2], [0.1], [0.1]], dtype=cp.float32)
    layer.V.value = cp.array([[-0.2], [0.2], [0.2]], dtype=cp.float32)

    # we want to pass in (1,1,3) - the most basic test possible
    input = GpuTensor(cp.array([0.2, 0.4, 0.1], dtype=cp.float32).reshape((1, 1, 3)))
    result = layer.forward(input)
    # now we're at (1,1,1), so we can just use error as a loss
    target = GpuTensor(cp.array([[[2]]], dtype=cp.float32))
    loss = result - target
    grads = loss.compute_gradient()
    relevant_grad = grads[layer.Q][1, 0]

    eps = 1e-4
    layer.Q.value[1, 0] += eps
    new_result = layer.forward(input)
    new_loss = new_result - target

    delta = (new_loss.value - loss.value) / eps

    print(delta)
    print(relevant_grad)
    assert cp.allclose(delta.value, relevant_grad)
