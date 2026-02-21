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


def test_self_attention_grad():
    layer = gpu_layers.SelfAttention(3, 1)
    # force the weights into a known state
    layer.Q.value = cp.array([[0.0], [0.0], [-0.0]], dtype=cp.float32)
    layer.K.value = cp.array([[0.2], [0.1], [0.1]], dtype=cp.float32)
    layer.V.value = cp.array([[-0.2], [0.2], [0.2]], dtype=cp.float32)
    print("Q", layer.Q)
    print("K", layer.K)
    print("V", layer.V)

    # we want to pass in (1,1,3) - the most basic test possible
    input = GpuTensor(cp.array([0.2, 0.4, 0.1], dtype=cp.float32).reshape((1, 1, 3)))
    result = layer.forward(input)
    # now we're at (1,1,1), so we can just use error as a loss
    target = GpuTensor(cp.array([[[2]]], dtype=cp.float32))
    loss = result - target
    grads = loss.compute_gradient()

    eps = 1e-4
    for weights in layer.params:
        for i in range(weights.value.shape[0]):
            for j in range(weights.value.shape[1]):
                old_val = float(weights.value[i, j])
                weights.value[i, j] += eps
                result = layer.forward(input)
                eps_loss = result - target
                empirical_grad = (eps_loss.value - loss.value) / eps
                computed_grad = grads[weights][i, j]
                assert cp.allclose(
                    empirical_grad, computed_grad, rtol=1e-3, atol=1e-3
                ), weights
                weights.value[i, j] = old_val


def test_linear_grad():
    layer = gpu_layers.Linear(3, 1, False)

    input = GpuTensor(cp.array([0.2, 0.4, 0.1], dtype=cp.float32).reshape((1, 1, 3)))
    result = layer.forward(input).sum()

    grads = result.compute_gradient()

    eps = 1e-4
    for weights in layer.params:
        for i in range(weights.value.shape[0]):
            for j in range(weights.value.shape[1]):
                old_val = float(weights.value[i, j])
                weights.value[i, j] += eps
                new_result = layer.forward(input).sum()
                empirical_grad = (new_result.value - result.value) / eps
                computed_grad = grads[weights][i, j]
                assert cp.allclose(
                    empirical_grad, computed_grad, rtol=1e-3, atol=1e-3
                ), weights
                weights.value[i, j] = old_val


def test_sigmoid_grad():
    layer = gpu_layers.Sigmoid()

    weights = GpuTensor(cp.array([0.2, 0.4, 0.1], dtype=cp.float32).reshape((1, 3)))
    result = layer.forward(weights).sum()

    grads = result.compute_gradient()

    eps = 1e-4
    for i in range(weights.value.shape[0]):
        for j in range(weights.value.shape[1]):
            old_val = float(weights.value[i, j])
            weights.value[i, j] += eps
            new_result = layer.forward(weights).sum()
            empirical_grad = (new_result.value - result.value) / eps
            computed_grad = grads[weights][i, j]
            assert cp.allclose(empirical_grad, computed_grad, rtol=1e-3, atol=1e-3), (
                weights
            )
            weights.value[i, j] = old_val


def test_layer_norm_grad():
    layer = gpu_layers.LayerNorm()

    weights = GpuTensor(
        cp.array(
            [
                [1, 0, -1],
                [1.3, 0.1, 0.1],
                [0.2, -0.4, 20.3],
            ],
            dtype=cp.float32,
        )
    )
    result = layer.forward(weights)
    # can't just sum result directly, by defn layernorm will have made that 0
    loss = (result * result).sum()

    grads = result.compute_gradient()

    eps = 1e-6
    for i in range(weights.value.shape[0]):
        for j in range(weights.value.shape[1]):
            old_val = float(weights.value[i, j])
            weights.value[i, j] += eps
            new_result = layer.forward(weights)
            new_loss = (new_result * new_result).sum()
            empirical_grad = (new_loss.value - loss.value) / eps
            computed_grad = grads[weights][i, j]
            assert cp.allclose(empirical_grad, computed_grad, rtol=1e-3, atol=1e-3), (
                weights
            )
            weights.value[i, j] = old_val
