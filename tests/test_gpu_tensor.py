import cupy as cp

from mytorch.gpu_tensor import GpuTensor

ONE = cp.array([1.0], dtype=cp.float32)


def build_gradient_lookup(ops):
    gradient_lookup = {}
    for variable, _, func in ops:
        gradient_lookup[variable] = func
    return gradient_lookup


def test_add():
    a = GpuTensor(cp.arange(8, dtype=cp.float32).reshape(2, 2, 2))
    b = GpuTensor(cp.array([3, 3], dtype=cp.float32).reshape(2, 1, 1))
    expected = a.value + b.value
    result = a + b
    assert cp.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    expected_a_grad = cp.ones_like(a.value)
    batch_one = cp.ones_like(a.value)
    actual_a_grad = gradient_lookup[a](batch_one)

    assert expected_a_grad.shape == actual_a_grad.shape
    assert cp.all(expected_a_grad == actual_a_grad)

    batch_one = cp.ones_like(b.value)
    expected_b_grad = cp.ones_like(b.value)
    actual_b_grad = gradient_lookup[b](batch_one)

    assert expected_b_grad.shape == actual_b_grad.shape
    assert cp.all(expected_b_grad == actual_b_grad)


def test_negation():
    a = GpuTensor(cp.array([1.0, 2.0, 3.0], dtype=cp.float32))
    expected = -(a.value)
    result = -a
    assert cp.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    expected_a_grad = cp.array([-1])
    actual_a_grad = gradient_lookup[a](ONE)

    assert cp.all(expected_a_grad == actual_a_grad)


def test_subtraction():
    a = GpuTensor(cp.array([1.0, 2.0, 3.0], dtype=cp.float32))
    b = GpuTensor(cp.array([3.0], dtype=cp.float32))
    expected = a.value - b.value
    result = a - b
    assert cp.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    expected_a_grad = cp.array([1])
    actual_a_grad = gradient_lookup[a](ONE)

    assert cp.all(expected_a_grad == actual_a_grad)

    expected_b_grad = cp.array([-1])
    actual_b_grad = gradient_lookup[b](ONE)

    assert cp.all(expected_b_grad == actual_b_grad)


def test_multiplication():
    a = GpuTensor(cp.array([1.0, 2.0, 3.0], dtype=cp.float32))
    b = GpuTensor(cp.array([3.0], dtype=cp.float32))
    expected = a.value * b.value
    result = a * b
    assert cp.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    expected_a_grad = cp.array([3.0])
    actual_a_grad = gradient_lookup[a](ONE)

    assert cp.all(expected_a_grad == actual_a_grad)

    expected_b_grad = cp.array([6.0])
    actual_b_grad = gradient_lookup[b](ONE)

    assert cp.all(expected_b_grad == actual_b_grad)


def test_division():
    a = GpuTensor(cp.array([1.0, 2.0, 3.0], dtype=cp.float32))
    b = GpuTensor(cp.array([3.0], dtype=cp.float32))
    expected = a.value / b.value
    result = a / b
    assert cp.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    expected_a_grad = cp.array([1 / 3])
    actual_a_grad = gradient_lookup[a](ONE)

    assert cp.allclose(expected_a_grad, actual_a_grad)

    expected_b_grad = cp.array([-2 / 3])
    actual_b_grad = gradient_lookup[b](ONE)

    assert cp.allclose(expected_b_grad, actual_b_grad)


def test_matmul():
    a = GpuTensor(cp.array([[1.0, 2.0, 3.0]], dtype=cp.float32))
    b = GpuTensor(cp.array([[3.0, 2.0, 1.0]], dtype=cp.float32).T)
    expected = a.value @ b.value
    result = a @ b
    assert cp.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    grad_input = cp.ones((1, 1), dtype=cp.float32)

    expected_a_grad = cp.array([[3.0, 2.0, 1.0]], dtype=cp.float32)
    actual_a_grad = gradient_lookup[a](grad_input)

    assert cp.all(expected_a_grad == actual_a_grad)

    expected_b_grad = cp.array([[1.0], [2.0], [3.0]], dtype=cp.float32)
    actual_b_grad = gradient_lookup[b](grad_input)

    assert cp.all(expected_b_grad == actual_b_grad)


def test_batch_matmul():
    a = GpuTensor(cp.arange(12, dtype=cp.float32).reshape(2, 2, 3))
    b = GpuTensor(cp.arange(12, dtype=cp.float32).reshape(2, 3, 2))
    expected = a.value @ b.value
    result = a @ b
    assert cp.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    grad_input = cp.ones((2, 2, 2), dtype=cp.float32)

    expected_a_grad = grad_input @ cp.swapaxes(b.value, -1, -2)
    actual_a_grad = gradient_lookup[a](grad_input)

    assert cp.all(expected_a_grad == actual_a_grad)

    expected_b_grad = cp.swapaxes(a.value, -1, -2) @ grad_input
    actual_b_grad = gradient_lookup[b](grad_input)

    assert cp.all(expected_b_grad == actual_b_grad)


def test_div_grad():
    weights = GpuTensor(cp.array([0.2, 0.4, 0.1], dtype=cp.float32).reshape((1, 3)))
    divisor = GpuTensor(cp.array([0.1, 0.2, 0.3], dtype=cp.float32).reshape((1, 3)))
    result = (weights / divisor).sum()

    grads = result.compute_gradient()

    eps = 1e-4
    for i in range(weights.value.shape[0]):
        for j in range(weights.value.shape[1]):
            old_val = float(weights.value[i, j])
            weights.value[i, j] += eps
            new_result = (weights / divisor).sum()
            empirical_grad = (new_result.value - result.value) / eps
            computed_grad = grads[weights][i, j]
            assert cp.allclose(empirical_grad, computed_grad, rtol=1e-3, atol=1e-3), (
                weights
            )
            weights.value[i, j] = old_val


def test_mean_grad():
    weights = GpuTensor(cp.array([0.2, 0.4, 0.1], dtype=cp.float32).reshape((1, 3)))
    result = weights.mean()

    grads = result.compute_gradient()

    eps = 1e-4
    for i in range(weights.value.shape[0]):
        for j in range(weights.value.shape[1]):
            old_val = float(weights.value[i, j])
            weights.value[i, j] += eps
            new_result = weights.mean()
            empirical_grad = (new_result.value - result.value) / eps
            computed_grad = grads[weights][i, j]
            assert cp.allclose(empirical_grad, computed_grad, rtol=1e-3, atol=1e-3), (
                weights
            )
            weights.value[i, j] = old_val


def test_var_grad():
    weights = GpuTensor(cp.array([0.2, 0.4, 0.1], dtype=cp.float32).reshape((1, 3)))
    result = weights.var(axis=-1)

    grads = result.compute_gradient()

    eps = 1e-4
    for i in range(weights.value.shape[0]):
        for j in range(weights.value.shape[1]):
            old_val = float(weights.value[i, j])
            weights.value[i, j] += eps
            new_result = weights.var(axis=-1)
            empirical_grad = (new_result.value - result.value) / eps
            computed_grad = grads[weights][i, j]
            assert cp.allclose(empirical_grad, computed_grad, rtol=1e-3, atol=1e-3), (
                weights
            )
            weights.value[i, j] = old_val


def test_sqrt_grad():
    weights = GpuTensor(cp.array([0.2, 0.4, 0.1], dtype=cp.float32).reshape((1, 3)))
    result = weights.sqrt().sum()

    grads = result.compute_gradient()

    eps = 1e-4
    for i in range(weights.value.shape[0]):
        for j in range(weights.value.shape[1]):
            old_val = float(weights.value[i, j])
            weights.value[i, j] += eps
            new_result = weights.sqrt().sum()
            empirical_grad = (new_result.value - result.value) / eps
            computed_grad = grads[weights][i, j]
            assert cp.allclose(empirical_grad, computed_grad, rtol=1e-3, atol=1e-3), (
                weights
            )
            weights.value[i, j] = old_val


def test_elu_grad():
    weights = GpuTensor(cp.array([0.2, -2.4, 0.1], dtype=cp.float32).reshape((1, 3)))
    result = weights.elu().sum()

    grads = result.compute_gradient()

    eps = 1e-4
    for i in range(weights.value.shape[0]):
        for j in range(weights.value.shape[1]):
            old_val = float(weights.value[i, j])
            weights.value[i, j] += eps
            new_result = weights.elu().sum()
            empirical_grad = (new_result.value - result.value) / eps
            computed_grad = grads[weights][i, j]
            assert cp.allclose(empirical_grad, computed_grad, rtol=1e-3, atol=1e-3), (
                weights
            )
            weights.value[i, j] = old_val


# FIXME: stop writin the same boilerplate code over and over
def test_cumsum_grad():
    weights = GpuTensor(
        cp.array(
            [
                [1, 0, -1],
                [1.3, -1.3, 0.1],
                [0.2, -0.4, 20.3],
            ],
            dtype=cp.float32,
        )
    )
    result = weights.cumsum(-1).sum()
    print(weights.cumsum(-1).value)

    grads = result.compute_gradient()

    eps = 1e-3
    for i in range(weights.value.shape[0]):
        for j in range(weights.value.shape[1]):
            old_val = float(weights.value[i, j])
            weights.value[i, j] += eps
            right_result = weights.cumsum(-1).sum()
            weights.value[i, j] = old_val - eps
            left_result = weights.cumsum(-1).sum()
            empirical_grad = (right_result.value - left_result.value) / (2 * eps)
            computed_grad = grads[weights][i, j]
            assert cp.allclose(empirical_grad, computed_grad, rtol=1e-3, atol=1e-3), (
                weights
            )
            weights.value[i, j] = old_val
