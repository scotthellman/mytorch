import cupy as cp

from mytorch.gpu_tensor import GpuTensor

ONE = cp.array([1.0], dtype=cp.float32)


def build_gradient_lookup(ops):
    gradient_lookup = {}
    for variable, _, func in ops:
        gradient_lookup[variable] = func
    return gradient_lookup


def test_add():
    a = GpuTensor(cp.array([1.0, 2.0, 3.0]))
    b = GpuTensor(cp.array([3.0]))
    expected = a.value + b.value
    result = a + b
    assert cp.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    expected_a_grad = cp.array([1])
    actual_a_grad = gradient_lookup[a](ONE)

    assert cp.all(expected_a_grad == actual_a_grad)

    expected_b_grad = cp.array([1])
    actual_b_grad = gradient_lookup[b](ONE)

    assert cp.all(expected_b_grad == actual_b_grad)


def test_negation():
    a = GpuTensor(cp.array([1.0, 2.0, 3.0]))
    expected = -(a.value)
    result = -a
    assert cp.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    expected_a_grad = cp.array([-1])
    actual_a_grad = gradient_lookup[a](ONE)

    assert cp.all(expected_a_grad == actual_a_grad)


def test_subtraction():
    a = GpuTensor(cp.array([1.0, 2.0, 3.0]))
    b = GpuTensor(cp.array([3.0]))
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
    a = GpuTensor(cp.array([1.0, 2.0, 3.0]))
    b = GpuTensor(cp.array([3.0]))
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
    a = GpuTensor(cp.array([1.0, 2.0, 3.0]))
    b = GpuTensor(cp.array([3.0]))
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
    a = GpuTensor(cp.array([[1.0, 2.0, 3.0]]))
    b = GpuTensor(cp.array([[3.0, 2.0, 1.0]]).T)
    expected = a.value @ b.value
    result = a @ b
    assert cp.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    grad_input = cp.ones((1, 1), dtype=cp.float32)

    expected_a_grad = cp.array([[3.0, 2.0, 1.0]])
    actual_a_grad = gradient_lookup[a](grad_input)

    assert cp.all(expected_a_grad == actual_a_grad)

    expected_b_grad = cp.array([[1.0], [2.0], [3.0]])
    actual_b_grad = gradient_lookup[b](grad_input)

    assert cp.all(expected_b_grad == actual_b_grad)
