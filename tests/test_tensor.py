import numpy as np

from mytorch.tensor import Tensor

ONE = np.array([1])


def build_gradient_lookup(ops):
    gradient_lookup = {}
    for variable, _, func in ops:
        gradient_lookup[variable] = func
    return gradient_lookup


def test_add():
    a = Tensor(np.array([1.0, 2.0, 3.0]))
    b = Tensor(np.array([3.0]))
    expected = a.value + b.value
    result = a + b
    assert np.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    expected_a_grad = np.array([1])
    actual_a_grad = gradient_lookup[a](ONE)

    assert np.all(expected_a_grad == actual_a_grad)

    expected_b_grad = np.array([1])
    actual_b_grad = gradient_lookup[b](ONE)

    assert np.all(expected_b_grad == actual_b_grad)


def test_negation():
    a = Tensor(np.array([1.0, 2.0, 3.0]))
    expected = -(a.value)
    result = -a
    assert np.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    expected_a_grad = np.array([-1])
    actual_a_grad = gradient_lookup[a](ONE)

    assert np.all(expected_a_grad == actual_a_grad)


def test_subtraction():
    a = Tensor(np.array([1.0, 2.0, 3.0]))
    b = Tensor(np.array([3.0]))
    expected = a.value - b.value
    result = a - b
    assert np.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    expected_a_grad = np.array([1])
    actual_a_grad = gradient_lookup[a](ONE)

    assert np.all(expected_a_grad == actual_a_grad)

    expected_b_grad = np.array([-1])
    actual_b_grad = gradient_lookup[b](ONE)

    assert np.all(expected_b_grad == actual_b_grad)


def test_multiplication():
    a = Tensor(np.array([1.0, 2.0, 3.0]))
    b = Tensor(np.array([3.0]))
    expected = a.value * b.value
    result = a * b
    assert np.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    expected_a_grad = np.array([3.0])
    actual_a_grad = gradient_lookup[a](ONE)

    assert np.all(expected_a_grad == actual_a_grad)

    expected_b_grad = np.array([6.0])
    actual_b_grad = gradient_lookup[b](ONE)

    assert np.all(expected_b_grad == actual_b_grad)


def test_division():
    a = Tensor(np.array([1.0, 2.0, 3.0]))
    b = Tensor(np.array([3.0]))
    expected = a.value / b.value
    result = a / b
    assert np.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    expected_a_grad = np.array([1 / 3])
    actual_a_grad = gradient_lookup[a](ONE)

    assert np.allclose(expected_a_grad, actual_a_grad)

    expected_b_grad = np.array([-2 / 3])
    actual_b_grad = gradient_lookup[b](ONE)

    assert np.allclose(expected_b_grad, actual_b_grad)


def test_matmul():
    a = Tensor(np.array([[1.0, 2.0, 3.0]]))
    b = Tensor(np.array([[3.0, 2.0, 1.0]]).T)
    expected = a.value @ b.value
    result = a @ b
    assert np.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    grad_input = np.ones((1, 1))

    expected_a_grad = np.array([[3.0, 2.0, 1.0]])
    actual_a_grad = gradient_lookup[a](grad_input)

    assert np.all(expected_a_grad == actual_a_grad)

    expected_b_grad = np.array([[1.0], [2.0], [3.0]])
    actual_b_grad = gradient_lookup[b](grad_input)

    assert np.all(expected_b_grad == actual_b_grad)
