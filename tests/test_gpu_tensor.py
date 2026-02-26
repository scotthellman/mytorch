import cupy as cp
from utils import evaluate_empirical_grad

from mytorch.tensor import Tensor

ONE = cp.array([1.0], dtype=cp.float32)


def build_gradient_lookup(ops):
    gradient_lookup = {}
    for variable, _, func in ops:
        gradient_lookup[variable] = func
    return gradient_lookup


def test_add():
    a = Tensor(cp.arange(8, dtype=cp.float32).reshape(2, 2, 2))
    b = Tensor(cp.array([3, 3], dtype=cp.float32).reshape(2, 1, 1))
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
    a = Tensor(cp.array([1.0, 2.0, 3.0], dtype=cp.float32))
    expected = -(a.value)
    result = -a
    assert cp.all(expected == result.value)

    gradient_lookup = build_gradient_lookup(result.operations)

    expected_a_grad = cp.array([-1])
    actual_a_grad = gradient_lookup[a](ONE)

    assert cp.all(expected_a_grad == actual_a_grad)


def test_subtraction():
    a = Tensor(cp.array([1.0, 2.0, 3.0], dtype=cp.float32))
    b = Tensor(cp.array([3.0], dtype=cp.float32))
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
    a = Tensor(cp.array([1.0, 2.0, 3.0], dtype=cp.float32))
    b = Tensor(cp.array([3.0], dtype=cp.float32))
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
    a = Tensor(cp.array([1.0, 2.0, 3.0], dtype=cp.float32))
    b = Tensor(cp.array([3.0], dtype=cp.float32))
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
    a = Tensor(cp.array([[1.0, 2.0, 3.0]], dtype=cp.float32))
    b = Tensor(cp.array([[3.0, 2.0, 1.0]], dtype=cp.float32).T)
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
    a = Tensor(cp.arange(12, dtype=cp.float32).reshape(2, 2, 3))
    b = Tensor(cp.arange(12, dtype=cp.float32).reshape(2, 3, 2))
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


def test_add_grad(two_d_tensor):
    right = Tensor(cp.copy(two_d_tensor.value + 5))

    def left_loss_func(x):
        return (x + right).sum()

    def right_loss_func(x):
        return (two_d_tensor + x).sum()

    evaluate_empirical_grad(two_d_tensor, left_loss_func)
    evaluate_empirical_grad(right, right_loss_func)


def test_sub_grad(two_d_tensor):
    right = Tensor(cp.copy(two_d_tensor.value + 5))

    def left_loss_func(x):
        return (x - right).sum()

    def right_loss_func(x):
        return (two_d_tensor - x).sum()

    evaluate_empirical_grad(two_d_tensor, left_loss_func)
    evaluate_empirical_grad(right, right_loss_func)


def test_mult_grad(two_d_tensor):
    right = Tensor(cp.copy(two_d_tensor.value + 2))

    def left_loss_func(x):
        return (x * right).sum()

    def right_loss_func(x):
        return (two_d_tensor * x).sum()

    evaluate_empirical_grad(two_d_tensor, left_loss_func)
    evaluate_empirical_grad(right, right_loss_func)


def test_matmul_grad(two_d_tensor):
    right = Tensor(cp.copy(two_d_tensor.value + 1))

    def left_loss_func(x):
        return (x @ right).sum()

    def right_loss_func(x):
        return (two_d_tensor @ x).sum()

    evaluate_empirical_grad(two_d_tensor, left_loss_func, eps=1e-2)
    evaluate_empirical_grad(right, right_loss_func, eps=1e-2)


def test_div_grad():
    numerator = Tensor(cp.array([0.2, 0.4, 0.1], dtype=cp.float32).reshape((1, 3)))
    denominator = Tensor(cp.array([0.1, 0.2, 0.3], dtype=cp.float32).reshape((1, 3)))

    def num_loss_func(x):
        return (x / denominator).sum()

    def den_loss_func(x):
        return (numerator / x).sum()

    evaluate_empirical_grad(numerator, num_loss_func)
    evaluate_empirical_grad(denominator, den_loss_func)


def test_add_constant_grad(two_d_tensor):
    def loss_func(x):
        return x.add_constant(3).sum()

    evaluate_empirical_grad(two_d_tensor, loss_func)


def test_exp_constant_grad(two_d_tensor):
    def loss_func(x):
        return x.exp().sum()

    evaluate_empirical_grad(two_d_tensor, loss_func)


def test_sum_grad(two_d_tensor):
    def loss_func(x):
        return x.sum()

    evaluate_empirical_grad(two_d_tensor, loss_func)


def test_mean_grad(two_d_tensor):
    def loss_func(x):
        return x.mean()

    evaluate_empirical_grad(two_d_tensor, loss_func)


def test_var_grad(two_d_tensor):
    def loss_func(x):
        return x.var(axis=-1).mean()

    evaluate_empirical_grad(two_d_tensor, loss_func)


def test_sqrt_grad(two_d_tensor):
    # can't have negatives in here
    two_d_tensor = two_d_tensor * two_d_tensor

    def loss_func(x):
        return x.sqrt().mean()

    evaluate_empirical_grad(two_d_tensor, loss_func)


def test_elu_grad(two_d_tensor):
    def loss_func(x):
        return x.elu().mean()

    evaluate_empirical_grad(two_d_tensor, loss_func)


def test_cumsum_grad(two_d_tensor):
    def loss_func(x):
        return x.cumsum(-1).mean()

    evaluate_empirical_grad(two_d_tensor, loss_func)


def test_transpose_last_grad(two_d_tensor):
    def loss_func(x):
        # need transposing to actually matter
        transposed = x.transpose_last()
        return transposed.cumsum(axis=-1).mean()

    evaluate_empirical_grad(two_d_tensor, loss_func)
