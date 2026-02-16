import cupy as cp

from mytorch.gpu_tensor import GpuTensor

ONE = cp.array([1])


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
