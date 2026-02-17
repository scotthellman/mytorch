import cupy as cp

from mytorch import kernels


def test_matmul_large_size():
    a = cp.random.random((1000, 1000), dtype=cp.float32)
    b = cp.random.random((1000, 1000), dtype=cp.float32)

    expected = a @ b
    actual = kernels.matmul(a, b)
    print(cp.max(cp.abs(expected - actual)))

    assert cp.allclose(expected, actual, atol=1e-2, rtol=1e-2)


def test_matmul_tall_size():
    a = cp.random.random((4, 1000), dtype=cp.float32)
    b = cp.random.random((1000, 1000), dtype=cp.float32)

    expected = a @ b
    actual = kernels.matmul(a, b)

    assert cp.allclose(expected, actual, atol=1e-2, rtol=1e-2)


def test_matmul_small_size():
    a = cp.random.random((10, 5), dtype=cp.float32)
    b = cp.random.random((5, 2), dtype=cp.float32)

    expected = a @ b
    actual = kernels.matmul(a, b)

    assert cp.allclose(expected, actual, atol=1e-2, rtol=1e-2)
