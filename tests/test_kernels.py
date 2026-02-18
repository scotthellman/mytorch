import math

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


def test_matmul_kernel_directly():
    a = cp.array([[1.0, 2.0, 3.0]], dtype=cp.float32)
    b = cp.array([[3.0, 2.0, 1.0]], dtype=cp.float32).T
    expected = a @ b
    result = cp.zeros_like(expected)
    # there's a footgun here: the x dimension of the grid
    # is handling columns, so we need to pull the shapes from
    # the "wrong" dimensions
    grid_size = (
        math.ceil(b.shape[-1] / (kernels.BLOCKSIZE * kernels.THREAD_WINDOW)),
        math.ceil(a.shape[-2] / (kernels.BLOCKSIZE * kernels.THREAD_WINDOW)),
    )
    block_size = (kernels.BLOCKSIZE, kernels.BLOCKSIZE)
    kernels.matmul_kernel(
        grid_size,
        block_size,
        (
            a.shape[-2],
            a.shape[-1],
            b.shape[-1],
            1,
            a,
            b,
            result,
        ),
    )
    assert cp.allclose(expected, result)


def test_cross_entropy():
    a = cp.array([[100, 100], [100, 10000]], dtype=cp.float32)
    y = cp.array([1, 1], dtype=cp.int32)

    # not sure why we need all this ceremony, but cp complains
    # about putting this into an array directly
    expected = cp.array([-float(cp.log(0.5)), 0.0], dtype=cp.float32)

    result, _ = kernels.cross_entropy(a, y)
    assert cp.allclose(expected, result)
