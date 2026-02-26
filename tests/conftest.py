import cupy as cp
import pytest

from mytorch.tensor import Tensor


@pytest.fixture
def two_d_tensor():
    return Tensor(
        cp.array(
            [
                [1, 0.1, -1],
                [1.3, -1.3, 0.1],
                [0.2, -0.4, 2.3],
            ],
            dtype=cp.float32,
        )
    )


@pytest.fixture
def one_d_tensor():
    return Tensor(
        cp.array(
            [
                [1, 0.1, -1],
            ],
            dtype=cp.float32,
        )
    )
