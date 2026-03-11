import itertools

import cupy as cp


def evaluate_empirical_grad(tensor, loss_func, eps=1e-3):
    tensor.reset()
    loss_func(tensor).compute_gradient()
    iters = [range(s) for s in tensor.value.shape]
    for key in itertools.product(*iters):
        old_val = float(tensor.value[key])
        tensor.value[key] -= eps
        left_loss = loss_func(tensor)
        tensor.value[key] = old_val + eps
        right_loss = loss_func(tensor)
        empirical_grad = (right_loss.value - left_loss.value) / (2 * eps)
        computed_grad = tensor.grad[key]
        # working with float32, so we can't be too particular about tolerances here
        assert cp.allclose(empirical_grad, computed_grad, rtol=1e-3, atol=1e-2)
        tensor.value[key] = old_val
