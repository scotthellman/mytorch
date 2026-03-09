import cupy as cp


def evaluate_empirical_grad(tensor, loss_func, eps=1e-3):
    tensor.reset()
    loss_func(tensor).compute_gradient()
    for i in range(tensor.value.shape[0]):
        for j in range(tensor.value.shape[1]):
            old_val = float(tensor.value[i, j])
            tensor.value[i, j] -= eps
            left_loss = loss_func(tensor)
            tensor.value[i, j] = old_val + eps
            right_loss = loss_func(tensor)
            empirical_grad = (right_loss.value - left_loss.value) / (2 * eps)
            computed_grad = tensor.grad[i, j]
            # working with float32, so we can't be too particular about tolerances here
            print(empirical_grad.ravel()[:5])
            print(computed_grad.ravel()[:5])
            assert cp.allclose(empirical_grad, computed_grad, rtol=1e-3, atol=1e-2)
            tensor.value[i, j] = old_val
