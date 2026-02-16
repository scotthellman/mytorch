from mytorch.tensor import Tensor


def sgd_step(loss: Tensor, step_size: float):
    gradients = loss.compute_gradient()
    for t, grad in gradients.items():
        if t.frozen:
            continue
        t.value += -grad * step_size
