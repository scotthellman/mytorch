import cupy as cp

from mytorch.tensor import Tensor


def sgd_step(loss: Tensor, step_size: float):
    gradients = loss.compute_gradient()
    for t, grad in gradients.items():
        if t.frozen:
            continue
        # FIXME: parameterize this
        max_val = cp.max(grad)
        if max_val > 10:
            grad *= 10 / max_val

        t.value += -grad * step_size
