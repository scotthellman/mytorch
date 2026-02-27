import cupy as cp
import numpy as np

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


class Adam:
    def __init__(
        self,
        tensors: list[tuple[list[Tensor], float, float]],  # lr, decay
        eps: float = 1e-6,
        clip: float | None = 1.0,
        warmup_steps: int = 0,
    ):
        self.tensors = tensors
        self.means = {}
        self.vars = {}
        self.b1 = 0.9
        self.b2 = 0.999
        self.t = 0
        self.eps = eps
        self.clip = clip
        self.warmup_steps = warmup_steps
        self.steps = 0

    def step(self):
        self.t += 1
        normalizer = None
        lr_adjustment = 1
        if self.steps < self.warmup_steps:
            lr_adjustment = (self.steps + 1) / (self.warmup_steps + 1)
        if self.clip:
            squared_sum = 0
            for tensors, _, __ in self.tensors:
                for tensor in tensors:
                    squared_sum += (tensor.grad**2).sum()
            rss = np.sqrt(squared_sum)
            if rss > self.clip:
                normalizer = self.clip / rss
        for tensors, lr, decay in self.tensors:
            for t in tensors:
                if t.frozen:
                    continue

                last_m = self.means.get(t, Tensor(cp.zeros_like(t.grad)))
                last_v = self.vars.get(t, Tensor(cp.zeros_like(t.grad)))
                if normalizer is not None:
                    grad = t.grad * normalizer
                grad = Tensor(t.grad)

                m = last_m.mult_constant(self.b1) + grad.mult_constant(1 - self.b1)

                v = grad * grad
                if last_v is not None:
                    v = last_v.mult_constant(self.b2) + v.mult_constant(1 - self.b2)

                # FIXME: need a proper detach
                m.operations = []
                v.operations = []
                self.means[t] = m
                self.vars[t] = v

                normed_m = m.mult_constant(1 / (1 - self.b1**self.t))
                normed_v = v.mult_constant(1 / (1 - self.b2**self.t))

                t.value -= (
                    lr_adjustment
                    * lr
                    * (
                        (normed_m / normed_v.sqrt().add_constant(self.eps)).value
                        - decay * t.value
                    )
                )
                t.reset()
