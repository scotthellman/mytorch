"""Optimizers that use gradient information to update model weights."""

import cupy as cp

from mytorch import kernels
from mytorch.tensor import Tensor


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
        for group in self.tensors:
            for t in group[0]:
                self.means[t] = cp.zeros_like(t.value)
                self.vars[t] = cp.zeros_like(t.value)

    def decay_lr(self, rate: float):
        new_tensors = []
        for group in self.tensors:
            print("lr decaying to ", group[1] * rate)
            new_tensors.append((group[0], group[1] * rate, group[2]))
        self.tensors = new_tensors

    def step(self):
        normalizer = 1.0
        lr_adjustment = 1
        if self.t < self.warmup_steps:
            lr_adjustment = (self.t + 1) / (self.warmup_steps + 1)
        self.t += 1
        if self.clip:
            seen = set()
            squared_sum = 0
            for tensors, _, __ in self.tensors:
                for tensor in tensors:
                    if tensor not in seen:
                        squared_sum += (tensor.grad**2).sum()
                        seen.add(tensor)
            rss = cp.sqrt(squared_sum)
            if rss > self.clip:
                normalizer = self.clip / rss
            squared_sum = 0
        seen = set()
        for tensors, lr, decay in self.tensors:
            # FIXME: I dropped handling the decay when I built the fused kernel
            adjusted_lr = lr * lr_adjustment
            for tensor in tensors:
                if tensor.frozen or tensor in seen:
                    continue
                seen.add(tensor)

                last_m = self.means[tensor]
                last_v = self.vars[tensor]
                kernels.adam_update(
                    last_m,
                    last_v,
                    tensor.grad,
                    self.b1,
                    self.b2,
                    self.t,
                    adjusted_lr,
                    self.eps,
                    float(normalizer),
                    tensor.value,
                )
                tensor.reset()
