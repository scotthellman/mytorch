import cupy as cp

from mytorch.gpu_tensor import GpuTensor


def sgd_step(loss: GpuTensor, step_size: float):
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
    def __init__(self, lr: float, decay: float = 0, eps: float = 1e-6):
        self.lr = lr
        self.means = {}
        self.vars = {}
        self.b1 = 0.99
        self.b2 = 0.999
        self.t = 0
        self.eps = eps
        # FIXME: actually implement decay

    def step(self, loss: GpuTensor):
        self.t += 1
        gradients = loss.compute_gradient()
        for t, grad in gradients.items():
            if t.frozen:
                continue

            grad = GpuTensor(grad)

            m = grad
            last_m = self.means.get(t)
            if last_m is not None:
                m = last_m.mult_constant(self.b1) + m.mult_constant(1 - self.b1)

            v = grad * grad
            last_v = self.vars.get(t)
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
                normed_m.mult_constant(self.lr) / normed_v.sqrt().add_constant(self.eps)
            ).value
