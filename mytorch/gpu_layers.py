import cupy as cp

from mytorch import kernels
from mytorch.gpu_tensor import GpuTensor


class Linear:
    # y = xA.T + b
    def __init__(self, in_size: int, out_size: int, bias: bool):
        # TODO: We probably want to be more flexible about how we do this
        weight_data = cp.random.normal(0, 0.02, (in_size, out_size), dtype=cp.float32)
        self.weights = GpuTensor(weight_data, frozen=False)
        if bias:
            self.bias = GpuTensor(
                cp.array([[0.0] * out_size], dtype=cp.float32), frozen=False
            )
        else:
            self.bias = None

    def forward(self, input: GpuTensor) -> GpuTensor:
        if self.bias:
            result = input @ self.weights + self.bias
        else:
            result = input @ self.weights
        return result


class Sigmoid:
    def forward(self, input: GpuTensor) -> GpuTensor:
        p = kernels.logistic(input.value)
        operations = [(input, "sigmoid", lambda acc: acc * (p * (1 - p)))]

        return GpuTensor(value=p, operations=operations)


class LayerNorm:
    def __init__(self, eps: float = 1e-5):
        # TODO: pytorch lets this learn an affine transform
        self.eps = eps

    def forward(self, input: GpuTensor) -> GpuTensor:
        # So, for this to work, we need:
        # mean, variance, sqrt
        # FIXME: We can worry about a fused kernel after, let's just get it working
        # FIXME: this will only work after linear layers, it's not as general as pytorch's
        # (assumes input is 2d)
        eps_vec = GpuTensor(cp.ones_like(input.value) * self.eps)
        demeaned = input - input.mean(axis=-1)
        variance = input.var(axis=-1)
        normed = demeaned / (variance + eps_vec).sqrt()
        return normed


class Embedding:
    def __init__(self, vocab_size: int, embedding_size: int):
        weight_data = cp.random.normal(
            0, 0.02, (vocab_size, embedding_size), dtype=cp.float32
        )
        self.weights = GpuTensor(weight_data, frozen=False)

    def forward(self, input: GpuTensor) -> GpuTensor:
        # making a hard assumption here: input is an int tensor
        embedded = self.weights.value[input.value]

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            # so. the local grad here is just selecting the relevant rows
            # so we need to expand acc out, 0ing any rows that weren't selected
            # TODO: shame we have to make this big matrix. Something to think about
            out = cp.zeros_like(self.weights.value)
            out[input.value] = acc
            return out

        return GpuTensor(
            value=embedded, operations=[(self.weights, "embed", local_grad)]
        )


class SelfAttention:
    def __init__(self, embedding_size, key_size):
        q_data = cp.random.normal(0, 0.02, (embedding_size, key_size), dtype=cp.float32)
        self.Q = GpuTensor(q_data, frozen=False)
        k_data = cp.random.normal(0, 0.02, (embedding_size, key_size), dtype=cp.float32)
        self.K = GpuTensor(k_data, frozen=False)
        v_data = cp.random.normal(0, 0.02, (embedding_size, key_size), dtype=cp.float32)
        self.V = GpuTensor(v_data, frozen=False)
        self.embedding_size = embedding_size
        self.key_size = key_size

    def forward(self, input: GpuTensor):
        # TODO: this can have some special caching behavior at inference time (3.3.2 of the paper)
        # let's be clear about shapes, as we go
        # input: (b, s, emb)
        # get our q,k,v values
        # q,k,v are all (b,s,key)
        q = input @ self.Q
        k = input @ self.K
        v = input @ self.V
        # now we need to calculate our similarities. Have to stop going off of AIAYN now,
        # switch to the linear attention paper
        # if we were doing AIAYN, we would get: result = softmax((Q@K.T) / sqrt(self.key_size)) @ V
        # but that softmax is very expensive, hence linear attention
        # The paper defines: result = V_i = elu(Q_i).T * sum(elu(K_j)V_j.T) / [num with V]
        # but they also nicely provide pseudocode, so I'll follow that
        # We are making a choice here: i'm going for the decoder side, so we need causal masking

        # these are still (b,s,key)
        transformed_q = q.elu()
        transformed_k = k.elu()

        # S and Z from the paper
        # these are updated as we go over the sequence, so they drop one sequence term
        # that gives us (b,key,key)
        s = cp.zeros(
            (input.value.shape[0], self.key_size, self.key_size), dtype=cp.float32
        )
        z = cp.zeros_like(s)
        # V' from the paper. shape is (b, s, key)

        # compute numerator and denominator separately, so that we can cleanly
        # split the custom backprop code out
        # FIXME: get consistent about my qkv ordering!!
        num = self.build_numerator(transformed_k, v, transformed_q)

        denom = transformed_q.transpose_last() @ transformed_k.cumsum(axis=1)

        result = num / denom

        pass

    def build_numerator(self, phi_k, v, phi_q):
        s = cp.zeros(
            (phi_k.value.shape[0], self.key_size, self.key_size), dtype=cp.float32
        )
        num = cp.zeros_like(phi_k)
        for i in range(phi_k.value.shape[-2]):
            s = s + phi_k[:, i, :, None] @ v[:, i, None, :]
            num[:, i] = (phi_q[:, i, None, :] @ s).squeeze(-2)

        # so we have the actual values, but we still need the gradients
        # This gets messy - we're going to want to fuse these together
        # in a kernel, but that will require some sort of closure to cache the
        # the results after our first invocation. #FIXME for now i'm just sticking
        # to python code that duplicates work
        def local_grad_q(acc: cp.ndarray) -> cp.ndarray:
            s = cp.zeros(
                (phi_k.value.shape[0], self.key_size, self.key_size),
                dtype=cp.float32,
            )
            grad = cp.zeros_like(phi_q)
            for i in range(phi_k.value.shape[-2]):
                s = s + phi_k[:, i, :, None] @ v[:, i, None, :]
                grad[:, i] = (acc[:, i, None, :] @ s.swapaxes(-1, -2)).squeeze(-2)
            return grad

        def local_grad_v(acc: cp.ndarray) -> cp.ndarray:
            s = cp.zeros(
                (phi_k.value.shape[0], self.key_size, self.key_size),
                dtype=cp.float32,
            )
            grad = cp.zeros_like(phi_k)
            for i in range(phi_q.value.shape[-2], -1, -1):
                s = s + phi_q[:, i, :, None] @ acc[:, i, None, :]
                grad[:, i] = (s.swapaxes(-1, -2) @ phi_k[:, i, :, None]).squeeze(-1)
            return grad

        def local_grad_k(acc: cp.ndarray) -> cp.ndarray:
            s = cp.zeros(
                (phi_k.value.shape[0], self.key_size, self.key_size),
                dtype=cp.float32,
            )
            grad = cp.zeros_like(phi_k)
            for i in range(phi_q.value.shape[-2], -1, -1):
                s = s + phi_q[:, i, :, None] @ acc[:, i, None, :]
                grad[:, i] = (s @ v[:, i, :, None]).squeeze(-1)
            return grad

        ops = [
            (phi_q, "attention_numerator", local_grad_q),
            (phi_k, "attention_numerator", local_grad_k),
            (v, "attention_numerator", local_grad_v),
        ]

        return GpuTensor(num, ops)


# FIXME: not really a layer. Need to fix my organization
class CrossEntropyLoss:
    def forward(self, input: GpuTensor, target: GpuTensor) -> GpuTensor:
        # This is not fully general - I'm only worrying about the case where the targets are not probas
        # at which point, this is essentially the lot of the softmax of the relevant index
        # we have to be a little fancy here to avoid numerical issues
        # can't just compute softmax and then log it
        # What shapes are at play here? input: (b, s, e). target: (b, s)
        # Target essentially selects what e-index we are computing softmax wrt to for that b,s
        # Then for each b,s, we compute log(softmax(x_target)). Define c = max(x[b,s]).
        # Then, with some trickery, our loss is:
        # x_target - c - log(sum(e^x_i-c))
        loss, grad_info = kernels.cross_entropy(input.value, target.value)

        # go ahead and assume we want to sum this
        loss = loss.sum()

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            return acc * grad_info

        # TODO: well this is a little clunky
        return GpuTensor(value=loss, operations=[(input, "CrossEntropy", local_grad)])
