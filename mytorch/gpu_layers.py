import cupy as cp

from mytorch import kernels
from mytorch.gpu_tensor import GpuTensor


class Linear:
    # y = xA.T + b
    def __init__(self, in_size: int, out_size: int, bias: bool):
        # TODO: We probably want to be more flexible about how we do this
        weight_data = cp.random.normal(0, 0.02, (in_size, out_size), dtype=cp.float32)
        self.weights = GpuTensor(weight_data, frozen=False)
        self.params = [self.weights]
        if bias:
            self.bias = GpuTensor(
                cp.array([[0.0] * out_size], dtype=cp.float32), frozen=False
            )
            self.params.append(self.bias)
        else:
            self.bias = None

    def forward(self, input: GpuTensor) -> GpuTensor:
        if self.bias:
            result = input @ self.weights + self.bias
        else:
            result = input @ self.weights
        return result


class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, input: GpuTensor) -> GpuTensor:
        p = kernels.logistic(input.value)
        operations = [(input, "sigmoid", lambda acc: acc * (p * (1 - p)))]

        return GpuTensor(value=p, operations=operations)


class LayerNorm:
    def __init__(self, eps: float = 1e-5):
        # TODO: pytorch lets this learn an affine transform
        self.eps = eps
        self.params = []

    def forward(self, input: GpuTensor) -> GpuTensor:
        # So, for this to work, we need:
        normed, grad_terms = kernels.layernorm(input.value, eps=self.eps)

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            return kernels.layernorm_back(acc, grad_terms[0], grad_terms[1])

        ops = [(input, "layernorm", local_grad)]
        return GpuTensor(normed, ops)


class Embedding:
    def __init__(self, vocab_size: int, embedding_size: int):
        weight_data = cp.random.normal(
            0, 0.02, (vocab_size, embedding_size), dtype=cp.float32
        )
        self.weights = GpuTensor(weight_data, frozen=False)
        self.params = [self.weights]

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


class AdditivePositionalEncoding:
    def forward(self, input: GpuTensor):
        # this is painfully inefficient
        positions = cp.zeros(input.value.shape[-2:], dtype=cp.float32)
        for s in range(positions.shape[0]):
            for d in range(positions.shape[1]):
                den = 100000 ** (2 * d / positions.shape[1])
                if d % 2 == 0:
                    positions[s, d] = cp.sin(s / den)
                else:
                    positions[s, d] = cp.cos(s / den)
        positions = GpuTensor(positions)
        input = input + positions
        return input


class SelfAttention:
    def __init__(self, embedding_size, n_heads=1):
        assert embedding_size % n_heads == 0
        weight_data = cp.random.normal(
            0, 0.1, (embedding_size, embedding_size * 3), dtype=cp.float32
        )
        self.weights = GpuTensor(weight_data, frozen=False)
        # q_data = cp.random.normal(0, 0.5, (embedding_size, key_size), dtype=cp.float32)
        # self.Q = GpuTensor(q_data, frozen=False)
        # k_data = cp.random.normal(0, 0.5, (embedding_size, key_size), dtype=cp.float32)
        # self.K = GpuTensor(k_data, frozen=False)
        # v_data = cp.random.normal(0, 0.1, (embedding_size, key_size), dtype=cp.float32)
        # self.V = GpuTensor(v_data, frozen=False)
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.key_size = embedding_size // n_heads
        self.params = [self.weights]

    def forward(self, input: GpuTensor):
        # TODO: this can have some special caching behavior at inference time (3.3.2 of the paper)
        # let's be clear about shapes, as we go
        # input: (b, s, emb)
        # get our q,k,v values
        # q,k,v are all (b,s,key)
        qwk = input @ self.weights
        q = qwk[..., : self.embedding_size]
        k = qwk[..., self.embedding_size : self.embedding_size * 2]
        v = qwk[..., self.embedding_size * 2 :]
        # now we need to calculate our similarities. Have to stop going off of AIAYN now,
        # switch to the linear attention paper
        # if we were doing AIAYN, we would get: result = softmax((Q@K.T) / sqrt(self.key_size)) @ V
        # but that softmax is very expensive, hence linear attention
        # The paper defines: result = V_i = elu(Q_i).T * sum(elu(K_j)V_j.T) / [num with V]
        # but they also nicely provide pseudocode, so I'll follow that
        # We are making a choice here: i'm going for the decoder side, so we need causal masking

        # multihead indexing reference: https://github.com/Whiax/BERT-Transformer-Pytorch/blob/main/train.py#L50

        # now we want to split into our heads
        q = q.reshape((input.value.shape[0], -1, self.n_heads, self.key_size))
        k = k.reshape((input.value.shape[0], -1, self.n_heads, self.key_size))
        v = v.reshape((input.value.shape[0], -1, self.n_heads, self.key_size))

        # shuffle n_heads to be one of the batch dims
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # these are still (b,s,key)
        transformed_q = q.elu().add_constant(1)
        transformed_k = k.elu().add_constant(1)

        # compute numerator and denominator separately, so that we can cleanly
        # split the custom backprop code out

        num = self.build_numerator(
            self.rotate(transformed_q), self.rotate(transformed_k), v
        )

        # denom is of shape (b, s, 1)
        # we want the dot product of q_i and the partial sum of k_i
        # We implement that with an elementwise muilt and then summing over the final dim
        denom = (
            (transformed_q * transformed_k.cumsum(axis=1))
            .sum(axis=-1, keepdims=True)
            .add_constant(1e-6)
        )

        result = num / denom

        # now we need to reassemble from the multiheads
        result = result.transpose(1, 2)
        result = result.reshape(
            (input.value.shape[0], input.value.shape[1], self.key_size * self.n_heads)
        )

        return result

    def rotate(self, tensor: GpuTensor):
        # NOTE: this assumes an even number of dimensions
        # we're specifically rotating the final dim based on the penultimate index
        # everything above that is just batched
        result = kernels.rope(tensor.value)

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            return kernels.rope(acc, backward=True)

        ops = [(tensor, "rope", local_grad)]

        return GpuTensor(result, ops)

    def build_numerator(
        self, phi_q_tensor: GpuTensor, phi_k_tensor: GpuTensor, v_tensor: GpuTensor
    ) -> GpuTensor:
        phi_k = phi_k_tensor.value
        phi_q = phi_q_tensor.value
        v = v_tensor.value
        s = cp.zeros(
            (phi_k.shape[0], phi_k.shape[1], self.key_size, self.key_size),
            dtype=cp.float32,
        )
        num = cp.zeros_like(phi_k)
        for i in range(phi_k.shape[-2]):
            s = s + phi_k[..., i, :, None] @ v[..., i, None, :]
            num[..., i, :] = (phi_q[..., i, None, :] @ s).squeeze(-2)

        # so we have the actual values, but we still need the gradients
        # This gets messy - we're going to want to fuse these together
        # in a kernel, but that will require some sort of closure to cache the
        # the results after our first invocation. #FIXME for now i'm just sticking
        # to python code that duplicates work
        def local_grad_q(acc: cp.ndarray) -> cp.ndarray:
            s = cp.zeros(
                (phi_k.shape[0], phi_k.shape[1], self.key_size, self.key_size),
                dtype=cp.float32,
            )
            grad = cp.zeros_like(phi_q)
            for i in range(phi_k.shape[-2]):
                s = s + phi_k[..., i, :, None] @ v[..., i, None, :]
                grad[..., i, :] = (acc[..., i, None, :] @ s.swapaxes(-1, -2)).squeeze(
                    -2
                )
            return grad

        def local_grad_v(acc: cp.ndarray) -> cp.ndarray:
            s = cp.zeros(
                (phi_k.shape[0], phi_k.shape[1], self.key_size, self.key_size),
                dtype=cp.float32,
            )
            grad = cp.zeros_like(phi_k)
            for i in range(phi_q.shape[-2] - 1, -1, -1):
                s = s + phi_q[..., i, :, None] @ acc[..., i, None, :]
                grad[..., i, :] = (s.swapaxes(-1, -2) @ phi_k[..., i, :, None]).squeeze(
                    -1
                )
            return grad

        def local_grad_k(acc: cp.ndarray) -> cp.ndarray:
            s = cp.zeros(
                (phi_k.shape[0], phi_k.shape[1], self.key_size, self.key_size),
                dtype=cp.float32,
            )
            grad = cp.zeros_like(phi_k)
            for i in range(phi_q.shape[-2] - 1, -1, -1):
                s = s + phi_q[..., i, :, None] @ acc[..., i, None, :]
                grad[..., i, :] = (s @ v[..., i, :, None]).squeeze(-1)
            return grad

        ops = [
            (phi_q_tensor, "attention_numerator_q", local_grad_q),
            (phi_k_tensor, "attention_numerator_k", local_grad_k),
            (v_tensor, "attention_numerator_v", local_grad_v),
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
        # FIXME: turns out this is using cupy operations still
        n = loss.size
        loss = loss.mean()

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            return acc * grad_info / n

        # TODO: well this is a little clunky
        return GpuTensor(value=loss, operations=[(input, "CrossEntropy", local_grad)])
