"""Neural Network layers that can work with Tensors"""

import cupy as cp
import cupyx

from mytorch import kernels
from mytorch.tensor import Tensor


class Linear:
    # y = xA.T + b
    def __init__(self, in_size: int, out_size: int, bias: bool):
        # TODO: We probably want to be more flexible about how we do this
        weight_data = cp.random.normal(0, 0.02, (in_size, out_size), dtype=cp.float32)
        self.weights = Tensor(weight_data, frozen=False)
        self.params = [self.weights]
        if bias:
            self.bias = Tensor(
                cp.array([[0.0] * out_size], dtype=cp.float32), frozen=False
            )
            self.params.append(self.bias)
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        if self.bias:
            result = input @ self.weights + self.bias
        else:
            result = input @ self.weights
        return result


class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, input: Tensor) -> Tensor:
        p = kernels.logistic(input.value)
        operations = [(input, "sigmoid", lambda acc: acc * (p * (1 - p)))]

        return Tensor(value=p, operations=operations)


class Elu:
    def __init__(self):
        self.params = []

    def forward(self, input: Tensor) -> Tensor:
        return input.elu()


class LayerNorm:
    def __init__(self, normed_shape: tuple[int], eps: float = 1e-5):
        self.eps = eps
        self.b = Tensor(cp.zeros(normed_shape, dtype=cp.float32), frozen=False)
        self.w = Tensor(cp.ones(normed_shape, dtype=cp.float32), frozen=False)
        self.params = [self.b, self.w]

    def forward(self, input: Tensor) -> Tensor:
        # So, for this to work, we need:
        normed, grad_terms = kernels.layernorm(input.value, eps=self.eps)

        # TODO: could pull b and w into the kernel

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            return kernels.layernorm_back(acc, grad_terms[0], grad_terms[1])

        ops = [(input, "layernorm", local_grad)]
        result = Tensor(normed, ops)
        return result.mul_and_add(self.w, self.b)


class Embedding:
    def __init__(self, vocab_size: int, embedding_size: int):
        weight_data = cp.random.normal(
            0, 0.02, (vocab_size, embedding_size), dtype=cp.float32
        )
        self.weights = Tensor(weight_data, frozen=False)
        self.params = [self.weights]

    def forward(self, input: Tensor) -> Tensor:
        # making a hard assumption here: input is an int tensor
        embedded = self.weights.value[input.value]

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            # so. the local grad here is just selecting the relevant rows
            # so we need to expand acc out, 0ing any rows that weren't selected
            # TODO: shame we have to make this big matrix. Something to think about
            out = cp.zeros_like(self.weights.value)
            cupyx.scatter_add(out, input.value, acc)
            return out

        return Tensor(value=embedded, operations=[(self.weights, "embed", local_grad)])


class AdditivePositionalEncoding:
    def forward(self, input: Tensor):
        # this is painfully inefficient
        positions = cp.zeros(input.value.shape[-2:], dtype=cp.float32)
        for s in range(positions.shape[0]):
            for d in range(positions.shape[1]):
                den = 100000 ** (2 * d / positions.shape[1])
                if d % 2 == 0:
                    positions[s, d] = cp.sin(s / den)
                else:
                    positions[s, d] = cp.cos(s / den)
        positions = Tensor(positions)
        input = input + positions
        return input


class LinearSelfAttention:
    def __init__(self, embedding_size, n_heads=1):
        assert embedding_size % n_heads == 0
        weight_data = cp.random.normal(
            0, 0.1, (embedding_size, embedding_size * 3), dtype=cp.float32
        )
        self.weights = Tensor(weight_data, frozen=False)
        self.out_weights = Tensor(
            cp.random.normal(
                0, 0.1, (embedding_size, embedding_size), dtype=cp.float32
            ),
            frozen=False,
        )
        # q_data = cp.random.normal(0, 0.5, (embedding_size, key_size), dtype=cp.float32)
        # self.Q = GpuTensor(q_data, frozen=False)
        # k_data = cp.random.normal(0, 0.5, (embedding_size, key_size), dtype=cp.float32)
        # self.K = GpuTensor(k_data, frozen=False)
        # v_data = cp.random.normal(0, 0.1, (embedding_size, key_size), dtype=cp.float32)
        # self.V = GpuTensor(v_data, frozen=False)
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.key_size = embedding_size // n_heads
        self.params = [self.weights, self.out_weights]

    def forward(self, input: Tensor):
        # TODO: this can have some special caching behavior at inference time (3.3.2 of the paper)
        # let's be clear about shapes, as we go
        # input: (b, s, emb)
        # get our q,k,v values
        # q,k,v are all (b,s,key)
        qwk = input @ self.weights
        # now we need to calculate our similarities. Have to stop going off of AIAYN now,
        # switch to the linear attention paper
        # if we were doing AIAYN, we would get: result = softmax((Q@K.T) / sqrt(self.key_size)) @ V
        # but that softmax is very expensive, hence linear attention
        # The paper defines: result = V_i = elu(Q_i).T * sum(elu(K_j)V_j.T) / [num with V]
        # but they also nicely provide pseudocode, so I'll follow that
        # We are making a choice here: i'm going for the decoder side, so we need causal masking

        # multihead indexing reference: https://github.com/Whiax/BERT-Transformer-Pytorch/blob/main/train.py#L50

        # now we want to split into our heads
        # and then shuffle n_heads to be one of the batch dims
        q = qwk.index_reshape_transpose(
            (Ellipsis, slice(None, self.embedding_size)),
            (input.value.shape[0], -1, self.n_heads, self.key_size),
            1,
            2,
        )
        k = qwk.index_reshape_transpose(
            (Ellipsis, slice(self.embedding_size, self.embedding_size * 2)),
            (input.value.shape[0], -1, self.n_heads, self.key_size),
            1,
            2,
        )
        v = qwk.index_reshape_transpose(
            (Ellipsis, slice(self.embedding_size * 2, None)),
            (input.value.shape[0], -1, self.n_heads, self.key_size),
            1,
            2,
        )

        # these are still (b,s,key)
        transformed_q = q.elu(1)
        transformed_k = k.elu(1)

        # compute numerator and denominator separately, so that we can cleanly
        # split the custom backprop code out

        num = self.build_numerator(
            self.rotate(transformed_q), self.rotate(transformed_k), v
        )

        # denom is of shape (b, s, 1)
        # we want the dot product of q_i and the partial sum of k_i
        # We implement that with an elementwise muilt and then summing over the final dim
        denom = (transformed_q * transformed_k.cumsum(axis=2)).sum(
            axis=-1, keepdims=True, constant_term=1e-6
        )

        result = num / denom

        # now we need to reassemble from the multiheads
        result = result.transpose(1, 2)
        result = result.reshape(
            (input.value.shape[0], input.value.shape[1], self.key_size * self.n_heads)
        )

        return result @ self.out_weights

    def rotate(self, tensor: Tensor):
        # NOTE: this assumes an even number of dimensions
        # we're specifically rotating the final dim based on the penultimate index
        # everything above that is just batched
        result = kernels.rope(tensor.value)

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            return kernels.rope(acc, backward=True)

        ops = [(tensor, "rope", local_grad)]

        return Tensor(result, ops)

    def build_numerator(
        self, phi_q_tensor: Tensor, phi_k_tensor: Tensor, v_tensor: Tensor
    ) -> Tensor:
        phi_k = phi_k_tensor.value
        phi_q = phi_q_tensor.value
        v = v_tensor.value
        s = cp.zeros(
            (phi_k.shape[0], phi_k.shape[1], self.key_size, self.key_size),
            dtype=cp.float32,
        )
        num = cp.zeros_like(phi_k)
        outer_product = phi_k[..., None] * v[..., None, :]
        s = cp.cumsum(outer_product, axis=-3)
        num = (phi_q[..., None, :] @ s).squeeze(-2)
        # time/space tradeoff here. Going for space right now
        # for i in range(phi_k.shape[-2]):
        #    s = s + phi_k[..., i, :, None] @ v[..., i, None, :]
        #    num[..., i, :] = (phi_q[..., i, None, :] @ s).squeeze(-2)

        # so we have the actual values, but we still need the gradients
        # There's some repeated work done by this backward pass funcs, so
        # use a closure to capture that
        grad_q = None
        grad_v = None
        grad_k = None

        def build_local_grads(acc: cp.ndarray) -> cp.ndarray:
            nonlocal grad_q
            nonlocal grad_v
            nonlocal grad_k
            nonlocal s
            # s = cp.zeros(
            #    (phi_k.shape[0], phi_k.shape[1], self.key_size, self.key_size),
            #    dtype=cp.float32,
            # )
            grad_q = (acc[..., None, :] @ s.swapaxes(-1, -2)).squeeze(-2)
            # grad_q = cp.zeros_like(phi_q)
            # for i in range(phi_k.shape[-2]):
            #    s = s + phi_k[..., i, :, None] @ v[..., i, None, :]
            #    grad_q[..., i, :] = (acc[..., i, None, :] @ s.swapaxes(-1, -2)).squeeze(
            #        -2
            #    )
            outer_product = phi_q[..., None] * acc[..., None, :]
            s = cp.flip(cp.cumsum(cp.flip(outer_product, axis=-3), axis=-3), axis=-3)
            grad_v = (s.swapaxes(-1, -2) @ phi_k[..., :, None]).squeeze(-1)
            grad_k = (s @ v[..., :, None]).squeeze(-1)
            # grad_v = cp.zeros_like(phi_k)
            # grad_k = cp.zeros_like(phi_k)
            # for i in range(phi_q.shape - 1, -1, -1):
            #    s = s + phi_q[..., i, :, None] @ acc[..., i, None, :]
            #    grad_v[..., i, :] = (
            #        s.swapaxes(-1, -2) @ phi_k[..., i, :, None]
            #    ).squeeze(-1)
            #    grad_k[..., i, :] = (s @ v[..., i, :, None]).squeeze(-1)

        def local_grad_q(acc: cp.ndarray) -> cp.ndarray:
            nonlocal grad_q
            if grad_q is None:
                build_local_grads(acc)
            return grad_q

        def local_grad_v(acc: cp.ndarray) -> cp.ndarray:
            nonlocal grad_v
            if grad_v is None:
                build_local_grads(acc)
            return grad_v

        def local_grad_k(acc: cp.ndarray) -> cp.ndarray:
            nonlocal grad_k
            if grad_k is None:
                build_local_grads(acc)
            return grad_k

        ops = [
            (phi_q_tensor, "attention_numerator_q", local_grad_q),
            (phi_k_tensor, "attention_numerator_k", local_grad_k),
            (v_tensor, "attention_numerator_v", local_grad_v),
        ]

        return Tensor(num, ops)


class SoftmaxSelfAttention:
    def __init__(self, embedding_size, n_heads=1):
        assert embedding_size % n_heads == 0
        weight_data = cp.random.normal(
            0, 0.1, (embedding_size, embedding_size * 3), dtype=cp.float32
        )
        self.weights = Tensor(weight_data, frozen=False)
        self.out_weights = Tensor(
            cp.random.normal(
                0, 0.1, (embedding_size, embedding_size), dtype=cp.float32
            ),
            frozen=False,
        )
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.key_size = embedding_size // n_heads
        self.params = [self.weights, self.out_weights]

    def forward(self, input: Tensor):
        # let's be clear about shapes, as we go
        # input: (b, s, emb)
        # get our q,k,v values
        # q,k,v are all (b,s,key)
        qwk = input @ self.weights
        # now we need to calculate our similarities. Have to stop going off of AIAYN now,
        # switch to the linear attention paper
        # if we were doing AIAYN, we would get: result = softmax((Q@K.T) / sqrt(self.key_size)) @ V

        # now we want to split into our heads
        # and then shuffle n_heads to be one of the batch dims
        q = qwk.index_reshape_transpose(
            (Ellipsis, slice(None, self.embedding_size)),
            (input.value.shape[0], -1, self.n_heads, self.key_size),
            1,
            2,
        )
        k = qwk.index_reshape_transpose(
            (Ellipsis, slice(self.embedding_size, self.embedding_size * 2)),
            (input.value.shape[0], -1, self.n_heads, self.key_size),
            1,
            2,
        )
        v = qwk.index_reshape_transpose(
            (Ellipsis, slice(self.embedding_size * 2, None)),
            (input.value.shape[0], -1, self.n_heads, self.key_size),
            1,
            2,
        )

        result = (
            self.rotate(q)
            @ self.rotate(k)
            .transpose_last()
            .mult_constant(1 / cp.sqrt(self.key_size, dtype=cp.float32))
        ).softmax() @ v

        # now we need to reassemble from the multiheads
        result = result.transpose(1, 2)
        result = result.reshape(
            (input.value.shape[0], input.value.shape[1], self.key_size * self.n_heads)
        )

        return result @ self.out_weights

    def rotate(self, tensor: Tensor):
        # FIXME: this shouldn't be part of the self attention class
        # NOTE: this assumes an even number of dimensions
        # we're specifically rotating the final dim based on the penultimate index
        # everything above that is just batched
        result = kernels.rope(tensor.value)

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            return kernels.rope(acc, backward=True)

        ops = [(tensor, "rope", local_grad)]

        return Tensor(result, ops)


class TransformerLayer:
    def __init__(self, embedding_size, n_heads=1, expansion_factor=1):
        self.attention = SoftmaxSelfAttention(embedding_size, n_heads)
        self.attention_norm = LayerNorm((embedding_size,))
        self.linear_expand = Linear(
            embedding_size, embedding_size * expansion_factor, True
        )
        self.linear_activation = Elu()
        self.linear_contract = Linear(
            embedding_size * expansion_factor, embedding_size, True
        )
        self.linear_norm = LayerNorm((embedding_size,))

        self.params = []
        self.params.extend(self.attention.params)
        self.params.extend(self.attention_norm.params)
        self.params.extend(self.linear_expand.params)
        self.params.extend(self.linear_contract.params)
        self.params.extend(self.linear_norm.params)

    def forward(self, tensor: Tensor) -> Tensor:
        attended = self.attention.forward(tensor)
        interim = self.attention_norm.forward(attended + tensor)
        processed = self.linear_contract.forward(
            self.linear_activation.forward(self.linear_expand.forward(interim))
        )
        return self.linear_norm.forward(processed + interim)


# TODO: not really a layer. Need to fix my organization
class CrossEntropyLoss:
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # This is not fully general - I'm only worrying about the case where the targets are not probas
        loss, grad_info = kernels.cross_entropy(input.value, target.value)

        # go ahead and assume we want to sum this
        # FIXME: turns out this is using cupy operations still
        n = loss.size
        loss = loss.mean()

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            return acc * grad_info / n

        return Tensor(value=loss, operations=[(input, "CrossEntropy", local_grad)])
