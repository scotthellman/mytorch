"""Code for working with arrays while building a compute graph.

Broadly equivalent to PyTorch's Tensor class, just much smaller in scope.

There are two big pieces here: `Tensor.compute_gradient` which runs
reverse-mode automatic differentiation, and then a variety of mathematical
operations on `Tensor`s that build an implicit compute graph by populating
`self.operations` with tuples of:
(parent, operation name, backward gradient function)
"""

from __future__ import annotations

import cupy as cp

from mytorch import kernels


class Tensor:
    def __init__(
        self,
        value: cp.ndarray,
        operations: list | None = None,
        frozen=True,
    ):
        self.value: cp.ndarray = value
        self.frozen = frozen
        if operations is None:
            self.operations = []
        else:
            self.operations = operations
        self.reset()

    def reset(self):
        self.grad = cp.zeros_like(self.value)

    def compute_gradient(self):
        """Populate self.grad on this tensor and all child tensors.

        Runs reverse-mode automatic differentiation. Every tensor tracks
        the operations that were used to create it (in `self.operations`),
        and this function reifies that implicit compute graph and then visits
        it in topological order.
        """
        visit_order = self.toposort()
        path_gradients = {self: cp.ones(self.value.shape, dtype=cp.float32)}
        for current_variable in visit_order:
            current_value = path_gradients[current_variable]
            if not current_variable.frozen:
                current_variable.grad += path_gradients[current_variable]
            for child, name, op in current_variable.operations:
                # G.add_edge(hash(current_variable), hash(child), label=name)
                child_value = op(current_value)
                if child not in path_gradients:
                    path_gradients[child] = child_value
                else:
                    path_gradients[child] += child_value
            # free up that memory
            del path_gradients[current_variable]

    def build_compute_graph(self) -> dict["Tensor", set["Tensor"]]:
        """Constructs the compute graph defined by `self.operations`.

        We only need to know the children of every node in the graph,
        so that's the only thing we actually construct here.
        """
        children: dict[Tensor, set[Tensor]] = {}
        stack: list[Tensor] = [self]
        while stack:
            current_variable = stack.pop()
            if current_variable in children:
                # we got this one already
                continue
            children[current_variable] = set()
            for group in current_variable.operations:
                child = group[0]
                children[current_variable].add(child)
                stack.append(child)
        return children

    def toposort(self) -> list["Tensor"]:
        """Get a topological ordering of the nodes in the compute graph."""
        children = self.build_compute_graph()

        ordering = self._toposort(children, set())
        return ordering[::-1]

    def _toposort(
        self, children: dict["Tensor"], seen: set["Tensor"]
    ) -> list["Tensor"]:
        # we know our graph is a tree, so this is a straightforward dfs
        seen.add(self)
        ordering = []
        for child in children[self]:
            if child not in seen:
                ordering.extend(child._toposort(children, seen))
        ordering.append(self)
        return ordering

    def __getitem__(self, key) -> Tensor:
        result = self.value[key]

        def local_grad(acc: cp.ndarray):
            grad = cp.zeros_like(self.value)
            grad[key] = acc
            return grad

        return Tensor(result, [(self, "index", local_grad)])

    def __add__(self, b: Tensor) -> Tensor:
        result = kernels.add(self.value, b.value)
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        b_broadcast_axes = compute_broadcast_axes(b.value.shape, result.shape)

        def local_grad_self_add(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting_backwards(
                acc, self_broadcast_axes, self.value.shape
            )

        def local_grad_b_add(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting_backwards(acc, b_broadcast_axes, b.value.shape)

        operations = [
            (self, "add", local_grad_self_add),  # conceptually, 1*acc
            (b, "add", local_grad_b_add),
        ]
        return Tensor(result, operations)

    def __sub__(self, b: Tensor) -> Tensor:
        result = kernels.sub(self.value, b.value)
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        b_broadcast_axes = compute_broadcast_axes(b.value.shape, result.shape)

        def local_grad_self_sub(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting_backwards(
                acc, self_broadcast_axes, self.value.shape
            )

        def local_grad_b_sub(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting_backwards(
                kernels.neg(acc), b_broadcast_axes, b.value.shape
            )

        operations = [(self, "sub", local_grad_self_sub), (b, "sub", local_grad_b_sub)]
        return Tensor(result, operations)

    def __mul__(self, b: Tensor) -> Tensor:
        # elementwise
        result = kernels.mul(self.value, b.value)
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        b_broadcast_axes = compute_broadcast_axes(b.value.shape, result.shape)

        def local_grad_self_mul(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting_backwards(
                kernels.mul(acc, b.value), self_broadcast_axes, self.value.shape
            )

        def local_grad_b_mul(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting_backwards(
                kernels.mul(acc, self.value), b_broadcast_axes, b.value.shape
            )

        operations = [
            (self, "mul", local_grad_self_mul),
            (b, "mul", local_grad_b_mul),
        ]
        return Tensor(result, operations)

    def __neg__(self) -> Tensor:
        result = kernels.neg(self.value)

        def local_grad_neg(acc: cp.ndarray) -> cp.ndarray:
            return kernels.neg(acc)

        operations = [(self, "neg", local_grad_neg)]
        return Tensor(result, operations)

    def __truediv__(self, b: Tensor) -> Tensor:
        # elementwise
        result = kernels.div(self.value, b.value)
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        b_broadcast_axes = compute_broadcast_axes(b.value.shape, result.shape)

        def local_grad_self_div(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting_backwards(
                kernels.div(acc, b.value), self_broadcast_axes, self.value.shape
            )

        def local_grad_b_div(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting_backwards(
                kernels.div_local_grad(acc, self.value, b.value),
                b_broadcast_axes,
                b.value.shape,
            )

        operations = [
            (self, "div", local_grad_self_div),
            (b, "div", local_grad_b_div),
        ]
        return Tensor(result, operations)

    def __matmul__(self, b: Tensor) -> Tensor:
        result = kernels.matmul(self.value, b.value)
        self_broadcast_axes = compute_broadcast_axes(
            self.value.shape, result.shape, matmul=True
        )
        b_broadcast_axes = compute_broadcast_axes(
            b.value.shape, result.shape, matmul=True
        )

        def local_grad_self_matmul(acc: cp.ndarray) -> cp.ndarray:
            # TODO: arguably I should implement this myself
            # FIXME: especially now that i have to copy to make hte view concrete
            grad = kernels.matmul(acc, cp.swapaxes(b.value, -2, -1).copy())
            return handle_broadcasting_backwards(
                grad, self_broadcast_axes, self.value.shape
            )

        def local_grad_b_matmul(acc: cp.ndarray) -> cp.ndarray:
            grad = kernels.matmul(cp.swapaxes(self.value, -2, -1).copy(), acc)
            return handle_broadcasting_backwards(grad, b_broadcast_axes, b.value.shape)

        operations = [
            (self, "matmul", local_grad_self_matmul),
            (b, "matmul", local_grad_b_matmul),
        ]
        return Tensor(result, operations)

    def mul_and_add(self, mul_term: Tensor, add_term: Tensor) -> Tensor:
        result = kernels.mul_and_add(self.value, mul_term.value, add_term.value)
        self_broadcast_axes = compute_broadcast_axes(self.value.shape, result.shape)
        mul_term_broadcast_axes = compute_broadcast_axes(
            mul_term.value.shape, result.shape
        )
        add_term_broadcast_axes = compute_broadcast_axes(
            add_term.value.shape, result.shape
        )

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting_backwards(
                kernels.mul(acc, mul_term.value), self_broadcast_axes, self.value.shape
            )

        def local_grad_add(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting_backwards(
                acc, add_term_broadcast_axes, add_term.value.shape
            )

        def local_grad_mul(acc: cp.ndarray) -> cp.ndarray:
            return handle_broadcasting_backwards(
                kernels.mul(acc, self.value),
                mul_term_broadcast_axes,
                mul_term.value.shape,
            )

        operations = [
            (self, "mul_and_add", local_grad_self),
            (mul_term, "mul_and_add", local_grad_mul),
            (add_term, "mul_and_add", local_grad_add),
        ]
        return Tensor(result, operations)

    def add_constant(self, c: float) -> Tensor:
        # FIXME: do this myself
        result = self.value + c
        operations = [(self, "+c", lambda acc: acc)]
        return Tensor(result, operations)

    def mult_constant(self, c: float) -> Tensor:
        # FIXME: do this myself
        result = self.value * c
        operations = [(self, "*c", lambda acc: c * acc)]
        return Tensor(result, operations)

    def exp(self) -> Tensor:
        result = kernels.exp(self.value)
        operations = [(self, "exp", lambda acc: acc * kernels.exp(self.value))]
        return Tensor(result, operations)

    def sum(
        self, axis: int | None = None, keepdims: bool = False, constant_term: float = 0
    ) -> Tensor:
        # FIXME: need to do this myself
        result = cp.sum(self.value, axis=axis, keepdims=keepdims).copy() + constant_term

        def local_grad_self(acc: cp.ndarray) -> cp.ndarray:
            grad = acc * cp.ones_like(self.value)
            return grad

        operations = [(self, "sum", local_grad_self)]
        return Tensor(result, operations)

    def sqrt(self) -> Tensor:
        result = kernels.sqrt(self.value)
        half = cp.ones_like(self.value) * 0.5
        operations = [(self, "sqrt", lambda acc: acc * half / kernels.sqrt(self.value))]
        return Tensor(result, operations)

    def mean(self, axis: int | None = None) -> Tensor:
        # FIXME: need to do this myself
        result = cp.mean(self.value, axis=axis, keepdims=True)

        def local_grad_self_mean(acc: cp.ndarray) -> cp.ndarray:
            n = self.value.size if axis is None else self.value.shape[axis]
            grad = acc * cp.ones_like(self.value) / n
            return grad

        operations = [(self, "sum", local_grad_self_mean)]
        return Tensor(result, operations)

    def var(self, axis: int) -> Tensor:
        # FIXME: need to do this myself
        result = cp.var(self.value, axis=axis, keepdims=True)
        mean = cp.mean(self.value, axis=axis, keepdims=True)
        n = self.value.shape[axis]

        def local_grad_self_var(acc: cp.ndarray) -> cp.ndarray:
            grad = acc * 2 / n * (self.value - mean)
            return grad

        operations = [(self, "sum", local_grad_self_var)]
        return Tensor(result, operations)

    def elu(self, added: int = 0) -> Tensor:
        result = kernels.elu(self.value, added)

        def local_grad_self_elu(acc: cp.ndarray) -> cp.ndarray:
            return kernels.elu_back(result, acc, added)

        operations = [(self, "elu", local_grad_self_elu)]

        return Tensor(result, operations)

    def cumsum(self, axis: int) -> Tensor:
        # FIXME: need to do this myself
        result = cp.cumsum(self.value, axis=axis).copy()

        def local_grad_self_cumsum(acc: cp.ndarray) -> cp.ndarray:
            return cp.flip(
                cp.cumsum(cp.flip(acc, axis=axis), axis=axis), axis=axis
            ).copy()

        operations = [(self, "cumsum", local_grad_self_cumsum)]

        return Tensor(result, operations)

    def reshape(self, shape: tuple[int]) -> Tensor:
        result = self.value.reshape(shape).copy()

        def local_grad(acc: cp.ndarray) -> cp.ndarray:
            return acc.reshape(self.value.shape)

        operations = [
            (self, "reshape", local_grad),
        ]

        return Tensor(result, operations)

    def transpose_last(self) -> Tensor:
        # NOTE: I give myself permission to not do this myself,
        # I've been leaving indexing stuff to cupy
        return self.transpose(-1, -2)

    def transpose(self, i: int, j: int) -> Tensor:
        # NOTE: I give myself permission to not do this myself,
        # I've been leaving indexing stuff to cupy
        result = self.value.swapaxes(i, j).copy()

        def local_grad_self_transpose(acc: cp.ndarray) -> cp.ndarray:
            return acc.swapaxes(i, j).copy()

        operations = [(self, "T", local_grad_self_transpose)]

        return Tensor(result, operations)

    def index_reshape_transpose(
        self, key: tuple, shape: tuple[int], i: int, j: int
    ) -> Tensor:
        # fusing reshape and transpose since they both force copies
        result = self.value[key].reshape(shape).swapaxes(i, j).copy()

        def local_grad_self_reshape_transpose(acc: cp.ndarray) -> cp.ndarray:
            grad = cp.zeros_like(self.value)
            grad[key] = acc.swapaxes(i, j).reshape(self.value[key].shape)
            return grad

        operations = [(self, "IRT", local_grad_self_reshape_transpose)]

        return Tensor(result, operations)

    def permute(self, indices) -> Tensor:
        # NOTE: I give myself permission to not do this myself,
        # I've been leaving indexing stuff to cupy
        result = self.value[..., indices].copy()

        def local_grad_self_permute(acc: cp.ndarray) -> cp.ndarray:
            return acc[..., indices].copy()

        operations = [(self, "swapaxes", local_grad_self_permute)]

        return Tensor(result, operations)

    def softmax(self) -> Tensor:
        result = kernels.softmax(self.value)

        def local_grad_self_softmax(acc: cp.ndarray) -> cp.ndarray:
            return kernels.softmax_back(acc, result)

        operations = [(self, "softmax", local_grad_self_softmax)]

        return Tensor(result, operations)


def compute_broadcast_axes(
    start_shape: tuple[int], broadcast_shape: tuple[int], matmul: bool = False
) -> tuple[int]:
    """Determines what axes need to be adjusted as part of broadcasting."""
    altered_axes = []
    start = -3 if matmul else -1
    for i in range(start, -len(broadcast_shape) - 1, -1):
        current = broadcast_shape[i]
        if abs(i) > len(start_shape) or current != start_shape[i]:
            altered_axes.append(len(broadcast_shape) + i)

    return tuple(altered_axes)


def handle_broadcasting_backwards(
    path_gradient: cp.ndarray,
    broadcast_axes: tuple[int],
    target_shape: tuple[int],
) -> cp.ndarray:
    """If we broadcast going forward, we need to sum over those axes going backwards.

    This function does that.
    """
    if broadcast_axes:
        return cp.sum(path_gradient, axis=broadcast_axes).reshape(target_shape)
    return path_gradient
