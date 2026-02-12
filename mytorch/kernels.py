import cupy as cp

add_code = r"""
extern "C" __global__
void add_kernel(const float* a, const float* b, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride){
        out[i] = a[i] + b[i];
    }
}
"""
add_kernel = cp.RawKernel(add_code, "add_kernel")

sub_code = r"""
extern "C" __global__
void sub_kernel(const float* a, const float* b, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride){
        out[i] = a[i] - b[i];
    }
}
"""
sub_kernel = cp.RawKernel(sub_code, "sub_kernel")

neg_code = r"""
extern "C" __global__
void neg_kernel(const float* a, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride){
        out[i] = -a[i];
    }
}
"""
neg_kernel = cp.RawKernel(neg_code, "neg_kernel")

mul_code = r"""
extern "C" __global__
void mul_kernel(const float* a, const float* b, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride){
        out[i] = a[i] * b[i];
    }
}
"""
mul_kernel = cp.RawKernel(mul_code, "mul_kernel")

div_code = r"""
extern "C" __global__
void div_kernel(const float* a, const float* b, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride){
        out[i] = a[i] / b[i];
    }
}
"""
div_kernel = cp.RawKernel(div_code, "div_kernel")

div_local_grad_code = r"""
extern "C" __global__
void div_local_grad_kernel(const float* path, const float* num, const float* den, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride){
        out[i] = -path[i] * num[i] / (den[i] * den[i]);
    }
}
"""
div_local_grad_kernel = cp.RawKernel(div_local_grad_code, "div_local_grad_kernel")

# FIXME: do better
matmul_code = r"""
extern "C" __global__
void matmul(int m, int n, int p, const float* A, const float* B, float* C) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // total number of threads in x
    const int x_stride = blockDim.x*gridDim.x;
    const int y_stride = blockDim.y*gridDim.y;

    for(int i = x; i <m; i += x_stride){
        for(int j = y; j<p; j += y_stride){
            // dot the xth row of A with the yth col of B
            float acc = 0.0;
            for(int k = 0; k < n; k++){
                acc += A[k + i*n] * B[k*p + j];
            }
            C[i*p + j] = acc;
        }
    }
}
"""
matmul_kernel = cp.RawKernel(matmul_code, "matmul")

exp_code = r"""
extern "C" __global__
void exp_kernel(const float* a, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride){
        out[i] = exp(a[i]);
    }
}
"""
exp_kernel = cp.RawKernel(exp_code, "exp_kernel")


def add(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    broadcast = cp.broadcast(a, b)
    result_shape = broadcast.shape
    result = cp.empty(result_shape, dtype=cp.float32)
    n = result.size
    grid_size = (4,)
    block_size = (256,)
    add_kernel(
        grid_size,
        block_size,
        (broadcast.values[0], broadcast.values[1], result, n),
    )
    return result


def sub(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    broadcast = cp.broadcast(a, b)
    result_shape = broadcast.shape
    result = cp.empty(result_shape, dtype=cp.float32)
    n = result.size
    grid_size = (4,)
    block_size = (256,)
    sub_kernel(
        grid_size,
        block_size,
        (broadcast.values[0], broadcast.values[1], result, n),
    )
    return result


def mul(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    broadcast = cp.broadcast(a, b)
    result_shape = broadcast.shape
    result = cp.empty(result_shape, dtype=cp.float32)
    n = result.size
    grid_size = (4,)
    block_size = (256,)
    mul_kernel(
        grid_size,
        block_size,
        (broadcast.values[0], broadcast.values[1], result, n),
    )
    return result


def div(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    broadcast = cp.broadcast(a, b)
    result_shape = broadcast.shape
    result = cp.empty(result_shape, dtype=cp.float32)
    n = result.size
    grid_size = (4,)
    block_size = (256,)
    div_kernel(
        grid_size,
        block_size,
        (broadcast.values[0], broadcast.values[1], result, n),
    )
    return result


def div_local_grad(path: cp.ndarray, num: cp.ndarray, den: cp.ndarray) -> cp.ndarray:
    broadcast = cp.broadcast(path, num, den)
    result = cp.empty(broadcast.shape, dtype=cp.float32)
    n = result.size
    grid_size = (4,)
    block_size = (256,)
    div_local_grad_kernel(
        grid_size,
        block_size,
        (broadcast.values[0], broadcast.values[1], broadcast.values[2], result, n),
    )
    return result


def neg(a: cp.ndarray) -> cp.ndarray:
    result_shape = a.shape
    result = cp.empty(result_shape, dtype=cp.float32)
    n = result.size
    grid_size = (4,)
    block_size = (256,)
    neg_kernel(
        grid_size,
        block_size,
        (a, result, n),
    )
    return result


def exp(a: cp.ndarray) -> cp.ndarray:
    result_shape = a.shape
    result = cp.empty(result_shape, dtype=cp.float32)
    n = result.size
    grid_size = (4,)
    block_size = (256,)
    exp_kernel(
        grid_size,
        block_size,
        (a, result, n),
    )
    return result


def matmul(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    assert a.shape[1] == b.shape[0]
    result = cp.empty((a.shape[0], b.shape[1]), dtype=cp.float32)
    n = result.size
    grid_size = (16, 16)
    block_size = (32, 32)
    matmul_kernel(
        grid_size,
        block_size,
        (a.shape[0], a.shape[1], b.shape[1], a, b, result),
    )
    return result
