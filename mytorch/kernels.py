import math

import cupy as cp

# These constants are used by the matmul code - we have to define them up here
# so that we can find/replace them in the cuda source. From cuda's perspective
# they have to be known at compile-time.
BLOCKSIZE = 32
THREAD_WINDOW = 2

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

matmul_code = r"""
extern "C" __global__
void matmul(int m, int k, int n, int batches, const float* A, const float* B, float* C) {
    // Implements matrix multiplication with shared memory and multiple C indices handled by one thread.
    // Works one chunk at a time, looking at a chunkSizexchunkSize block of A and B, and storing their
    // various rowxcol dot products in threadResults. By sliding chunk along the cols of A and the rows
    // of B, we accumulate the entire dot product for each row and col.
    // One extra wrinkle: We need to handle batched multiplications. We assume the last two dimensions
    // are the two actually being multiplied, everything else is batch dims. That's collapsed down into
    // the batches argument. We're in row-major world here, so moving to a new batch means we need to advance
    // our pointers by the product of the two final dims
    const int batchStepA = m*k;
    const int batchStepB = k*n;
    const int batchStepC = m*n;

    // Some template variables:
    // BLOCKSIZE is the underlying block size - assuming our block is square
    // THREAD_WINDOW is how many X or Y indices each thread is responsible for (we're making square windows)
    // Therefore, each block is really responsible for (BLOCKSIZE*THREAD_WINDOW)^2 entries

    // How big one chunk is
    const int chunkSize = BLOCKSIZE * THREAD_WINDOW;

    // Where this block's work starts, in global coords
    const int blockRow = blockIdx.y * chunkSize;
    const int blockCol = blockIdx.x * chunkSize;

    // Go ahead and adjust A and B to account for where the blocks start
    A += blockRow * k;
    B += blockCol;

    // Shared memory for this block - we'll load chunks of A and B into here so that we don't
    // waste time repeatedly querying global memory
    // This is chunkSize^2, but we can't define it as such since that's a variable
    __shared__ float As[BLOCKSIZE * BLOCKSIZE * THREAD_WINDOW * THREAD_WINDOW];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE * THREAD_WINDOW * THREAD_WINDOW];

    // Where this thread starts in chunk coordinates. Each thread will handle a THREAD_WINDOWxTHREAD_WINDOW
    // square of dot products - this lets each thread handle more math, reducing memory pressure
    const int threadX = (threadIdx.x % BLOCKSIZE) * THREAD_WINDOW;
    const int threadY = (threadIdx.y % BLOCKSIZE) * THREAD_WINDOW;

    for(int batch = 0; batch < batches; batch++){
        // accumulate our partial dot products here
        float threadResults[THREAD_WINDOW * THREAD_WINDOW] = {0.0};

        for(int chunkIdx = 0; chunkIdx < k; chunkIdx += chunkSize){
            // Assume that our pointers for A and B are always pointing to the first entry of the chunk we care about. 

            // Load the next chunk of A and B into shared memory. If we've run off the edge of the matrix load 0s instead
            // Each thread is responsible for loading THREAD_WINDOW^2 values
            // And sync immediately before and after writing to shared mem, to make sure all threads are viewing 
            // fresh data
            __syncthreads();
            for(int colOffset = 0; colOffset < THREAD_WINDOW; colOffset += 1){
                for(int rowOffset = 0; rowOffset < THREAD_WINDOW; rowOffset += 1){
                    int localIndex = (threadY + rowOffset) * chunkSize + (threadX + colOffset);
                    bool inBoundsKA = chunkIdx + threadX + colOffset < k;
                    bool inBoundsKB = chunkIdx + threadY + rowOffset < k;
                    bool inBoundsM = (blockRow+threadY + rowOffset) < m;
                    bool inBoundsN = (blockCol+threadX + colOffset) < n;
                    if(inBoundsM && inBoundsKA){
                        As[localIndex] = A[(threadY + rowOffset) * k + (threadX + colOffset + chunkIdx)];
                    } else{
                        As[localIndex] = 0.0;
                    }
                    if(inBoundsN && inBoundsKB){
                        Bs[localIndex] = B[(threadY + rowOffset + chunkIdx) * n + (threadX + colOffset)];
                    } else {
                        Bs[localIndex] = 0.0;
                    }
                }
            }
            __syncthreads();

            //for each thread, we need to dot the xth col of b with the yth row of a
            // but with two extra iterations - really we care about xth + stride and yth+stride
            for(int dotIndex = 0; dotIndex < chunkSize; dotIndex++){
                for(int threadOffsetY = 0; threadOffsetY < THREAD_WINDOW; threadOffsetY += 1){
                    int aIndex = (threadY + threadOffsetY) * chunkSize + (dotIndex);
                    float aVal = As[aIndex];
                    for(int threadOffsetX = 0; threadOffsetX < THREAD_WINDOW; threadOffsetX += 1){
                        int bIndex = (dotIndex) * chunkSize + (threadX + threadOffsetX);
                        threadResults[threadOffsetY*THREAD_WINDOW + threadOffsetX] += aVal * Bs[bIndex];
                    }
                }
            }
        }
        // Now we have to pull threadResults into C
        for(int threadOffsetY = 0; threadOffsetY < THREAD_WINDOW; threadOffsetY += 1){
            for(int threadOffsetX = 0; threadOffsetX < THREAD_WINDOW; threadOffsetX += 1){
                int cX = (blockCol + threadX + threadOffsetX);
                int cY = (blockRow + threadY + threadOffsetY);
                if(cX < n && cY < m){
                    C[cY * n + cX] = threadResults[threadOffsetY * THREAD_WINDOW + threadOffsetX];
                }
            }
        }
        A += batchStepA;
        B += batchStepB;
        C += batchStepC;
    }
}
""".replace("BLOCKSIZE", str(BLOCKSIZE)).replace("THREAD_WINDOW", str(THREAD_WINDOW))

matmul_kernel = cp.RawKernel(matmul_code, "matmul")

low_k_matmul_code = r"""
extern "C" __global__
void matmul(int m, int n, int p, int outer_loop, const float* A, const float* B, float* C) {
    // This is a naive matmul, but it works well for matrices with a very small inner dimension
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // total number of threads in x
    const int x_stride = blockDim.x*gridDim.x;
    const int y_stride = blockDim.y*gridDim.y;

    for(int batch = 0; batch < outer_loop; batch += 1){
        const int batch_offset_A = batch * m * n;
        const int batch_offset_B = batch * p * n;
        const int batch_offset_C = batch * m * p;
        for(int i = x; i <m; i += x_stride){
            for(int j = y; j<p; j += y_stride){
                // dot the xth row of A with the yth col of B
                float acc = 0.0;
                for(int k = 0; k < n; k++){
                    acc += A[batch_offset_A + k + i*n] * B[batch_offset_B + k*p + j];
                }
                C[batch_offset_C + i*p + j] = acc;
            }
        }
    }
}
"""
low_k_matmul_kernel = cp.RawKernel(low_k_matmul_code, "matmul")

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

logistic_code = r"""
extern "C" __global__
void logistic_kernel(const float* a, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride){
        out[i] = 1.0 / (1 + exp(-a[i]));
    }
}
"""
logistic_kernel = cp.RawKernel(logistic_code, "logistic_kernel")

sqrt_code = r"""
extern "C" __global__
void sqrt_kernel(const float* a, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride){
        out[i] = sqrtf(a[i]);
    }
}
"""
sqrt_kernel = cp.RawKernel(sqrt_code, "sqrt_kernel")

# TODO: it's probably better to split up the max and ce computations
cross_entropy_code = r"""
extern "C" __global__
void cross_entropy_kernel(const float* a, const int* target, float* out, float* grad_out, const int emb, const int batch) {
    // assume we launched a grid that is large enough to cover all of the batches.
    // So each thread is responsible for computing the cross entropy for
    // one cell. 
    // Note that this computes the gradient terms at the same time, since that reuses the softmax denominator
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int batch_index = tid * emb;

    // When is a thread out of bounds?
    // when tid is bigger than batch
    if(tid < batch){
        // first we have to know the maximum value
        int target_index = target[tid];
        float target_value = a[batch_index + target_index];
        float max_val = target_value;
        for (int i = 0; i < emb; i++){
            float current_val = a[batch_index + i];
            max_val = max_val > current_val ? max_val : current_val;
        }

        // now we can compute the actual loss. This us using the LogSumExp trick
        // for numerical stability in all of those exponents
        float exp_sum = 0;
        for (int i = 0; i < emb; i++){
            exp_sum += exp(a[batch_index + i] - max_val);
        }

        // now we need to populate the outputs
        // we need grad_out for everything, and out for the true values
        out[tid] = -target_value + max_val + log(exp_sum);
        for(int i = 0 ; i < emb; i++){
            float true_label = i == target_index ? 1.0 : 0.0;
            float log_prob = a[batch_index + i] - max_val - log(exp_sum);
            grad_out[batch_index + i] = exp(log_prob) - true_label;
        }
    }
}
"""
cross_entropy_kernel = cp.RawKernel(cross_entropy_code, "cross_entropy_kernel")

# having each thread deal with one whole layer seems like a good balance of easy and fast
layernorm_code = r"""
extern "C" __global__
void layernorm_kernel(const float* a, float* out, float* inv_vars, float* norms, int emb_size, int total_size, float eps) {
    int embless_tid = (blockDim.x * blockIdx.x + threadIdx.x);
    int embless_stride = blockDim.x * gridDim.x;
    int batch_dim_size = total_size / emb_size;

    for (int stride_index = 0; stride_index + embless_tid < batch_dim_size; stride_index += embless_stride){
        // we need to compute the mean and variance of this layer
        // i is this thread's index into the batch dims
        int i = embless_tid + stride_index;
        float mean = 0.0;
        float var = 0.0;
        // ok so unfortunately this is very bad coalescing behavior. Something to think about for a v2
        for (int j = 0; j < emb_size; j++) {
            float value = a[i*emb_size+j];
            float delta = value - mean;
            mean += (delta)/float(j+1);
            var += delta * (value - mean);
        }
        var /= emb_size;

        // and now, for every value, we subtract the mean, divided by sqrt(var+eps)
        for (int j = 0; j < emb_size; j++) {
            float norm = a[i*emb_size+j] - mean;
            out[i*emb_size+j] = (norm) / sqrt(var + eps);
            norms[i*emb_size+j] = norm;
        }

        inv_vars[i] = 1.0/var;
    }
}
"""
layernorm_kernel = cp.RawKernel(layernorm_code, "layernorm_kernel")

# https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
layernorm_back_code = r"""
extern "C" __global__
void layernorm_back_kernel(const float* a, const float* inv_vars, const float* norms, float* out, int emb_size, int total_size) {
    int embless_tid = (blockDim.x * blockIdx.x + threadIdx.x);
    int embless_stride = blockDim.x * gridDim.x;
    int batch_dim_size = total_size / emb_size;

    for (int stride_index = 0; stride_index + embless_tid < batch_dim_size; stride_index += embless_stride){
        // first we need the mean of the incoming grads, and of the incoming grads * the norms
        int i = embless_tid + stride_index;
        float grad_mean = 0.0;
        float mult_mean = 0.0;
        for (int j = 0; j < emb_size; j++) {
            float grad_value = a[i*emb_size+j];
            float grad_delta = grad_value - grad_mean;
            grad_mean += (grad_delta)/float(j+1);

            float norm_value = norms[i*emb_size+j];
            float mult_mean_delta = (norm_value * grad_value) - mult_mean;
            mult_mean += mult_mean_delta/float(j+1);
        }
        
        // and now, for every value: grad - grad_mean - norm*mult_mean
        // all that is then * inv var
        for (int j = 0; j < emb_size; j++) {
            float result = sqrt(inv_vars[i]) * (a[i*emb_size+j] - grad_mean - norms[i*emb_size+j]*mult_mean) * inv_vars[i];
            out[i*emb_size+j] = result;
        }
    }
}
"""
layernorm_back_kernel = cp.RawKernel(layernorm_back_code, "layernorm_back_kernel")


rope_code = r"""
extern "C" __global__
void rope_kernel(const float* a, float* out, int emb_size, int seq_size, int batch_size, bool backward) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const float theta_sign = backward ? -1.0: 1.0;

    // for coalescing purposes: x indexes into embedding, y into sequence
    // z handles the batch dims
    // arguably one thread should handle 2 indices, but that's future work

    bool in_bounds = z < batch_size && y < seq_size && x < emb_size;

    if(in_bounds) {
        float theta = theta_sign * pow(10000.0, -2.0*float(x/2)/float(emb_size));

        int true_index = z*seq_size*emb_size + y*emb_size + x;
        int sin_index = true_index + (1 - 2*(true_index/2));

        float cos_val = a[true_index] * cos((y+1.0) * theta);
        float sin_val = a[sin_index] * sin((y+1.0) * theta);
        float sign = (-2 * (sin_index%2)) + 1;
        out[true_index] = cos_val + sign * sin_val;
    }
}
"""
rope_kernel = cp.RawKernel(rope_code, "rope_kernel")


def add(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    broadcast_shape = cp.broadcast_shapes(a.shape, b.shape)
    result = cp.empty(broadcast_shape, dtype=cp.float32)
    a = broadcast_if_needed(a, broadcast_shape)
    b = broadcast_if_needed(b, broadcast_shape)
    n = result.size
    block_size = (512,)
    grid_size = (math.ceil(result.size / block_size[0]),)
    add_kernel(
        grid_size,
        block_size,
        (a, b, result, n),
    )
    return result


def sub(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    broadcast_shape = cp.broadcast_shapes(a.shape, b.shape)
    result = cp.empty(broadcast_shape, dtype=cp.float32)
    a = broadcast_if_needed(a, broadcast_shape)
    b = broadcast_if_needed(b, broadcast_shape)
    n = result.size
    block_size = (512,)
    grid_size = (math.ceil(result.size / block_size[0]),)
    sub_kernel(
        grid_size,
        block_size,
        (a, b, result, n),
    )
    return result


def mul(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    broadcast_shape = cp.broadcast_shapes(a.shape, b.shape)
    result = cp.empty(broadcast_shape, dtype=cp.float32)
    a = broadcast_if_needed(a, broadcast_shape)
    b = broadcast_if_needed(b, broadcast_shape)
    n = result.size
    block_size = (512,)
    grid_size = (math.ceil(result.size / block_size[0]),)
    mul_kernel(
        grid_size,
        block_size,
        (a, b, result, n),
    )
    return result


def div(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    broadcast_shape = cp.broadcast_shapes(a.shape, b.shape)
    result = cp.empty(broadcast_shape, dtype=cp.float32)
    a = broadcast_if_needed(a, broadcast_shape)
    b = broadcast_if_needed(b, broadcast_shape)
    n = result.size
    block_size = (512,)
    grid_size = (math.ceil(result.size / block_size[0]),)
    div_kernel(
        grid_size,
        block_size,
        (a, b, result, n),
    )
    return result


def div_local_grad(path: cp.ndarray, num: cp.ndarray, den: cp.ndarray) -> cp.ndarray:
    broadcast_shape = cp.broadcast_shapes(path.shape, num.shape, den.shape)
    result = cp.empty(broadcast_shape, dtype=cp.float32)
    n = result.size
    block_size = (512,)
    grid_size = (math.ceil(result.size / block_size[0]),)
    path = broadcast_if_needed(path, broadcast_shape)
    num = broadcast_if_needed(num, broadcast_shape)
    den = broadcast_if_needed(den, broadcast_shape)
    div_local_grad_kernel(
        grid_size,
        block_size,
        (
            path,
            num,
            den,
            result,
            n,
        ),
    )
    return result


def neg(a: cp.ndarray) -> cp.ndarray:
    result_shape = a.shape
    result = cp.empty(result_shape, dtype=cp.float32)
    n = result.size
    block_size = (512,)
    grid_size = (math.ceil(result.size / block_size[0]),)
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
    block_size = (512,)
    grid_size = (math.ceil(result.size / block_size[0]),)
    exp_kernel(
        grid_size,
        block_size,
        (a, result, n),
    )
    return result


def logistic(a: cp.ndarray) -> cp.ndarray:
    result_shape = a.shape
    result = cp.empty(result_shape, dtype=cp.float32)
    n = result.size
    block_size = (512,)
    grid_size = (math.ceil(result.size / block_size[0]),)
    logistic_kernel(
        grid_size,
        block_size,
        (a, result, n),
    )
    return result


def sqrt(a: cp.ndarray) -> cp.ndarray:
    result_shape = a.shape
    result = cp.empty(result_shape, dtype=cp.float32)
    n = result.size
    block_size = (512,)
    grid_size = (math.ceil(result.size / block_size[0]),)
    sqrt_kernel(
        grid_size,
        block_size,
        (a, result, n),
    )
    return result


def cross_entropy(a: cp.ndarray, targets: cp.ndarray) -> cp.ndarray:
    # For now I should be in very direct control of what's passed in here,
    # so i'm ignoring broadcasting
    result_shape = targets.shape
    result = cp.empty(result_shape, dtype=cp.float32)
    grad_result = cp.empty(a.shape, dtype=cp.float32)
    # we want the grid to cover all of targets
    n = result.size
    block_size = (512,)
    grid_size = (math.ceil(n / block_size[0]),)
    cross_entropy_kernel(
        grid_size,
        block_size,
        (a, targets, result, grad_result, a.shape[-1], n),
    )
    return result, grad_result


def matmul(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    assert a.shape[-1] == b.shape[-2]
    # FIXME: tune this, there probably is some level at which the simpler one is faster
    use_simple_kernel = False
    if use_simple_kernel:
        kernel = low_k_matmul_kernel
        grid_size = (
            math.ceil(a.shape[-2] / (BLOCKSIZE)),
            math.ceil(b.shape[-1] / (BLOCKSIZE)),
        )
    else:
        kernel = matmul_kernel
        # there's a footgun here: the x dimension of the grid
        # is handling columns, so we need to pull the shapes from
        # the "wrong" dimensions
        grid_size = (
            math.ceil(b.shape[-1] / (BLOCKSIZE * THREAD_WINDOW)),
            math.ceil(a.shape[-2] / (BLOCKSIZE * THREAD_WINDOW)),
        )
    a_batch = a.shape[:-2]
    b_batch = b.shape[:-2]
    broadcast_shape = cp.broadcast_shapes(a_batch, b_batch)

    # FIXME: we really don't want these copies, but that'll
    # make the kernel a lot more complex
    # bandaid fix for now: only copy if we really have to
    a = broadcast_if_needed(a, broadcast_shape + a.shape[-2:])
    b = broadcast_if_needed(b, broadcast_shape + b.shape[-2:])
    result = cp.empty(broadcast_shape + (a.shape[-2], b.shape[-1]), dtype=cp.float32)
    num_batches = 1
    for d in broadcast_shape:
        num_batches *= d

    block_size = (BLOCKSIZE, BLOCKSIZE)
    kernel(
        grid_size,
        block_size,
        (
            a.shape[-2],
            a.shape[-1],
            b.shape[-1],
            num_batches,
            a,
            b,
            result,
        ),
    )
    return result


def broadcast_if_needed(a: cp.ndarray, broadcast_shape: tuple[int]) -> cp.ndarray:
    # TODO: in the long run, we'd prefer not to copy at all. But that will
    # (seemingly) complicate the kernels considerably - they'd need to be
    # aware of the shape/stride info in cupy
    if a.shape == broadcast_shape:
        return a
    return cp.broadcast_to(a, broadcast_shape).copy()


def layernorm(a: cp.ndarray, eps: float = 1e-6) -> cp.ndarray:
    # Hard assumption here that we are operating on the last dimension
    # TODO: probably want the affine transformation part of this too
    emb_size = a.shape[-1]
    result_shape = a.shape
    result = cp.empty(result_shape, dtype=cp.float32)
    # TODO: storing norms is a big memory hog, but simplifies the back kernel. Maybe change that
    norms = cp.empty(result_shape, dtype=cp.float32)
    inv_vars = cp.empty(result_shape[:-1], dtype=cp.float32)
    # we want the grid to cover all of targets
    n = result.size
    block_size = (512,)
    grid_size = (
        math.ceil(
            n / (emb_size * block_size[0]),
        ),
    )
    layernorm_kernel(
        grid_size,
        block_size,
        (a, result, inv_vars, norms, a.shape[-1], n, eps),
    )
    grad_data = (inv_vars, norms)
    return result, grad_data


def layernorm_back(
    a: cp.ndarray, inv_vars: cp.ndarray, norms: cp.ndarray
) -> cp.ndarray:
    emb_size = a.shape[-1]
    result_shape = a.shape
    result = cp.empty(result_shape, dtype=cp.float32)
    # we want the grid to cover all of targets
    n = result.size
    block_size = (512,)
    grid_size = (
        math.ceil(
            n / (emb_size * block_size[0]),
        ),
    )
    layernorm_back_kernel(
        grid_size,
        block_size,
        (a, inv_vars, norms, result, a.shape[-1], n),
    )
    return result


def rope(a: cp.ndarray, backward: bool = False) -> cp.ndarray:
    emb_size = a.shape[-1]
    seq_size = a.shape[-2]
    batch_size = 1
    for s in a.shape[:-2]:
        batch_size *= s
    result_shape = a.shape
    result = cp.empty(result_shape, dtype=cp.float32)
    # we want the grid to cover all of targets
    # x is emb, y is seq, z is batch
    block_size = (32, 4, 2)
    grid_size = (
        math.ceil(
            emb_size / block_size[0],
        ),
        math.ceil(
            seq_size / block_size[1],
        ),
        math.ceil(
            batch_size / block_size[2],
        ),
    )
    rope_kernel(
        grid_size,
        block_size,
        (a, result, emb_size, seq_size, batch_size, backward),
    )
    return result
