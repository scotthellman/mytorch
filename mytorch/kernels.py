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

        for(int chunkIdx = 0; chunkIdx < n; chunkIdx += chunkSize){
            // Assume that our pointers for A and B are always pointing to the first entry of the chunk we care about. 

            // Load the next chunk of A and B into shared memory. If we've run off the edge of the matrix load 0s instead
            // Each thread is responsible for loading THREAD_WINDOW^2 values
            // And sync immediately before and after writing to shared mem, to make sure all threads are viewing 
            // fresh data
            __syncthreads();
            for(int colOffset = 0; colOffset < THREAD_WINDOW; colOffset += 1){
                for(int rowOffset = 0; rowOffset < THREAD_WINDOW; rowOffset += 1){
                    int localIndex = (threadY + rowOffset) * chunkSize + (threadX + colOffset);
                    bool inBoundsK = chunkIdx+threadY < k && (chunkIdx+threadX) < k;
                    bool inBoundsM = (blockRow+threadY + rowOffset) < m;
                    bool inBoundsN = (blockCol+threadX + rowOffset) < n;
                    if(inBoundsM && inBoundsK){
                        As[localIndex] = A[(threadY + rowOffset) * k + (threadX + colOffset + chunkIdx)];
                    } else{
                        As[localIndex] = 0.0;
                    }
                    if(inBoundsN && inBoundsK){
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
    assert a.shape[-1] == b.shape[-2]
    use_simple_kernel = a.shape[-1] < 32
    # FIXME: tune this
    if use_simple_kernel:
        kernel = low_k_matmul_kernel
        grid_size = (
            math.ceil(a.shape[-1] / (BLOCKSIZE)),
            math.ceil(a.shape[-1] / (BLOCKSIZE)),
        )
    else:
        kernel = matmul_kernel
        grid_size = (
            math.ceil(a.shape[-1] / (BLOCKSIZE * THREAD_WINDOW)),
            math.ceil(a.shape[-1] / (BLOCKSIZE * THREAD_WINDOW)),
        )

    a_batch = a.shape[:-2]
    b_batch = b.shape[:-2]
    broadcast_shape = cp.broadcast_shapes(a_batch, b_batch)

    # FIXME: we really don't want these copies, but that'll
    # make the kernel a lot more complex
    # FIXME: I Wonder how often we actually need to broadcast. It might be
    # worth checking and only copying if there's actually a bcast happening
    a_broadcast = cp.broadcast_to(a, broadcast_shape + a.shape[-2:]).copy()
    b_broadcast = cp.broadcast_to(b, broadcast_shape + b.shape[-2:]).copy()
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
            a_broadcast,
            b_broadcast,
            result,
        ),
    )
    return result
