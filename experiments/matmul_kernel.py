import math
import time

import cupy as cp
from cupyx.profiler import benchmark

BLOCKSIZE = 32
THREAD_WINDOW = 2

strided_matmul_code = r"""
extern "C" __global__
void strided_matmul(int m, int n, int p, const float* A, const float* B, float* C) {
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

strided_kernel = cp.RawKernel(strided_matmul_code, "strided_matmul")

# non-square blocks will break this
coalescing_matmul_code = r"""
extern "C" __global__
void coalescing_matmul(int m, int n, int p, const float* A, const float* B, float* C) {
    const int x = blockIdx.x * blockDim.x + (threadIdx.x / blockDim.x);
    const int y = blockIdx.y * blockDim.x + (threadIdx.x % blockDim.x);

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

coalescing_kernel = cp.RawKernel(coalescing_matmul_code, "coalescing_matmul")
# FIXME: think harder about x and y - why is my first version faster?

shared_mem_matmul_code = r"""
extern "C" __global__
void shared_matmul(int m, int n, int p, const float* A, const float* B, float* C) {
    const int cRow = blockIdx.y * BLOCKSIZE;
    const int cCol = blockIdx.x * BLOCKSIZE;

    // so what needs to happen:
    // we build a small block of shared memory and load the relevant A and B values into it
    // each thread works through its designated x,y in that small block
    // when that's done, we iterate onwards to the next block of shared mem
    // conceptually not hard, just need to track the indices carefully

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const int threadX = threadIdx.x % BLOCKSIZE;
    const int threadY = threadIdx.y % BLOCKSIZE;


    // I could move the pointers around but I think that would just confuse me more for now

    float acc = 0.0;
    for(int chunkIdx = 0; chunkIdx < n; chunkIdx += BLOCKSIZE){
        // prepare As and Bs
        // Every thread is responsible for pulling one element
        // But only do that if we're in bounds! Use 0 if not
        // make sure everyone is done reading from As and Bs before we get back to it
        __syncthreads();
        if((chunkIdx+threadY < n) && (chunkIdx+threadX) < n) {
            As[threadY * BLOCKSIZE + threadX] = A[(cRow + threadY) * n + (chunkIdx + threadX)];
            Bs[threadY * BLOCKSIZE + threadX] = B[(threadY + chunkIdx) * p + (cCol + threadX)];
        }
        else {
            As[threadY * BLOCKSIZE + threadX] = 0.0;
            Bs[threadY * BLOCKSIZE + threadX] = 0.0;
        }
        __syncthreads();


        // dot the xth row of A with the yth col of B
        for(int k = 0; k < BLOCKSIZE; k++){
            int aIndex = (threadY * BLOCKSIZE) + k;
            int bIndex =  k*BLOCKSIZE + threadX;
            acc += As[aIndex] * Bs[bIndex];
        }
    }
    // Now we need to get back to the original coordinates
    if(threadX < m && threadY < p){
        C[(cRow+threadY)*p + cCol + threadX] = acc;
    }
}
""".replace("BLOCKSIZE", str(BLOCKSIZE))

shared_kernel = cp.RawKernel(shared_mem_matmul_code, "shared_matmul")

multiple_per_thread_matmul_code = r"""
extern "C" __global__
void multi_per_thread_matmul(int m, int k, int n, const float* A, const float* B, float* C) {
    // Some template variables:
    // BLOCKSIZE is the underlying block size - assuming our block is square
    // THREAD_WINDOW is how many X or Y indices each thread is responsible for (we're making square windows)
    // Therefore, each block is really responsible for (BLOCKSIZE*THREAD_WINDOW)^2 entries

    // Incidentally, this is also how much we're moving the chunk when we iterate
    const int chunkStride = BLOCKSIZE * THREAD_WINDOW;

    // Where the block starts in global coords
    const int blockRow = blockIdx.y * chunkStride;
    const int blockCol = blockIdx.x * chunkStride;

    // Go ahead and adjust A and B to account for where the blocks start
    A += blockRow * k;
    B += blockCol;

    // Shared memory for this block - we'll load chunks of A and B into here so that we don't
    // waste time repeatedly querying global memory
    // This is chunkStride^2, but we can't define it as such since that's a variable
    __shared__ float As[BLOCKSIZE * BLOCKSIZE * THREAD_WINDOW * THREAD_WINDOW];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE * THREAD_WINDOW * THREAD_WINDOW];

    // Where this thread starts in block coordinates. We will then tile
    // this over the whole block, to let each thread be responsible for more math
    const int threadX = threadIdx.x % BLOCKSIZE;
    const int threadY = threadIdx.y % BLOCKSIZE;

    float threadResults[THREAD_WINDOW * THREAD_WINDOW] = {0.0};
    // We want to minimize how much we touch the shared memory, so we'll use these arrays to
    // store the immediate rows/cols we want
    for(int chunkIdx = 0; chunkIdx < n; chunkIdx += chunkStride){
        // Assume that our pointers for A and B are always pointing to the first entry of the
        // chunk we care about. 

        // Load the next chunk of A and B into shared memory
        // If we've run off the edge of the matrix load 0s instead
        // Each thread is responsible for loading THREAD_WINDOW^2 values
        __syncthreads();
        for(int colOffset = 0; colOffset < THREAD_WINDOW; colOffset += 1){
            for(int rowOffset = 0; rowOffset < THREAD_WINDOW; rowOffset += 1){
                // We want to load, from A: A[(threadY+rowOffset) * chunkStride + (threadX+colOffset)]
                int localIndex = (threadY*THREAD_WINDOW + rowOffset) * chunkStride + (threadX*THREAD_WINDOW + colOffset);
                bool inBoundsK = chunkIdx+threadY*THREAD_WINDOW < k && (chunkIdx+threadX*THREAD_WINDOW) < k;
                bool inBoundsM = (blockRow+threadY*THREAD_WINDOW + rowOffset) < m;
                bool inBoundsN = (blockCol+threadX*THREAD_WINDOW + rowOffset) < n;
                if(inBoundsM && inBoundsK){
                    As[localIndex] = A[(threadY*THREAD_WINDOW + rowOffset) * k + (threadX*THREAD_WINDOW + colOffset)];
                } else{
                    As[localIndex] = 0.0;
                }
                if(inBoundsN && inBoundsK){
                    Bs[localIndex] = B[(threadY*THREAD_WINDOW + rowOffset) * n + (threadX*THREAD_WINDOW + colOffset)];
                } else {
                    Bs[localIndex] = 0.0;
                }
            }
        }
        __syncthreads();


        //for each thread, we need to dot the xth col of b with the yth row of a
        // but with an extra iteration - really we care about xth + stride and yth+stride
        for(int dotIndex = 0; dotIndex < chunkStride; dotIndex++){
            for(int threadOffsetY = 0; threadOffsetY < THREAD_WINDOW; threadOffsetY += 1){
                for(int threadOffsetX = 0; threadOffsetX < THREAD_WINDOW; threadOffsetX += 1){
                    int aIndex = (threadY*THREAD_WINDOW + threadOffsetY) * chunkStride + (dotIndex);
                    int bIndex = (dotIndex) * chunkStride + (threadX*THREAD_WINDOW + threadOffsetX);
                    threadResults[threadOffsetY*THREAD_WINDOW + threadOffsetX] += As[aIndex] * Bs[bIndex];
                }
            }
        }
        // Shuffle A and B along as we go - simplifies the indexing to move the pointer directly
        A += chunkStride;
        B += chunkStride * n;
    }
    // Now we have to pull threadResults into C
    for(int threadOffsetY = 0; threadOffsetY < THREAD_WINDOW; threadOffsetY += 1){
        for(int threadOffsetX = 0; threadOffsetX < THREAD_WINDOW; threadOffsetX += 1){
            int cX = (blockCol + threadX*THREAD_WINDOW + threadOffsetX);
            int cY = (blockRow + threadY*THREAD_WINDOW + threadOffsetY);
            if(cX < n && cY < m){
                C[cY * n + cX] = threadResults[threadOffsetY * THREAD_WINDOW + threadOffsetX];
            }
        }
    }
}
""".replace("BLOCKSIZE", str(BLOCKSIZE)).replace("THREAD_WINDOW", str(THREAD_WINDOW))

multi_per_thread_kernel = cp.RawKernel(
    multiple_per_thread_matmul_code, "multi_per_thread_matmul"
)


size = 10000
A = cp.random.random((size, size), dtype=cp.float32)
B = cp.random.random((size, size), dtype=cp.float32)
C = cp.empty((A.shape[0], B.shape[1]), dtype=cp.float32)

# A = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
# B = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
# A = cp.arange(16, dtype=cp.float32).reshape(4, 4)
# B = cp.arange(16, dtype=cp.float32).reshape(4, 4)
nA = cp.asnumpy(A)
nB = cp.asnumpy(B)
start = time.time()
result = nA @ nB
print(result)
duration = time.time() - start
print("CPU took", duration)
multigrid = (
    math.ceil(B.shape[-1] / (BLOCKSIZE * THREAD_WINDOW)),
    math.ceil(A.shape[-1] / (BLOCKSIZE * THREAD_WINDOW)),
)
grid = (
    math.ceil(B.shape[-1] / (BLOCKSIZE * THREAD_WINDOW)),
    math.ceil(A.shape[-1] / (BLOCKSIZE * THREAD_WINDOW)),
)

print(
    benchmark(
        shared_kernel,
        (
            grid,
            (BLOCKSIZE, BLOCKSIZE),
            (A.shape[0], A.shape[1], B.shape[1], A, B, C),
        ),
        n_repeat=10,
        n_warmup=3,
    )
)

print(
    benchmark(
        multi_per_thread_kernel,
        (
            multigrid,
            (BLOCKSIZE, BLOCKSIZE),
            (A.shape[0], A.shape[1], B.shape[1], A, B, C),
        ),
        n_repeat=10,
        n_warmup=3,
    )
)
