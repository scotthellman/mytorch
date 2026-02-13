import math
import time

import cupy as cp
from cupyx.profiler import benchmark

BLOCKSIZE = 32

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
    // FIXME: fix these comments, they're varying amounts of stale

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


size = 5000
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
# print(result)
duration = time.time() - start
print("took", duration)
grid = (math.ceil(B.shape[-1] / BLOCKSIZE), math.ceil(A.shape[-1] / BLOCKSIZE))
print(grid)

start = time.time()
strided_kernel(
    grid,
    (
        32,
        32,
    ),
    (A.shape[0], A.shape[1], B.shape[1], A, B, C),
)
duration = time.time() - start
print("took", duration)
print(C)
C = cp.empty((A.shape[0], B.shape[1]), dtype=cp.float32)


start = time.time()
shared_kernel(
    grid,
    (
        BLOCKSIZE,
        BLOCKSIZE,
    ),
    (A.shape[0], A.shape[1], B.shape[1], A, B, C),
)
duration = time.time() - start
print("took", duration)
print(C)


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
        strided_kernel,
        (
            grid,
            (BLOCKSIZE, BLOCKSIZE),
            (A.shape[0], A.shape[1], B.shape[1], A, B, C),
        ),
        n_repeat=10,
        n_warmup=3,
    )
)
