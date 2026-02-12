import cupy as cp

naive_matmul_code = r"""
extern "C" __global__
void naive_matmul(int m, int n, int p, const float* A, const float* B, float* C) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < m && y < p) {
        // dot the xth row of A with the yth col of B
        float acc = 0.0;
        for(int i = 0; i < n; i++){
            acc += A[i + x*n] * B[i*p + y];
        }
        C[x*p + y] = acc;
    }
}
"""

naive_kernel = cp.RawKernel(naive_matmul_code, "naive_matmul")

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


A = cp.array([[1, 2, 3], [9, 8, 7]], dtype=cp.float32)
B = cp.array([[-1, -2], [1, 2], [0, 1]], dtype=cp.float32)
A = cp.random.random((10, 20), dtype=cp.float32)
B = cp.random.random((20, 3), dtype=cp.float32)
C = cp.empty((10, 3), dtype=cp.float32)

print(A @ B)

strided_kernel(
    (
        1,
        1,
    ),
    (
        2,
        2,
    ),
    (A.shape[0], A.shape[1], B.shape[1], A, B, C),
)
print(C)
