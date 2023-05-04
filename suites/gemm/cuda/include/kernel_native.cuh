#include<stdio.h>
#include<stdlib.h>
#define A(i,j) A[(i) + (j)*lda] //column major
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
// naive version
__global__  __launch_bounds__(1024)
void mysgemm_native(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    if (tx < M && ty < N) {
        float tmp = 0.0;
        for (int k_count = 0; k_count<K; k_count++){
            tmp += A(tx, k_count) * B(k_count, ty);
        }
        C(tx,ty) = alpha * tmp + beta*C(tx,ty);
    }
}