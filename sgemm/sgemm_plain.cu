#include<xmath.h>
#include<sgemm.h>

__global__
void sgemm_plain_kernel(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float *matA,
    float *matB,
    float *matC)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    int gidx = blockIdx.x * blockDim.x + tidx;
    int gidy = blockIdx.y * blockDim.y + tidy;

    if (gidx >= N || gidy >= M) return;

    float tmp = 0.;
    for (int k = 0; k < K; k++) {
        tmp += matA[gidy * lda + k] * matB[k * ldb + gidx];
    }
    matC[gidy * ldc + gidx] = tmp;
}

int sgemm_plain(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float *matA,
    float *matB,
    float *matC)
{
    dim3 blockSize(16, 16);
    dim3 gridSize(N, M);

    sgemm_plain_kernel<<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);

    return 0;
}
