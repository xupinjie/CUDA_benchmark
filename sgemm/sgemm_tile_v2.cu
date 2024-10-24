#include <cuda_runtime.h>
#include <cstdio>

#include<xmath.h>
#include<sgemm.h>

template<int mItem, int nItem, int kItem>
__global__
void sgemm_tile_v2_smem_kernel(
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

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    int gidx = bidx * blockDim.x + tidx;
    int gidy = bidy * blockDim.y + tidy;

    // if (gidx >= N || gidy >= M) return;

    __shared__ float tmpA[mItem][kItem];
    __shared__ float tmpB[kItem][nItem];
    // float tmpC[mItem][nItem] = {0};
    float tmpC = 0.;
    for (int k = 0; k < K; k += kItem) {
        __syncthreads();
        tmpA[tidy][tidx] = matA[gidy * lda + (k+tidx)];
        tmpB[tidy][tidx] = matB[(k+tidy)*ldb + gidx];
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < kItem; kk++) {
            tmpC += tmpA[tidy][kk] * tmpB[kk][tidx];
        }
    }
    
    matC[gidy * ldc + gidx] = tmpC;
}

int sgemm_tile_v2(
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
    int mItem = 32, nItem = 32, kItem = 32;
    dim3 blockSize(32, 32);
    dim3 gridSize(N/32, M/32);

    if (mItem==32 && nItem==32 && kItem==32) sgemm_tile_v2_smem_kernel<32,32,32><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);

    return 0;
}
