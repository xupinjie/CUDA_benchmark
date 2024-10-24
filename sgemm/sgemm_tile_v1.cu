#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>

#include<xmath.h>
#include<sgemm.h>

template<int mItem, int nItem, int kItem>
__global__
void sgemm_tile_v1_kernel(
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
    // if(gidy==0)printf("%d %d %d %d %d %d\n", tidx, tidy, bidx, bidy, gidx, gidy);
    // if(gidy==0 && gidx ==0)printf("%d %d\n", gridDim.x, gridDim.y);

    float tmpC[mItem][nItem] = {0};
    for (int k = 0; k < K; k += kItem) {
        
        float tmpA[mItem][kItem];
        float tmpB[kItem][nItem];

        #pragma unroll
        for (int mm = 0; mm < mItem; mm++) {
            #pragma unroll
            for (int kk = 0; kk < kItem; kk++) {
                tmpA[mm][kk] = matA[(gidy * mItem + mm) * lda + (k+kk)];
            }
        }
        #pragma unroll
        for (int kk = 0; kk < kItem; kk++) {
            #pragma unroll
            for (int nn = 0; nn < nItem; nn++) {
                tmpB[kk][nn] = matB[(k+kk)*ldb + (gidx*nItem+nn)];
            }
        }

        #pragma unroll
        for (int kk = 0; kk < kItem; kk++) {
            #pragma unroll
            for (int mm = 0; mm < mItem; mm++) {
                #pragma unroll
                for (int nn = 0; nn < nItem; nn++) {
                    tmpC[mm][nn] += tmpA[mm][kk] * tmpB[kk][nn];
                }
            }
        }
    }
    
    #pragma unroll
    for (int mm = 0; mm < mItem; mm++) {
        #pragma unroll
        for (int nn = 0; nn < nItem; nn++) {
            matC[(gidy*mItem+mm) * ldc + (gidx*nItem+nn)] = tmpC[mm][nn];
        }
    }
}

int sgemm_tile_v1(
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
    const int mItem = 16, nItem = 4, kItem = 8;
    const int ls0 = 16, ls1 = 16; 
    dim3 blockSize(ls0, ls1);
    dim3 gridSize(N/nItem/ls0, M/mItem/ls1);

    cudaProfilerStart();
    if (mItem==4 && nItem==4 && kItem==4) sgemm_tile_v1_kernel<4,4,4><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);
    if (mItem==4 && nItem==4 && kItem==8) sgemm_tile_v1_kernel<4,4,8><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);
    if (mItem==4 && nItem==8 && kItem==4) sgemm_tile_v1_kernel<4,8,4><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);
    if (mItem==8 && nItem==4 && kItem==4) sgemm_tile_v1_kernel<8,4,4><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);

    if (mItem==4 && nItem==8 && kItem==8) sgemm_tile_v1_kernel<4,8,8><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);
    if (mItem==8 && nItem==4 && kItem==8) sgemm_tile_v1_kernel<8,4,8><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);
    if (mItem==8 && nItem==8 && kItem==4) sgemm_tile_v1_kernel<8,8,4><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);

    if (mItem==8 && nItem==8 && kItem==8) sgemm_tile_v1_kernel<8,8,8><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);

    if (mItem==8 && nItem==4 && kItem==16) sgemm_tile_v1_kernel<8,4,16><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);
    if (mItem==16 && nItem==4 && kItem==8) sgemm_tile_v1_kernel<16,4,8><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);
    if (mItem==32 && nItem==4 && kItem==8) sgemm_tile_v1_kernel<32,4,8><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);

    cudaProfilerStop();

    return 0;
}
