#include <cuda_runtime.h>
#include <cstdio>

#include<xmath.h>
#include<sgemm.h>

template<int MTILE, int NTILE, int KTILE, int mItem, int nItem, int kItem>
__global__
void sgemm_tile_v3_smem_kernel(
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

    int M_begin = gidy * mItem;
    int N_begin = gidx * nItem;

    
    __shared__ float s_tmpA[MTILE][KTILE];
    __shared__ float s_tmpB[KTILE][NTILE];
    __shared__ float s_tmpC[MTILE][NTILE];

    for (int kt = 0; kt < K; kt += KTILE) {
        
    }
    
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

int sgemm_tile_smem_v3(
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

    if (mItem==32 && nItem==32 && kItem==32) sgemm_tile_v3_smem_kernel<32,32,32><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);

    return 0;
}
