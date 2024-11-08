#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>

#include<xmath.h>
#include<sgemm.h>

template<int TM, int TN, int KT>
__global__
void sgemm_tile_v1_kernel1(
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

    float tmpC[TM][TN] = {0};
    for (int k = 0; k < K; k += KT) {
        
        float tmpA[TM][KT];
        float tmpB[KT][TN];

        #pragma unroll
        for (int mm = 0; mm < TM; mm++) {
            #pragma unroll
            for (int kk = 0; kk < KT; kk++) {
                tmpA[mm][kk] = matA[(gidy * TM + mm) * lda + (k+kk)];
            }
        }
        #pragma unroll
        for (int kk = 0; kk < KT; kk++) {
            #pragma unroll
            for (int nn = 0; nn < TN; nn++) {
                tmpB[kk][nn] = matB[(k+kk)*ldb + (gidx*TN+nn)];
            }
        }

        #pragma unroll
        for (int kk = 0; kk < KT; kk++) {
            #pragma unroll
            for (int mm = 0; mm < TM; mm++) {
                #pragma unroll
                for (int nn = 0; nn < TN; nn++) {
                    tmpC[mm][nn] += tmpA[mm][kk] * tmpB[kk][nn];
                }
            }
        }
    }
    
    #pragma unroll
    for (int mm = 0; mm < TM; mm++) {
        #pragma unroll
        for (int nn = 0; nn < TN; nn++) {
            matC[(gidy*TM+mm) * ldc + (gidx*TN+nn)] = tmpC[mm][nn];
        }
    }
}

template<int TM, int TN>
__global__
void sgemm_tile_v1_kernel2(
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

    float tmpC[TM][TN] = {0};
    float tmpA[TM];
    float tmpB[TN];
    for (int k = 0; k < K; k++) {
        #pragma unroll
        for (int mm = 0; mm < TM; mm++) {
            tmpA[mm] = matA[(gidy * TM + mm) * lda + k];
        }
        #pragma unroll
        for (int nn = 0; nn < TN; nn++) {
            tmpB[nn] = matB[k*ldb + (gidx*TN+nn)];
        }

        #pragma unroll
        for (int mm = 0; mm < TM; mm++) {
            #pragma unroll
            for (int nn = 0; nn < TN; nn++) {
                tmpC[mm][nn] += tmpA[mm] * tmpB[nn];
            }
        }
    }
    
    #pragma unroll
    for (int mm = 0; mm < TM; mm++) {
        #pragma unroll
        for (int nn = 0; nn < TN; nn++) {
            matC[(gidy*TM+mm) * ldc + (gidx*TN+nn)] = tmpC[mm][nn];
        }
    }
}

template<int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int TM, int TN>
__global__
void sgemm_tile_v1_kernel3(
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
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const int gidx = bidx * BLOCK_SIZE_X + tidx;
    const int gidy = bidy * BLOCK_SIZE_Y + tidy;

    float tmpC[TM][TN] = {0};
    float tmpA[TM];
    float tmpB[TN];
    for (int k = 0; k < K; k++) {
        #pragma unroll
        for (int mm = 0; mm < TM; mm++) {
            tmpA[mm] = matA[(gidy * TM + mm) * lda + k];
        }
        #pragma unroll
        for (int nn = 0; nn < TN; nn++) {
            tmpB[nn] = matB[k*ldb + (bidx*BLOCK_SIZE_X*TN + BLOCK_SIZE_X*nn + tidx)];
        }

        #pragma unroll
        for (int mm = 0; mm < TM; mm++) {
            #pragma unroll
            for (int nn = 0; nn < TN; nn++) {
                tmpC[mm][nn] += tmpA[mm] * tmpB[nn];
            }
        }
    }
    
    #pragma unroll
    for (int mm = 0; mm < TM; mm++) {
        #pragma unroll
        for (int nn = 0; nn < TN; nn++) {
            matC[(gidy*TM+mm) * ldc + (bidx*BLOCK_SIZE_X*TN + BLOCK_SIZE_X*nn + tidx)] = tmpC[mm][nn];
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
    #define TM 8
    #define TN 8
    #define TK 16
    #define BLOCK_SIZE_X 32
    #define BLOCK_SIZE_Y 16
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize(N/TN/BLOCK_SIZE_X, M/TM/BLOCK_SIZE_Y);

    cudaProfilerStart();

    // sgemm_tile_v1_kernel1<TM, TN, TK><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);
    // sgemm_tile_v1_kernel2<TM, TN><<<gridSize, blockSize>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);
    sgemm_tile_v1_kernel3<BLOCK_SIZE_X, BLOCK_SIZE_Y, TM, TN><<<gridSize, blockSize, 0, 0>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);

    cudaProfilerStop();

    #undef TM
    #undef TN
    #undef TK
    #undef BLOCK_SIZE_X
    #undef BLOCK_SIZE_Y
    return 0;
}
