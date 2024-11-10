#include <cuda_runtime.h>
#include <cstdio>

#include<xmath.h>
#include<sgemm.h>

//!!!
template<int BM, int BN, int BK>
__global__ void sgemm_tile_v2_smem_kernel1(
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
    const int gidx = bidx * BN + tidx;
    const int gidy = bidy * BM + tidy;

    // if (gidx >= N || gidy >= M) return;

    __shared__ float smemA[BM][BK];
    __shared__ float smemB[BK][BN];
    float tmpC = 0.;
    for (int k = 0; k < K; k += BK) {
        smemA[tidy][tidx] = matA[gidy*lda + (k+tidx)];
        #pragma unroll
        for (int i = 0; i < BK; i+=BM) {
            smemB[i+tidy][tidx] = matB[(k+i+tidy)*ldb + gidx];
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            tmpC += smemA[tidy][kk] * smemB[kk][tidx];
        }
        __syncthreads();
    }
    
    matC[gidy * ldc + gidx] = tmpC;
}

//!!!
template<int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_tile_v2_smem_kernel2(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    const float * __restrict__ matA,
    const float * __restrict__ matB,
    float * __restrict__ matC)
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    // if (gidx >= N || gidy >= M) return;

    __shared__ float smemA[BM*TM][BK];
    __shared__ float smemB[BK][BN*TN];
    float tmpC[TM][TN] = {0.};
    for (int k = 0; k < K; k += BK) {
        for (int i = 0; i < TM; i++) { //TM个BM
            smemA[i*BM+tidy][tidx] = matA[(bidy*BM*TM+i*BM+tidy)*lda + (k+tidx)];
        }
        #pragma unroll
        for (int i = 0; i < BK; i+=BM) { //重复BK/BM次
            for (int j = 0; j < TN; j++) { //TN个BN
                smemB[i+tidy][j*BN+tidx] = matB[(k+i+tidy)*ldb + (bidx*BN*TN+j*BN+tidx)];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            for (int i = 0; i < TM; i++) { //TM个BM
                for (int j = 0; j < TN; j++) {
                    tmpC[i][j] += smemA[i*BM+tidy][kk] * smemB[kk][j*BN+tidx];
                }
            }
        }
        __syncthreads();
    }
    
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            matC[(bidy*BM*TM+i*BM+tidy) * ldc + (bidx*BN*TN+j*BN+tidx)] = tmpC[i][j];
        }
    }
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
    #define WARP_SIZE 32
    #define BM 8
    #define BN WARP_SIZE
    #define BK WARP_SIZE
    #define TM 8
    #define TN 8

    // #define WARP_SIZE 32
    // #define BM 8
    // #define BN WARP_SIZE
    // #define BK WARP_SIZE
    // #define TM 8
    // #define TN 8

    dim3 blockSize(BN, BM);

    // dim3 gridSize(N/BN, M/BM);
    // sgemm_tile_v2_smem_kernel1<BM, BN, BK><<<gridSize, blockSize, 0, 0>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);

    dim3 gridSize(N/BN/TN, M/BM/TM);
    sgemm_tile_v2_smem_kernel2<BM, BN, BK, TM, TN><<<gridSize, blockSize, 0, 0>>>(M, N, K, lda, ldb, ldc, matA, matB, matC);

    #undef WARP_SIZE
    #undef BM
    #undef BN
    #undef BK
    #undef TM
    #undef TN

    return 0;
}
