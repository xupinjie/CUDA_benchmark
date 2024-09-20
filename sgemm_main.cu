#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "xmath.h"
#include "fp_diff.h"
#include "sgemm.h"

#define GEMM_KERNEL sgemm_tile_v2

static void sgemm_host(
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
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.;
            for (int k = 0; k < K; k++) {
                sum += matA[i*lda+k] * matB[k*ldb+j];
            }
            matC[i*ldc+j] = sum;
        }
    }
}

int sgemm_main()
{
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    printf("M:%d N:%d K:%d\n", M, N, K);

    float *matA = (float *)malloc(M * lda * sizeof(float));
    float *matB = (float *)malloc(K * ldb * sizeof(float));
    float *matC = (float *)malloc(M * ldc * sizeof(float));
    float *refC = (float *)malloc(M * ldc * sizeof(float));

    dataRandom2D(matA, M, K, lda);
    dataRandom2D(matB, K, N, ldb);
    valueSet2D(matC, M, N, ldc, -1.);
    valueSet2D(refC, M, N, ldc, -1.);

    sgemm_host(M, N, K, lda, ldb, ldc, matA, matB, refC);

    float *matA_device, *matB_device, *matC_device;
    cudaMalloc(&matA_device, M * lda * sizeof(float));
    cudaMalloc(&matB_device, K * ldb * sizeof(float));
    cudaMalloc(&matC_device, M * ldc * sizeof(float));
    cudaMemcpy(matA_device, matA, M * lda * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matB_device, matB, K * ldb * sizeof(float), cudaMemcpyHostToDevice);


    // check res
    GEMM_KERNEL(M, N, K, lda, ldb, ldc, matA_device, matB_device, matC_device);
    cudaMemcpy(matC, matC_device, M * ldc * sizeof(float), cudaMemcpyDeviceToHost);
    fp_diff(matC, refC, M * N, 1e-6, 1e-6, 0.999);
    // int errno = 0;
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         if (errno>10) break;

    //         float res = matC[i*ldc+j];
    //         float ref = refC[i*ldc+j];
    //         if (abs(res-ref) > 1e-4) {
    //             printf("[%d,%d] %f vs. %f\n", i, j, res, ref);
    //             errno++;
    //         }
    //     }
    // }

    // profiling
    const int warmup = 5;
    const int trials = 5;

    for (int i = 0; i < warmup; i++) {
        GEMM_KERNEL(M, N, K, lda, ldb, ldc, matA_device, matB_device, matC_device);
    }

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < trials; i++) {
        GEMM_KERNEL(M, N, K, lda, ldb, ldc, matA_device, matB_device, matC_device);
    }
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    elapsedTime /= trials;

    double FLOPs = 2. * M * N * K;
    printf("cudaEventElapsedTime: %fms %fgflops\n", elapsedTime, FLOPs / elapsedTime * 1e-6);

    free(matA);
    free(matB);
    free(matC);
    free(refC);
    cudaFree(matA_device);
    cudaFree(matB_device);
    cudaFree(matC_device);

    return 0;
}
