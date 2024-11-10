#include <stdio.h>
#include <stdlib.h>

#include "xmath.h"

template<typename T, int BLOCK_SIZE, int WARP_SIZE>
__global__ void reduce_shuffle(
    const T* __restrict__ src,
    int len,
    T* __restrict__ dst)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lid = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    const int warp_num = BLOCK_SIZE / WARP_SIZE;
    
    const int idx = bid * BLOCK_SIZE + tid;
    T sum = idx < len ? src[idx] : (T)0;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    __shared__ T smem[warp_num];
    if (lid==0) smem[wid] = sum;
    __syncthreads();

    sum = (T)0;
    if (tid < warp_num) {
        sum = smem[tid];
        if (warp_num >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
        if (warp_num >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
        if (warp_num >= 8)  sum += __shfl_down_sync(0xffffffff, sum, 4);
        if (warp_num >= 4)  sum += __shfl_down_sync(0xffffffff, sum, 2);
        if (warp_num >= 2)  sum += __shfl_down_sync(0xffffffff, sum, 1);
    }
    
    if (tid == 0) dst[bid] = sum;
}

static int reduce_sum(float *inData, int len, float *res) 
{
    const int block_size     = 512;
    const int elem_per_block = block_size * 1;
    const int block_num      = ceil(len, elem_per_block);
    const int warp_size      = 32;

    // printf("block_size:%d elem_per_block:%d block_num:%d\n", block_size, elem_per_block, block_num);

    float *resBufferDevice;
    cudaMalloc(&resBufferDevice, sizeof(float) * block_num);
    cudaMemset(resBufferDevice, '\0', sizeof(float) * block_num);

    reduce_shuffle<float ,block_size, warp_size><<<block_num, block_size>>>(inData, len, resBufferDevice);
    reduce_shuffle<float, block_size, warp_size><<<1, block_size>>>(resBufferDevice, block_num, res);
    
    cudaDeviceSynchronize();

    cudaFree(resBufferDevice);

    return 0;
}

/////////////////////////////////////////////////////////////
static void init_data(float *inData, int DSIZE)
{
    for (int i = 0; i < DSIZE; i++) {
        // inData[i] = (float)1;
        inData[i] = (float)i;
    }
}

static void reduce_sum_host(float *inData, int DSIZE, float &res)
{
    res = 0;
    for (int i = 0; i < DSIZE; i++) {
        res += inData[i];
    }
}

int main()
{
    const int warm   = 5;
    const int trials = 10;

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int DSIZE = 51200;
    float *inData = (float *)malloc(sizeof(float) * DSIZE);

    /* init inData */
    init_data(inData, DSIZE);
    
    /* get ref */
    float ref = 0.;
    reduce_sum_host(inData, DSIZE, ref);

    float *inDataDevice;
    cudaMalloc(&inDataDevice, sizeof(float) * DSIZE);
    cudaMemcpy(inDataDevice, inData, sizeof(float)*DSIZE, cudaMemcpyHostToDevice);

    float *res = (float *)malloc(1*sizeof(float));
    float *resDevice;
    cudaMalloc(&resDevice, 1*sizeof(float));

    for (int i = 0; i < warm; i++) {
        reduce_sum(inDataDevice, DSIZE, resDevice);
    }

    cudaEventRecord(start, 0);
    for (int i = 0; i < trials; i++) {
        reduce_sum(inDataDevice, DSIZE, resDevice);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("cudaEventElapsedTime: %f\n", elapsedTime/trials);

    cudaMemcpy(res, resDevice, 1*sizeof(float), cudaMemcpyDeviceToHost);

    printf("ref:%f res:%f\n", ref, *res);

    free(inData);
    free(res);
    cudaFree(inDataDevice);
    cudaFree(resDevice);

    return 0;
}
