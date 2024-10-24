#include <stdio.h>
#include <stdlib.h>

#include "xmath.h"

inline
__device__ void warp_reduce_srm(float *inData, int tid) 
{
    /* 从计算能力7开始，warp中的线程是有可能不同步的了，若步调不一致，则计算结果会出现错误 */
    //if (tid < 32) 
        inData[tid] += inData[tid+32];__syncwarp();
    //if (tid < 16) 
        inData[tid] += inData[tid+16];__syncwarp();
    //if (tid < 8)  
        inData[tid] += inData[tid+8]; __syncwarp();
    //if (tid < 4)  
        inData[tid] += inData[tid+4]; __syncwarp();
    //if (tid < 2)  
        inData[tid] += inData[tid+2]; __syncwarp();
    //if (tid < 1)  
        inData[tid] += inData[tid+1]; __syncwarp();
}

inline
__device__ float warp_reduce_shfl(float sum) 
{
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum,  8); 
    sum += __shfl_down_sync(0xffffffff, sum,  4); 
    sum += __shfl_down_sync(0xffffffff, sum,  2); 
    sum += __shfl_down_sync(0xffffffff, sum,  1); 
    return sum;
}

template<int BLOCK_SIZE, int WARP_SIZE>
__global__ void reduce_sharememory(float *inData, float *res) 
{
    int i   = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ float sdata[BLOCK_SIZE];
    sdata[tid] = inData[i] + inData[i+BLOCK_SIZE];
    __syncthreads();

    for (int s = BLOCK_SIZE/2; s >= 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }

    // if (tid < 32) warp_reduce_srm(sdata, tid);
    if (tid < 32) sdata[0] = warp_reduce_shfl(sdata[tid]); /*如果条件为tid < 16，则后16个线程的寄存器sum并没有值，shuffle直接出错*/

    if (tid == 0) {
        res[blockIdx.x] = sdata[0];
    }
}

template<int BLOCK_SIZE, int WARP_SIZE>
__global__ void reduce_shuffle(float *inData, float *res) 
{
    int i   = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    int tid = threadIdx.x;
    int lid = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    float sum = inData[i] + inData[i+BLOCK_SIZE];
    __syncthreads();

    sum = warp_reduce_shfl(sum);
    __syncthreads();

    __shared__ float sdata[BLOCK_SIZE / WARP_SIZE];
    if (lid == 0) {
        sdata[wid] = sum;
    }
    __syncthreads();

    if (wid == 0) {
        sum = warp_reduce_shfl(sdata[lid]);
    }

    if (tid == 0) {
        res[blockIdx.x] = sum;
    }
}

static int reduce_sum(float *inData, int len, float *res) 
{
    const int block_size     = 512;
    const int elem_per_block = block_size * 2;
    const int block_num      = ceil(len, elem_per_block);
    // printf("block_size:%d elem_per_block:%d block_num:%d\n", block_size, elem_per_block, block_num);

    float *resBufferDevice;
    cudaMalloc(&resBufferDevice, sizeof(float) * round_up(block_num, block_size * 2));
    cudaMemset(resBufferDevice, '\0', sizeof(float) * round_up(block_num, block_size * 2));

    // reduce_sharememory<block_size, 32><<<block_num, block_size>>>(inData, resBufferDevice);
    reduce_shuffle<block_size, 32><<<block_num, block_size>>>(inData, resBufferDevice);
    reduce_sharememory<block_size,32 ><<<1, block_size>>>(resBufferDevice, res);
    
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

    const int DSIZE = 5000;
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
