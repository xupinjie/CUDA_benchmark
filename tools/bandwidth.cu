#include <cuda_runtime.h>
#include <stdio.h>

// #define N (1024 * 1024 * 128)
#define N (6400 * 6400)

__global__ void bandwidthTest(float *d_a, float *d_b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 简单的内存拷贝操作，读取 d_a 的数据并写入 d_b
        d_b[idx] = d_a[idx];
    }
}

int main() {
    float *h_a, *h_b;          // 主机端内存指针
    float *d_a, *d_b;          // 设备端内存指针
    int size = N * sizeof(float);  // 数组大小（字节）

    // 分配主机端内存
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);

    // 初始化主机端数组
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
    }

    // 分配设备端内存
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);

    // 将主机端数据拷贝到设备端
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // 设置 CUDA 事件以测量时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 启动内核之前记录时间
    cudaEventRecord(start, 0);

    // 启动内核
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    bandwidthTest<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);

    // 停止时间记录
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 计算带宽 (字节/秒)
    float bandwidth = (2.0f * size) / (milliseconds * 1e6);  // 乘以2是因为有读和写

    printf("Global memory bandwidth = %f GB/s\n", bandwidth);

    // 清理内存
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
