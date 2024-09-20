#include <cuda_runtime.h>
#include <iostream>

__global__ void memoryAccessTest(float* read_data, float* write_data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 保证线程索引不超过数组边界
    if (idx < N) {
        // 从 read_data 读取，写入 write_data
        float temp = read_data[idx];  // 从read_data读取
        write_data[idx] = temp * 2.0f;  // 写入write_data
    }
}

int main() {
    // 假设我们有1024个float
    int N = 32;
    size_t size = N * sizeof(float);

    // 分配主机和设备内存
    float* h_read_data = (float*)malloc(size);
    float* h_write_data = (float*)malloc(size);
    float* d_read_data;
    float* d_write_data;

    cudaMalloc(&d_read_data, size);
    cudaMalloc(&d_write_data, size);

    // 初始化主机读取数据
    for (int i = 0; i < N; ++i) {
        h_read_data[i] = static_cast<float>(i);
    }

    // 将读取数据复制到设备
    cudaMemcpy(d_read_data, h_read_data, size, cudaMemcpyHostToDevice);

    // 启动kernel，32线程每个block，32个block，共1024线程
    int threadsPerBlock = 32;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    memoryAccessTest<<<blocks, threadsPerBlock>>>(d_read_data, d_write_data, N);

    // 复制结果回主机
    cudaMemcpy(h_write_data, d_write_data, size, cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 32; ++i) {
        std::cout << "h_write_data[" << i << "] = " << h_write_data[i] << std::endl;
    }

    // 释放内存
    free(h_read_data);
    free(h_write_data);
    cudaFree(d_read_data);
    cudaFree(d_write_data);

    return 0;
}
