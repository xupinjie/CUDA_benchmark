#include <cuda_runtime.h>
#include <iostream>

#define HEIGHT 6400  // 矩阵高度
#define WIDTH 6400   // 矩阵宽度

template<typename T, int BLOCK_SIZE>
__global__ void transpose(
    const T *src,
    int height,
    int width,
    T *dst)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x >= width || y >= height) return;

    __shared__ T smem[BLOCK_SIZE][BLOCK_SIZE+1];

    smem[tidy][tidx] = src[y * width + x];
    __syncthreads();
    
    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    dst[y * height + x] = smem[tidx][tidy];
}

// template<typename T, int BLOCK_SIZE>
// __global__ void transpose(float* input, int height, int width, float* output) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x < width && y < height) {
//         output[x * height + y] = input[y * width + x];
//     }
// }

// 封装核函数调用、网格设置和内存传输的函数，并计算带宽
void run_transpose(float* d_input, float* d_output, float* h_output, int width, int height) {
    // 设置线程块和网格维度
    const int BLOCK_SIZE = 32;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 启动CUDA核函数
    transpose<float, BLOCK_SIZE><<<grid, block>>>(d_input, height, width, d_output);
}

// 验证转置结果是否正确
void verify_result(float* original, float* transposed, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (original[i * width + j] != transposed[j * height + i]) {
                std::cerr << "Error: Element (" << i << ", " << j << ") is incorrect!" << std::endl;
                return;
            }
        }
    }
    std::cout << "Transpose successful!" << std::endl;
}

int main() {
    // 矩阵大小
    int width = WIDTH;
    int height = HEIGHT;
    size_t bytes = width * height * sizeof(float);

    // 在主机端分配内存
    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);

    // 初始化输入矩阵
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            h_input[i * width + j] = i * width + j;
        }
    }

    // 在设备端分配内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // 将数据从主机拷贝到设备
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // 创建CUDA事件，计算时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start);

    run_transpose(d_input, d_output, h_output, width, height);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算核函数执行时间（毫秒）
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 将结果从设备拷贝回主机
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // 清理事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 计算内存带宽：数据传输量 (读 + 写)，单位是字节
    float dataSizeGB = 2.0f * bytes / (1024.0f * 1024.0f * 1024.0f); // 转换为 GB
    float bandwidth = dataSizeGB / (milliseconds / 1000.0f);          // GB/s


    // 验证转置结果是否正确
    verify_result(h_input, h_output, width, height);

    // 输出计算带宽
    std::cout << "Memory Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // 释放设备和主机内存
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
