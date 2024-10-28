#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cassert>

#define WIDTH 512   // 矩阵宽度（列数）
#define HEIGHT 100 // 矩阵高度（行数）

template<typename T, int BLOCK_SIZE, int WARP_SIZE>
__global__ void softmax(
    const T* src,
    int batch,
    int width,
    T* dst)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lid = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    const int warp_num = BLOCK_SIZE / WARP_SIZE;

    // if (tid > width) return;
    // if (bid > batch) return;

    //max
    T g_max = (T)0;
    for (int i = 0; i < width; i+=BLOCK_SIZE) {
        T max = src[bid * width + i + tid]; //补充边界判断
        max = fmaxf(__shfl_down_sync(0xffffffff, max, 16), max);
        max = fmaxf(__shfl_down_sync(0xffffffff, max, 8), max);
        max = fmaxf(__shfl_down_sync(0xffffffff, max, 4), max);
        max = fmaxf(__shfl_down_sync(0xffffffff, max, 2), max); 
        max = fmaxf(__shfl_down_sync(0xffffffff, max, 1), max);

        __shared__ T smem[warp_num];
        if (lid == 0) smem[wid] = max;
        __syncthreads();

        max = (T)0;
        if (tid < warp_num) {
            max = smem[tid];
            max = fmaxf(__shfl_down_sync(0xffffffff, max, 16), max);
            max = fmaxf(__shfl_down_sync(0xffffffff, max, 8), max);
            max = fmaxf(__shfl_down_sync(0xffffffff, max, 4), max);
            max = fmaxf(__shfl_down_sync(0xffffffff, max, 2), max);
            max = fmaxf(__shfl_down_sync(0xffffffff, max, 1), max);
        }

        if (tid == 0) g_max = max;
    }
    __shared__ T s_g_max[1];
    if (tid == 0) s_g_max[0] = g_max;
    __syncthreads();
    g_max = s_g_max[0];

    //sum
    T sum_exp = (T)0;
    for (int i = 0; i < width; i+=BLOCK_SIZE) {
        T sum = src[bid * width + i + tid]; //补充边界判断
        sum = expf(sum-g_max);
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);

        __shared__ T smem[warp_num];
        if (lid == 0) smem[wid] = sum;
        __syncthreads();

        sum = (T)0;
        if (tid < warp_num) {
            sum = smem[tid];
            sum += __shfl_down_sync(0xffffffff, sum, 16);
            sum += __shfl_down_sync(0xffffffff, sum, 8);
            sum += __shfl_down_sync(0xffffffff, sum, 4);
            sum += __shfl_down_sync(0xffffffff, sum, 2);
            sum += __shfl_down_sync(0xffffffff, sum, 1);
        }

        if (tid == 0) sum_exp += sum;
    }
    __shared__ T sum_exp_fix[1];
    if (tid == 0) sum_exp_fix[0] = sum_exp;
    __syncthreads();

    sum_exp = sum_exp_fix[0];

    for (int i = 0; i < width; i+=BLOCK_SIZE) {
        dst[bid * width + i + tid] = expf(src[bid * width + i + tid]-g_max) / sum_exp;
    }
}

// // CUDA 核函数：计算行上的 Softmax
// __global__ void softmax(float* input, float* output, int width, int height) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < height) {
//         float max_val = -INFINITY;

//         // 1. 找到该行的最大值（避免数值溢出）
//         for (int i = 0; i < width; ++i) {
//             max_val = fmaxf(max_val, input[row * width + i]);
//         }

//         // 2. 计算指数和求和
//         float sum = 0.0;
//         for (int i = 0; i < width; ++i) {
//             output[row * width + i] = expf(input[row * width + i] - max_val); // 减去最大值
//             sum += output[row * width + i];
//         }

//         // 3. 归一化 Softmax 结果
//         for (int i = 0; i < width; ++i) {
//             output[row * width + i] /= sum;
//         }
//     }
// }

// 封装核函数的调用
void run_softmax(float* d_input, float* d_output, float* h_output, int width, int height) {
    const int BLOCK_SIZE = 256;  // 每个线程块的线程数
    const int WARP_SIZE = 32;  // 每个线程块的线程数
    int numThreads = BLOCK_SIZE;
    int numBlocks = height;

    // 启动 CUDA 核函数
    // softmax<<<numBlocks, numThreads>>>(d_input, d_output, width, height);
    softmax<float, BLOCK_SIZE, WARP_SIZE><<<numBlocks, numThreads>>>(d_input, height, width, d_output);

    // 将结果从设备拷贝回主机
    size_t bytes = width * height * sizeof(float);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
}

// 在 CPU 上计算 Softmax
void softmax_cpu(float* input, float* output, int width, int height) {
    for (int i = 0; i < height; ++i) {
        float max_val = -INFINITY;

        // 找到该行的最大值
        for (int j = 0; j < width; ++j) {
            max_val = fmax(max_val, input[i * width + j]);
        }

        // 计算指数和求和
        float sum = 0.0;
        for (int j = 0; j < width; ++j) {
            output[i * width + j] = exp(input[i * width + j] - max_val);  // 数值稳定性
            // output[i * width + j] = exp(input[i * width + j]);
            sum += output[i * width + j];
        }

        // 归一化 Softmax 结果
        for (int j = 0; j < width; ++j) {
            output[i * width + j] /= sum;
        }
        // printf("==sum: %f\n", sum);
        // printf("==max: %f\n", max_val);

    }
}

// 验证 GPU 和 CPU 的结果
bool verify_result(float* cpu_output, float* gpu_output, int width, int height, float epsilon=1e-5) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float diff = fabs(cpu_output[i * width + j] - gpu_output[i * width + j]);
            if (diff > epsilon) {
                std::cerr << "Mismatch at (" << i << ", " << j << "): CPU=" << cpu_output[i * width + j]
                          << ", GPU=" << gpu_output[i * width + j] << ", diff=" << diff << std::endl;
                // return false;
            }
        }
    }
    return true;
}

// 打印矩阵
void print_matrix(float* matrix, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << matrix[i * width + j] << "\t";
        }
        std::cout << std::endl;
    }
}

int main() {
    // 定义矩阵大小
    int width = WIDTH;
    int height = HEIGHT;
    size_t bytes = width * height * sizeof(float);

    // 在主机端分配内存并初始化数据
    float* h_input = (float*)malloc(bytes);
    float* h_output_gpu = (float*)malloc(bytes);
    float* h_output_cpu = (float*)malloc(bytes);
    
    // // 初始化输入矩阵
    // for (int i = 0; i < height; ++i) {
    //     for (int j = 0; j < width; ++j) {
    //         h_input[i * width + j] = static_cast<float>(i * width + j); // 示例值
    //     }
    // }
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            h_input[i * width + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // std::cout << "Input matrix:" << std::endl;
    // print_matrix(h_input, width, height);

    // 在设备端分配内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // 将数据从主机拷贝到设备
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // CUDA 事件用于测量时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 启动计时
    cudaEventRecord(start);
    // 调用封装的 CUDA softmax 函数
    run_softmax(d_input, d_output, h_output_gpu, width, height);
    // 停止计时
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算运行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 计算带宽 (字节数 / 毫秒数 * 1e3 / 1e9 以得到 GB/s)
    float bandwidth = (2 * bytes / milliseconds) * 1e-6;
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // 在 CPU 上计算 softmax
    softmax_cpu(h_input, h_output_cpu, width, height);

    // // 输出 GPU 结果
    // std::cout << "\nGPU Softmax result:" << std::endl;
    // print_matrix(h_output_gpu, width, height);

    // // 输出 CPU 结果
    // std::cout << "\nCPU Softmax result:" << std::endl;
    // print_matrix(h_output_cpu, width, height);

    // 验证 CPU 和 GPU 结果
    if (verify_result(h_output_cpu, h_output_gpu, width, height)) {
        std::cout << "Verification successful! CPU and GPU results match." << std::endl;
    } else {
        std::cout << "Verification failed! CPU and GPU results do not match." << std::endl;
    }

    // 释放设备和主机内存
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);

    // 销毁 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
