#include <cuda_runtime.h>
#include <iostream>

void get_cuda_info()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;

        // 查询每个SM的共享内存大小
        int sharedMemPerBlock;
        cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);

        std::cout << "Shared Memory per Block: " << sharedMemPerBlock / 1024.0f << " KB" << std::endl;

        // 打印每个SM的最大共享内存
        std::cout << "Shared Memory per SM: " << deviceProp.sharedMemPerMultiprocessor / 1024.0f << " KB" << std::endl;
        std::cout << std::endl;

        // 获取每个线程块的最大线程数
        int maxThreadsPerBlock;
        cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device);
        std::cout << "Max threads per block: " << maxThreadsPerBlock << std::endl;

        // 每个线程束的最大维度（X、Y、Z方向）
        std::cout << "Max threads dimensions (block): " 
                  << deviceProp.maxThreadsDim[0] << " x " 
                  << deviceProp.maxThreadsDim[1] << " x " 
                  << deviceProp.maxThreadsDim[2] << std::endl;

        // 每个网格的最大维度（X、Y、Z方向）
        std::cout << "Max grid size: " 
                  << deviceProp.maxGridSize[0] << " x " 
                  << deviceProp.maxGridSize[1] << " x " 
                  << deviceProp.maxGridSize[2] << std::endl;

        // 获取整个设备上的最大线程数
        std::cout << "Max threads per SM: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;

        std::cout << std::endl;
    }

}