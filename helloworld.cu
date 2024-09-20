#include <stdio.h>
#include <iostream>
//#include <cuda_runtime.h>

__global__ void foo()
{
    printf("CUDA!\n");
}

void useCUDA()
{
    
    foo<<<1,5>>>();
    cudaDeviceSynchronize();

}

int main()
{

    std::cout<<"Hello NVCC"<<std::endl;
    useCUDA();
    return 0;
}
