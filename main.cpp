#include <iostream>

#include "kernels/cuda_kernel.h"

int main()
{
    get_cuda_info();
    // reduce_main();
    sgemm_main();

    return 0;
}