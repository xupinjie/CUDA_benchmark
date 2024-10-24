cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

project(CUDAProject LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "90")
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
include_directories(${CMAKE_SOURCE_DIR}/kernels)
include_directories(${CMAKE_SOURCE_DIR}/)

set(COMMON_KERNELS
    tools/cuda_info.cu
    kernels/sgemm_plain.cu
    kernels/sgemm_tile_v1.cu
    kernels/sgemm_tile_v2.cu
)

# bandwidth kernel
set(SOURCES_BANDWIDTH
    tools/bandwidth.cu
    ${COMMON_KERNELS}
)
add_executable(bandwidth ${SOURCES_BANDWIDTH})
target_link_libraries(bandwidth ${CUDA_LIBRARIES})
set_target_properties(bandwidth PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# reduce kernel
set(SOURCES_REDUCE
    kernels/reduce_main.cu
    ${COMMON_KERNELS}
)
add_executable(reduce ${SOURCES_REDUCE})
target_link_libraries(reduce ${CUDA_LIBRARIES})
set_target_properties(reduce PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# gemm kernel
set(SOURCES_GEMM
    kernels/sgemm_main.cu
    ${COMMON_KERNELS}
)
add_executable(gemm ${SOURCES_GEMM})
target_link_libraries(gemm ${CUDA_LIBRARIES})
set_target_properties(gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
