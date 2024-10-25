cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

project(CUDAProject LANGUAGES CXX CUDA)

# set(CMAKE_CUDA_ARCHITECTURES "90")
set(CMAKE_CUDA_ARCHITECTURES "61")
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# common
include_directories(${CMAKE_SOURCE_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/common)
set(COMMON_KERNELS
    tools/cuda_info.cu
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
include_directories(${CMAKE_SOURCE_DIR}/reduce)
set(SOURCES_REDUCE
    # reduce/reduce_main.cu
    reduce/reduce_main_v2.cu
    ${COMMON_KERNELS}
)
add_executable(reduce ${SOURCES_REDUCE})
target_link_libraries(reduce ${CUDA_LIBRARIES})
set_target_properties(reduce PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# sgemm kernel
include_directories(${CMAKE_SOURCE_DIR}/sgemm)
set(SOURCES_GEMM
    sgemm/sgemm_main.cu
    sgemm/sgemm_plain.cu
    sgemm/sgemm_tile_v1.cu
    sgemm/sgemm_tile_v2.cu
    ${COMMON_KERNELS}
)
add_executable(sgemm ${SOURCES_GEMM})
target_link_libraries(sgemm ${CUDA_LIBRARIES})
set_target_properties(sgemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# transpose kernel
include_directories(${CMAKE_SOURCE_DIR}/transpose)
set(SOURCES_TRANSPOSE
    transpose/transpose_main.cu
    ${COMMON_KERNELS}
)
add_executable(transpose ${SOURCES_TRANSPOSE})
target_link_libraries(transpose ${CUDA_LIBRARIES})
set_target_properties(transpose PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
