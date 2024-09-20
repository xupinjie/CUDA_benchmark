cmake_minimum_required(VERSION 3.10)

# 设置策略 CMP0104
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()
# 设置 CUDA 架构（根据你的 GPU 进行设置）
set(CMAKE_CUDA_ARCHITECTURES "61"  # 替换为适合你的架构
)

project(CUDAProject LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# 启用CUDA
# find_package(CUDA REQUIRED)

# 添加CUDA和C++文件
set(SOURCES
    main.cpp
    tools/cuda_info.cu
    sgemm_main.cu
    sgemm_plain.cu
    sgemm_tile_v1.cu
    sgemm_tile_v2.cu
    reduce_main.cu
)

# 设置编译标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)



#添加头文件
include_directories(${CMAKE_SOURCE_DIR})

# 添加可执行文件
add_executable(test_cuda_kernel ${SOURCES})
# 链接CUDA库（如果需要）
target_link_libraries(test_cuda_kernel ${CUDA_LIBRARIES})
# 指定CUDA的编译选项
set_target_properties(test_cuda_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
