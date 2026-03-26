cmake_minimum_required(VERSION 3.18) # 建议 3.18+ 以获得更好的 CUDA 支持
project(MiniDecoder CXX CUDA)        # 声明项目语言包含 CUDA

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 告诉编译器针对你的显卡架构进行优化
set(CMAKE_CUDA_ARCHITECTURES 100)

find_package(CUDAToolkit REQUIRED)

macro(add_cuda_target target)
    # 添加可执行文件
    add_executable(${target} ${ARGN})

    # 链接库
    target_link_libraries(${target} PRIVATE CUDA::cudart cublas)

    # 统一输出路径到 build 根目录
    set_target_properties(${target} PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    message(STATUS "Successfully added CUDA target: ${target}")
endmacro()

# add_cuda_target(mini_gpt_basic_attention mini_gpt_basic_attention.cu)
# add_cuda_target(mini_gpt_decoder mini_gpt_decoder.cu)
add_executable(cpu_decoder cpu_decoder.cpp decoder.cpp common.cpp)
add_cuda_target(gpu_decoder gpu_decoder.cu decoder.cpp common.cpp)