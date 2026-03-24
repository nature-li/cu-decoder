cmake_minimum_required(VERSION 3.18) # 建议 3.18+ 以获得更好的 CUDA 支持
project(MiniGptDecoder CUDA)        # 声明项目语言包含 CUDA

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 告诉编译器针对你的显卡架构进行优化
# 例如：RTX 30系列是 86，RTX 40系列是 89
# 如果不确定，可以设为 75 (T4/RTX 20) 或 80 (A100)
set(CMAKE_CUDA_ARCHITECTURES 86)

# 添加可执行文件
add_executable(mini_gpt_basic_attention mini_gpt_basic_attention.cu)

# 如果你用到了数学库 (如 cublas)，需要链接它们
# target_link_libraries(mini_gpt_basic_attention PRIVATE cublas)

# 指定输出位置（可选）
set_target_properties(mini_gpt_basic_attention PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")