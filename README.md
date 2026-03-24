# Mini-GPT CUDA Decoder

这是一个基于 **NVIDIA CUDA** 原生实现的极简 GPT 解码推理引擎。本项目不依赖任何深度学习框架（如 PyTorch 或 TensorFlow），直接在 GPU 硬件层面上手写 Transformer 核心算子，实现了从词向量嵌入到自回归字符生成的完整链路。

## 目录结构说明

| 文件名 | 用途描述 |
| :--- | :--- |
| **`mini_gpt_decoder.cu`** | **核心程序**。包含完整的 12 层 Transformer Decoder 逻辑、KV Cache 管理及自回归生成循环。 |
| **`mini_gpt_basic_attention.cu`** | 实验性质的单层 Attention 实现，用于前期验证算子正确性。 |
| **`comm.h`** | 公共头文件，包含 `GPTConfig` 定义、CUDA 错误检查宏（`CHECK_CUDA`）及工具函数。 |
| **`CMakeLists.txt`** | 项目构建配置文件，支持快速编译。 |

## 核心特性

* **原生 CUDA 核函数**：手写 `embedding`, `rms_norm`, `attention`, `ffn`, `residual` 等核心算子。
* **KV Cache 优化**：实现了键值对缓存，避免了自回归生成过程中重复计算历史 Token，显著提升推理速度。
* **自回归生成逻辑**：模型能将自身输出作为下一步输入，实现连续文本生成。
* **带温度的随机采样 (Stochastic Sampling)**：
    * 内置 `expf` 差异放大器。
    * 支持 `Temperature` 参数调节，有效避免模型陷入 `aaaaa` 的死循环。
    * 数值稳定性处理：自动减去 `Max Logit` 防止指数溢出。

## 快速开始

### 1. 环境依赖
* Ubuntu / Linux 环境
* NVIDIA GPU (支持 CUDA 11.0+)
* CUDA Toolkit

### 2. 编译
使用 NVCC 直接编译：
```bash
nvcc mini_gpt_decoder.cu -o mini_gpt_decoder
```
或者使用 CMake（推荐）：
```bash
mkdir build && cd build
cmake ..
make
```

### 3. 运行
```bash
./mini_gpt_decoder
```

## 运行效果

程序启动后会初始化一个随机权重的 12 层 Transformer 模型，并以字符 `'a'` 为起始符进行自回归推理。由于引入了随机采样，模型每次都会生成一段独一无二的字符序列：

> **Starting Generation:** a... [随机生成的 20-30 个字符] ...Done.
