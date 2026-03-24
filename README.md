***

# Mini-GPT CUDA Decoder

这是一个基于 **NVIDIA CUDA** 原生实现的极简 GPT 解码推理引擎。本项目不依赖任何深度学习框架（如 PyTorch 或 TensorFlow），直接在 GPU 硬件层面上手写 Transformer 核心算子，并结合 **cuBLAS** 实现了从词向量嵌入到自回归字符生成的完整高性能链路。

## 目录结构说明

| 文件名 | 用途描述 |
| :--- | :--- |
| **`mini_gpt_decoder.cu`** | **核心程序**。包含完整的 12 层 Transformer Decoder 逻辑、cuBLAS 线性层加速、KV Cache 管理及自回归生成循环。 |
| **`mini_gpt_basic_attention.cu`** | 实验性质的单层 Attention 实现，用于前期验证算子正确性。 |
| **`comm.h`** | 公共头文件，包含 `GPTConfig` 定义、CUDA 错误检查宏（`CHECK_CUDA`）及工具函数。 |
| **`CMakeLists.txt`** | 项目构建配置文件，支持快速编译。 |

## 核心特性

* **高性能矩阵运算 (cuBLAS 加速)**
  全面废弃朴素的循环矩阵乘法，接入工业标准的 `cublasSgemm`，极致利用 GPU 显存带宽与算力。
* **PyTorch 权重格式完全对齐**
  底层内存布局支持标准的 `[out_features, in_features]` 行优先权重存储，通过 CUDA 视角的列优先转置操作 (`CUBLAS_OP_T`)，为后续直接加载 HuggingFace 模型权重扫清了障碍。
* **健壮的原生 CUDA 核函数**
  采用 **Grid-Stride Loop (网格跨步循环)** 编写 Attention 算子，彻底解除长序列对 `blockDim.x` 的硬件限制；支持计算与激活的轻量级算子融合。
* **完备的模型架构与 KV Cache 安全**
  实现了严谨的 Transformer 解码器结构（包含防止数值爆炸的 **Final RMSNorm**）；带有内存越界拦截的 KV Cache，避免在达到上下文上限时触发非法显存访问。
* **先进的自回归解码策略**
  内置数值稳定的 Softmax 处理（自动减去 Max Logit 防止指数溢出）；支持 **Top-K 截断采样** 与 **Temperature (温度)** 调节，有效避免模型陷入重复生成的死循环。

## 快速开始

### 1. 环境依赖
* Ubuntu / Linux 环境
* NVIDIA GPU (支持 CUDA 11.0+)
* CUDA Toolkit (必须包含 cuBLAS 库)

### 2. 编译
由于引入了 cuBLAS 矩阵加速，使用 NVCC 编译时**必须链接 `-lcublas` 库**：
```bash
nvcc mini_gpt_decoder.cu -lcublas -o mini_gpt_decoder
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

程序启动后会初始化一个随机权重的 12 层 Transformer 模型，并以字符 `'a'` 为起始符进行自回归推理。由于引入了 Top-K 和随机采样，模型每次都会生成一段独一无二的字符序列。当生成达到设定的上下文长度上限（如 1024）时，程序会安全停止并退出。

> **Starting Generation:** a... [随机生成的序列] ...
> [Warning] Reached maximum context length (1024). Stopping generation.
> Done.