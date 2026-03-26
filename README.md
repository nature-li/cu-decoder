# cu-decoder

一个从零手写的 LLM 推理引擎，支持加载 [llama2.c](https://github.com/karpathy/llama2.c) 格式的模型，提供 C++ CPU、CUDA GPU、PyTorch 三个版本实现，方便对照学习。

## 项目结构

| 文件 | 说明 |
| :--- | :--- |
| `common.h` | 公共数据结构（Config、Weights、Tokenizer、ModelFile） |
| `common.cpp` | 公共函数（模型加载、BPE tokenizer、采样） |
| `decoder.h` | Decoder 基类定义 |
| `decoder.cpp` | Decoder::generate 实现（prefill + 自回归生成） |
| `cpu_decoder.h` | CPUDecoder 类定义 |
| `cpu_decoder.cpp` | CPUDecoder 实现 + main |
| `gpu_decoder.h` | GPUDecoder 类定义 |
| `gpu_decoder.cu` | CUDA kernel + GPUDecoder 实现 + main |
| `py_decoder.py` | PyTorch 版本实现，逻辑与 C++ 版本对齐 |
| `CMakeLists.txt` | 构建配置 |

## 实现的核心模块

- **BPE Tokenizer**：encode/decode，支持原始字节 token（`<0xXX>`）
- **Embedding lookup**
- **RMSNorm**
- **QKV 投影**（matmul）
- **RoPE 位置编码**
- **KV Cache**
- **Multi-Head Attention**（支持 GQA）
- **SwiGLU FFN**
- **Temperature + Top-K 采样**

## 环境依赖

- Linux
- NVIDIA GPU + CUDA 12.x
- CMake 3.18+
- Python 3.x + PyTorch（PyTorch 版本）

## 编译
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

## 运行

准备模型文件（以 stories15M 为例）：
```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
```

CPU 版本：
```bash
./cpu_decoder stories15M.bin tokenizer.bin
```

GPU 版本：
```bash
./gpu_decoder stories15M.bin tokenizer.bin
```

PyTorch 版本：
```bash
python3 py_decoder.py stories15M.bin tokenizer.bin
```

## 性能对比（stories15M，RTX 5060 Ti）

| 版本 | tokens/s |
| :--- | :--- |
| CPU C++ | 157 |
| PyTorch GPU | 265 |
| CUDA C++ | 422 |

## 参考

- [llama2.c](https://github.com/karpathy/llama2.c) - Andrej Karpathy
- [llama2.c 模型下载](https://huggingface.co/karpathy/tinyllamas)