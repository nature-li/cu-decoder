## 1. 整体性能分布（nsys - Timeline）
sudo nsys profile --stats=true -o profile_v1_110m \
    ./gpu_decoder_v1 ../stories110M.bin ../tokenizer.bin <<< "Once upon a time"

nsys-ui profile_v1_110m.nsys-rep

## 2. 单个 kernel 深度分析（ncu - Kernel Profiling）
### 2.1 抓取 matmul_kernel 的详细指标
sudo ncu --kernel-name matmul_kernel \
         --launch-skip 0 --launch-count 1 \
         --set full \
         -o profile_matmul_v1 \
         ./gpu_decoder_v1 ../stories110M.bin ../tokenizer.bin <<< "Once upon a time"

### 2.2 查看分析结果
ncu-ui profile_matmul_v1.ncu-rep

## 3.整体分析v3
sudo nsys profile --stats=true -o profile_v3_110m \
    ./gpu_decoder_v3 ../stories110M.bin ../tokenizer.bin <<< "Once upon a time"
nsys-ui profile_v3_110m.nsys-rep


## 3.整体分析v4
sudo nsys profile --stats=true -o profile_v4_110m \
    ./gpu_decoder_v4 ../stories110M.bin ../tokenizer.bin <<< "Once upon a time"
nsys-ui profile_v4_110m.nsys-rep
