#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <random>

struct Config {
  int dim;         // 模型隐藏层维度
  int hidden_dim;  // FFN 中间层维度
  int n_layers;    // Transformers 层数
  int n_heads;     // attention 头数
  int n_kv_heads;  // KV 头数, GQA 时小于 n_heads
  int vocab_size;  // 词表大小
  int seq_len;     // 最大序列化长度
};

struct Weights {
  float* token_embedding;  // [vocab_size, dim] 词嵌入表
  float* rms_att;          // [n_layers, dim] 每层 attention 前的 RMSNorm 权重
  float* wq;               // [n_layers, dim, dim] Query 投影矩阵
  float* wk;               // [n_layers, n_kv_heads*head_dim, dim] Key 投影矩阵
  float* wv;       // [n_layers, n_kv_heads*head_dim, dim] Value 投影矩阵
  float* wo;       // [n_layers, dim, dim] attention 输出投影矩阵
  float* rms_ffn;  // [n_layers, dim] 每层 FFN 前的 RMSNorm 权重
  float* w1;  // [n_layers, hidden_dim, dim] FFN gate 矩阵 (SwiGLU 的门控路径)
  float* w2;  // [n_layers, dim, hidden_dim] FFN down 矩阵 (降维回 dim)
  float* w3;  // [n_layers, hidden_dim, dim] FFN up 矩阵 (SwiGLU 的值路径)
  float* rms_final;      // [dim] 最后一层输出的 RMSNorm 权重
  float* freq_cis_real;  // [seq_len, head_dim/2] RoPE 位置编码的 cos 分量
  float* freq_cis_imag;  // [seq_len, head_dim/2] RoPE 位置编码的 sin 分量
  float* wcls;           // [vocab_size, dim] 输出层投影矩阵
};

struct ModelFile {
  int fd;       // 文件描述符
  void* data;   // mmap 映射的内存起始地址
  size_t size;  // 文件总大小（字节）
};

struct Tokenizer {
  int vocab_size;        // 词表大小
  int max_token_length;  // 最长 token 的字符数，用于分配 encode 缓冲区
  char** vocab;          // 词表字符串数组，vocab[i] 是第 i 个 token 的字符串
  // BPE merge 优先级分数，vocab_scores[i] 对应第 i 个 token
  float* vocab_scores;
};

int load_config(Config& config, std::string& model_file);
int open_model(const std::string& model_file, ModelFile& mf);
void close_model(ModelFile& mf);
int load_weights(Weights& w, const Config& config, float* data,
                 const ModelFile& mf);
int load_tokenizer(Tokenizer& t, const std::string& tokenizer_file,
                   int vocab_size);
void free_tokenizer(Tokenizer& t);
const char* decode(Tokenizer& t, int prev_token, int token);
int encode(Tokenizer& t, const std::string& text, int* tokens);
int vocab_lookup(Tokenizer& t, const char* str);
int argmax(const float* logits, int size);
int sample(const float* logits, int size, float temperature, std::mt19937& rng);
int sample_topk(const float* logits, int size, int k, float temperature,
                std::mt19937& rng);