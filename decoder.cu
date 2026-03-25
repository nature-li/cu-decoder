#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

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

int load_config(Config& config, std::string& model_file) {
  FILE* f = fopen(model_file.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "failed to open: %s\n", model_file.c_str());
    return -1;
  }

  if (fread(&config, sizeof(Config), 1, f) != 1) {
    fprintf(stderr, "failed to read config\n");
    fclose(f);
    return -1;
  }

  fclose(f);
  return 0;
}

struct ModelFile {
  int fd;
  void* data;
  size_t size;
};

int load_weights(Weights& w, const Config& config, float* data) {
  int head_dim = config.dim / config.n_heads;
  float* ptr = data;

  w.token_embedding = ptr;
  ptr += config.vocab_size * config.dim;
  w.rms_att = ptr;
  ptr += config.n_layers * config.dim;
  w.wq = ptr;
  ptr += config.n_layers * config.dim * config.dim;
  w.wk = ptr;
  ptr += config.n_layers * config.n_kv_heads * head_dim * config.dim;
  w.wv = ptr;
  ptr += config.n_layers * config.n_kv_heads * head_dim * config.dim;
  w.wo = ptr;
  ptr += config.n_layers * config.dim * config.dim;
  w.rms_ffn = ptr;
  ptr += config.n_layers * config.dim;
  w.w1 = ptr;
  ptr += config.n_layers * config.hidden_dim * config.dim;
  w.w2 = ptr;
  ptr += config.n_layers * config.dim * config.hidden_dim;
  w.w3 = ptr;
  ptr += config.n_layers * config.hidden_dim * config.dim;
  w.rms_final = ptr;
  ptr += config.dim;
  w.freq_cis_real = ptr;
  ptr += config.seq_len * head_dim / 2;
  w.freq_cis_imag = ptr;
  ptr += config.seq_len * head_dim / 2;

  // vocab_size 为负数时与 token_embedding 区享
  if (config.vocab_size > 0) {
    w.wcls = ptr;
  } else {
    w.wcls = w.token_embedding;
  }

  return 0;
}

int open_model(const std::string& model_file, ModelFile& mf) {
  mf.fd = open(model_file.c_str(), O_RDONLY);
  if (mf.fd < 0) {
    fprintf(stderr, "failed to open: %s\n", model_file.c_str());
    return -1;
  }

  mf.size = lseek(mf.fd, 0, SEEK_END);
  mf.data = mmap(nullptr, mf.size, PROT_READ, MAP_PRIVATE, mf.fd, 0);
  if (mf.data == MAP_FAILED) {
    fprintf(stderr, "mmap failed\n");
    close(mf.fd);
    return -1;
  }

  return 0;
}

void close_model(ModelFile& mf) {
  munmap(mf.data, mf.size);
  close(mf.fd);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model_file>\n", argv[0]);
    return 1;
  }

  Config config;
  std::string model_file = argv[1];
  if (load_config(config, model_file) != 0) {
    return 1;
  }

  printf("dim        = %d\n", config.dim);
  printf("hidden_dim = %d\n", config.hidden_dim);
  printf("n_layers   = %d\n", config.n_layers);
  printf("n_heads    = %d\n", config.n_heads);
  printf("n_kv_heads = %d\n", config.n_kv_heads);
  printf("vocab_size = %d\n", config.vocab_size);
  printf("seq_len    = %d\n", config.seq_len);

  ModelFile mf;
  if (open_model(model_file, mf) != 0) {
    return 1;
  }

  float* data = (float*)((char*)mf.data + sizeof(Config));
  Weights w;
  load_weights(w, config, data);

  printf("token_embedding[0] = %f\n", w.token_embedding[0]);
  printf("rms_final[0]       = %f\n", w.rms_final[0]);

  close_model(mf);

  return 0;
}