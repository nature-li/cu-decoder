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

struct RunState {
  // [dim] 当前 token 的隐藏状态，每层 attention/FFN 都在这上边做 in-place 更新
  float* x;
  float* xb;  // [dim] RSMNorm 输出缓冲区，避免覆盖 x
  float* q;   // [dim] 当前 token 的 Query 向量
  float* k;   // [dim] 当前 token 的 Key 向量
  float* v;   // [dim] 当前 token 的 Value 向量
  //[n_heads, seq_len] 每个 head 对所有历史 token 的 attention score
  float* att;
  float* logits;  // [vocab_size] 最终输出的 logits, 用来采样下一个 token
  // [n_layers, seq_len, dim] 所有层的 Key Cache, 避免重复计算历史 token
  float* k_cache;
  // [n_layers, seq_len, dim] 所有层的 Value Cache，避免重复计算历史 token
  float* v_cache;
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

int load_tokenizer(Tokenizer& t, const std::string& tokenizer_file,
                   int vocab_size) {
  t.vocab_size = vocab_size;
  t.vocab = new char*[vocab_size];
  t.vocab_scores = new float[vocab_size];

  FILE* f = fopen(tokenizer_file.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "failed to open: %s\n", tokenizer_file.c_str());
    return -1;
  }

  if (fread(&t.max_token_length, sizeof(int), 1, f) != 1) {
    fprintf(stderr, "failed to read max_token_length\n");
    fclose(f);
    return -1;
  }

  for (int i = 0; i < vocab_size; i++) {
    if (fread(&t.vocab_scores[i], sizeof(float), 1, f) != 1) {
      fprintf(stderr, "failed to read vocab_scores[%d]\n", i);
      fclose(f);
      return -1;
    }

    int len;
    if (fread(&len, sizeof(int), 1, f) != 1) {
      fprintf(stderr, "failed to read token length[%d]\n", i);
      fclose(f);
      return -1;
    }

    t.vocab[i] = new char[len + 1];  // +1 for '\0'
    if (fread(t.vocab[i], sizeof(char), len, f) != (size_t)len) {
      fprintf(stderr, "failed to read token string[%d]\n", i);
      fclose(f);
      return -1;
    }
    t.vocab[i][len] = '\0';
  }

  fclose(f);
  return 0;
}

void free_tokenizer(Tokenizer& t) {
  for (int i = 0; i < t.vocab_size; i++) {
    delete[] t.vocab[i];
  }

  delete[] t.vocab;
  delete[] t.vocab_scores;
}

const char* decode(Tokenizer& t, int prev_token, int token) {
  char* piece = t.vocab[token];

  /**
   * BOS(1) 后面跟 token 如果以空格开头, 去掉空格, 比如:
   * " hello" 在句首应该输出 "hello"
   */
  if (prev_token == 1 && piece[0] == ' ') {
    piece++;
  }

  /**
   * 处理 <0x61> 这种原始字节 token, 转成对应字符
   * 格式固定是 <0xXX>, 长度 6
   */
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhx>", &byte_val) == 1) {
    static char byte_piece[2];
    byte_piece[0] = (char)byte_val;
    byte_piece[1] = '\0';
    return byte_piece;
  }

  return piece;
}

int alloc_run_state(RunState& s, const Config& config) {
  int dim = config.dim;
  int n_layers = config.n_layers;
  int n_heads = config.n_heads;
  int seq_len = config.seq_len;

  s.x = new float[dim];                             // 隐藏状态
  s.xb = new float[dim];                            // RUMNorm 输出
  s.q = new float[dim];                             // Query
  s.k = new float[dim];                             // Key
  s.v = new float[dim];                             // Value
  s.att = new float[n_heads * seq_len];             // attention scores
  s.logits = new float[config.vocab_size];          // 输出 logits
  s.k_cache = new float[n_layers * seq_len * dim];  // KV Cache: Key
  s.v_cache = new float[n_layers * seq_len * dim];  // KV Cache: Value

  return 0;
}

void free_run_state(RunState& s) {
  delete[] s.x;
  delete[] s.xb;
  delete[] s.q;
  delete[] s.k;
  delete[] s.v;
  delete[] s.att;
  delete[] s.logits;
  delete[] s.k_cache;
  delete[] s.v_cache;
}

/**
 * rms(x) = sqrt(mean(x^2) + 1e-6)
 * RMSNorm(x) = x / rms(x) * weight
 */
void rmsnorm(float* out, const float* x, const float* weight, int dim) {
  // 计算 x 的均方根
  float ss = 0.0f;
  for (int i = 0; i < dim; i++) {
    ss += x[i] * x[i];
  }
  ss = 1.0f / sqrtf(ss / dim + 1e-6f);

  // 归一化
  for (int i = 0; i < dim; i++) {
    out[i] = x[i] * ss * weight[i];
  }
}

void forward(Config& config, Weights& w, RunState& s, int token, int pos) {
  int dim = config.dim;

  // 1.Embedding lookup
  // 从 token embedding 表里取出第 token 行，作为初始隐藏状态
  float* emb = w.token_embedding + token * dim;
  memcpy(s.x, emb, dim * sizeof(float));

  // 2.过第一层
  for (int l = 0; l < config.n_layers; l++) {
    // attention 前的 RMSNorm
    rmsnorm(s.xb, s.x, w.rms_att + l * dim, dim);
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <model_file> <tokenizer_file>\n", argv[0]);
    return 1;
  }
  std::string model_file = argv[1];
  std::string tokenizer_file = argv[2];

  Config config;
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

  Tokenizer tokenizer;
  if (load_tokenizer(tokenizer, tokenizer_file, abs(config.vocab_size)) != 0) {
    return 1;
  }
  // 验证一下几个 token
  printf("vocab[0]     = %s\n", tokenizer.vocab[0]);
  printf("vocab[1]     = %s\n", tokenizer.vocab[1]);
  printf("vocab[100]   = %s\n", tokenizer.vocab[100]);
  printf("vocab[1000]  = %s\n", tokenizer.vocab[1000]);

  // prev_token 传上一个 token id，第一个 token 传 1 (BOS)
  int prev_token = 1;
  int token = 1000;
  printf("%s\n", decode(tokenizer, prev_token, token));

  RunState state;
  alloc_run_state(state, config);

  // BOS token = 1，pos = 0
  forward(config, w, state, 1, 0);
  printf("x[0] = %f\n", state.x[0]);

  free_run_state(state);
  free_tokenizer(tokenizer);
  close_model(mf);
  return 0;
}