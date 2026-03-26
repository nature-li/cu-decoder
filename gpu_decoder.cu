#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <string>

#include "common.h"

struct GPUWeights {
  float* token_embedding;
  float* rms_att;
  float* wq;
  float* wk;
  float* wv;
  float* wo;
  float* rms_ffn;
  float* w1;
  float* w2;
  float* w3;
  float* rms_final;
  float* freq_cis_real;
  float* freq_cis_imag;
  float* wcls;
};

struct RunState {
  // [dim] 当前 token 的隐藏状态，每层 attention/FFN 都在这上边做 in-place 更新
  float* x;
  float* xb;   // [dim] RSMNorm 输出缓冲区，避免覆盖 x
  float* xb2;  // [dim] attention/FFN 输出投影的临时缓冲区
  float* q;    // [dim] 当前 token 的 Query 向量
  float* k;    // [kv_dim] 当前 token 的 Key 向量
  float* v;    // [kv_dim] 当前 token 的 Value 向量
  //[n_heads, seq_len] 每个 head 对所有历史 token 的 attention score
  float* att;
  float* hb;      // [hidden_dim] FFN 中间结果: SiLU(w1(x))
  float* hb2;     // [hidden_dim] FFN 中间结果: w3(x)
  float* logits;  // [vocab_size] 最终输出的 logits, 用来采样下一个 token
  // [n_layers, seq_len, kv_dim] 所有层的 Key Cache, 避免重复计算历史 token
  float* k_cache;
  // [n_layers, seq_len, kv_dim] 所有层的 Value Cache，避免重复计算历史 token
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

int load_weights(Weights& w, const Config& config, float* data,
                 const ModelFile& mf) {
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

  size_t wcls_offset = (ptr - (float*)data) * sizeof(float) + sizeof(Config);
  if (wcls_offset < mf.size) {
    w.wcls = ptr;
  } else {
    w.wcls = w.token_embedding;
  }

  return 0;
}

void upload_weights(GPUWeights& gw, const Weights& w, const Config& config) {}

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
  int kv_dim = (config.dim / config.n_heads) * config.n_kv_heads;
  int n_layers = config.n_layers;
  int n_heads = config.n_heads;
  int seq_len = config.seq_len;

  s.x = new float[dim];                  // 隐藏状态
  s.xb = new float[dim];                 // RUMNorm 输出
  s.xb2 = new float[dim];                // attention/FFN 输出投影的临时缓冲区
  s.q = new float[dim];                  // Query
  s.k = new float[kv_dim];               // Key
  s.v = new float[kv_dim];               // Value
  s.att = new float[n_heads * seq_len];  // attention scores
  s.hb = new float[config.hidden_dim];   // FFN 中间结果: SiLU(w1(x))
  s.hb2 = new float[config.hidden_dim];  // FFN 中间结果: w3(x)
  s.logits = new float[abs(config.vocab_size)];        // 输出 logits
  s.k_cache = new float[n_layers * seq_len * kv_dim];  // KV Cache: Key
  s.v_cache = new float[n_layers * seq_len * kv_dim];  // KV Cache: Value

  return 0;
}

void free_run_state(RunState& s) {
  delete[] s.x;
  delete[] s.xb;
  delete[] s.xb2;
  delete[] s.q;
  delete[] s.k;
  delete[] s.v;
  delete[] s.att;
  delete[] s.hb;
  delete[] s.hb2;
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

/**
 * q = xb @ wq
 * k = xb @ wk
 * v = xb @ wv
 *
 * 参数
 * - n: 输入维度
 * - d: 输出维度
 * - w: [d, n]
 */
void matmul(float* out, const float* x, const float* w, int n, int d) {
  for (int i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += x[j] * w[i * n + j];
    }
    out[i] = val;
  }
}

/**
 * [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
 *
 * 参数:
 * - freq_real: [seq_len, head_dim/2]
 * - freq_imag: [seq_len, head_dim/2]
 */
void rope(float* q, float* k, const float* freq_real, const float* freq_imag,
          int pos, int dim, int kv_dim, int head_dim) {
  // 每两个元素做一次旋转, Q 和 K 都要做
  for (int i = 0; i < dim; i += 2) {
    int head_pos = i % head_dim;
    float cos_val = freq_real[pos * head_dim / 2 + head_pos / 2];
    float sin_val = freq_imag[pos * head_dim / 2 + head_pos / 2];

    float q0 = q[i];
    float q1 = q[i + 1];
    q[i] = q0 * cos_val - q1 * sin_val;
    q[i + 1] = q0 * sin_val + q1 * cos_val;
  }

  for (int i = 0; i < kv_dim; i += 2) {
    int head_pos = i % head_dim;
    float cos_val = freq_real[pos * head_dim / 2 + head_pos / 2];
    float sin_val = freq_imag[pos * head_dim / 2 + head_pos / 2];

    float k0 = k[i];
    float k1 = k[i + 1];
    k[i] = k0 * cos_val - k1 * sin_val;
    k[i + 1] = k0 * sin_val + k1 * cos_val;
  }
}

void attention(RunState& s, const Config& config, int layer, int pos,
               int kv_dim, int head_dim) {
  // GQA 时每个 KV 头服务几个 Q 头
  int kv_mul = config.n_heads / config.n_kv_heads;
  float scale = 1.0f / sqrtf((float)head_dim);

  float* k_cache_layer = s.k_cache + layer * config.seq_len * kv_dim;
  float* v_cache_layer = s.v_cache + layer * config.seq_len * kv_dim;

  for (int h = 0; h < config.n_heads; h++) {
    // 当前 head 的 Q
    float* q_head = s.q + h * head_dim;
    // 存储当前 head 的 scores
    float* att_head = s.att + h * config.seq_len;

    // Q @ K^T -> scores
    for (int t = 0; t <= pos; t++) {
      float* k_head = k_cache_layer + t * kv_dim + (h / kv_mul) * head_dim;
      float score = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        score += q_head[d] * k_head[d];
      }
      att_head[t] = score * scale;
    }

    // softmax
    float max_val = att_head[0];
    for (int t = 1; t <= pos; t++) {
      max_val = fmaxf(max_val, att_head[t]);
    }
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
      att_head[t] = expf(att_head[t] - max_val);
      sum += att_head[t];
    }
    for (int t = 0; t <= pos; t++) {
      att_head[t] /= sum;
    }

    // score @ V -> 输出
    float* out_head = s.xb + h * head_dim;
    memset(out_head, 0, head_dim * sizeof(float));
    for (int t = 0; t <= pos; t++) {
      float* v_head = v_cache_layer + t * kv_dim + (h / kv_mul) * head_dim;
      for (int d = 0; d < head_dim; d++) {
        out_head[d] += att_head[t] * v_head[d];
      }
    }
  }
}

/**
 * 激活函数 SiLU: x * sigmoid(x)
 */
float silu(float x) { return x * (1.0f / (1.0f + expf(-x))); }

void forward(Config& config, Weights& w, RunState& s, int token, int pos) {
  int dim = config.dim;

  // 1.Embedding lookup
  // 从 token embedding 表里取出第 token 行，作为初始隐藏状态
  float* emb = w.token_embedding + token * dim;
  memcpy(s.x, emb, dim * sizeof(float));

  // 2.过第一层
  for (int l = 0; l < config.n_layers; l++) {
    int head_dim = config.dim / config.n_heads;
    // K/V 向量的总长度, 等于 n_kv_heads * head_dim
    int kv_dim = config.n_kv_heads * head_dim;

    // attention 前的 RMSNorm
    rmsnorm(s.xb, s.x, w.rms_att + l * dim, dim);

    // QKV 投影
    matmul(s.q, s.xb, w.wq + l * dim * dim, dim, dim);
    matmul(s.k, s.xb, w.wk + l * kv_dim * dim, dim, kv_dim);
    matmul(s.v, s.xb, w.wv + l * kv_dim * dim, dim, kv_dim);

    // RoPE
    rope(s.q, s.k, w.freq_cis_real, w.freq_cis_imag, pos, dim, kv_dim,
         head_dim);

    // 写入 KV Cache
    // 当前层 KV Cache 的起始指针
    float* k_cache_layer = s.k_cache + l * config.seq_len * kv_dim;
    float* v_cache_layer = s.v_cache + l * config.seq_len * kv_dim;
    // 把当前 pos 的 K/V 写入 cache
    memcpy(k_cache_layer + pos * kv_dim, s.k, kv_dim * sizeof(float));
    memcpy(v_cache_layer + pos * kv_dim, s.v, kv_dim * sizeof(float));

    // attention
    attention(s, config, l, pos, kv_dim, head_dim);
    // attention 输出投影: xb = xb @ wo
    matmul(s.xb2, s.xb, w.wo + l * dim * dim, dim, dim);
    for (int i = 0; i < dim; i++) {
      s.x[i] += s.xb2[i];
    }

    // FFN 前的 RMSNorm
    rmsnorm(s.xb, s.x, w.rms_ffn + l * dim, dim);

    /**
     * FFN(x) = w2(SiLU(w1(x)) * w3(x))
     */
    matmul(s.hb, s.xb, w.w1 + l * config.hidden_dim * dim, dim,
           config.hidden_dim);
    matmul(s.hb2, s.xb, w.w3 + l * config.hidden_dim * dim, dim,
           config.hidden_dim);
    for (int i = 0; i < config.hidden_dim; i++) {
      s.hb[i] = silu(s.hb[i]) * s.hb2[i];
    }

    // FFN 输出投影 + 残差连接
    matmul(s.xb2, s.hb, w.w2 + l * dim * config.hidden_dim, config.hidden_dim,
           dim);
    for (int i = 0; i < dim; i++) {
      s.x[i] += s.xb2[i];
    }
  }

  // 最终 RMSNorm
  rmsnorm(s.xb, s.x, w.rms_final, dim);

  // 输出 logits
  matmul(s.logits, s.xb, w.wcls, dim, abs(config.vocab_size));
}

int argmax(const float* logits, int size) {
  int max_idx = 0;
  float max_val = logits[0];
  for (int i = 1; i < size; i++) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      max_idx = i;
    }
  }

  return max_idx;
}

/**
 * temperature 采样
 * temperature 越高分布越平坦 (更随机), 越低越集中 (更确定)
 * temperature = 0 退化为 argmax
 */
int sample(const float* logits, int size, float temperature,
           std::mt19937& rng) {
  if (temperature == 0.0f) {
    return argmax(logits, size);
  }

  // 除以 temperature，缩放 logits
  std::vector<float> probs(size);
  float max_val = logits[0];
  for (int i = 1; i < size; i++) {
    max_val = fmaxf(max_val, logits[i]);
  }

  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    probs[i] = expf((logits[i] - max_val) / temperature);
    sum += probs[i];
  }

  for (int i = 0; i < size; i++) {
    probs[i] /= sum;
  }

  // 轮盘赌采样
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  float r = dis(rng);
  float cur = 0.0f;
  for (int i = 0; i < size; i++) {
    cur += probs[i];
    if (r <= cur) {
      return i;
    }
  }

  return size - 1;
}

/**
 * top-k 采样
 * 只保留概率最高的 k 个 token，其余归零，再做 temperature 采样
 */
int sample_topk(const float* logits, int size, int k, float temperature,
                std::mt19937& rng) {
  if (k <= 0 || k >= size) {
    return sample(logits, size, temperature, rng);
  }

  // 找第 k 大的阈值
  std::vector<float> tmp(logits, logits + size);
  std::nth_element(tmp.begin(), tmp.begin() + k - 1, tmp.end(),
                   std::greater<float>());
  float threshold = tmp[k - 1];

  // 低于阈值的归零，其余做 softmax
  std::vector<float> probs(size);
  float max_val = logits[0];
  for (int i = 1; i < size; i++) {
    max_val = fmaxf(max_val, logits[i]);
  }

  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    if (logits[i] >= threshold) {
      probs[i] = expf((logits[i] - max_val) / temperature);
    } else {
      probs[i] = 0.0f;
    }
    sum += probs[i];
  }

  for (int i = 0; i < size; i++) {
    probs[i] /= sum;
  }

  // 轮盘赌采样
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  float r = dis(rng);
  float cur = 0.0f;
  for (int i = 0; i < size; i++) {
    cur += probs[i];
    if (r <= cur) return i;
  }
  return size - 1;
}

/**
 * 在词表里查找 str，返回 token id，找不到返回 -1
 */
int vocab_lookup(Tokenizer& t, const char* str) {
  for (int i = 0; i < t.vocab_size; i++) {
    if (strcmp(t.vocab[i], str) == 0) {
      return i;
    }
  }
  return -1;
}

/**
 * BPE encode: string -> token ids
 * 结果写入 tokens，返回 token 数量
 */
int encode(Tokenizer& t, const std::string& text, int* tokens) {
  int n_tokens = 0;

  // 每个字符先单独变成 token
  char buf[16] = {};
  for (unsigned char c : text) {
    snprintf(buf, sizeof(buf), "%c", c);
    int id = vocab_lookup(t, buf);
    if (id == -1) {
      // 找不到就用字节 token <0xXX>
      snprintf(buf, sizeof(buf), "<0x%02X>", c);
      id = vocab_lookup(t, buf);
    }
    if (id != -1) {
      tokens[n_tokens++] = id;
    }
  }

  // 反复合并 score 最高的相邻 pair
  char merge_buf[512];
  while (true) {
    int best_id = -1;
    float best_score = -1e10f;
    int best_idx = -1;

    for (int i = 0; i < n_tokens - 1; i++) {
      // 拼接相邻两个 token 的字符串
      snprintf(merge_buf, sizeof(merge_buf), "%s%s", t.vocab[tokens[i]],
               t.vocab[tokens[i + 1]]);
      int id = vocab_lookup(t, merge_buf);
      if (id != -1 && t.vocab_scores[id] > best_score) {
        best_score = t.vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    // 没有可合并的 pair 了，结束
    if (best_idx == -1) {
      break;
    }

    // 合并 best_idx 和 best_idx+1
    tokens[best_idx] = best_id;
    for (int i = best_idx + 1; i < n_tokens - 1; i++) {
      tokens[i] = tokens[i + 1];
    }
    n_tokens--;
  }

  return n_tokens;
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
  load_weights(w, config, data, mf);

  Tokenizer tokenizer;
  if (load_tokenizer(tokenizer, tokenizer_file, abs(config.vocab_size)) != 0) {
    return 1;
  }
  // BOS 和 EOS
  printf("vocab[1] = %s\n", tokenizer.vocab[1]);
  printf("vocab[2] = %s\n", tokenizer.vocab[2]);

  RunState state;
  alloc_run_state(state, config);

  int vocab_size = abs(config.vocab_size);
  int steps = config.seq_len;
  std::mt19937 rng(time(nullptr));
  float temperature = 0.8f;
  int top_k = 40;

  // 读用户输入
  std::string prompt;
  printf("Enter prompt: ");
  std::getline(std::cin, prompt);

  // encode prompt -> token ids
  std::vector<int> prompt_tokens(prompt.size() + 10);
  int n_prompt = encode(tokenizer, prompt, prompt_tokens.data());

  // prefill
  int token = 1;  // BOS
  int pos = 0;
  forward(config, w, state, token, pos++);  // 先跑 BOS
  for (int i = 0; i < n_prompt; i++) {
    if (pos >= config.seq_len) {
      fprintf(stderr, "prompt too long, max %d tokens\n", config.seq_len - 1);
      return 1;
    }
    token = prompt_tokens[i];
    forward(config, w, state, token, pos++);
  }

  // 生成阶段
  for (; pos < steps; pos++) {
    int next_token =
        sample_topk(state.logits, vocab_size, top_k, temperature, rng);
    // EOS
    if (next_token == 1 || next_token == 2) {
      break;
    }

    printf("%s", decode(tokenizer, token, next_token));
    fflush(stdout);

    token = next_token;
    forward(config, w, state, token, pos);
  }
  printf("\n");

  free_run_state(state);
  free_tokenizer(tokenizer);
  close_model(mf);
  return 0;
}