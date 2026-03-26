#include "common.h"

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
