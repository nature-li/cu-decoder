#include "decoder.h"

#include <chrono>
#include <cstdio>
#include <vector>

void Decoder::generate(const Tokenizer& tokenizer, const std::string& prompt,
                       int steps, float temperature, int top_k,
                       std::mt19937& rng) {
  int vocab_size = abs(config.vocab_size);

  std::vector<int> prompt_tokens(prompt.size() + 10);
  int n_prompt =
      encode(const_cast<Tokenizer&>(tokenizer), prompt, prompt_tokens.data());

  int token = 1;
  int pos = 0;
  forward(token, pos++);

  for (int i = 0; i < n_prompt; i++) {
    if (pos >= config.seq_len) {
      fprintf(stderr, "prompt too long\n");
      return;
    }
    token = prompt_tokens[i];
    forward(token, pos++);
  }

  // 生成阶段开始计时
  int n_generated = 0;
  auto t_start = std::chrono::steady_clock::now();

  for (; pos < steps; pos++) {
    float* logits = get_logits();
    int next_token = sample_topk(logits, vocab_size, top_k, temperature, rng);
    if (next_token == 1 || next_token == 2) break;

    printf("%s", decode(const_cast<Tokenizer&>(tokenizer), token, next_token));
    fflush(stdout);

    token = next_token;
    forward(token, pos);
    n_generated++;
  }
  printf("\n");

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();
  fprintf(stderr, "\n%.2f tokens/s (%d tokens in %.2fs)\n",
          n_generated / elapsed, n_generated, elapsed);
}