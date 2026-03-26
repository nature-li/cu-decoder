#pragma once
#include "decoder.h"

struct RunState {
  float* x;
  float* xb;
  float* xb2;
  float* q;
  float* k;
  float* v;
  float* att;
  float* hb;
  float* hb2;
  float* logits;
  float* k_cache;
  float* v_cache;
};

class CPUDecoder : public Decoder {
 public:
  CPUDecoder(const std::string& model_file);
  ~CPUDecoder();

  void forward(int token, int pos) override;
  float* get_logits() override { return state.logits; }

 private:
  ModelFile mf;
  Weights w;
  RunState state;
};