#pragma once
#include <cublas_v2.h>

#include "decoder.h"

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

struct GPURunState {
  float* x;
  float* xb;
  float* xb2;
  float* q;
  float* k;
  float* v;
  float* att;
  float* hb;
  float* hb2;
  float* logits;  // pinned memory
  float* k_cache;
  float* v_cache;
};

class GPUDecoder : public Decoder {
 public:
  GPUDecoder(const std::string& model_file);
  ~GPUDecoder();

  void forward(int token, int pos) override;
  float* get_logits() override;

 private:
  ModelFile mf;
  Weights w;
  GPUWeights gw;
  GPURunState gpu_state;
  cublasHandle_t cublas_handle;
};