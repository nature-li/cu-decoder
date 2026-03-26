import torch
import struct
import numpy as np
from dataclasses import dataclass

# ============================================================================
# Config & 权重加载
# ============================================================================

@dataclass
class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int

def load_model(model_file: str):
    with open(model_file, 'rb') as f:
        # 读 config
        config = Config(*struct.unpack('7i', f.read(28)))
        vocab_size = abs(config.vocab_size)
        head_dim = config.dim // config.n_heads
        kv_dim = config.n_kv_heads * head_dim

        # 读权重，直接 mmap
        data = np.memmap(model_file, dtype=np.float32, mode='r', offset=28)
        ptr = 0

        def take(n):
            nonlocal ptr
            t = torch.from_numpy(data[ptr:ptr+n].copy())
            ptr += n
            return t

        w = {}
        w['token_embedding'] = take(vocab_size * config.dim).reshape(vocab_size, config.dim)
        w['rms_att']         = take(config.n_layers * config.dim).reshape(config.n_layers, config.dim)
        w['wq']              = take(config.n_layers * config.dim * config.dim).reshape(config.n_layers, config.dim, config.dim)
        w['wk']              = take(config.n_layers * kv_dim * config.dim).reshape(config.n_layers, kv_dim, config.dim)
        w['wv']              = take(config.n_layers * kv_dim * config.dim).reshape(config.n_layers, kv_dim, config.dim)
        w['wo']              = take(config.n_layers * config.dim * config.dim).reshape(config.n_layers, config.dim, config.dim)
        w['rms_ffn']         = take(config.n_layers * config.dim).reshape(config.n_layers, config.dim)
        w['w1']              = take(config.n_layers * config.hidden_dim * config.dim).reshape(config.n_layers, config.hidden_dim, config.dim)
        w['w2']              = take(config.n_layers * config.dim * config.hidden_dim).reshape(config.n_layers, config.dim, config.hidden_dim)
        w['w3']              = take(config.n_layers * config.hidden_dim * config.dim).reshape(config.n_layers, config.hidden_dim, config.dim)
        w['rms_final']       = take(config.dim)
        w['freq_cis_real']   = take(config.seq_len * head_dim // 2).reshape(config.seq_len, head_dim // 2)
        w['freq_cis_imag']   = take(config.seq_len * head_dim // 2).reshape(config.seq_len, head_dim // 2)

        # wcls 共享判断
        remaining = len(data) - ptr
        if remaining >= vocab_size * config.dim:
            w['wcls'] = take(vocab_size * config.dim).reshape(vocab_size, config.dim)
        else:
            w['wcls'] = w['token_embedding']

    return config, w

# ============================================================================
# Tokenizer
# ============================================================================

def load_tokenizer(tokenizer_file: str, vocab_size: int):
    vocab = []
    scores = []
    with open(tokenizer_file, 'rb') as f:
        max_token_length = struct.unpack('i', f.read(4))[0]
        for _ in range(vocab_size):
            score = struct.unpack('f', f.read(4))[0]
            length = struct.unpack('i', f.read(4))[0]
            token = f.read(length).decode('utf-8', errors='replace')
            vocab.append(token)
            scores.append(score)
    return vocab, scores

def decode(vocab, prev_token, token):
    piece = vocab[token]
    if prev_token == 1 and piece.startswith(' '):
        piece = piece[1:]
    # 处理 <0xXX> 字节 token
    if piece.startswith('<0x') and piece.endswith('>'):
        byte_val = int(piece[3:-1], 16)
        piece = chr(byte_val)
    return piece

def encode(vocab, scores, text: str):
    # 先拆成单字符
    tokens = []
    for c in text:
        if c in vocab:
            tokens.append(vocab.index(c))
        else:
            byte_str = f'<0x{ord(c):02X}>'
            if byte_str in vocab:
                tokens.append(vocab.index(byte_str))

    # BPE merge
    while True:
        best_score = -1e10
        best_idx = -1
        best_id = -1
        for i in range(len(tokens) - 1):
            merged = vocab[tokens[i]] + vocab[tokens[i+1]]
            if merged in vocab:
                idx = vocab.index(merged)
                if scores[idx] > best_score:
                    best_score = scores[idx]
                    best_idx = i
                    best_id = idx
        if best_idx == -1:
            break
        tokens[best_idx] = best_id
        tokens.pop(best_idx + 1)

    return tokens

# ============================================================================
# Forward
# ============================================================================

def rmsnorm(x, weight):
    # x: [dim], weight: [dim]
    norm = x * torch.rsqrt(x.pow(2).mean() + 1e-6)
    return norm * weight

def forward(config, w, k_cache, v_cache, token, pos):
    head_dim = config.dim // config.n_heads
    kv_mul = config.n_heads // config.n_kv_heads

    # 1. Embedding lookup
    x = w['token_embedding'][token]  # [dim]

    for l in range(config.n_layers):
        # 2. Attention 前 RMSNorm
        xb = rmsnorm(x, w['rms_att'][l])  # [dim]

        # 3. QKV 投影
        q = xb @ w['wq'][l].T   # [dim]
        k = xb @ w['wk'][l].T   # [kv_dim]
        v = xb @ w['wv'][l].T   # [kv_dim]

        # reshape 成多头
        q = q.reshape(config.n_heads, head_dim)           # [n_heads, head_dim]
        k = k.reshape(config.n_kv_heads, head_dim)        # [n_kv_heads, head_dim]
        v = v.reshape(config.n_kv_heads, head_dim)        # [n_kv_heads, head_dim]

        # 4. RoPE
        cos = w['freq_cis_real'][pos]  # [head_dim/2]
        sin = w['freq_cis_imag'][pos]  # [head_dim/2]

        # 对 q 和 k 的每对元素做旋转
        def apply_rope(x):
            x0 = x[..., 0::2]  # 偶数位
            x1 = x[..., 1::2]  # 奇数位
            return torch.stack([x0*cos - x1*sin, x0*sin + x1*cos], dim=-1).flatten(-2)

        q = apply_rope(q)
        k = apply_rope(k)

        # 5. KV Cache 写入
        k_cache[l][pos] = k  # [n_kv_heads, head_dim]
        v_cache[l][pos] = v

        # 6. Attention
        # GQA: 把 k/v 重复 kv_mul 次对齐 n_heads
        k_all = k_cache[l][:pos+1].repeat_interleave(kv_mul, dim=1)  # [pos+1, n_heads, head_dim]
        v_all = v_cache[l][:pos+1].repeat_interleave(kv_mul, dim=1)

        # q: [n_heads, head_dim], k_all: [pos+1, n_heads, head_dim]
        scores = torch.einsum('hd,thd->ht', q, k_all) / head_dim**0.5  # [n_heads, pos+1]
        scores = torch.softmax(scores, dim=-1)
        attn_out = torch.einsum('ht,thd->hd', scores, v_all)  # [n_heads, head_dim]
        attn_out = attn_out.reshape(config.dim)  # [dim]

        # 7. 输出投影 + 残差
        x = x + attn_out @ w['wo'][l].T

        # 8. FFN 前 RMSNorm
        xb = rmsnorm(x, w['rms_ffn'][l])

        # 9. SwiGLU FFN
        h = torch.nn.functional.silu(xb @ w['w1'][l].T) * (xb @ w['w3'][l].T)

        # 10. FFN 输出投影 + 残差
        x = x + h @ w['w2'][l].T

    # 11. 最终 RMSNorm
    x = rmsnorm(x, w['rms_final'])

    # 12. 输出 logits
    logits = x @ w['wcls'].T  # [vocab_size]
    return logits

# ============================================================================
# 采样
# ============================================================================

def sample_topk(logits, k, temperature):
    if temperature == 0:
        return logits.argmax().item()
    topk_vals, topk_idx = torch.topk(logits, k)
    probs = torch.softmax(topk_vals / temperature, dim=-1)
    chosen = torch.multinomial(probs, 1).item()
    return topk_idx[chosen].item()

# ============================================================================
# 生成
# ============================================================================

def generate(config, w, vocab, scores, prompt, steps=256, temperature=0.8, top_k=40):
    head_dim = config.dim // config.n_heads
    vocab_size = abs(config.vocab_size)

    # KV Cache
    k_cache = [torch.zeros(config.seq_len, config.n_kv_heads, head_dim) for _ in range(config.n_layers)]
    v_cache = [torch.zeros(config.seq_len, config.n_kv_heads, head_dim) for _ in range(config.n_layers)]

    # encode prompt
    prompt_tokens = encode(vocab, scores, prompt)

    token = 1  # BOS
    pos = 0
    forward(config, w, k_cache, v_cache, token, pos)
    pos += 1

    for t in prompt_tokens:
        token = t
        forward(config, w, k_cache, v_cache, token, pos)
        pos += 1

    # 生成
    import time
    t_start = time.time()
    n_generated = 0

    while pos < steps:
        logits = forward(config, w, k_cache, v_cache, token, pos)
        next_token = sample_topk(logits, top_k, temperature)
        if next_token in (1, 2):
            break

        piece = decode(vocab, token, next_token)
        print(piece, end='', flush=True)

        token = next_token
        pos += 1
        n_generated += 1

    print()
    elapsed = time.time() - t_start
    print(f'\n{n_generated / elapsed:.2f} tokens/s ({n_generated} tokens in {elapsed:.2f}s)',
          flush=True)

# ============================================================================
# main
# ============================================================================

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <model_file> <tokenizer_file>')
        sys.exit(1)

    config, w = load_model(sys.argv[1])
    vocab, scores = load_tokenizer(sys.argv[2], abs(config.vocab_size))

    prompt = input('Enter prompt: ')
    generate(config, w, vocab, scores, prompt)