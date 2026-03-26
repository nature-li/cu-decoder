#!/bin/bash
# benchmark.sh

# 脚本所在目录
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
BUILD_DIR=$SCRIPT_DIR/build

MODEL=$SCRIPT_DIR/stories110M.bin
TOKENIZER=$SCRIPT_DIR/tokenizer.bin
PROMPT="Once upon a time"

echo "=== CPU ==="
echo $PROMPT | $BUILD_DIR/cpu_decoder $MODEL $TOKENIZER

echo "=== GPU v1 (naive) ==="
echo $PROMPT | $BUILD_DIR/gpu_decoder_v1 $MODEL $TOKENIZER

echo "=== GPU v2 (tiling) ==="
echo $PROMPT | $BUILD_DIR/gpu_decoder_v2 $MODEL $TOKENIZER

echo "=== GPU v3 (tiling) ==="
echo $PROMPT | $BUILD_DIR/gpu_decoder_v3 $MODEL $TOKENIZER

echo "=== GPU v4 (tiling) ==="
echo $PROMPT | $BUILD_DIR/gpu_decoder_v4 $MODEL $TOKENIZER
