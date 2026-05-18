#!/usr/bin/env bash
set -euo pipefail

# Real-neuron MAIA run for one A100 and low host RAM.
# Required env vars:
#   OPENAI_API_KEY
# Optional:
#   HF_TOKEN or HUGGING_FACE_HUB_TOKEN for gated FLUX weights

MODEL="${MODEL:-clip-RN50}"
LAYER="${LAYER:-layer4}"
UNIT="${UNIT:-1673}"
AGENT="${AGENT:-gpt-4o}"
DEVICE="${DEVICE:-0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-1024}"
MAX_ROUNDS="${MAX_ROUNDS:-15}"
RESULTS_DIR="${RESULTS_DIR:-./results_one_a100_real}"
PROMPTS_DIR="${PROMPTS_DIR:-./prompts/open}"
EXEMPLARS_DIR="${EXEMPLARS_DIR:-./exemplars}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python main.py \
  --agent "${AGENT}" \
  --model "${MODEL}" \
  --unit_mode manual \
  --units "${LAYER}=${UNIT}" \
  --path2prompts "${PROMPTS_DIR}" \
  --path2exemplars "${EXEMPLARS_DIR}" \
  --path2save "${RESULTS_DIR}" \
  --device "${DEVICE}" \
  --text2image_device "cuda:${DEVICE}" \
  --img2img_device "cuda:${DEVICE}" \
  --disable_cpu_offload \
  --max_output_tokens "${MAX_OUTPUT_TOKENS}" \
  --max_rounds "${MAX_ROUNDS}" \
  --debug
