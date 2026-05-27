#!/usr/bin/env bash
set -euo pipefail

# Real-neuron MAIA run for a low-host-RAM Kubernetes pod.
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
RESULTS_DIR="${RESULTS_DIR:-./results_pod_real}"
PROMPTS_DIR="${PROMPTS_DIR:-./prompts/open}"
EXEMPLARS_DIR="${EXEMPLARS_DIR:-./exemplars}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent|--AGENT)
      AGENT="$2"
      shift 2
      ;;
    --model|--MODEL)
      MODEL="$2"
      shift 2
      ;;
    --layer|--LAYER)
      LAYER="$2"
      shift 2
      ;;
    --unit|--UNIT)
      UNIT="$2"
      shift 2
      ;;
    --device|--DEVICE)
      DEVICE="$2"
      shift 2
      ;;
    --max-output-tokens|--MAX_OUTPUT_TOKENS)
      MAX_OUTPUT_TOKENS="$2"
      shift 2
      ;;
    --max-rounds|--MAX_ROUNDS)
      MAX_ROUNDS="$2"
      shift 2
      ;;
    --results-dir|--RESULTS_DIR)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --prompts-dir|--PROMPTS_DIR)
      PROMPTS_DIR="$2"
      shift 2
      ;;
    --exemplars-dir|--EXEMPLARS_DIR)
      EXEMPLARS_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

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
