#!/bin/bash
# Script to run full evaluation with sampling on models from models.txt

if [ -z "${OPENROUTER_API_KEY}" ]; then
  echo "OPENROUTER_API_KEY is not set."
  echo "Export it first, for example:"
  echo "  export OPENROUTER_API_KEY=\"your_key_here\""
  exit 1
fi

python3 run_comparison.py \
  --models_file models.txt \
  --full_evaluation \
  --request_workers "${REQUEST_WORKERS:-6}" \
  --checkpoint_chunk_size "${CHECKPOINT_CHUNK_SIZE:-10}" \
  --model_workers "${MODEL_WORKERS:-1}" \
  ${SHOW_RATELIMIT:+--show_ratelimit} \
  ${VERBOSE_WORKERS:+--verbose_workers} \
  --ratelimit_log_every "${RATELIMIT_LOG_EVERY:-20}"
