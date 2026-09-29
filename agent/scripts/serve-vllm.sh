#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local directory or Hugging Face model id}"

MODEL_PROFILE="${MODEL_PROFILE:-local}"
if [[ "${MODEL_PROFILE}" != "local" ]]; then
  echo "MODEL_PROFILE must be local" >&2
  exit 2
fi
DEFAULT_SERVED_NAME="local-model"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME:-${DEFAULT_SERVED_NAME}}" \
  --host "${MODEL_HOST:-127.0.0.1}" \
  --port "${MODEL_PORT:-8000}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --dtype "${MODEL_DTYPE:-bfloat16}" \
  --max-model-len "${MODEL_CONTEXT_WINDOW:-65536}" \
  --limit-mm-per-prompt "${MODEL_MM_LIMIT:-image=16}"
