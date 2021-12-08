#!/usr/bin/env bash

# List all pre-trained model checkoints on GSC

MODELS_DIR='gs://t5-codex/models/'
hash gsutil 2>/dev/null || { echo >&2 "Please install https://cloud.google.com/storage/docs/gsutil_install"; exit 1; }

gsutil ls "${MODELS_DIR}" |  awk -F ${MODELS_DIR} '{print $2}'