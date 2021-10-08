#!/usr/bin/env bash

# Downloads a given model from GSC to ./models

if [[ $# -ne 2 ]] ; then
    echo 'Usage: ./cp_model.sh <model_name> <checkpoint>'
    echo 'Run ./ls_models.py to get those values'
    exit 1
fi

hash gsutil 2>/dev/null || { echo >&2 "Please install https://cloud.google.com/storage/docs/gsutil_install"; exit 1; }

model=$1
checkpoint=$2

mkdir -p "models/${model}"
gsutil cp "gs://t5-codex/models/${model}/operative_config.gin" "models/${model}"
gsutil cp "gs://t5-codex/models/${model}/checkpoint" "models/${model}"
gsutil -m cp "gs://t5-codex/models/${model}/model.ckpt-${checkpoint}.*" "models/${model}"
echo "models/${model}"
ls -la "models/${model}"