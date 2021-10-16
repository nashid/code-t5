#!/usr/bin/env bash

set -x
set -e

for temp in 0 0.2 0.4 0.6 0.8 1; do
    time python3 ./mtf_model_inference.py -t $temp -m 'gs://t5-codex/models' --arch 'arch-t5.1.1-prefix_lm-1k'
    echo "---------"
done
