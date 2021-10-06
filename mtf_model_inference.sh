#!/usr/bin/env bash

set -x

for temp in 0 0.2 0.4 0.6 0.8 1; do
    time python3 ./mtf_model_inference.py -t $temp --arch 'large-512'
done
