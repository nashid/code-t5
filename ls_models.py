#!/usr/bin/env python3
import os
import re

import tensorflow as tf

def sizeof_fmt(num):
    for unit in ["", "K", "M", "B"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0

print("\t{:35} {:6} {:6}/{} {}".format("Model", "tokens", "steps", "ch", "tokens_per_batch"))
print("\t----------------------")

MODELS_DIR="gs://t5-codex/models" # or "./models"
for dir in tf.io.gfile.glob(f"{MODELS_DIR}/*"):
    if not os.path.isdir(dir):
        continue
    model = os.path.basename(dir)
    if model.endswith("-top5k"):
        continue
    checkpoints = tf.io.gfile.glob(f"{MODELS_DIR}/{model}/model.ckpt-*.meta")
    checkpoints = [ int(os.path.basename(ch).split("-")[-1].removesuffix(".meta")) for ch in checkpoints ]
    checkpoints = sorted(checkpoints)

    num_checkpoints=len(checkpoints)
    steps=int(checkpoints[-1]) if num_checkpoints > 0 else 0

    batch_size=None
    with tf.io.gfile.GFile(f"{MODELS_DIR}/{model}/operative_config.gin") as f:
        config = f.read()
        m = re.findall("tokens_per_batch = (\d+)", config)
        if len(m):
            batch_size=int(m[0])
    trained_on = sizeof_fmt(steps*batch_size)

    print(f"\t{dir.removeprefix(MODELS_DIR)[1:]:35} {trained_on:6} {steps:6}/{num_checkpoints:<3} {batch_size}")