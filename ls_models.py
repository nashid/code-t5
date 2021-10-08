#!/usr/bin/env python3
import os
import re
import argparse

import tensorflow as tf

"""List pre-trained model checkpoints on GCS or local dir"""

def sizeof_fmt(num):
    for unit in ["", "K", "M", "B"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0

def main(args):
    MODELS_DIR=args.models_dir
    print(f"Listing all models at '{MODELS_DIR}'")
    print("\t{:35} {:6} {:6}/{} {}".format("Model", "tokens", "steps", "ch", "tokens_per_batch"))
    print("\t----------------------")

    for dir in tf.io.gfile.glob(f"{MODELS_DIR}/*"):
        if not (dir.startswith("gs://") or os.path.isdir(dir)):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Print examples for a give Task or Mixture', )

    parser.add_argument(
        '-m', '--models_dir',
        type=str, default='gs://t5-codex/models', # or './models'
        help='Path to the dir with sub-directories for individual checkpoints')

    args = parser.parse_args()
    main(args)
