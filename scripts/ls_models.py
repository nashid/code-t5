#!/usr/bin/env python3
"""List pre-trained model checkpoints on GCS or local dir"""
import os
import re
import argparse

import tensorflow as tf


def sizeof_fmt(num):
    for unit in ["", "K", "M", "B"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0


def remove_suffix(s, suffix):
    if suffix and s.endswith(suffix):
        return s[: -len(suffix)]
    return s


def remove_prefix(s, prefix):
    if prefix and s.startswith(prefix):
        return s[len(prefix) :]
    return s


def main(model_dir: str):
    print(f"Listing all models at '{model_dir}'")
    print("\t{:35} {:6} {:>6}/{} {}".format("Model", "tokens", "steps", "ch", "tokens_per_batch"))
    print("\t----------------------")

    for dir_name in tf.io.gfile.glob(f"{model_dir}/*"):
        if not (dir_name.startswith("gs://") or os.path.isdir(dir_name)):
            continue
        model = os.path.basename(dir_name)
        if model.endswith("-top5k"):
            continue
        checkpoints = tf.io.gfile.glob(f"{model_dir}/{model}/model.ckpt-*.meta")
        checkpoints = [int(remove_suffix(os.path.basename(ch).split("-")[-1], ".meta")) for ch in checkpoints]
        checkpoints = sorted(checkpoints)

        num_checkpoints = len(checkpoints)
        steps = int(checkpoints[-1]) if num_checkpoints > 0 else 0

        batch_size = 0
        with tf.io.gfile.GFile(f"{model_dir}/{model}/operative_config.gin") as f:
            config = f.read()
            m = re.findall(r"tokens_per_batch = (\d+)", config)
            if len(m):
                batch_size = int(m[0])
        trained_on = sizeof_fmt(steps * batch_size)

        print(
            f"\t{remove_prefix(dir_name, model_dir)[1:]:35} {trained_on:6} {steps:6}/{num_checkpoints:<3} {batch_size}"
        )


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="List all saved models")

    _parser.add_argument(
        "-m",
        "--models_dir",
        type=str,
        default="gs://t5-codex/models",  # or './models'
        help="Path to the dir with sub-directories for individual checkpoints",
    )

    _args = _parser.parse_args()
    main(_args.models_dir)
