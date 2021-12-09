#!/usr/bin/env python3
# coding=utf-8
# Copyright 2021 JetBrains.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Print data examples from specified holdout for a given Task or Mixture.
"""

import argparse
from typing import Optional

import gin
import seqio
import t5
import tensorflow as tf
import tensorflow_datasets as tfds
from t5.data import preprocessors

from code_t5.constants import NEWLINE
from code_t5.tasks import register_all_known_tasks


def main(
    data_dir: str,
    task_name: Optional[str],
    split: str,
    input_lengths: int,
    target_lengths: int,
    n_samples: int,
    cache_dir: str,
    vocab_path: str,
):
    if not task_name:
        print("No Task name was provided with --task/-t\nLoading all available tasks...")
        print("Available Mixtures")
        print(seqio.MixtureRegistry.names())
        print()
        print("Available Tasks")
        print(seqio.TaskRegistry.names())
        return

    print("Loading a list of the datasets (takes 10sec)")
    register_all_known_tasks(data_dir, with_test=True)
    print("Done")

    vocab = seqio.SentencePieceVocabulary(vocab_path, t5.data.DEFAULT_EXTRA_IDS)

    def decode(s):
        return vocab.decode(s.tolist()).replace(NEWLINE, "\n")

    task = seqio.TaskRegistry().get(task_name)
    with gin.unlock_config():
        gin.bind_parameter(
            "preprocessors.unsupervised.preprocessors",
            [
                preprocessors.select_random_chunk,
                preprocessors.reduce_concat_tokens,
                preprocessors.split_tokens_to_targets_length,
            ],
        )
        gin.bind_parameter("preprocessors.select_random_chunk.max_length", 65536)

    if cache_dir != "null":
        seqio.utils.add_global_cache_dirs([cache_dir])

    ds = task.get_dataset(
        split=split,
        sequence_length={"inputs": input_lengths, "targets": target_lengths},
        use_cached=cache_dir != "null",
    )
    print(f"\nsequence_length = {{'inputs': {input_lengths}, 'targets': {target_lengths}}}")
    print(f"Printing {n_samples} of {task.source.num_input_examples(split)} examples from '{split}'")
    for ex in tfds.as_numpy(ds.take(n_samples)):
        print("----")
        print(ex)
        if "inputs" in ex:
            print(f"Inputs: {tf.size(ex['inputs'])}\n'{decode(ex['inputs'])}'")
        print(f"Targets: {tf.size(ex['targets'])}\n'{decode(ex['targets'])}'")
        print("----")
        print()


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Print examples for a give Task or Mixture",
    )

    _parser.add_argument("--data_dir", type=str, default="gs://t5-codex/data", help="Path to the data directory")
    _parser.add_argument("-t", "--task", help="Name of the Task/Mixture to read")
    _parser.add_argument(
        "--split", type=str, default="validation", help="Name of the dataset split to use (train/validation)"
    )
    _parser.add_argument("-l", "--limit", type=int, default=10, help="limit number of examples to print")
    _parser.add_argument("--inputs", type=int, default=128, help="length of the Inputs")
    _parser.add_argument("--targets", type=int, default=128, help="length of the Targets")
    _parser.add_argument(
        "--cache_dir", type=str, default="gs://t5-codex/cache", help="Path to the preprocessed dataset cache"
    )
    _parser.add_argument("--vocab", type=str, default="gs://t5-codex/data", help="Path to the vocabulary file")

    _args = _parser.parse_args()
    main(
        _args.data_dir, _args.task, _args.split, _args.limit, _args.inputs, _args.targets, _args.cache_dir, _args.vocab
    )
