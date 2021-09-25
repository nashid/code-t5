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

import argparse

import tensorflow as tf



def main(args):
    if not args.task:
        print("No Task name was provided wiht --task/-t\nLoading available tasks...")
        import seqio
        import codeT5.tasks  # pylint:disable=unused-import
        print("Available Mixtures")
        print(seqio.MixtureRegistry.names())
        print()
        print("Available Tasks")
        print(seqio.TaskRegistry.names())
        return

    print("Loading a list of the datasets (takes 10sec)")
    import seqio
    import t5
    import tensorflow_datasets as tfds
    import codeT5.tasks  # pylint:disable=unused-import
    print("Done")

    vocab = seqio.SentencePieceVocabulary("data/py5k-50.model",
                                          t5.data.DEFAULT_EXTRA_IDS)
    def decode(s):
        return vocab.decode(s.tolist()).replace('Ċ', '\n')

    task = seqio.TaskRegistry().get(args.task)
    ds = task.get_dataset(split=args.split,
                          sequence_length={
                              "inputs": args.inputs,
                              "targets": args.targets
                          })  #4k samples
    print(f"\nsequence_length = {{'inputs': {args.inputs}, 'targets': {args.targets}}}")
    print(f"{args.limit} of {task.source.num_input_examples(args.split)} examples from '{args.split}'")
    for ex in tfds.as_numpy(ds.take(args.limit)):
        print("----")
        print(ex)
        print(f"Intputs: {tf.size(ex['inputs'])}\n'{decode(ex['inputs'])}'")
        print(f"Targets: {tf.size(ex['targets'])}\n'{decode(ex['targets'])}'")
        print("----")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Print examples for a give Task or Mixture', )

    parser.add_argument(
        '-t', '--task',
        help="Name of the Task/Mixture to read")
    parser.add_argument(
        '--split', type=str, default="validation",
        help="Name of the dataset split to use (train/validation)")
    parser.add_argument(
        '-l', '--limit', type=int, default=10,
        help="limit number of examples to print")
    parser.add_argument(
        '--inputs', type=int, default=128,
        help="length of the Inputs")
    parser.add_argument(
        '--targets', type=int, default=128,
        help="length of the Targets")



    args = parser.parse_args()
    main(args)
