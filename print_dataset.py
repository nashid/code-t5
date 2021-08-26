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

from codeT5.tasks import py5k_dataset_fn as dataset_fn
import tensorflow_datasets as tfds

limit = 2

print("A few raw validation examples...")
for ex in tfds.as_numpy(dataset_fn("validation").take(limit)):
  print(ex)

print()

#----

import codeT5.tasks # pylint:disable=unused-import
import t5.data.mixtures;

print(t5.data.MixtureRegistry.names())
print()
print(t5.data.TaskRegistry.names())
print()

#----
import seqio;

vocab = seqio.SentencePieceVocabulary("data/py5k-50.model", t5.data.DEFAULT_EXTRA_IDS)
task = seqio.TaskRegistry().get("py5k_prefix_lm")
ds = task.get_dataset(split="validation", sequence_length={"inputs": 128, "targets": 128}) #4k samples

print("A few raw validation examples...")
for ex in tfds.as_numpy(ds.take(limit)):
    print(ex)
    print()
    print(vocab.decode_tf(ex['inputs']))
    print(vocab.decode_tf(ex['targets']))
    print()

