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

import functools
import os

import seqio
import t5.data
import tensorflow_datasets as tfds
from t5.data import preprocessors
from t5.evaluation import metrics
import tensorflow as tf

TaskRegistry = seqio.TaskRegistry

def get_sentencepiece_model_path():
    return "gs://t5-codex/py5k-50.model"

vocab = seqio.SentencePieceVocabulary(get_sentencepiece_model_path(), t5.data.DEFAULT_EXTRA_IDS)

DEFAULT_OUTPUT_FEATURES = {
    "inputs":
    seqio.Feature(vocabulary=vocab,
                  add_eos=True,
                  required=False),
    "targets":
    seqio.Feature(vocabulary=vocab, add_eos=True)
}

DATA_DIR = "gs://t5-codex/data"

py_txt_path = {
    "train": os.path.join(DATA_DIR, "py-50stars-top5k-2019", "py5k-50.train-*.txt"),
    "validation": os.path.join(DATA_DIR, "py-50stars-top5k-2019", "py5k-50.test.txt")
}


def py5k_dataset_fn(split, shuffle_files=False):
    # We only have one file for each split.
    del shuffle_files

    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(py_txt_path[split])
    ds = ds.map(lambda ex: {"text": ex},
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds

def fl_preprocessor(ds):
    def _to_inputs_and_targets(ex):
        return {
            "inputs": "",
            "targets": ex,
        }
    return ds.map(_to_inputs_and_targets,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


# Prefix language modeling pretraining task used in Raffel et al., 2019.
TaskRegistry.add(
    "py_50stars_top5k_2019",
    source=seqio.TextLineDataSource(
        split_to_filepattern=py_txt_path,
        num_input_examples={"train": 170000, "validation":40815},
    ),
    preprocessors=[
        fl_preprocessor,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.prefix_lm,
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[t5.evaluation.metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)
