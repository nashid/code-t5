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

import os

import seqio
import t5.data
from t5.data import preprocessors
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

py_50_top5k_txt_path = {
    "train": os.path.join(DATA_DIR, "py-50stars-top5k-2019", "py5k-50.train.txt-*"),
    "validation": os.path.join(DATA_DIR, "py-50stars-top5k-2019", "py5k-50.test.txt")
}

py_50_txt_path = {
    "train": os.path.join(DATA_DIR, "py-50stars-2019", "py_50stars_2019.train.txt-*"),
    "validation": os.path.join(DATA_DIR, "py-50stars-2019", "py_50stars_2019.test.txt-*")
}

py_10_txt_path = {
    "train": os.path.join(DATA_DIR, "py-10stars-2019", "py_10stars_2019.train.txt-*"),
    "validation": os.path.join(DATA_DIR, "py-10stars-2019", "py_10stars_2019.test.txt-*")
}

github_python_minus_ethpy150open_path = {
    "train": os.path.join(DATA_DIR, "github_python_minus_ethpy150open_dedup", "github_py_minus_ethpy150.train.*.txt"),
    "validation": os.path.join(DATA_DIR, "github_python_minus_ethpy150open_dedup", "github_py_minus_ethpy150.validation.*.txt")
}



def py5k_dataset_fn(split, shuffle_files=False):
    # In case when we only have one file for each split.
    del shuffle_files

    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(py_50_top5k_txt_path[split])
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


# FullLine dataset for Prefix language modeling pretraining, as in Raffel et al., 2019.
TaskRegistry.add(
    "fl_py_50stars_top5k_2019",
    source=seqio.TextLineDataSource(
        split_to_filepattern=py_50_top5k_txt_path,
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

TaskRegistry.add(
    "fl_py_50stars_2019",
    source=seqio.TextLineDataSource(
        split_to_filepattern=py_50_txt_path,
        num_input_examples={"train": 700000, "validation":352596},
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

TaskRegistry.add(
    "fl_py_10stars_2019",
    source=seqio.TextLineDataSource(
        split_to_filepattern=py_10_txt_path,
        num_input_examples={"train": 1100000, "validation":559698},
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

# BigQuery Github dataset
TaskRegistry.add(
    "bq_py_2016_minus_ethpy150",
    source=seqio.TextLineDataSource(
        split_to_filepattern=github_python_minus_ethpy150open_path,
        num_input_examples={"train": 5884757, "validation": 1292044},
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
