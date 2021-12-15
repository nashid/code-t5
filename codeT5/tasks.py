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
import tensorflow as tf

from codeT5.data import preprocessors
from codeT5.data.dataset_providers import GzTextLineDataSource

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

fl_py_50stars_top5k_txt_paths = {
    "train": os.path.join(DATA_DIR, "fl_py_50stars_top5k_2019", "py5k-50.train.txt-*"),
    "validation": os.path.join(DATA_DIR, "fl_py_50stars_top5k_2019", "py5k-50.test.txt")
}

fl_py_50stars_2019_txt_paths = {
    "train": os.path.join(DATA_DIR, "fl_py_50stars_2019", "py_50stars_2019.train.txt-*"),
    "validation": os.path.join(DATA_DIR, "fl_py_50stars_2019", "py_50stars_2019.test.txt-*")
}

fl_py_10stars_2019_txt_paths = {
    "train": os.path.join(DATA_DIR, "fl_py_10stars_2019", "py_10stars_2019.train.txt-*"),
    "validation": os.path.join(DATA_DIR, "fl_py_10stars_2019", "py_10stars_2019.test.txt-*")
}

bq_py_2016_minus_ethpy150_paths = {
    "train": os.path.join(DATA_DIR, "bq_py_2016_minus_ethpy150", "github_py_minus_ethpy150.train.*.txt"),
    "validation": os.path.join(DATA_DIR, "bq_py_2016_minus_ethpy150", "github_py_minus_ethpy150.validation.*.txt")
}

bq_py_2016_dedup_paths = {
    "train": os.path.join(DATA_DIR, "bq_py_2016_dedup", "txt", "gh_py.train.*.txt.gz"),
    "validation": os.path.join(DATA_DIR, "bq_py_2016_dedup", "txt", "gh_py.valid.txt.gz")
}

# Theoretically, there is https://www.tensorflow.org/io/api_docs/python/tfio/experimental/serialization/decode_json
# it could be interesting to add such a preprocessor, coupled with tf.strings.regex_replace(, "\n", "Ċ")
# and compare the pipeline performance.
# at_py_2020_jsonl = {
#     "train": os.path.join(DATA_DIR, "at_py_2020", "jsonl", "20211130_085041_00017_*.train.gz"),
#     "validation": os.path.join(DATA_DIR, "at_py_2020", "jsonl", "20211130_085041_00017_*.valid.gz")
#}

at_py_2020_txt = {
    "train": os.path.join(DATA_DIR, "at_py_2020", "txt", "at_py.train.*.txt.gz"),
    "validation": os.path.join(DATA_DIR, "at_py_2020", "txt", "at_py.valid.txt.gz")
}


def py5k_dataset_fn(split, shuffle_files=False):
    # In case when we only have one file for each split.
    del shuffle_files

    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(fl_py_50stars_top5k_txt_paths[split])
    ds = ds.map(lambda ex: {"text": ex},
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


# FullLine dataset for Prefix language modeling pretraining, as in Raffel et al., 2019.
TaskRegistry.add(
    "fl_py_50stars_top5k_2019",
    source=seqio.TextLineDataSource(
        split_to_filepattern=fl_py_50stars_top5k_txt_paths,
        num_input_examples={"train": 170000, "validation":40815},
    ),
    preprocessors=[
        preprocessors.fl_preprocessor,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.unsupervised,
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[t5.evaluation.metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)

TaskRegistry.add(
    "fl_py_50stars_2019",
    source=seqio.TextLineDataSource(
        split_to_filepattern=fl_py_50stars_2019_txt_paths,
        num_input_examples={"train": 700000, "validation":352596},
    ),
    preprocessors=[
        preprocessors.fl_preprocessor,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.unsupervised,
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[t5.evaluation.metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)

TaskRegistry.add(
    "fl_py_10stars_2019",
    source=seqio.TextLineDataSource(
        split_to_filepattern=fl_py_10stars_2019_txt_paths,
        num_input_examples={"train": 1100000, "validation":559698},
    ),
    preprocessors=[
        preprocessors.fl_preprocessor,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.unsupervised,
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[t5.evaluation.metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)

# BigQuery Github dataset
TaskRegistry.add(
    "bq_py_2016_minus_ethpy150",
    source=seqio.TextLineDataSource(
        split_to_filepattern=bq_py_2016_minus_ethpy150_paths,
        num_input_examples={"train": 5884757, "validation": 1292044},
    ),
    preprocessors=[
        preprocessors.fl_preprocessor,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.unsupervised,
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[t5.evaluation.metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)

# BigQuery Github re-splited and de-duplicated
TaskRegistry.add(
    "bq_py_2016_dedup",
    source=GzTextLineDataSource(
        split_to_filepattern=bq_py_2016_dedup_paths,
        num_input_examples={"train": 3800000, "validation": 20448},
    ),
    preprocessors=[
        preprocessors.fl_preprocessor,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.unsupervised,
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[t5.evaluation.metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)

# Athena Github dataset
TaskRegistry.add(
    "at_py_2020", # at_java_2020 is coming...
    source=GzTextLineDataSource(
        split_to_filepattern=at_py_2020_txt,
        num_input_examples={"train": 3788418, "validation": 10373},
    ),
    preprocessors=[
        preprocessors.fl_preprocessor,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.unsupervised,
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[t5.evaluation.metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)
